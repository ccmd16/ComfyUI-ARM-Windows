"""Opt-in SD 1.5 VAE decoding through a precompiled Snapdragon QNN model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from comfy import qnn_runtime
from comfy.snapdragon_qnn_backend import prebuilt_checkpoint_matches

_lock = Lock()
_session = None
_session_path = None
_session_error = None
_quant_params = None


def _get_session(path: str):
    global _session, _session_path, _session_error, _quant_params
    path = str(Path(path).resolve(strict=True))
    with _lock:
        if _session_path == path and _session is not None:
            return _session, _quant_params
        if _session_path == path and _session_error is not None:
            raise RuntimeError(_session_error)
        _session_path = path
        try:
            model = onnx.load(path, load_external_data=False)
            initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
            params = tuple(float(initializers[name]) for name in ("latent_scale", "latent_zp", "image_scale", "image_zp"))
            handle = qnn_runtime.build_session_handle(path, backend="htp", strict_backend=True)
            if handle.input_names != ["latent"] or handle.output_names != ["image"]:
                raise ValueError("Unexpected QNN SD 1.5 VAE input or output")
            _session, _quant_params, _session_error = handle, params, None
            logging.info("Snapdragon QNN SD 1.5 VAE ready: %s", qnn_runtime.session_summary(handle))
            return _session, _quant_params
        except Exception as exc:
            _session_error = str(exc)
            raise


def try_decode(vae, latent: torch.Tensor) -> torch.Tensor | None:
    model_path = os.environ.get("COMFYUI_QNN_SD15_VAE_ONNX")
    if not model_path or latent.ndim != 4 or tuple(latent.shape[1:]) != (4, 64, 64):
        return None
    patcher = getattr(vae, "patcher", None)
    if patcher is None or getattr(patcher, "patches", None) or getattr(patcher, "object_patches", None):
        return None
    if not prebuilt_checkpoint_matches(getattr(vae, "_qnn_checkpoint_path", None)):
        return None
    try:
        handle, (input_scale, input_zero, output_scale, output_zero) = _get_session(model_path)
        values = latent.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(values).all():
            return None
        output = []
        for item in values:
            # ComfyUI sampler returns SD1.5 latents after dividing by 0.18215;
            # the Qualcomm decoder contains that division internally.
            nhwc = np.transpose(item[None] * np.float32(0.18215), (0, 2, 3, 1))
            q = np.clip(np.rint(nhwc / input_scale + input_zero), 0, 65535).astype(np.uint16)
            pixels = handle.session.run(["image"], {"latent": q})[0].astype(np.float32)
            output.append(np.clip((pixels - output_zero) * output_scale, 0.0, 1.0))
        return torch.from_numpy(np.concatenate(output, axis=0))
    except Exception as exc:
        logging.warning("Snapdragon QNN VAE unavailable; falling back to original VAE: %s", exc)
        return None
