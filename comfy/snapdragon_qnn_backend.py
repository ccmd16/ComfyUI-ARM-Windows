from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import folder_paths
from comfy import qnn_runtime
from comfy.windows_arm import preferred_windows_arm_runtime

try:
    from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from onnxruntime.quantization.execution_providers import qnn as qnn_quant
except Exception as exc:  # pragma: no cover - host/package dependent
    CalibrationMethod = None
    QuantType = None
    CalibrationDataReader = object  # type: ignore[assignment]
    quantize = None
    qnn_quant = None
    onnxruntime_quantization_import_error = exc
else:
    onnxruntime_quantization_import_error = None

try:
    import onnx
except Exception as exc:  # pragma: no cover - host/package dependent
    onnx = None
    onnxruntime_onnx_import_error = exc
else:
    onnxruntime_onnx_import_error = None

QNN_UNET_DISABLE_ENV = "COMFYUI_DISABLE_QNN_UNET"
QNN_UNET_ENABLE_ENV = "COMFYUI_ENABLE_QNN_UNET"
SUPPORTED_MODEL_TYPES = {"EPS", "V_PREDICTION", "V_PREDICTION_EDM"}
_PREBUILT_CHECKPOINT_HASHES: dict[tuple[str, int, int], str] = {}


def prebuilt_checkpoint_matches(checkpoint_path: Any) -> bool:
    expected = os.environ.get("COMFYUI_QNN_PREBUILT_CHECKPOINT_SHA256", "").lower()
    if not expected or not checkpoint_path:
        return False
    try:
        path = Path(checkpoint_path).resolve(strict=True)
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        if cache_key not in _PREBUILT_CHECKPOINT_HASHES:
            with path.open("rb") as checkpoint:
                _PREBUILT_CHECKPOINT_HASHES[cache_key] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        return _PREBUILT_CHECKPOINT_HASHES[cache_key] == expected
    except (OSError, TypeError, ValueError):
        return False


def _prebuilt_matches_model(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> bool:
    """Do not silently apply a fixed UNet to a different checkpoint or LoRA."""
    if tuple(xc.shape) != (1, 4, 64, 64) or tuple(t.shape) != (1,) or tuple(context.shape) != (1, 77, 768):
        return False
    patcher = getattr(model, "current_patcher", None)
    if patcher is None or getattr(patcher, "patches", None) or getattr(patcher, "object_patches", None):
        return False
    cached_init = getattr(patcher, "cached_patcher_init", None)
    if not cached_init or not cached_init[1]:
        return False
    return prebuilt_checkpoint_matches(cached_init[1][0])


def qnn_unet_runtime_enabled() -> bool:
    if os.environ.get(QNN_UNET_DISABLE_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return False

    if os.environ.get(QNN_UNET_ENABLE_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return True

    return (
        preferred_windows_arm_runtime() == "qnn"
        and qnn_runtime.is_qnn_npu_available()
        and qnn_runtime.get_runtime_hint() == "qnn"
        and quantize is not None
        and qnn_quant is not None
        and onnxruntime_onnx_import_error is None
    )


def _tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str]:
    return tuple(int(dim) for dim in value.shape), str(value.dtype)


def _build_model_signature(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> str:
    patcher = getattr(model, "current_patcher", None)
    patcher_uuid = getattr(patcher, "patches_uuid", None)
    cached_init = getattr(patcher, "cached_patcher_init", None) if patcher is not None else None

    signature_parts = [
        model.__class__.__module__,
        model.__class__.__qualname__,
        getattr(getattr(model, "model_type", None), "name", "UNKNOWN"),
        str(patcher_uuid) if patcher_uuid is not None else "no-patcher",
        repr(cached_init[0].__name__ if cached_init and cached_init[0] is not None else None),
        repr(cached_init[1] if cached_init and len(cached_init) > 1 else None),
        repr(_tensor_signature(xc)),
        repr(_tensor_signature(t)),
        repr(_tensor_signature(context)),
        os.environ.get("COMFYUI_QNN_EXPORT_CACHE_TAG", ""),
    ]
    return hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()


def _build_cache_directory(model_signature: str) -> Path:
    cache_root = Path(folder_paths.get_temp_directory()) / "snapdragon-qnn-unet"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / model_signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _to_numpy_tensor(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy()


def _make_calibration_samples(
    xc: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    input_names: tuple[str, ...],
) -> list[dict[str, np.ndarray]]:
    samples: list[dict[str, np.ndarray]] = []
    xc_variants = [xc, xc * 0.9, xc * 1.1]
    t_variants = [t, t * 0.5, t * 1.5]
    context_variants = [context, context * 0.9, context * 1.1]

    if os.environ.get("COMFYUI_QNN_CALIBRATE_W8A16") == "1":
        # SD1.x uses timesteps from 0 to 999 throughout a denoising run.
        # Calibrate across that range instead of only around the first step.
        t_variants = [torch.full_like(t, value) for value in (0, 250, 500, 750, 999)]
        xc_variants = [xc * scale for scale in (0.8, 0.9, 1.0, 1.1, 1.2)]
        context_variants = [context] * len(t_variants)

    for i in range(len(t_variants)):
        sample: dict[str, np.ndarray] = {}
        if "xc" in input_names:
            sample["xc"] = _to_numpy_tensor(xc_variants[i])
        if "t" in input_names:
            sample["t"] = _to_numpy_tensor(t_variants[i])
        if "context" in input_names:
            sample["context"] = _to_numpy_tensor(context_variants[i])
        samples.append(sample)

    return samples


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self._samples = samples
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._samples):
            return None
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def __len__(self) -> int:
        return len(self._samples)

    def set_range(self, start_index: int, end_index: int):
        self._samples = self._samples[start_index:end_index]
        self._index = 0


class _OnnxExportWrapper(torch.nn.Module):
    def __init__(self, diffusion_model: torch.nn.Module):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.diffusion_model(
            xc,
            t,
            context=context,
            control=None,
            transformer_options={},
        )


@dataclass(slots=True)
class SnapdragonQnnUnetBackend:
    model_signature: str
    cache_dir: Path
    export_path: Path
    preprocessed_path: Path
    quantized_path: Path
    session_id: str
    providers: list[str]
    input_names: tuple[str, ...]
    shape_signature: tuple[tuple[int, ...], tuple[str, str], tuple[str, str], tuple[str, str]]
    official_quantization: dict[str, tuple[float, float]] | None = None

    def close(self) -> None:
        try:
            qnn_runtime.close_session(self.session_id)
        except Exception:
            pass

    def run(self, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.official_quantization is not None:
            def quantize(name: str, value: np.ndarray) -> np.ndarray:
                scale, zero = self.official_quantization[name]
                return np.clip(np.rint(value / scale + zero), 0, 65535).astype(np.uint16)

            feed = {
                "latent": quantize("latent", np.transpose(_to_numpy_tensor(xc), (0, 2, 3, 1))),
                "timestep": quantize("timestep", _to_numpy_tensor(t).reshape(1, 1)),
                "text_emb": quantize("text_emb", _to_numpy_tensor(context)),
            }
            result = qnn_runtime.run_session(self.session_id, feed, output_name="output_latent")
            scale, zero = self.official_quantization["output_latent"]
            output = (result.numpy().astype(np.float32) - zero) * scale
            return torch.from_numpy(np.ascontiguousarray(np.transpose(output, (0, 3, 1, 2))))
        canonical_inputs = {
            "xc": xc,
            "t": t,
            "context": context,
        }
        feed = {name: canonical_inputs[name] for name in self.input_names if name in canonical_inputs}
        return qnn_runtime.run_session(
            self.session_id,
            feed,
        )


def _export_model(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor, export_path: Path) -> None:
    wrapper = _OnnxExportWrapper(model.diffusion_model)
    wrapper.eval()

    dynamic_axes = {
        "xc": {0: "batch", 2: "height", 3: "width"},
        "t": {0: "batch"},
        "context": {0: "batch", 1: "tokens"},
        "model_output": {0: "batch", 2: "height", 3: "width"},
    }
    export_kwargs = dict(
        args=(xc, t, context),
        f=str(export_path),
        input_names=["xc", "t", "context"],
        output_names=["model_output"],
        # 16-bit QDQ requires opset 21; exporting it directly avoids the ONNX
        # version converter trying to serialize multi-GB weights into a proto.
        opset_version=21 if os.environ.get("COMFYUI_QNN_CALIBRATE_W8A16") == "1" else 18,
        dynamic_axes=None if os.environ.get("COMFYUI_QNN_STATIC_EXPORT") == "1" else dynamic_axes,
        external_data=True,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )

    with torch.no_grad():
        torch.onnx.export(wrapper, **export_kwargs)


def _preprocess_model(export_path: Path, preprocessed_path: Path) -> Path:
    if qnn_quant is None:
        return export_path

    try:
        created = qnn_quant.qnn_preprocess_model(
            export_path,
            preprocessed_path,
            fuse_layernorm=True,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            external_data_location=preprocessed_path.with_suffix(".data").name,
        )
    except Exception as exc:
        logging.debug("QNN preprocessing failed, using the exported ONNX model directly: %s", exc)
        return export_path

    if created and preprocessed_path.exists():
        return preprocessed_path
    return export_path


def _quantize_model(
    source_path: Path,
    quantized_path: Path,
    xc: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    input_names: tuple[str, ...],
) -> None:
    if quantize is None or qnn_quant is None:
        raise RuntimeError("onnxruntime quantization is unavailable")

    reader = _CalibrationReader(_make_calibration_samples(xc, t, context, input_names))
    quant_config = qnn_quant.get_qnn_qdq_config(
        source_path,
        reader,
        activation_type=QuantType.QUInt16 if os.environ.get("COMFYUI_QNN_CALIBRATE_W8A16") == "1" else QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibration_providers=["CPUExecutionProvider"],
    )
    quantize(source_path, quantized_path, quant_config)


def _build_backend(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> SnapdragonQnnUnetBackend:
    model_signature = _build_model_signature(model, xc, t, context)
    cache_dir = _build_cache_directory(model_signature)
    export_path = cache_dir / "denoiser.onnx"
    preprocessed_path = cache_dir / "denoiser.preprocessed.onnx"
    quantized_path = cache_dir / "denoiser.qdq.onnx"
    official = os.environ.get("COMFYUI_QNN_OFFICIAL_UNET_ONNX")
    prebuilt = official or os.environ.get("COMFYUI_QNN_PREBUILT_ONNX")
    if prebuilt:
        quantized_path = Path(prebuilt).resolve(strict=True)
    source_path = export_path
    input_names: tuple[str, ...] = ("xc", "t", "context")

    if not quantized_path.exists():
        _export_model(model, xc, t, context, export_path)
        source_path = _preprocess_model(export_path, preprocessed_path)

        if os.environ.get("COMFYUI_QNN_EXPORT_ONLY") == "1":
            raise RuntimeError(f"Export-only mode completed: {source_path}")

        if onnx is None:
            raise RuntimeError("onnx is unavailable")

        # Only the graph inputs are needed here; loading external weights would
        # retain a second multi-gigabyte copy during calibration.
        exported_model = onnx.load(str(source_path), load_external_data=False)
        input_names = tuple(input_value.name for input_value in exported_model.graph.input)
        _quantize_model(source_path, quantized_path, xc, t, context, input_names)
    else:
        if onnx is None:
            raise RuntimeError("onnx is unavailable")
        exported_model = onnx.load(str(quantized_path), load_external_data=False)
        input_names = tuple(input_value.name for input_value in exported_model.graph.input)

    official_quantization = None
    if official:
        from onnx import numpy_helper
        if input_names != ("latent", "timestep", "text_emb"):
            raise RuntimeError(f"Unexpected Qualcomm SD 1.5 UNet inputs: {input_names}")
        values = {item.name: numpy_helper.to_array(item) for item in exported_model.graph.initializer}
        official_quantization = {
            name: (float(values[name + "_scale"]), float(values[name + "_zp"]))
            for name in ("latent", "timestep", "text_emb", "output_latent")
        }

    backend_handle = qnn_runtime.build_session_handle(
        str(quantized_path),
        backend="auto",
        allow_cpu_fallback=True,
        strict_backend=False,
    )

    if qnn_runtime.QNN_PROVIDER_NAME not in backend_handle.providers:
        qnn_runtime.close_session(backend_handle.session_id)
        raise RuntimeError(
            f"QNN session was built without the QNN provider. providers={backend_handle.providers}"
        )

    session_id = qnn_runtime.register_session(backend_handle)
    shape_signature = (
        tuple(int(dim) for dim in xc.shape),
        _tensor_signature(t),
        _tensor_signature(context),
        (str(xc.dtype), str(context.dtype)),
    )
    logging.info("Snapdragon QNN UNet backend ready: %s", qnn_runtime.session_summary(backend_handle))
    return SnapdragonQnnUnetBackend(
        model_signature=model_signature,
        cache_dir=cache_dir,
        export_path=export_path,
        preprocessed_path=preprocessed_path,
        quantized_path=quantized_path,
        session_id=session_id,
        providers=backend_handle.providers,
        input_names=input_names,
        shape_signature=shape_signature,
        official_quantization=official_quantization,
    )



def _supports_standard_unet_call(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor, c_concat=None, control=None, transformer_options=None, extra_conds=None) -> bool:
    # Already a QNN-backed SDXL wrapper; tracing it would capture runtime
    # results as constants and needlessly nest a second QNN backend.
    if getattr(model, "_qnn_fixed_sdxl", False):
        return False
    if not qnn_unet_runtime_enabled():
        return False

    if not torch.is_tensor(xc) or not torch.is_tensor(t) or not torch.is_tensor(context):
        return False

    if xc.ndim != 4:
        return False

    if c_concat is not None or control is not None:
        return False

    if transformer_options:
        if not isinstance(transformer_options, dict):
            return False
        sampler_metadata = {"cond_or_uncond", "uuids", "sigmas", "sample_sigmas"}
        behavioral_options = {
            key: value for key, value in transformer_options.items()
            if key not in sampler_metadata and value
        }
        if behavioral_options:
            return False

    if extra_conds:
        return False

    model_type_name = getattr(getattr(model, "model_type", None), "name", None)
    if model_type_name not in SUPPORTED_MODEL_TYPES:
        return False

    return True


def _get_backend(model: Any, xc: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> SnapdragonQnnUnetBackend | None:
    backend = getattr(model, "_snapdragon_qnn_backend", None)
    disabled = getattr(model, "_snapdragon_qnn_backend_disabled", False)
    if disabled:
        return None

    shape_signature = (
        tuple(int(dim) for dim in xc.shape),
        _tensor_signature(t),
        _tensor_signature(context),
        (str(xc.dtype), str(context.dtype)),
    )
    model_signature = _build_model_signature(model, xc, t, context)

    if backend is not None:
        if backend.model_signature == model_signature and backend.shape_signature == shape_signature:
            return backend
        try:
            backend.close()
        except Exception:
            pass

    try:
        backend = _build_backend(model, xc, t, context)
    except Exception as exc:
        logging.warning("Snapdragon QNN UNet backend unavailable; falling back to PyTorch. Details: %s", exc)
        setattr(model, "_snapdragon_qnn_backend_disabled", True)
        setattr(model, "_snapdragon_qnn_backend_error", str(exc))
        return None

    setattr(model, "_snapdragon_qnn_backend", backend)
    setattr(model, "_snapdragon_qnn_backend_disabled", False)
    setattr(model, "_snapdragon_qnn_backend_error", None)
    return backend


def try_apply_qnn_backend(
    model: Any,
    xc: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    c_concat=None,
    control=None,
    transformer_options=None,
    extra_conds=None,
) -> torch.Tensor | None:
    if (os.environ.get("COMFYUI_QNN_PREBUILT_ONNX") or os.environ.get("COMFYUI_QNN_OFFICIAL_UNET_ONNX")) and not _prebuilt_matches_model(model, xc, t, context):
        if not getattr(model, "_qnn_prebuilt_mismatch_reported", False):
            logging.warning("Prebuilt QNN model does not match checkpoint, patches, or input shape; using PyTorch.")
            model._qnn_prebuilt_mismatch_reported = True
        return None
    if not _supports_standard_unet_call(
        model,
        xc,
        t,
        context,
        c_concat=c_concat,
        control=control,
        transformer_options=transformer_options,
        extra_conds=extra_conds,
    ):
        return None

    backend = _get_backend(model, xc, t, context)
    if backend is None:
        return None

    try:
        output = backend.run(xc, t, context)
    except Exception as exc:
        logging.warning("Snapdragon QNN UNet execution failed; disabling QNN backend and falling back to PyTorch. Details: %s", exc)
        try:
            backend.close()
        except Exception:
            pass
        setattr(model, "_snapdragon_qnn_backend_disabled", True)
        setattr(model, "_snapdragon_qnn_backend_error", str(exc))
        return None

    return output


def describe_model_backend(model: Any) -> str:
    backend = getattr(model, "_snapdragon_qnn_backend", None)
    if backend is None:
        disabled = getattr(model, "_snapdragon_qnn_backend_disabled", False)
        if disabled:
            return f"disabled: {getattr(model, '_snapdragon_qnn_backend_error', 'unknown error')}"
        return "uninitialized"
    return f"session={backend.session_id}; providers={backend.providers}; cache={backend.cache_dir}"
