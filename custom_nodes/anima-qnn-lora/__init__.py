"""Load and chain ordinary Anima LoRAs through independent QNN graph inputs."""

import math
import os

import numpy as np

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.utils
import folder_paths
from comfy.anima_qnn import LORA_RANK, lora_slots
from comfy.weight_adapter.lora import LoRAAdapter


class AnimaAdapter:
    def __init__(self, model, state, strength):
        key_map = comfy.lora.model_lora_keys_unet(model, {})
        state = comfy.lora_convert.convert_lora(state)
        patches = comfy.lora.load_lora(state, key_map, log_missing=False)
        if state and not patches:
            raise ValueError("No Anima weights matched this LoRA.")

        zeros = {}
        used = set()
        feeds = []

        def unpack(key, patch, layer, rank_limit=None):
            if not isinstance(patch, LoRAAdapter):
                raise ValueError(f"Anima QNN supports ordinary linear LoRA only: {key}")
            up, down, alpha, mid, dora, reshape = patch.weights
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError(f"Anima QNN requires two linear LoRA matrices: {key}")
            rank = down.shape[0]
            if (
                any(x is not None for x in (mid, dora, reshape))
                or rank == 0 or (rank_limit is not None and rank > rank_limit)
                or down.shape != (rank, layer.in_features)
                or up.shape != (layer.out_features, rank)
            ):
                raise ValueError(f"Unsupported LoRA shape or rank: {key}")
            gain = strength if alpha is None else strength * alpha / rank
            if not math.isfinite(gain):
                raise ValueError(f"Non-finite LoRA scale: {key}")
            used.update(patch.loaded_keys)
            if alpha is not None:
                used.update(k for k in state if k.endswith(".alpha") and key_map.get(k[:-6]) == key)
            return down, up, rank, gain

        for block_index, block in enumerate(model.diffusion_model.blocks):
            block_feed = {}
            for slot, (path, layer) in enumerate(lora_slots(block)):
                key = f"diffusion_model.blocks.{block_index}.{path}.weight"
                shapes = ((layer.in_features, LORA_RANK), (LORA_RANK, layer.out_features))
                for shape in shapes:
                    if shape not in zeros:
                        zeros[shape] = np.zeros(shape, dtype=np.float32)
                down, up = (zeros[shape] for shape in shapes)
                if key in patches:
                    down_weight, up_weight, rank, gain = unpack(key, patches.pop(key), layer, LORA_RANK)
                    down, up = (a.copy() for a in (down, up))
                    down[:, :rank] = down_weight.T.float().numpy()
                    up[:rank, :] = up_weight.T.float().numpy() * gain
                    if not (np.isfinite(down).all() and np.isfinite(up).all()):
                        raise ValueError(f"Non-finite LoRA weights: {key}")
                block_feed[f"lora_{slot}_down"] = down
                block_feed[f"lora_{slot}_up"] = up
            feeds.append(block_feed)
        self.bridge_patches = {}
        for path, layer in lora_slots(model.diffusion_model.llm_adapter):
            key = f"diffusion_model.llm_adapter.{path}.weight"
            if key in patches:
                patch = patches.pop(key)
                down, up, _, _ = unpack(key, patch, layer)
                if not (down.isfinite().all() and up.isfinite().all()):
                    raise ValueError(f"Non-finite LoRA weights: {key}")
                self.bridge_patches[key] = patch
        unmatched = set(state) - used
        if patches or unmatched:
            names = sorted(str(k) for k in (*patches, *unmatched))[:5]
            raise ValueError("Anima QNN LoRA cannot apply these weights: " + ", ".join(names))
        self.feeds = tuple(feeds)
        self.has_weights = bool(used) and strength != 0


class AnimaAdapterStack:
    """Combine chainable LoRAs while keeping their QNN rank slots separate."""

    def __init__(self, adapters):
        self.adapters = tuple(adapters)
        if not self.adapters:
            raise ValueError("Anima QNN adapter stack cannot be empty")

        block_count = len(self.adapters[0].feeds)
        if any(len(adapter.feeds) != block_count for adapter in self.adapters):
            raise ValueError("Anima QNN LoRAs have incompatible block layouts")

        feeds = []
        for block_index in range(block_count):
            block_feed = {}
            for adapter_index, adapter in enumerate(self.adapters):
                for name, value in adapter.feeds[block_index].items():
                    _, slot, kind = name.split("_")
                    block_feed[f"lora_{adapter_index}_{slot}_{kind}"] = value
            feeds.append(block_feed)
        self.feeds = tuple(feeds)
        self.has_weights = any(adapter.has_weights for adapter in self.adapters)
        self.patch_revision = None


class AnimaQNNLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (["(none)", *folder_paths.get_filename_list("loras")],),
            "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = "Chain several nodes to apply multiple Anima LoRAs. Each image LoRA must be a standard linear LoRA with rank at most 32."

    def load(self, model, lora_name, strength):
        if os.environ.get("COMFYUI_ANIMA_QNN") != "1":
            raise RuntimeError("Start ComfyUI using start.cmd with Anima QNN enabled to load QNN LoRA.")
        if not isinstance(model.model, comfy.model_base.Anima):
            raise ValueError("Anima QNN LoRA requires an Anima diffusion model.")
        options = model.model_options["transformer_options"]
        previous = options.get("anima_qnn_adapter")
        if previous is None:
            if model.patches:
                raise ValueError("Connect the first Anima QNN LoRA node to an unpatched model.")
            previous_adapters = ()
        else:
            if not isinstance(previous, AnimaAdapterStack):
                raise ValueError("The model already has an unsupported Anima QNN adapter.")
            if model.patches_uuid != previous.patch_revision:
                raise ValueError("Chain Anima QNN LoRA nodes directly; another model patch changed this model.")
            previous_adapters = previous.adapters

        if lora_name == "(none)" or strength == 0:
            return (model,)

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        weights = comfy.utils.load_torch_file(path, safe_load=True)
        result = model.clone()
        current = AnimaAdapter(model.model, weights, strength)
        if current.bridge_patches:
            applied = result.add_patches(current.bridge_patches, strength)
            if set(applied) != set(current.bridge_patches):
                raise ValueError("Anima QNN LoRA could not apply all CPU text-bridge weights.")
        adapter = AnimaAdapterStack((*previous_adapters, current))
        adapter.patch_revision = result.patches_uuid
        result.model_options["transformer_options"]["anima_qnn_adapter"] = adapter
        return (result,)


NODE_CLASS_MAPPINGS = {"AnimaQNNLoraLoader": AnimaQNNLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaQNNLoraLoader": "Load Anima LoRA (QNN, chainable)"}
