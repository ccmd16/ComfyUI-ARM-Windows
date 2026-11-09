"""Static, image-only Anima transformer blocks for Qualcomm QNN."""

import hashlib
import io
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
from torch import nn


_RESIDUAL_SCALE = 256.0
QNN_IMAGE_SIZES = (256, 512, 1024)


def _normalize(x, eps, *, center=False, weight=None, input_scale=1.0):
    # Scale before squaring. HTP FP16 can overflow on large residuals and
    # flush the tiny epsilon to zero for the padded text tokens.
    scale = x.abs().amax(-1, keepdim=True).clamp_min(2 ** -8)
    x = x / scale
    if center:
        x = x - x.mean(-1, keepdim=True)
    variance = x.square().mean(-1, keepdim=True)
    epsilon = (eps ** 0.5 / (scale * input_scale)).square()
    x = x * (variance + epsilon).clamp_min(2 ** -14).rsqrt()
    return x if weight is None else x * weight


class ImageBlock(nn.Module):
    """Flatten the image grid and express attention with ordinary ONNX ops."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def _attention(self, attn, x, context, cos=None, sin=None):
        def project(layer, source):
            return layer(source).reshape(source.shape[0], source.shape[1], attn.n_heads, attn.head_dim).transpose(1, 2)

        q = _normalize(project(attn.q_proj, x), attn.q_norm.eps, weight=attn.q_norm.weight)
        k = _normalize(project(attn.k_proj, context), attn.k_norm.eps, weight=attn.k_norm.weight)
        v = project(attn.v_proj, context)
        if cos is not None:
            def rotate(z):
                first, second = z.chunk(2, dim=-1)
                return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

            q, k = rotate(q), rotate(k)
        q = q * (attn.head_dim ** -0.5)
        keys = k.transpose(-2, -1)
        if q.shape[-2] > 1024 and k.shape[-2] > 1024:
            # Every query still sees every key. Limit the size of each score tensor.
            attended = torch.cat([(tile @ keys).softmax(-1) @ v for tile in q.split(256, dim=-2)], dim=-2)
        else:
            attended = (q @ keys).softmax(-1) @ v
        attended = attended.transpose(1, 2).flatten(2)
        return attn.output_proj(attended)

    def forward(self, x, time, context, modulation, cos, sin):
        block = self.block

        def condition(norm, adapter, residual):
            shift, scale, gate = (adapter(time) + modulation).chunk(3, dim=-1)
            y = _normalize(residual, norm.eps, center=True, input_scale=_RESIDUAL_SCALE)
            return y * (1 + scale) + shift, gate / _RESIDUAL_SCALE

        y, gate = condition(block.layer_norm_self_attn, block.adaln_modulation_self_attn, x)
        self_delta = gate * self._attention(block.self_attn, y, y, cos, sin)
        x = x + self_delta
        y, gate = condition(block.layer_norm_cross_attn, block.adaln_modulation_cross_attn, x)
        cross_delta = gate * self._attention(block.cross_attn, y, context)
        x = x + cross_delta
        y, gate = condition(block.layer_norm_mlp, block.adaln_modulation_mlp, x)
        return self_delta + cross_delta + gate * block.mlp(y)


_NAMES = ("x", "step", "context", "ada", "cos", "sin")
LORA_RANK = 32


def lora_slots(block):
    return [(name, layer) for name, layer in block.named_modules() if isinstance(layer, nn.Linear)]


class _AdapterLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapters = ()

    def forward(self, x):
        output = self.base(x)
        for down, up in self.adapters:
            output = output + (x @ down) @ up
        return output


class AdapterBlock(nn.Module):
    def __init__(self, block, adapter_count):
        super().__init__()
        self.graph = ImageBlock(block)
        self.adapter_count = adapter_count
        self.slots = []
        for path, layer in lora_slots(block):
            parent, _, name = path.rpartition(".")
            wrapper = _AdapterLinear(layer)
            setattr(block.get_submodule(parent), name, wrapper)
            self.slots.append(wrapper)

    def forward(self, x, step, context, ada, cos, sin, *adapters):
        expected = self.adapter_count * len(self.slots) * 2
        if len(adapters) != expected:
            raise ValueError(f"Expected {expected} Anima QNN LoRA inputs, got {len(adapters)}")
        for slot_index, slot in enumerate(self.slots):
            slot.adapters = tuple(
                (adapters[2 * (adapter_index * len(self.slots) + slot_index)],
                 adapters[2 * (adapter_index * len(self.slots) + slot_index) + 1])
                for adapter_index in range(self.adapter_count)
            )
        return self.graph(x, step, context, ada, cos, sin)


class ImageBlockRuntime:
    """Keep small sessions in memory and cache large contexts on disk."""

    def __init__(self):
        self.lock = threading.Lock()
        self.shape = None
        self.sessions = []
        self.context_store = None
        self.prefetch_contexts = True
        self.weights_digest = None
        self.shared_session = None
        self.shared_weights = None

    @staticmethod
    def _block_layout(block):
        return (block.x_dim, block.cross_attn.context_dim, block.self_attn.n_heads,
                block.mlp._hidden_dim / block.x_dim, block.adaln_modulation_self_attn[1].out_features)

    @classmethod
    def _portable_block(cls, block):
        from comfy.ldm.cosmos.predict2 import Block
        from comfy.ops import disable_weight_init

        x_dim, context_dim, heads, mlp_ratio, lora_dim = cls._block_layout(block)
        portable = Block(
            x_dim, context_dim, heads, mlp_ratio=mlp_ratio, use_adaln_lora=True,
            adaln_lora_dim=lora_dim,
            operations=disable_weight_init, device="meta", dtype=torch.float32,
        )
        portable.load_state_dict({k: v.detach().float() for k, v in block.state_dict().items()}, assign=True)
        return portable

    @classmethod
    def _compile(cls, block, inputs, opts, target=None):
        return cls._compile_graph(ImageBlock(cls._portable_block(block)).eval(), inputs, _NAMES, opts, target)

    @classmethod
    def _adapter_template(cls, block, inputs, adapter_inputs):
        slot_count = len(lora_slots(block))
        if slot_count == 0 or len(adapter_inputs) % (2 * slot_count):
            raise ValueError("Anima QNN LoRA inputs do not match the block's linear layers")
        adapter_count = len(adapter_inputs) // (2 * slot_count)
        adapter_names = tuple(
            f"lora_{adapter_index}_{slot}_{kind}"
            for adapter_index in range(adapter_count)
            for slot in range(slot_count)
            for kind in ("down", "up")
        )
        if set(adapter_names) != set(adapter_inputs):
            raise ValueError("Anima QNN LoRA inputs have an unexpected layout")
        names = _NAMES + adapter_names
        inputs += tuple(torch.from_numpy(adapter_inputs[name]) for name in adapter_names)
        buffer = io.BytesIO()
        torch.onnx.export(
            AdapterBlock(cls._portable_block(block), adapter_count).eval(), inputs, buffer,
            input_names=list(names), output_names=["y"], opset_version=17,
            # Equal export-time tensors can have different values in other blocks.
            keep_initializers_as_inputs=True,
            dynamo=False, export_params=False, do_constant_folding=False,
        )
        return buffer.getvalue(), names

    @staticmethod
    def _weight_key(name):
        return name.removeprefix("graph.block.").replace(".base.", ".")

    def _prepare_shared_session(self, block, example, adapter_inputs, opts):
        import onnxruntime as ort

        names = set(_NAMES).union(adapter_inputs)

        def load(context):
            session = ort.InferenceSession(context, sess_options=opts)
            inputs = {item.name for item in session.get_inputs()}
            weights = {name: self._weight_key(name) for name in inputs - names}
            if not names <= inputs or set(weights.values()) != block.state_dict().keys():
                raise ValueError("Unexpected shared Anima QNN graph inputs")
            self.shared_session = session
            self.shared_weights = weights

        cache = None
        if os.environ.get("COMFYUI_ANIMA_QNN_CACHE_SHARED", "1") == "1":
            try:
                root, key = self._cache_key(example, adapter_inputs, ("shared", self._block_layout(block)))
                cache = root / f"shared-{key}.onnx"
                started = perf_counter()
                load(cache.read_bytes())
                logging.info("Anima QNN: loaded shared graph from disk in %.1fs", perf_counter() - started)
                return
            except FileNotFoundError:
                pass
            except Exception:
                logging.warning("Anima QNN: shared cache unavailable, recompiling", exc_info=True)

        blob, _ = self._adapter_template(block, example, adapter_inputs)
        started = perf_counter()
        compiler = ort.ModelCompiler(
            opts, blob, embed_compiled_data_into_model=True,
            graph_optimization_level=opts.graph_optimization_level,
        )
        context = compiler.compile_to_bytes()
        load(context)
        logging.info("Anima QNN: compiled shared RAM graph in %.1fs (%.1f MiB)",
                     perf_counter() - started, len(context) / 2**20)
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="shared-", dir=cache.parent) as directory:
                    staged = Path(directory) / cache.name
                    staged.write_bytes(context)
                    staged.replace(cache)
            except OSError:
                logging.warning("Anima QNN: could not save the shared graph cache", exc_info=True)

    @classmethod
    def _compile_adapter_block(cls, block, template, opts, target=None, *, memory=False):
        import onnx

        blob, names = template
        graph = onnx.load_model_from_string(blob)
        weights = block.state_dict()
        with (nullcontext() if memory else cls._scratch()) as directory:
            data = None if memory else Path(directory) / "weights.bin"
            with (nullcontext() if memory else data.open("wb")) as stream:
                for argument in list(graph.graph.input):
                    if argument.name in names:
                        continue
                    key = cls._weight_key(argument.name)
                    weight = weights[key].detach().float().contiguous().numpy()
                    initializer = graph.graph.initializer.add()
                    initializer.name = argument.name
                    initializer.data_type = onnx.TensorProto.FLOAT
                    initializer.dims.extend(weight.shape)
                    if memory:
                        initializer.raw_data = weight.tobytes()
                    else:
                        initializer.data_location = onnx.TensorProto.EXTERNAL
                        offset = stream.tell()
                        stream.write(memoryview(weight).cast("B"))
                        for name, value in (("location", data.name), ("offset", offset), ("length", weight.nbytes)):
                            entry = initializer.external_data.add()
                            entry.key, entry.value = name, str(value)
                    graph.graph.input.remove(argument)
            payload = graph.SerializeToString()
            if memory:
                import onnxruntime as ort

                del graph  # The compiler only needs the serialized model.
                compiler = ort.ModelCompiler(
                    opts, payload, embed_compiled_data_into_model=True,
                    graph_optimization_level=opts.graph_optimization_level,
                )
                return compiler.compile_to_bytes()
            source = Path(directory) / "source.onnx"
            source.write_bytes(payload)
            return cls._load_context(source, opts, target)

    @staticmethod
    def _compile_graph(graph, inputs, names, opts, target=None):
        with ImageBlockRuntime._scratch() as directory:
            source = Path(directory) / "source.onnx"
            torch.onnx.export(
                graph, inputs, str(source),
                input_names=list(names), output_names=["y"], opset_version=17, dynamo=False,
            )
            return ImageBlockRuntime._load_context(source, opts, target)

    @staticmethod
    def _scratch():
        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="anima-qnn-", dir=scratch)

    @staticmethod
    def _load_context(source, opts, target=None):
        import onnxruntime as ort

        # Keep the small context sessions and release the source graph's FP32 weights.
        context = target if target is not None else source.with_name("context.onnx")
        compiled = ort.InferenceSession(str(source), sess_options=ImageBlockRuntime._options(context))
        del compiled
        return context if target is not None else ort.InferenceSession(str(context), sess_options=opts)

    @staticmethod
    def _options(context=None):
        import onnxruntime as ort
        import onnxruntime_qnn as qnn

        provider = "QNNExecutionProvider"
        if not any(d.ep_name == provider for d in ort.get_ep_devices()):
            ort.register_execution_provider_library(provider, qnn.get_library_path())
        devices = [
            d for d in ort.get_ep_devices()
            if d.ep_name == provider and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            raise RuntimeError("Anima QNN: no Qualcomm NPU device available")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        opts.add_provider_for_devices(devices, {
            "backend_path": qnn.get_qnn_htp_path(),
            "enable_htp_fp16_precision": "1",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "1",
        })
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if context is not None:
            opts.add_session_config_entry("ep.context_enable", "1")
            opts.add_session_config_entry("ep.context_embed_mode", "0")
            opts.add_session_config_entry("ep.context_file_path", str(context))
        return opts

    @staticmethod
    def _cache_key(inputs, adapter_inputs, layout):
        import onnxruntime_qnn as qnn

        digest = hashlib.sha256()
        module = Path(__file__).resolve()
        for source in (module, module.parent / "ldm" / "cosmos" / "predict2.py"):
            digest.update(source.read_bytes())
        for library in (qnn.get_library_path(), qnn.get_qnn_htp_path()):
            path = Path(library).resolve()
            stat = path.stat()
            digest.update(repr((str(path), stat.st_size, stat.st_mtime_ns)).encode())
        digest.update(repr((
            torch.__version__, version("onnxruntime"), version("onnxruntime-qnn"), layout,
            tuple((tuple(t.shape), str(t.dtype)) for t in inputs),
            tuple((name, value.shape, str(value.dtype)) for name, value in adapter_inputs.items()),
        )).encode())
        return module.parents[1] / "user" / "anima-qnn-cache", digest.hexdigest()

    def _cache_directory(self, blocks, inputs, adapters):
        if self.weights_digest is None:
            weights = hashlib.sha256()
            for block in blocks:
                for name, tensor in block.state_dict().items():
                    value = tensor.detach().float().contiguous().numpy()
                    weights.update(repr((name, value.shape)).encode())
                    weights.update(memoryview(value).cast("B"))
            self.weights_digest = weights.hexdigest()
        root, key = self._cache_key(inputs, {} if adapters is None else adapters.feeds[0], self.weights_digest)
        return root / key

    @staticmethod
    def _cached_blocks(directory, count):
        paths = [directory / f"block-{index:02d}.onnx" for index in range(1, count + 1)]
        for path in paths:
            binary = path.with_name(f"{path.stem}_qnn.bin")
            try:
                if not path.stat().st_size or not binary.stat().st_size:
                    return []
            except FileNotFoundError:
                return []
        return paths

    def _prepare_sessions(self, blocks, example, adapters, paged):
        persist = paged and os.environ.get("COMFYUI_ANIMA_QNN_PERSIST_CACHE") == "1"
        cache = self._cache_directory(blocks, example, adapters) if persist else None
        if cache is not None:
            sessions = self._cached_blocks(cache, len(blocks))
            if sessions:
                logging.info("Anima QNN: reusing %d compiled blocks from %s", len(sessions), cache)
                return sessions

        opts = self._options()  # Register the provider before creating any worker sessions.
        memory = paged and adapters is not None and not persist and os.environ.get("COMFYUI_ANIMA_QNN_CONTEXT_MEMORY", "1") == "1"
        if memory and os.environ.get("COMFYUI_ANIMA_QNN_SHARED_GRAPH", "1") == "1":
            try:
                self._prepare_shared_session(blocks[0], example, adapters.feeds[0], opts)
                return list(blocks)
            except Exception:
                logging.warning("Anima QNN: shared graph compilation failed, compiling individual blocks", exc_info=True)
        template = None if adapters is None else self._adapter_template(blocks[0], example, adapters.feeds[0])
        parallel = paged and template is not None and os.environ.get("COMFYUI_ANIMA_QNN_COMPILE_WORKERS", "2") == "2"
        if memory:
            # Compiled blocks are about 150 MiB each. Leave headroom for the
            # two compiler sessions, runtime activations and the rest of ComfyUI.
            required = len(blocks) * (160 << 20) + (5 << 30)
            if psutil.virtual_memory().available >= required:
                try:
                    return self._compile_blocks(blocks, example, template, opts, None, parallel, memory=True)
                except Exception:
                    logging.warning("Anima QNN: RAM compilation failed, retrying with temporary files", exc_info=True)
            else:
                logging.info("Anima QNN: less than %.1f GiB RAM available, using temporary files", required / 2**30)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.context_store = tempfile.TemporaryDirectory(prefix="building-", dir=cache.parent)
        elif paged:
            self.context_store = self._scratch()
        store = Path(self.context_store.name) if self.context_store is not None else None

        try:
            sessions = self._compile_blocks(blocks, example, template, opts, store, parallel)
            if cache is not None and self._cached_blocks(store, len(blocks)):
                try:
                    store.rename(cache)
                except OSError:
                    cached = self._cached_blocks(cache, len(blocks))
                    if not cached:
                        logging.warning("Anima QNN: could not publish context cache at %s; using temporary contexts", cache)
                        return sessions
                self.context_store.cleanup()
                self.context_store = None
                logging.info("Anima QNN: compiled contexts available at %s", cache)
                return self._cached_blocks(cache, len(blocks))
            return sessions
        except BaseException:
            if self.context_store is not None:
                self.context_store.cleanup()
                self.context_store = None
            raise

    def _compile_blocks(self, blocks, example, template, opts, store, parallel, *, memory=False):
        started = perf_counter()
        logging.info("Anima QNN: compiling %d blocks for %d image tokens (%d workers, %s)",
                     len(blocks), example[0].shape[1], 2 if parallel else 1, "RAM" if memory else "file-backed")

        def target_for(index):
            return store / f"block-{index + 1:02d}.onnx" if store is not None else None

        @torch.no_grad()
        def compile_block(index):
            target = target_for(index)
            if template is None:
                return self._compile(blocks[index], example, opts, target)
            return self._compile_adapter_block(blocks[index], template, opts, target, memory=memory)

        def record(index, session):
            sessions.append(session)
            if index == 0 or (index + 1) % 5 == 0:
                logging.info("Anima QNN: %d/%d blocks ready (%.1fs)", index + 1, len(blocks), perf_counter() - started)

        sessions = []
        if parallel:
            # Each worker serializes its own weights and creates a separate QNN context.
            with ThreadPoolExecutor(max_workers=2) as pool:
                while len(sessions) < len(blocks) and parallel:
                    indices = range(len(sessions), min(len(sessions) + 2, len(blocks)))
                    futures = [pool.submit(compile_block, index) for index in indices]
                    wait(futures)
                    for index, future in zip(indices, futures):
                        try:
                            session = future.result()
                        except Exception:
                            logging.warning("Anima QNN: parallel compilation failed at block %d; retrying serially",
                                            index + 1, exc_info=True)
                            parallel = False
                            target = target_for(index)
                            if target is not None:
                                target.unlink(missing_ok=True)
                                target.with_name(f"{target.stem}_qnn.bin").unlink(missing_ok=True)
                            session = compile_block(index)
                        record(index, session)
        for index in range(len(sessions), len(blocks)):
            record(index, compile_block(index))
        logging.info("Anima QNN: compiled %d blocks in %.1fs", len(sessions), perf_counter() - started)
        return sessions

    def run(self, blocks, x, time, context, modulation, rope, adapters=None):
        # RoPE has one position grid shared by all CFG batch entries.
        cos = rope[..., 0, 0].transpose(1, 2)
        sin = rope[..., 1, 0].transpose(1, 2)
        x_flat = x.flatten(1, 3)
        adapter_signature = None if adapters is None else tuple(
            (name, tuple(value.shape), str(value.dtype)) for name, value in adapters.feeds[0].items()
        )
        key = (tuple(x_flat.shape[1:]), tuple(context.shape[1:]), adapter_signature)
        # Page large contexts instead of keeping all block sessions resident.
        # A 512px image has exactly 1024 tokens. Page that boundary too:
        # keeping all 40 per-block QNN contexts resident can exhaust HTP memory.
        paged = x_flat.shape[1] >= 1024
        with self.lock:
            if self.shape != key:
                self.sessions.clear()
                self.shape = None
                self.shared_session = None
                self.shared_weights = None
                if self.context_store is not None:
                    self.context_store.cleanup()
                    self.context_store = None
                example = tuple(t[:1].contiguous() for t in (x_flat / _RESIDUAL_SCALE, time, context, modulation, cos, sin))
                self.sessions = self._prepare_sessions(blocks, example, adapters, paged)
                self.shape = key

            residuals = []
            feeds = []
            shared_feed = {"cos": cos.contiguous().numpy(), "sin": sin.contiguous().numpy()}
            for index in range(x.shape[0]):
                residuals.append(x_flat[index:index + 1].contiguous().numpy().copy())
                feed = {
                    name: value[index:index + 1].contiguous().numpy()
                    for name, value in zip(_NAMES[1:4], (time, context, modulation))
                }
                feed.update(shared_feed)
                feeds.append(feed)
            scaled_input = np.empty_like(residuals[0])
            load_wait = 0.0
            started = perf_counter()
            if self.shared_session is not None:
                @torch.no_grad()
                def load_block(block):
                    state = block.state_dict()
                    weights = {
                        name: state[key].detach().float().contiguous().numpy()
                        for name, key in self.shared_weights.items()
                    }
                    return self.shared_session, weights
            elif paged:
                import onnxruntime as ort

                opts = self._options()

                def load_block(context):
                    source = str(context) if isinstance(context, Path) else context
                    return ort.InferenceSession(source, sess_options=opts), {}

            # Only the current and next block can have live sessions.
            prefetch = paged and self.prefetch_contexts
            with (ThreadPoolExecutor(max_workers=1) if prefetch else nullcontext()) as worker:
                pending = None
                for block_index, cached in enumerate(self.sessions):
                    session, weights = cached, {}
                    if paged:
                        wait_started = perf_counter()
                        if pending is None:
                            session, weights = load_block(cached)
                        else:
                            try:
                                session, weights = pending.result()
                            except Exception:
                                logging.warning("Anima QNN: concurrent block load failed, using serial loads", exc_info=True)
                                self.prefetch_contexts = False
                                session, weights = load_block(cached)
                            finally:
                                pending = None
                        load_wait += perf_counter() - wait_started
                        if worker is not None and self.prefetch_contexts and block_index + 1 < len(self.sessions):
                            pending = worker.submit(load_block, self.sessions[block_index + 1])
                    for residual, feed in zip(residuals, feeds):
                        np.divide(residual, _RESIDUAL_SCALE, out=scaled_input)
                        feed["x"] = scaled_input
                        block_feed = feed if adapters is None else {**feed, **adapters.feeds[block_index], **weights}
                        delta = session.run(None, block_feed)[0]
                        np.multiply(delta, _RESIDUAL_SCALE, out=delta)
                        np.add(residual, delta, out=residual)
                    del session
            if paged:
                logging.info("Anima QNN: %d blocks, batch %d, %.1fs total, %.1fs waiting for block data",
                             len(self.sessions), len(residuals), perf_counter() - started, load_wait)
            result = torch.cat([torch.from_numpy(residual) for residual in residuals]).reshape_as(x)
            if not torch.isfinite(result).all():
                raise RuntimeError("Anima QNN: non-finite output after transformer blocks")
            return result
