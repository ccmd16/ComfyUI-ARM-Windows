"""Compare a real Anima block with a strict QNN session."""

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from safetensors import safe_open

from comfy import options

parser = argparse.ArgumentParser()
parser.add_argument("--block", type=int, default=1)
parser.add_argument("--size", type=int, default=256, choices=(256, 512, 1024))
parser.add_argument("--lora", action="store_true", help="Swap rank-16 adapters on one compiled block")
args = parser.parse_args()
sys.argv[1:] = ["--cpu", "--use-pytorch-cross-attention"]
options.enable_args_parsing()

from comfy.anima_qnn import ImageBlock, ImageBlockRuntime, LORA_RANK, _RESIDUAL_SCALE, lora_slots
from comfy.ldm.cosmos.predict2 import Block
from comfy.ops import disable_weight_init


def probe_adapters(block, inputs, x, names):
    paths = ("self_attn.q_proj", "cross_attn.v_proj", "mlp.layer1", "mlp.layer2")
    runtime = ImageBlockRuntime()
    slots = lora_slots(block)
    zeros = {}
    for i, (_, layer) in enumerate(slots):
        zeros[f"lora_{i}_down"] = np.zeros((layer.in_features, LORA_RANK), dtype=np.float32)
        zeros[f"lora_{i}_up"] = np.zeros((LORA_RANK, layer.out_features), dtype=np.float32)
    template = runtime._adapter_template(block, inputs, zeros)
    session = runtime._compile_adapter_block(block, template, runtime._options())
    outputs = []
    for seed in (0, 41, 82, 0):
        torch.manual_seed(seed)
        adapter = dict(zeros)
        merged = copy.deepcopy(block)
        for i, (path, layer) in enumerate(slots):
            if seed and path in paths:
                down = torch.zeros_like(torch.from_numpy(zeros[f"lora_{i}_down"]))
                up = torch.zeros_like(torch.from_numpy(zeros[f"lora_{i}_up"]))
                down[:, :16] = torch.randn(layer.in_features, 16) * .035
                up[:16, :] = torch.randn(16, layer.out_features) * .035
                adapter.update({f"lora_{i}_down": down.numpy(), f"lora_{i}_up": up.numpy()})
                merged.get_submodule(path).weight.add_((down @ up).T)
        ref = x + ImageBlock(merged)(*inputs) * _RESIDUAL_SCALE
        feed = {**{name: t.numpy() for name, t in zip(names, inputs)}, **adapter}
        start = time.perf_counter()
        for _ in range(4):
            result = x.numpy() + session.run(None, feed)[0] * _RESIDUAL_SCALE
        err = np.linalg.norm(result - ref.numpy()) / np.linalg.norm(ref.numpy())
        print(f"Adapter seed {seed}: {(time.perf_counter()-start)*250:.2f} ms, relative error {err:.6f}", flush=True)
        assert err < .015
        outputs.append(result)
    assert np.array_equal(outputs[0], outputs[-1])
    assert not np.allclose(outputs[1], outputs[2], atol=.002)
    print("One QNN session handled base / LoRA A / LoRA B / base.")


def probe():
    torch.manual_seed(1)
    torch.set_num_threads(3)
    weight_file = ROOT / "models/diffusion_models/SilvermoonMix-Anima29B-Evolved-V2.3-INT8-ConvRot.safetensors"
    block = Block(2048, 1024, 16, use_adaln_lora=True, operations=disable_weight_init, device="cpu", dtype=torch.float32)
    weights = {}
    with safe_open(weight_file, framework="pt", device="cpu") as source:
        for name in block.state_dict():
            key = f"net.blocks.{args.block}.{name}"
            w = source.get_tensor(key)
            if w.dtype == torch.int8:
                w = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(w, source.get_tensor(key + "_scale"), 256)
            weights[name] = w.float()
    block.load_state_dict(weights)
    del weights
    block.eval()
    flat = ImageBlock(block).eval()
    side = args.size // 16
    tokens = side ** 2
    x = torch.randn(1, tokens, 2048) * .2
    step = torch.randn(1, 1, 2048) * .2
    context = torch.randn(1, 512, 1024) * .2
    ada = torch.randn(1, 1, 6144) * .2
    angles = torch.randn(1, 1, tokens, 64)
    cos, sin = angles.cos(), angles.sin()
    c, s = cos.transpose(1, 2), sin.transpose(1, 2)
    rope = torch.stack((c, -s, s, c), -1).unflatten(-1, (2, 2))
    reference = block(x.reshape(1, 1, side, side, 2048), step, context, rope, ada).flatten(1, 3)
    inputs = (x / _RESIDUAL_SCALE, step, context, ada, cos, sin)
    torch.testing.assert_close(x + flat(*inputs) * _RESIDUAL_SCALE, reference, atol=2e-5, rtol=2e-4)
    names = ("x", "step", "context", "ada", "cos", "sin")
    if args.lora:
        probe_adapters(block, inputs, x, names)
        return
    runtime = ImageBlockRuntime()
    session = runtime._compile(block, inputs, runtime._options())
    feed = {key: tensor.numpy() for key, tensor in zip(names, inputs)}
    output = x.numpy() + session.run(None, feed)[0] * _RESIDUAL_SCALE
    start = time.perf_counter()
    for _ in range(5):
        session.run(None, feed)
    err = np.linalg.norm(output - reference.numpy()) / np.linalg.norm(reference.numpy())
    print(f"QNN {args.size}px block {args.block}: {(time.perf_counter()-start)*200:.2f} ms, rel RMSE {err:.6f}")


with torch.no_grad():
    probe()
