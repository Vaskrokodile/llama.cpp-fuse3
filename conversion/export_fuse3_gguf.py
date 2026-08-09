"""Standalone GGUF exporter for Fuse3.

Exports a Fuse3 HuggingFace model to GGUF format without requiring
modifications to llama.cpp's converter infrastructure. Writes the
GGUF file directly using the gguf Python package.

The resulting GGUF file uses the `fuse3` architecture and contains:
- Standard LFM2 host tensors (attention, conv, dense FFN)
- Custom Fuse3 expert tensors (router, scale, expert weights)

Usage:
    python export_gguf.py --model-dir /path/to/fuse3 --output fuse3.gguf --outtype f16

Note: To actually RUN inference with this GGUF, you need the custom
llama.cpp fork with Fuse3 architecture support (see INTEGRATION.md).
The GGUF file can be created without the fork, but cannot be loaded
by stock llama.cpp.
"""
import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np

# gguf package from llama.cpp
try:
    import gguf
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def load_weights(model_dir: str) -> dict:
    """Load all safetensors weights."""
    from safetensors.torch import load_file
    import glob

    weights = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        weights.update(load_file(shard))
    return weights


def compute_ff_dim(config: dict) -> int:
    """Compute the actual FF dim (same as LFM2)."""
    ff_dim = config.get("block_ff_dim", config.get("intermediate_size", 0))
    auto_adjust = config.get("block_auto_adjust_ff_dim", False)
    multiplier = config.get("block_ffn_dim_multiplier")
    multiple_of = config.get("block_multiple_of", 1)

    if auto_adjust:
        ff_dim = int(2 * ff_dim / 3)
        if multiplier is not None:
            ff_dim = int(multiplier * ff_dim)
        ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

    return ff_dim


def export_gguf(model_dir: str, output_path: str, outtype: str = "f16"):
    """Export Fuse3 model to GGUF format."""
    config = load_config(model_dir)
    weights = load_weights(model_dir)

    print(f"[export] Config: {json.dumps(config, indent=2)[:500]}...")
    print(f"[export] Loaded {len(weights)} tensors")

    # Create GGUF writer
    writer = gguf.GGUFWriter(output_path, "fuse3")

    # ── Set model parameters ──
    n_vocab = config["vocab_size"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]
    n_head = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    n_ff = compute_ff_dim(config)
    norm_eps = config.get("norm_eps", 1e-5)
    conv_L_cache = config.get("conv_L_cache", 3)

    # Per-layer KV heads (0 for conv layers)
    layer_types = config.get("layer_types", [])
    n_kv_heads_list = [
        n_kv_heads if lt != "conv" else 0
        for lt in layer_types
    ]

    writer.add_vocab_size(n_vocab)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_feed_forward_length(n_ff)
    writer.add_shortconv_l_cache(conv_L_cache)
    writer.add_layer_norm_rms_eps(norm_eps)

    # Per-layer KV heads — use raw key name since gguf package version varies
    try:
        writer.add_key_value_heads(n_kv_heads_list)
    except AttributeError:
        # Use add_array with raw key string
        writer.add_array("attention.key_value_head_count", n_kv_heads_list)

    # Fuse3-specific parameters
    experts_per_layer = config.get("experts_per_layer", {})
    expert_intermediate_size = config.get("expert_intermediate_size", 512)
    top_k_experts = config.get("top_k_experts", 8)
    swiglu_limit = config.get("swiglu_limit", 10.0)
    expert_scale_init = config.get("expert_scale_init", -5.0)

    # Convert to int keys
    experts_per_layer_int = {
        int(k): len(v) if isinstance(v, list) else v
        for k, v in experts_per_layer.items()
    }

    augmented_layers = sorted(experts_per_layer_int.keys())
    expert_counts = [experts_per_layer_int.get(i, 0) for i in range(n_layer)]

    writer.add_expert_feed_forward_length(expert_intermediate_size)
    writer.add_expert_count(max(expert_counts) if expert_counts else 0)
    # expert_top_k — use raw key since gguf package may not have this method
    try:
        writer.add_expert_top_k(top_k_experts)
    except AttributeError:
        writer.add_uint32("expert_top_k", top_k_experts)

    # Custom Fuse3 keys
    writer.add_array("fuse3.augmented_layers", augmented_layers)
    writer.add_array("fuse3.expert_counts", expert_counts)
    # swiglu_limit as a float
    try:
        writer.add_float32("fuse3.swiglu_limit", swiglu_limit)
    except AttributeError:
        pass  # Non-critical, default is 10.0 in C++

    print(f"[export] Augmented layers: {augmented_layers}")
    print(f"[export] Expert counts: {expert_counts}")
    print(f"[export] Expert intermediate: {expert_intermediate_size}")
    print(f"[export] Top-k: {top_k_experts}")

    # ── Determine output dtype ──
    dtype_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
        "bf16": gguf.GGMLQuantizationType.BF16,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }
    out_dtype = dtype_map.get(outtype, gguf.GGMLQuantizationType.F16)

    # ── Write tensors ──
    tensor_count = 0

    def add_tensor(name, data, raw_dtype=None):
        nonlocal tensor_count
        if data.dim() == 0:
            data = data.unsqueeze(0)
        # Convert bfloat16 to float16 (numpy doesn't support bfloat16)
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float16)
        data_np = data.numpy()
        # Use the specified dtype or infer from data
        if raw_dtype is not None:
            writer.add_tensor(name, data_np, raw_dtype=raw_dtype)
        else:
            writer.add_tensor(name, data_np, raw_dtype=out_dtype)
        tensor_count += 1

    # Token embedding
    embd = weights["model.embed_tokens.weight"]
    add_tensor("token_embd.weight", embd)

    # Output norm (LFM2 uses token_embd_norm)
    output_norm = weights.get("model.embedding_norm.weight", weights.get("model.output_norm.weight"))
    if output_norm is not None:
        add_tensor("token_embd_norm.weight", output_norm)

    # Output (tied weights)
    lm_head = weights.get("lm_head.weight")
    if lm_head is not None:
        add_tensor("output.weight", lm_head)

    # Per-layer tensors
    for i in range(n_layer):
        prefix = f"model.layers.{i}"

        # Check if this is an augmented layer (has host_layer prefix)
        is_augmented = i in experts_per_layer_int and experts_per_layer_int[i] > 0
        host_prefix = f"{prefix}.host_layer" if is_augmented else prefix

        # FFN norm
        ffn_norm = weights.get(f"{host_prefix}.ffn_norm.weight", weights.get(f"{prefix}.ffn_norm.weight"))
        if ffn_norm is not None:
            add_tensor(f"blk.{i}.ffn_norm.weight", ffn_norm)

        # Dense FFN (host)
        for tensor_name, gguf_name in [
            ("feed_forward.w1.weight", "ffn_gate.weight"),
            ("feed_forward.w3.weight", "ffn_up.weight"),
            ("feed_forward.w2.weight", "ffn_down.weight"),
        ]:
            w = weights.get(f"{host_prefix}.{tensor_name}", weights.get(f"{prefix}.{tensor_name}"))
            if w is not None:
                add_tensor(f"blk.{i}.{gguf_name}", w)

        # Attention norm (operator_norm)
        attn_norm = weights.get(f"{host_prefix}.operator_norm.weight", weights.get(f"{prefix}.operator_norm.weight"))
        if attn_norm is not None:
            add_tensor(f"blk.{i}.attn_norm.weight", attn_norm)

        # Check if attention or conv layer
        is_attn = layer_types[i] == "full_attention" if i < len(layer_types) else True

        if is_attn:
            # Attention tensors
            for tensor_name, gguf_name in [
                ("self_attn.q_proj.weight", "attn_q.weight"),
                ("self_attn.k_proj.weight", "attn_k.weight"),
                ("self_attn.v_proj.weight", "attn_v.weight"),
                ("self_attn.o_proj.weight", "attn_output.weight"),
                ("self_attn.q_layernorm.weight", "attn_q_norm.weight"),
                ("self_attn.k_layernorm.weight", "attn_k_norm.weight"),
            ]:
                w = weights.get(f"{host_prefix}.{tensor_name}", weights.get(f"{prefix}.{tensor_name}"))
                if w is not None:
                    add_tensor(f"blk.{i}.{gguf_name}", w)
        else:
            # ShortConv tensors
            for tensor_name, gguf_name in [
                ("conv.conv.weight", "shortconv_conv.weight"),
                ("conv.in_proj.weight", "shortconv_inproj.weight"),
                ("conv.out_proj.weight", "shortconv_outproj.weight"),
            ]:
                w = weights.get(f"{host_prefix}.{tensor_name}", weights.get(f"{prefix}.{tensor_name}"))
                if w is not None:
                    # Conv weight needs squeezing (2D for ggml)
                    if "conv.conv" in tensor_name and w.dim() == 3:
                        w = w.squeeze(1)
                    add_tensor(f"blk.{i}.{gguf_name}", w)

        # Fuse3 expert tensors (only for augmented layers)
        if is_augmented:
            n_exp = experts_per_layer_int[i]

            # Router
            router_w = weights.get(f"{prefix}.router.gate.weight")
            if router_w is not None:
                add_tensor(f"blk.{i}.fuse3_router.weight", router_w)

            # Expert scale
            scale = weights.get(f"{prefix}.expert_scale")
            if scale is not None:
                add_tensor(f"blk.{i}.fuse3_expert_scale.weight", scale.unsqueeze(0) if scale.dim() == 0 else scale)

            # Expert weights — stack per-expert weights
            for proj_name, gguf_proj in [
                ("gate_proj", "gate"),
                ("up_proj", "up"),
                ("down_proj", "down"),
            ]:
                expert_weights = []
                for eid in range(n_exp):
                    key = f"{prefix}.experts.{eid}.{proj_name}.weight"
                    if key in weights:
                        expert_weights.append(weights[key])
                    else:
                        print(f"[export] WARNING: Missing {key}")

                if expert_weights:
                    # Stack: {n_exp, ...dims} -> transpose to {...dims, n_exp}
                    stacked = torch.stack(expert_weights, dim=-1)
                    add_tensor(f"blk.{i}.fuse3_experts.{gguf_proj}.weight", stacked)

    print(f"[export] Written {tensor_count} tensors")

    # Write the GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path) / 1e9
    print(f"[export] GGUF written: {output_path} ({file_size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Export Fuse3 to GGUF")
    parser.add_argument("--model-dir", required=True, help="Path to Fuse3 model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0"],
                        help="Output tensor dtype")
    args = parser.parse_args()

    export_gguf(args.model_dir, args.output, args.outtype)


if __name__ == "__main__":
    main()
