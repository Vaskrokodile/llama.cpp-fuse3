"""Fuse3 GGUF converter for llama.cpp.

Converts a Fuse3 (LFM2 host + Qwen3.6 coding experts) HuggingFace model
to GGUF format. The host LFM2 weights use standard LFM2 tensor naming;
the expert/router/scale weights use custom `fuse3_*` tensor names that
are consumed by the custom `LLM_ARCH_FUSE3` graph builder in llama.cpp.

Usage:
    python convert_hf_to_gguf.py /path/to/fuse3/model --outtype f16

This file should be placed in llama.cpp's `conversion/` directory.
"""
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf
from .lfm2 import LFM2Model


@ModelBase.register("Fuse3ForCausalLM")
class Fuse3Model(TextModel):
    """Fuse3 model converter: LFM2 host + Qwen3.6 coding experts.

    Extends the LFM2 converter to handle:
    - host_layer. prefix on augmented layers (strip it for GGUF)
    - Expert weights (gate_proj, up_proj, down_proj per expert)
    - Router weights (gate per augmented layer)
    - Expert scale (scalar per augmented layer)
    """

    model_arch = gguf.MODEL_ARCH.FUSE3

    def set_gguf_parameters(self):
        # Set num_key_value_heads per layer (0 for conv layers)
        self.hparams["num_key_value_heads"] = [
            self.hparams["num_key_value_heads"] if layer_type != "conv" else 0
            for layer_type in self.hparams["layer_types"]
        ]

        super().set_gguf_parameters()

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_shortconv_l_cache(self.hparams["conv_L_cache"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["norm_eps"])

        # Feed forward length (same as LFM2)
        ff_dim = self.find_hparam(["block_ff_dim", "intermediate_size"])
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["block_multiple_of"]
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.gguf_writer.add_feed_forward_length(ff_dim)

        # Fuse3-specific parameters
        experts_per_layer = self.hparams.get("experts_per_layer", {})
        expert_intermediate_size = self.hparams.get("expert_intermediate_size", 512)
        top_k_experts = self.hparams.get("top_k_experts", 8)
        swiglu_limit = self.hparams.get("swiglu_limit", 10.0)
        expert_scale_init = self.hparams.get("expert_scale_init", -5.0)

        # Convert experts_per_layer keys to int (JSON saves them as strings)
        experts_per_layer_int = {
            int(k): len(v) if isinstance(v, list) else v
            for k, v in experts_per_layer.items()
        }

        self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)
        self.gguf_writer.add_expert_count(
            max(experts_per_layer_int.values()) if experts_per_layer_int else 0
        )
        self.gguf_writer.add_expert_top_k(top_k_experts)

        # Store per-layer expert counts as a custom array
        # We use a naming convention: fuse3_augmented_layers = list of layer indices
        augmented_layers = sorted(experts_per_layer_int.keys())
        self.gguf_writer.add_array(
            "fuse3.augmented_layers",
            augmented_layers,
        )
        # Per-layer expert counts
        expert_counts = [experts_per_layer_int.get(i, 0) for i in range(self.hparams["num_hidden_layers"])]
        self.gguf_writer.add_array(
            "fuse3.expert_counts",
            expert_counts,
        )
        self.gguf_writer.add_float("fuse3.swiglu_limit", swiglu_limit)
        self.gguf_writer.add_float("fuse3.expert_scale_init", expert_scale_init)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # Strip host_layer. prefix — in GGUF, the host components are directly on the layer
        new_name = name.replace(".host_layer.", ".")

        # Conv weight: squeeze to 2d (same as LFM2)
        if "conv.conv" in new_name:
            data_torch = data_torch.squeeze(1)

        # Expert weights: rename to fuse3 convention
        # HF: model.layers.{bid}.experts.{eid}.gate_proj.weight
        # GGUF: blk.{bid}.fuse3_experts.{eid}.gate.weight
        if ".experts." in new_name and "router" not in new_name:
            # Parse: model.layers.{bid}.experts.{eid}.{gate_proj|up_proj|down_proj}.weight
            parts = new_name.split(".")
            # Find "experts" in parts
            exp_idx = parts.index("experts")
            eid = int(parts[exp_idx + 1])
            proj_name = parts[exp_idx + 2]
            # Map proj names: gate_proj -> gate, up_proj -> up, down_proj -> down
            proj_map = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
            gguf_proj = proj_map.get(proj_name, proj_name)
            gguf_name = f"blk.{bid}.fuse3_experts.{eid}.{gguf_proj}.weight"
            yield gguf_name, data_torch
            return

        # Router weights: rename to fuse3 convention
        # HF: model.layers.{bid}.router.gate.weight
        # GGUF: blk.{bid}.fuse3_router.weight
        if ".router." in new_name:
            gguf_name = f"blk.{bid}.fuse3_router.weight"
            yield gguf_name, data_torch
            return

        # Expert scale: rename to fuse3 convention
        # HF: model.layers.{bid}.expert_scale
        # GGUF: blk.{bid}.fuse3_expert_scale.weight
        if new_name.endswith(".expert_scale"):
            gguf_name = f"blk.{bid}.fuse3_expert_scale.weight"
            # Ensure it's a 1d tensor
            if data_torch.dim() == 0:
                data_torch = data_torch.unsqueeze(0)
            yield gguf_name, data_torch
            return

        # All other tensors: use standard LFM2 naming
        yield from super().modify_tensors(data_torch, new_name, bid)
