# llama.cpp with Fuse3 architecture support

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds support for the **Fuse3** model architecture.

## What is Fuse3?

Fuse3 is a hybrid model that combines an LFM2 host with per-layer coding expert augmentation. After each augmented layer's dense FFN, a router selects coding experts whose output is normalized, scaled, and added to the residual stream.

- **Host**: LFM2 (linear attention + short conv hybrid)
- **Experts**: Qwen3.6 coding experts (32 experts per augmented layer, top-k=8)
- **Router**: sqrtsoftplus scoring with normalization
- **Scale**: softplus activation clamped to [0, 0.1]

## Files added/modified

### Added
- `src/models/fuse3.cpp` - Fuse3 model implementation (graph builder)
- `conversion/fuse3.py` - HuggingFace to GGUF converter (for `convert_hf_to_gguf.py`)
- `conversion/export_fuse3_gguf.py` - Standalone GGUF exporter (no llama.cpp build needed)

### Modified
- `gguf-py/gguf/constants.py` - Added `FUSE3` to `MODEL_ARCH` enum and tensor list
- `src/llama-arch.h` - Added `LLM_ARCH_FUSE3` to the arch enum
- `src/llama-arch.cpp` - Registered `"fuse3"` architecture name
- `src/llama-model.cpp` - Added `LLM_ARCH_FUSE3` dispatch to `llama_model_fuse3`
- `src/models/models.h` - Added `llama_model_fuse3` struct (extends `llama_model_lfm2`)

## Usage

### Convert a Fuse3 model to GGUF

```bash
# Using the standalone exporter (no build needed):
python conversion/export_fuse3_gguf.py --model-dir /path/to/fuse3/model --output fuse3-f16.gguf --outtype f16

# Using llama.cpp's converter:
python convert_hf_to_gguf.py /path/to/fuse3/model --outtype f16
```

### Quantize

```bash
./build/bin/llama-quantize fuse3-f16.gguf fuse3-Q4_K_M.gguf q4_k_m
```

### Run inference

```bash
./build/bin/llama-cli -m fuse3-Q4_K_M.gguf -p "Write a Python fizzbuzz"
```

## Building

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # or OFF for CPU
cmake --build . --config Release -j
```

## GGUF tensor layout

Standard LFM2 tensors (token_embd, attn_q/k/v, ffn_gate/up/down, shortconv, etc.) plus Fuse3-specific tensors per augmented layer:

- `blk.{i}.fuse3_router.weight` - Router weights `{n_embd, n_experts}`
- `blk.{i}.fuse3_expert_scale.weight` - Expert scale `{1}`
- `blk.{i}.fuse3_experts.gate.weight` - Stacked gate weights `{n_embd, n_ff_exp, n_experts}`
- `blk.{i}.fuse3_experts.up.weight` - Stacked up weights `{n_embd, n_ff_exp, n_experts}`
- `blk.{i}.fuse3_experts.down.weight` - Stacked down weights `{n_ff_exp, n_embd, n_experts}`

## Custom GGUF metadata

- `fuse3.augmented_layers` - Array of layer indices that have experts
- `fuse3.expert_counts` - Array of expert counts per layer
- `fuse3.swiglu_limit` - SwiGLU clamp limit (default 10.0)
