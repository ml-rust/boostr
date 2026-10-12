# Architecture

This page covers the module layout, the layer stack, and the design rules of boostr.
For the feature list, see [capabilities](capabilities.md).

## Layer Stack

```
┌───────────────────────────────────────────────────────┐
│                    boostr                             │
│   (attention, RoPE, MoE, quantization, model loaders) │
└──────────────────────────┬────────────────────────────┘
                           │
                        (uses)
                           │
┌──────────────────────────▼───────────────────────────┐
│                      numr                            │
│   (tensors, ops, runtime, autograd, linalg, FFT)     │
└──────────────────────────────────────────────────────┘
```

## Design Principles

- **Extension traits**: ML operations such as `AttentionOps` and `RoPEOps` attach to numr's clients. boostr does not wrap numr types.
- **`QuantTensor`**: a separate type for quantized data, with its own kernels.
- **`impl_generic`**: composite ops built from numr primitives. Every backend runs the same logic.
- **Per-backend kernels**: dequantization, quantized matmul, and fused attention use SIMD, PTX, or WGSL kernels.
- **No vendor libraries**: boostr uses no cuBLAS, cuDNN, or MKL.

## Module Overview

- **`ops/`**: ML-specific operations (attention, RoPE, MoE, and others)
- **`quant/`**: Quantized tensors and kernels (27 formats)
- **`nn/`**: Neural network modules (Embedding, RmsNorm, LayerNorm, GroupNorm, Conv1d, Conv2d, LoRA linear)
- **`model/`**: Model architectures (LLaMA, Mamba2, hybrid, Qwen3.5, Qwen3-VL, encoders, audio models, spaCy tagger)
- **`format/`**: Model loaders (SafeTensors, GGUF, TCF, PyTorch `.pt`)
- **`inference/`**: Inference infrastructure (KV caches, scheduling, speculative decoding, prefix caching)
- **`optimizer/`**: Training optimizers and gradient utilities
- **`trainer/`**: Training loop, checkpoints, and mixed precision
- **`distributed/`**: Multi-GPU coordination, including ZeRO, tensor parallelism, and pipeline parallelism

For the rules on file layout and the `impl_generic` pattern, see [CONTRIBUTING.md](../CONTRIBUTING.md#architecture).
