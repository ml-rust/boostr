# Third-Party Notices

boostr is licensed under Apache-2.0 (see `LICENSE`). Parts of its source port,
transcribe, or reimplement code from the projects below. Each section names the
upstream, its copyright line, its license, and the boostr paths that draw on it.

The full Apache License 2.0 text is in `LICENSE`. Each MIT section carries the
full MIT license text, as that license requires.

## Apache License 2.0

### Hugging Face Transformers

- Upstream: https://github.com/huggingface/transformers
- Copyright 2018- The Hugging Face team. All rights reserved.
- License: Apache-2.0
- boostr paths:
  - `src/model/audio/vits/` (the VITS / MMS-TTS model)
  - `src/model/audio/clap/` (the CLAP audio tower, including `swin_block.rs`,
    `window_attention.rs`, `stage.rs`, `audio_encoder.rs`, `projection.rs`)
  - `src/model/audio/whisper_model/alignment.rs` (Whisper word-timestamp alignment)
  - `src/model/audio/mel/` (log-mel feature extraction conventions)

### hexgrad Kokoro

- Upstream: https://github.com/hexgrad/kokoro
- Copyright hexgrad. The upstream LICENSE file carries no copyright line.
- License: Apache-2.0
- boostr paths: `src/model/audio/kokoro/`, including `kokoro_v2.rs`, and the
  defaults in `src/model/config/tts.rs`

### vLLM

- Upstream: https://github.com/vllm-project/vllm
- Copyright contributors to the vLLM project.
- License: Apache-2.0
- boostr paths:
  - `src/ops/cuda/kernels/cache/reshape_and_cache.cu`
  - `src/ops/cuda/kernels/attention/decode_attention.cu`

### OpenBMB VoxCPM

- Upstream: https://github.com/OpenBMB/VoxCPM
- Copyright OpenBMB. The upstream LICENSE file carries no copyright line.
- License: Apache-2.0
- boostr paths: `src/model/audio/voxcpm/`

### Neuphonic NeuCodec

- Upstream: https://github.com/neuphonic/neucodec
- Copyright Neuphonic. The upstream LICENSE file carries no copyright line.
- License: Apache-2.0
- boostr paths: `src/model/audio/neucodec/`

## MIT License

### llama.cpp / ggml

- Upstream: https://github.com/ggml-org/llama.cpp
- License: MIT
- boostr paths:
  - `src/quant/cuda/kernels/` (the `.cu` / `.cuh` files that cite
    llama.cpp or ggml, including `iq_grid.cuh`, `mma_int8.cuh`, `gemm/` and `gemv/`)
  - `src/quant/cpu/kernels/` (GGUF block dequantization and quantization,
    `iq_grid/` codebook tables, `quantize/`)
  - `src/ops/impl_generic/architecture/gated_delta_net/` and
    `src/ops/traits/architecture/gated_delta_net.rs`
  - `src/ops/impl_generic/position/mrope.rs` and `src/ops/traits/position/mrope.rs`
  - `src/model/hybrid/blocks/gdn/`
  - `src/format/gguf/hadamard_contract/`
  - `src/ops/cuda/kernels/attention/decode_attention.cu`
  - `src/ops/cuda/kernels/inference/sampling_penalties.cu`

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### spaCy

- Upstream: https://github.com/explosion/spaCy
- License: MIT
- boostr paths: `src/model/text/spacy_tagger/`

```text
The MIT License (MIT)

Copyright (C) 2016-2024 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### thinc

- Upstream: https://github.com/explosion/thinc
- License: MIT
- boostr paths: `src/model/text/spacy_tagger/` (`Maxout`, `HashEmbed`,
  `seq2col` and the MurmurHash3 variant in `hash.rs`)

```text
The MIT License (MIT)

Copyright (C) 2016 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### vector-quantize-pytorch

- Upstream: https://github.com/lucidrains/vector-quantize-pytorch
- License: MIT
- boostr paths: `src/nn/fsq/` (finite scalar quantization and residual FSQ)

```text
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Silero VAD

- Upstream: https://github.com/snakers4/silero-vad
- License: MIT
- boostr paths: `src/model/audio/vad/`

```text
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### OpenAI Whisper

- Upstream: https://github.com/openai/whisper
- License: MIT
- boostr paths: `src/model/audio/whisper.rs`, `src/model/audio/whisper_model/`,
  `src/model/audio/whisper_decoder/`

```text
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
