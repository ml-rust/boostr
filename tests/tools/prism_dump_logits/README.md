# prism_dump_logits

Oracle for `tests/qwen35_parity.rs`: loads a GGUF model via the PrismML
llama.cpp fork, prefills fixed ids, greedily decodes 3 more, dumps every
position's logit row to a binary file.

Build (`<fork>` = a checkout of `PrismML-Eng/llama.cpp`, branch `prism`, built with `-DGGML_CUDA=ON`):
```bash
g++ -std=c++17 -O2 -I<fork>/include -I<fork>/ggml/include dump_logits.cpp \
  -L<fork>/build/bin -lllama -lggml -lggml-cpu -lggml-cuda -lggml-base \
  -Wl,-rpath,<fork>/build/bin -o dump_logits
```

Fixture for `qwen35_parity.rs` (ids: `"The capital of France is"`, no BOS).
Output goes to `$BOOSTR_BONSAI2_DIR/fixtures/fork_logits_capital_of_france_pq2_0.bin`:
```bash
./dump_logits Ternary-Bonsai-2-27B-PQ2_0.gguf \
  fork_logits_capital_of_france_pq2_0.bin 99 760 6511 314 9338 369
```
