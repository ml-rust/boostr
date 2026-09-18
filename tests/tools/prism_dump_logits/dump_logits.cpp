// dump_logits.cpp
//
// Pure model oracle: loads a GGUF model via the PrismML llama.cpp fork,
// prefills a fixed sequence of already-tokenized ids, greedily decodes 3
// more tokens, and dumps every position's full logit row to a binary file.
//
// No tokenizer use: all token ids are supplied on the command line.
// Deterministic: greedy argmax selection only, no sampling, no RNG.
//
// Usage:
//   dump_logits <model.gguf> <out.bin> <n_gpu_layers> <token_id> [token_id ...]
//
// Output file format (little-endian):
//   u32 n_positions
//   u32 n_vocab
//   n_positions * n_vocab  f32   (row-major, one row per position, in order:
//                                 prefill positions first, then the 3 decode
//                                 positions)
//   u32 n_decoded            (always 3)
//   n_decoded * u32           (the 3 greedily chosen token ids)

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr int kNumDecodeSteps = 3;

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s <model.gguf> <out.bin> <n_gpu_layers> <token_id> [token_id ...]\n",
        argv0);
}

// Finds the argmax over a logits row and returns (token_id, logit_value).
std::pair<llama_token, float> argmax_logits(const float * logits, int32_t n_vocab) {
    llama_token best_id = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id  = i;
        }
    }
    return { best_id, best_val };
}

void write_u32_le(std::ofstream & out, uint32_t v) {
    out.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

void write_f32_row_le(std::ofstream & out, const float * row, int32_t n) {
    // Host is assumed little-endian (x86_64 / aarch64 default), matching the
    // f32 layout llama.cpp already uses internally.
    out.write(reinterpret_cast<const char *>(row), sizeof(float) * static_cast<size_t>(n));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string out_path   = argv[2];

    int n_gpu_layers = 0;
    {
        char * end = nullptr;
        n_gpu_layers = static_cast<int>(std::strtol(argv[3], &end, 10));
        if (end == argv[3]) {
            std::fprintf(stderr, "error: invalid n_gpu_layers '%s'\n", argv[3]);
            return 1;
        }
    }

    std::vector<llama_token> prompt_tokens;
    prompt_tokens.reserve(static_cast<size_t>(argc - 4));
    for (int i = 4; i < argc; i++) {
        char * end = nullptr;
        long id = std::strtol(argv[i], &end, 10);
        if (end == argv[i]) {
            std::fprintf(stderr, "error: invalid token id '%s'\n", argv[i]);
            return 1;
        }
        prompt_tokens.push_back(static_cast<llama_token>(id));
    }

    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "error: no token ids supplied\n");
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "error: unable to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 512;
    ctx_params.n_batch   = 512;
    ctx_params.n_threads = 8;
    ctx_params.n_threads_batch = 8;
    // Flash attention left off (default LLAMA_FLASH_ATTN_TYPE_AUTO is not
    // forced on); KV cache type left at context defaults.
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    std::printf("n_vocab: %d\n", n_vocab);

    const int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());
    const int32_t n_positions = n_prompt + kNumDecodeSteps;

    // All logit rows we will dump, in position order: prefill rows first,
    // then the kNumDecodeSteps decode rows.
    std::vector<std::vector<float>> logit_rows;
    logit_rows.reserve(static_cast<size_t>(n_positions));

    std::vector<llama_token> decoded_ids;
    decoded_ids.reserve(kNumDecodeSteps);

    // --- Prefill: one llama_decode call, logits requested at every position ---
    {
        llama_batch batch = llama_batch_init(n_prompt, /*embd=*/0, /*n_seq_max=*/1);
        batch.n_tokens = n_prompt;
        for (int32_t i = 0; i < n_prompt; i++) {
            batch.token[i]     = prompt_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = true; // request logits for every position
        }

        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "error: llama_decode failed during prefill\n");
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        for (int32_t i = 0; i < n_prompt; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits == nullptr) {
                std::fprintf(stderr, "error: llama_get_logits_ith(%d) returned null during prefill\n", i);
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            logit_rows.emplace_back(logits, logits + n_vocab);

            auto [id, val] = argmax_logits(logits, n_vocab);
            std::printf("pos %d (prefill): argmax_id=%d logit=%.6f\n", i, id, val);
        }

        llama_batch_free(batch);
    }

    // --- Greedy decode: 3 more tokens, one llama_decode per step ---
    {
        // Seed with the argmax of the last prefill position's logits.
        llama_token next_id = argmax_logits(logit_rows.back().data(), n_vocab).first;

        for (int step = 0; step < kNumDecodeSteps; step++) {
            llama_batch batch = llama_batch_get_one(&next_id, 1);
            // llama_batch_get_one leaves batch.logits == nullptr, which means
            // "only the last token is output" -- exactly what we want here
            // since there is only one token in this batch anyway.

            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "error: llama_decode failed during decode step %d\n", step);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }

            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                std::fprintf(stderr, "error: llama_get_logits_ith(-1) returned null during decode step %d\n", step);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            logit_rows.emplace_back(logits, logits + n_vocab);

            auto [id, val] = argmax_logits(logits, n_vocab);
            std::printf("pos %d (decode step %d): argmax_id=%d logit=%.6f\n",
                        n_prompt + step, step, id, val);

            decoded_ids.push_back(id);
            next_id = id;
        }
    }

    std::printf("decoded ids:");
    for (llama_token id : decoded_ids) {
        std::printf(" %d", id);
    }
    std::printf("\n");

    // --- Write out.bin ---
    {
        std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "error: unable to open '%s' for writing\n", out_path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        write_u32_le(out, static_cast<uint32_t>(n_positions));
        write_u32_le(out, static_cast<uint32_t>(n_vocab));
        for (const auto & row : logit_rows) {
            write_f32_row_le(out, row.data(), n_vocab);
        }
        write_u32_le(out, static_cast<uint32_t>(kNumDecodeSteps));
        for (llama_token id : decoded_ids) {
            write_u32_le(out, static_cast<uint32_t>(id));
        }

        if (!out) {
            std::fprintf(stderr, "error: write to '%s' failed\n", out_path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
