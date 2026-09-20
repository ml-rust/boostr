// Oracle for `tests/qwen35_vision_parity.rs`: the reference decoder's
// last-position logits for a prompt holding one image, plus the token ids
// the prompt tokenizes to and the greedy continuation.
//
// Build (from this directory, FORK = the reference llama.cpp checkout with a
// built `build/bin`):
//   gcc -O1 -o mtmd_logits_dump main.c -I$FORK/include -I$FORK/ggml/include \
//       -I$FORK/tools/mtmd -L$FORK/build/bin -lmtmd -lllama -lggml -lggml-base -lm \
//       -Wl,-rpath,$FORK/build/bin
// Run:
//   ./mtmd_logits_dump <text.gguf> <mmproj.gguf> <image> <n_gpu_layers> \
//       <out.logits> <out.json>
//
// The prompt is fixed to the chat form below with the media marker in the
// user turn. The vision tower runs on the CPU; `n_gpu_layers` applies to
// the text model only.
//
// `out.logits`: u32 n_vocab, then n_vocab f32 — the logits of the last
//               prompt row (the row after `<|vision_end|>...assistant\n`).
// `out.json`:   `ids` (prompt ids with one `image_pad` in place of the
//               image), `nx`, `ny`, `n_image_tokens`, the three marker ids,
//               `n_past` after the prompt, `argmax` of the dumped row, and
//               `greedy` (that argmax followed by 3 greedy decode ids).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define N_DECODE 3

static const char *PROMPT =
    "<|im_start|>user\n<__media__>Describe.<|im_end|>\n<|im_start|>assistant\n";

static int32_t single_token(const struct llama_vocab *vocab, const char *text) {
    llama_token out[4];
    int32_t n = llama_tokenize(vocab, text, (int32_t)strlen(text), out, 4, false, true);
    if (n != 1) {
        fprintf(stderr, "'%s' tokenizes to %d tokens, want 1\n", text, n);
        exit(1);
    }
    return out[0];
}

static int32_t argmax(const float *row, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <text.gguf> <mmproj.gguf> <image> <n_gpu_layers> <out.logits> <out.json>\n",
                argv[0]);
        return 2;
    }
    const char *text_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = argv[3];
    int n_gpu_layers = atoi(argv[4]);
    const char *out_logits = argv[5];
    const char *out_json = argv[6];

    ggml_backend_load_all();
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    struct llama_model *model = llama_model_load_from_file(text_path, mp);
    if (!model) { fprintf(stderr, "model load failed: %s\n", text_path); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.no_perf = true;
    struct llama_context *lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "context init failed\n"); return 1; }

    struct mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false;
    mcp.n_threads = 8;
    mcp.print_timings = false;
    mcp.warmup = false;
    mcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_context *mctx = mtmd_init_from_file(mmproj_path, model, mcp);
    if (!mctx) { fprintf(stderr, "mtmd init failed: %s\n", mmproj_path); return 1; }

    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, image_path, false);
    if (!bw.bitmap) { fprintf(stderr, "bitmap load failed: %s\n", image_path); return 1; }

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    struct mtmd_input_text text = { PROMPT, strlen(PROMPT), true, true };
    const mtmd_bitmap *bitmaps[1] = { bw.bitmap };
    if (mtmd_tokenize(mctx, chunks, &text, bitmaps, 1) != 0) {
        fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }

    const int32_t vision_start = single_token(vocab, "<|vision_start|>");
    const int32_t vision_end = single_token(vocab, "<|vision_end|>");
    const int32_t image_pad = single_token(vocab, "<|image_pad|>");

    // Prompt ids with one image_pad per image chunk.
    int32_t *ids = NULL;
    size_t n_ids = 0, cap_ids = 0;
    uint32_t nx = 0, ny = 0, n_image_tokens = 0;
    int n_images = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token *toks = mtmd_input_chunk_get_tokens_text(c, &n);
            for (size_t k = 0; k < n; k++) {
                if (n_ids == cap_ids) {
                    cap_ids = cap_ids ? cap_ids * 2 : 64;
                    ids = (int32_t *)realloc(ids, cap_ids * sizeof(int32_t));
                }
                ids[n_ids++] = toks[k];
            }
        } else if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens *img = mtmd_input_chunk_get_tokens_image(c);
            n_image_tokens = (uint32_t)mtmd_image_tokens_get_n_tokens(img);
            struct mtmd_decoder_pos last_pos =
                mtmd_image_tokens_get_decoder_pos(img, 0, n_image_tokens - 1);
            nx = (uint32_t)last_pos.x + 1;
            ny = (uint32_t)last_pos.y + 1;
            if (n_ids == cap_ids) {
                cap_ids = cap_ids ? cap_ids * 2 : 64;
                ids = (int32_t *)realloc(ids, cap_ids * sizeof(int32_t));
            }
            ids[n_ids++] = image_pad;
            n_images++;
        } else {
            fprintf(stderr, "unexpected chunk type\n");
            return 1;
        }
    }
    if (n_images != 1) { fprintf(stderr, "%d image chunks, want 1\n", n_images); return 1; }

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, (int32_t)cp.n_batch, true, &n_past) != 0) {
        fprintf(stderr, "mtmd_helper_eval_chunks failed\n");
        return 1;
    }
    const float *last = llama_get_logits_ith(lctx, -1);
    if (!last) { fprintf(stderr, "no logits at the last row\n"); return 1; }
    float *row = (float *)malloc((size_t)n_vocab * sizeof(float));
    memcpy(row, last, (size_t)n_vocab * sizeof(float));
    int32_t greedy[N_DECODE + 1];
    greedy[0] = argmax(row, n_vocab);

    // Greedy continuation: each token at rope position n_past, the value
    // the helper leaves after the prompt (behind the KV slot count).
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int step = 0; step < N_DECODE; step++) {
        batch.n_tokens = 1;
        batch.token[0] = greedy[step];
        batch.pos[0] = n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "decode step %d failed\n", step); return 1; }
        greedy[step + 1] = argmax(llama_get_logits_ith(lctx, -1), n_vocab);
    }
    llama_batch_free(batch);

    FILE *f = fopen(out_logits, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_logits); return 1; }
    uint32_t nv = (uint32_t)n_vocab;
    fwrite(&nv, 4, 1, f);
    fwrite(row, sizeof(float), (size_t)n_vocab, f);
    fclose(f);

    f = fopen(out_json, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_json); return 1; }
    fprintf(f, "{\n  \"ids\": [");
    for (size_t i = 0; i < n_ids; i++) fprintf(f, "%s%d", i ? ", " : "", ids[i]);
    fprintf(f, "],\n  \"nx\": %u,\n  \"ny\": %u,\n  \"n_image_tokens\": %u,\n", nx, ny, n_image_tokens);
    fprintf(f, "  \"vision_start\": %d,\n  \"vision_end\": %d,\n  \"image_pad\": %d,\n",
            vision_start, vision_end, image_pad);
    fprintf(f, "  \"n_past\": %d,\n  \"n_vocab\": %d,\n  \"argmax\": %d,\n  \"greedy\": [",
            (int)(n_past - N_DECODE), n_vocab, greedy[0]);
    for (int i = 0; i <= N_DECODE; i++) fprintf(f, "%s%d", i ? ", " : "", greedy[i]);
    fprintf(f, "],\n  \"n_gpu_layers\": %d\n}\n", n_gpu_layers);
    fclose(f);

    printf("n_ids=%zu nx=%u ny=%u n_image_tokens=%u n_past=%d argmax=%d greedy=%d %d %d %d\n",
           n_ids, nx, ny, n_image_tokens, (int)(n_past - N_DECODE), greedy[0],
           greedy[0], greedy[1], greedy[2], greedy[3]);

    free(row);
    free(ids);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
