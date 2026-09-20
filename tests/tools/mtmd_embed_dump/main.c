// Oracle dump for the Qwen3-VL vision encoder in the reference mtmd library.
//
// Build (from this directory, FORK = the reference llama.cpp checkout with a
// built `build/bin`):
//   gcc -O1 -o mtmd_embed_dump main.c -I$FORK/include -I$FORK/ggml/include \
//       -I$FORK/tools/mtmd -L$FORK/build/bin -lmtmd -lllama -lggml -lggml-base -lm \
//       -Wl,-rpath,$FORK/build/bin
// Run:
//   ./mtmd_embed_dump <text.gguf> <mmproj.gguf> <image.png> <out.embd> <out.u8chw>
// Fixture PNGs come from `gen_fixture_images.py`; `img_*.f32.embd` fixtures
// come from the same run against an F32 copy of the mmproj made with
// `mmproj_to_f32.py`.
//
// `out.embd`:  u32 n_tokens, u32 n_embd, u32 nx, u32 ny, then
//              n_tokens*n_embd f32 row-major (token-major).
// `out.u8chw`: u32 width, u32 height, then 3*height*width u8 planar CHW —
//              the resized, padded image the encoder consumed, recovered from
//              the first im2col node of the graph (the graph input is not a
//              node, so the eval callback never sees it; im2col holds every
//              pixel once, cast to f16, and the u8 grid is recovered exactly
//              because the f16 step is far below one u8 step after
//              normalization).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

struct capture {
    int done;
    int64_t ne[4];
    enum ggml_type type;
    float *data; // f32, ne[0]*ne[1]*ne[2]*ne[3]
};

static bool eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    struct capture *cap = (struct capture *)user_data;
    if (t->op != GGML_OP_IM2COL || cap->done) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    size_t n = (size_t)ggml_nelements(t);
    cap->type = t->type;
    for (int i = 0; i < 4; i++) cap->ne[i] = t->ne[i];
    cap->data = (float *)malloc(n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, cap->data, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t *tmp = (ggml_fp16_t *)malloc(n * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(t, tmp, 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp, cap->data, (int64_t)n);
        free(tmp);
    } else {
        fprintf(stderr, "im2col node has unexpected type %d\n", (int)t->type);
        exit(3);
    }
    cap->done = 1;
    return true;
}

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }

// The vocab-only text model reports no input width, so the projector width
// comes from the mmproj header.
static uint32_t read_projection_dim(const char *mmproj_path) {
    struct gguf_init_params ip = { true, NULL };
    struct gguf_context *g = gguf_init_from_file(mmproj_path, ip);
    if (!g) return 0;
    int64_t key = gguf_find_key(g, "clip.vision.projection_dim");
    uint32_t v = key < 0 ? 0 : gguf_get_val_u32(g, key);
    gguf_free(g);
    return v;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <text.gguf> <mmproj.gguf> <image> <out.embd> <out.u8chw>\n", argv[0]);
        return 2;
    }
    const char *text_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = argv[3];
    const char *out_embd = argv[4];
    const char *out_u8 = argv[5];

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    struct llama_model *model = llama_model_load_from_file(text_path, mp);
    if (!model) { fprintf(stderr, "text model load failed: %s\n", text_path); return 1; }

    struct capture cap;
    memset(&cap, 0, sizeof(cap));

    struct mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.n_threads = 8;
    cp.print_timings = false;
    cp.warmup = false;
    // The non-flash path keeps K/V in f32; flash attention casts them to f16.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &cap;
    mtmd_context *ctx = mtmd_init_from_file(mmproj_path, model, cp);
    if (!ctx) { fprintf(stderr, "mtmd init failed: %s\n", mmproj_path); return 1; }

    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image_path, false);
    if (!bw.bitmap) { fprintf(stderr, "bitmap load failed: %s\n", image_path); return 1; }

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const char *prompt = mtmd_default_marker();
    struct mtmd_input_text text = { prompt, strlen(prompt), false, true };
    const mtmd_bitmap *bitmaps[1] = { bw.bitmap };
    int32_t rc = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
    if (rc != 0) { fprintf(stderr, "mtmd_tokenize failed: %d\n", rc); return 1; }

    const mtmd_input_chunk *img_chunk = NULL;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img_chunk = c; break; }
    }
    if (!img_chunk) { fprintf(stderr, "no image chunk\n"); return 1; }

    const mtmd_image_tokens *img = mtmd_input_chunk_get_tokens_image(img_chunk);
    size_t n_tokens = mtmd_image_tokens_get_n_tokens(img);
    struct mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(img, 0, n_tokens - 1);
    uint32_t nx = last.x + 1;
    uint32_t ny = last.y + 1;

    rc = mtmd_encode_chunk(ctx, img_chunk);
    if (rc != 0) { fprintf(stderr, "mtmd_encode_chunk failed: %d\n", rc); return 1; }
    const float *embd = mtmd_get_output_embd(ctx);
    uint32_t n_embd = read_projection_dim(mmproj_path);
    if (n_embd == 0) { fprintf(stderr, "clip.vision.projection_dim missing in %s\n", mmproj_path); return 1; }

    FILE *f = fopen(out_embd, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_embd); return 1; }
    write_u32(f, (uint32_t)n_tokens);
    write_u32(f, n_embd);
    write_u32(f, nx);
    write_u32(f, ny);
    fwrite(embd, sizeof(float), n_tokens * n_embd, f);
    fclose(f);

    if (!cap.done) { fprintf(stderr, "im2col node not captured\n"); return 1; }
    // im2col ne = [IC*KH*KW, OW, OH, N]; patch 16x16, 3 channels.
    const int64_t kk = cap.ne[0];
    const int64_t ow = cap.ne[1];
    const int64_t oh = cap.ne[2];
    if (kk != 3 * 16 * 16) { fprintf(stderr, "im2col inner dim %lld != 768\n", (long long)kk); return 1; }
    uint32_t width = (uint32_t)(ow * 16);
    uint32_t height = (uint32_t)(oh * 16);
    uint8_t *chw = (uint8_t *)malloc((size_t)3 * width * height);
    for (int c = 0; c < 3; c++) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                int64_t p = (int64_t)(y / 16) * ow + (x / 16);
                int64_t idx = p * kk + c * 256 + (y % 16) * 16 + (x % 16);
                float v = cap.data[idx];
                long u = lroundf((v * 0.5f + 0.5f) * 255.0f);
                if (u < 0) u = 0;
                if (u > 255) u = 255;
                chw[((size_t)c * height + y) * width + x] = (uint8_t)u;
            }
        }
    }
    f = fopen(out_u8, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_u8); return 1; }
    write_u32(f, width);
    write_u32(f, height);
    fwrite(chw, 1, (size_t)3 * width * height, f);
    fclose(f);

    printf("n_tokens=%zu n_embd=%u nx=%u ny=%u image=%ux%u\n", n_tokens, n_embd, nx, ny, width, height);

    free(chw);
    free(cap.data);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
