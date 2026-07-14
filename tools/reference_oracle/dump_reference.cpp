// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
//
// Arch-agnostic llama.cpp / GGUF per-layer fp32 reference-oracle dumper.
//
// A thin variant of llama.cpp's `examples/eval-callback` that, instead of
// pretty-printing a 3-element slice of every graph tensor, writes the FULL
// value of graph activation tensors to disk as contiguous fp32 blobs + a
// manifest. It runs the stock llama.cpp graph for WHATEVER arch the GGUF
// declares (`qwen35`, `qwen3next`, `gemma3`, `glm4moe`, `deepseek2`, `lfm2`,
// `hunyuan-moe`, `gemma4`, …) on a FIXED prompt, and is meant to run on CPU
// (`-ngl 0`) so every op is computed in fp32 — the highest-precision, most
// trustworthy oracle for an fni8 / fni8-serve kernel or model port to TDD
// against. No llama.cpp source is modified; we link its prebuilt libraries.
//
// This is the GENERALIZED successor to flint8's Qwen3.5-only dump_reference.cpp
// (flint8 #105). The only Qwen3.5-specific thing there was the hardcoded tensor
// name-set and prompt; here BOTH are parameters:
//   - prompt / model / n_ctx: ordinary llama.cpp CLI flags (-p, -m, -c ...).
//   - which tensors to dump: env FNI8_DUMP_PATTERNS (comma-separated name
//     prefixes, each matched up to end-of-string or a '-<layer>' digit). If
//     UNSET, dump EVERY named, non-quantized f32/f16/bf16/i32 graph activation
//     (the full eval-callback set) — always safe, the manifest records shapes.
//
// Build: see build_dump.sh (links the prebuilt on-box llama.cpp libraries).
// Run:   FNI8_DUMP_DIR=<out> ./dump_reference -m <model.gguf> -ngl 0 \
//            -c 512 --temp 0 -p "<fixed prompt>"
//
// Output layout (in FNI8_DUMP_DIR):
//   <sanitized_name>.f32   raw little-endian fp32, ne0 (fastest) .. ne3
//   manifest.tsv           name<TAB>ne0<TAB>ne1<TAB>ne2<TAB>ne3<TAB>op<TAB>file
//   meta.txt               model path, prompt, add_bos, token ids, n_tokens,
//                          llama.cpp commit (recorded by build_dump.sh)
//
// The logical shape of each blob is [ne3, ne2, ne1, ne0] row-major (ne0 is the
// contiguous/fastest dimension, matching ggml). Tensors that recur under the
// same name in the graph settle to their LAST value (e.g. Qcur-/Kcur- are the
// post-RoPE result) — exactly what a kernel checks its output against.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

std::string          g_outdir;
std::vector<std::string> g_patterns;  // from FNI8_DUMP_PATTERNS; empty => dump-all
std::map<std::string, std::string> g_manifest;  // name -> row (last-wins)
std::vector<uint8_t> g_raw;   // scratch for backend tensor fetch
std::vector<float>   g_flat;  // scratch for de-strided fp32 output

// A name matches a pattern if it starts with the pattern AND the next char is
// end-of-string or a digit (the "-<layer>" index): "gate-" matches "gate-7"
// but NOT "gate_sigmoid-7". With no patterns configured, everything is wanted.
bool wanted(const char* name) {
  if (g_patterns.empty()) return true;
  for (const std::string& p : g_patterns) {
    const size_t lp = p.size();
    if (std::strncmp(name, p.c_str(), lp) == 0) {
      const char c = name[lp];
      if (c == '\0' || (c >= '0' && c <= '9')) return true;
    }
  }
  return false;
}

std::string sanitize(const char* name) {
  std::string s(name);
  for (char& c : s) {
    if (!(std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.')) c = '_';
  }
  return s;
}

// Read one element as fp32 regardless of source dtype.
float elem_f32(const uint8_t* base, ggml_type type, const size_t* nb,
               int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
  const size_t off = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
  const uint8_t* d = base + off;
  switch (type) {
    case GGML_TYPE_F32:  return *(const float*)d;
    case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t*)d);
    case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t*)d);
    case GGML_TYPE_I32:  return (float)*(const int32_t*)d;
    default:             return 0.0f;  // unsupported -> caller skips
  }
}

bool dump_cb(ggml_tensor* t, bool ask, void* /*ud*/) {
  if (ask) return true;                         // yes, we want the data
  if (!t->name[0]) return true;                 // skip unnamed nodes
  if (!wanted(t->name)) return true;
  if (ggml_is_quantized(t->type)) return true;  // never dump quantized nodes
  if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 &&
      t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_I32) return true;

  const bool is_host = ggml_backend_buffer_is_host(t->buffer);
  const uint8_t* base;
  if (is_host) {
    base = (const uint8_t*)t->data;
  } else {
    const size_t n = ggml_nbytes(t);
    g_raw.resize(n);
    ggml_backend_tensor_get(t, g_raw.data(), 0, n);
    base = g_raw.data();
  }

  const int64_t* ne = t->ne;
  const size_t*  nb = t->nb;
  const size_t   count = (size_t)ne[0] * ne[1] * ne[2] * ne[3];
  g_flat.resize(count);
  size_t idx = 0;
  for (int64_t i3 = 0; i3 < ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < ne[2]; ++i2)
      for (int64_t i1 = 0; i1 < ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < ne[0]; ++i0)
          g_flat[idx++] = elem_f32(base, t->type, nb, i0, i1, i2, i3);

  const std::string fname = sanitize(t->name) + ".f32";
  const std::string path  = g_outdir + "/" + fname;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "dump: cannot open %s\n", path.c_str()); return true; }
  std::fwrite(g_flat.data(), sizeof(float), count, f);
  std::fclose(f);

  char row[512];
  snprintf(row, sizeof(row), "%s\t%lld\t%lld\t%lld\t%lld\t%s\t%s", t->name,
           (long long)ne[0], (long long)ne[1], (long long)ne[2], (long long)ne[3],
           ggml_op_desc(t), fname.c_str());
  g_manifest[t->name] = row;   // last-wins: matches the settled on-disk .f32
  return true;
}

void load_patterns() {
  const char* env = std::getenv("FNI8_DUMP_PATTERNS");
  if (!env || !*env) return;   // dump-all
  std::string s(env), cur;
  for (char c : s) {
    if (c == ',') { if (!cur.empty()) g_patterns.push_back(cur); cur.clear(); }
    else if (!std::isspace((unsigned char)c)) cur += c;
  }
  if (!cur.empty()) g_patterns.push_back(cur);
}

}  // namespace

static bool run(llama_context* ctx, const common_params& params) {
  const llama_model* model = llama_get_model(ctx);
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const bool add_bos = llama_vocab_get_add_bos(vocab);

  std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
  if (tokens.empty()) { LOG_ERR("no input tokens\n"); return false; }

  // meta.txt: everything needed to reproduce this dump byte-for-byte.
  const std::string meta_path = g_outdir + "/meta.txt";
  if (FILE* m = std::fopen(meta_path.c_str(), "wb")) {
    fprintf(m, "model\t%s\n", params.model.path.c_str());
    fprintf(m, "prompt\t%s\n", params.prompt.c_str());
    fprintf(m, "add_bos\t%d\n", (int)add_bos);
    fprintf(m, "n_tokens\t%zu\n", tokens.size());
    fprintf(m, "tokens\t");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(m, "%d%s", tokens[i], i + 1 < tokens.size() ? "," : "");
    fprintf(m, "\n");
    const char* commit = std::getenv("FNI8_LLAMA_COMMIT");
    fprintf(m, "llama_commit\t%s\n", commit ? commit : "unknown");
    fprintf(m, "dump_patterns\t%s\n", g_patterns.empty() ? "(all)" : std::getenv("FNI8_DUMP_PATTERNS"));
    std::fclose(m);
  }

  LOG_INF("dump: %zu tokens -> %s\n", tokens.size(), g_outdir.c_str());
  if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
    LOG_ERR("decode failed\n");
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  std::setlocale(LC_NUMERIC, "C");

  const char* od = std::getenv("FNI8_DUMP_DIR");
  if (!od) { fprintf(stderr, "set FNI8_DUMP_DIR to the output directory\n"); return 1; }
  g_outdir = od;
  load_patterns();

  common_params params;
  common_init();
  if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

  llama_backend_init();
  llama_numa_init(params.numa);

  params.cb_eval           = dump_cb;
  params.cb_eval_user_data = nullptr;
  params.warmup            = false;

  auto llama_init = common_init_from_params(params);
  auto* model = llama_init->model();
  auto* ctx   = llama_init->context();
  if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }

  const bool ok = run(ctx, params);

  // Write the manifest once, last-wins per tensor name (matches on-disk .f32).
  const std::string man_path = g_outdir + "/manifest.tsv";
  if (FILE* mf = std::fopen(man_path.c_str(), "wb")) {
    fprintf(mf, "name\tne0\tne1\tne2\tne3\top\tfile\n");
    for (const auto& kv : g_manifest) fprintf(mf, "%s\n", kv.second.c_str());
    std::fclose(mf);
  }
  llama_backend_free();
  return ok ? 0 : 1;
}
