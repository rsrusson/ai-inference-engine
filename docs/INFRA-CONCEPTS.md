# INFRA-CONCEPTS.md — Serving & memory cheat sheet

Companion reference to this repo's experiments (see `loadtest/README.md` for the
measured numbers and `PLAN.md` for the roadmap). Written to prepare for
AI-infrastructure interviews: it explains the levers, units, and arithmetic that
come up when operating a vLLM (or similar) serving engine on a small GPU.

All concrete figures below use `Qwen/Qwen2.5-0.5B-Instruct` on an RTX 3070 Laptop
8 GB — the model/card this repo is pinned to.

---

## 1. The three length/capacity knobs (don't confuse them)

| Knob | Set where | Controls | Analogy |
|---|---|---|---|
| `--max-model-len` | server startup | longest **any one sequence** may be (prompt + generated) | size of one container |
| `--max-num-seqs` | server startup | how many **sequences run concurrently** | how many containers fit side by side |
| `--max-tokens` | per request (JSON) | how many NEW tokens **that caller** may generate | how full one container can be filled |

- Total context length of a request (= its prompt tokens **+** its `max_tokens`)
  must be ≤ `--max-model-len`, or the request is rejected:
  ```
  max_tokens=5000 cannot be greater than max_model_len=4096  (HTTP 400)
  ```
- `--max-model-len` also sizes CUDA-graph capture; set too high it shrinks cache
  headroom (Phase 2.2 §B). `--max-num-seqs` bounds simultaneous resident sequences
  to keep memory/latency predictable under load.

---

## 2. `--ignore_eos` / EOS (what "end of generation" means)

During decoding the model emits one token per step. **EOS** (end-of-sequence, e.g.
`<|endoftext|>`) is a special vocabulary token; when the model emits it, generation
**stops** and the response ends. That is why a short prompt stops ~100 tokens even
when you allowed 128.

`ignore_eos=true` tells the model **"don't use EOS as a stop signal"** — it keeps
decoding until it hits `--max-tokens`. It has *nothing* to do with padding/filler
*between* sequences; it only prevents early stop *within* one sequence.

Why experiments need it: a normal request stops quickly, so vLLM frees its KV
blocks fast and you never see the cache fill. Forcing a long generation keeps the
sequence resident so PagedAttention usage is observable (`vram_watch.py --ignore-eos`).

---

## 3. Sequence = request (roughly)

A **sequence** is one generation run: the prompt plus everything it generates, the
atomic scheduling unit. **One user `POST` = one sequence.** N concurrent requests
⇒ N resident sequences (governed by scheduler and `--max-num-seqs`).

A single human user can send many requests => many sequences. Sequence counting is
about requests in flight, not user identity.

---

## 4. What a token is (and "how big")

A token is the discrete unit the **tokenizer** produces. It is deterministic (BPE),
not a guess:
- ~3/4 of an English word on average; punctuation/spaces count.
- Common words can be 1 token; long/rare words split into several.
- Rule of thumb ~4 chars ≈ 1 token (≈0.75 words) is an *estimate for sizing/cost*,
  not an internal unit. Exact counts come from the tokenizer's fixed vocabulary.

For a given prompt you can query the exact count:
```bash
curl -s http://127.0.0.1:8000/tokenize -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain what an inference-serving engine does."}'
```

---

## 5. What a GPU "block" (PagedAttention page) is

A **block** is vLLM's fixed-size KV-cache allocation unit: K & V for a fixed run of
tokens, for **every** KV layer, all at once.

- Block size for these kernels = **16 tokens** (vLLM's standard default; our
  `/metrics` reported `block_size="16"`).
- Physical size of one block here = `16 tokens × 12 KB/token` = **192 KB** of KV.
- A sequence's KV spans many **non-contiguous** blocks (that is the point of
  paging: no giant contiguous carve-out, low fragmentation). The LAST block of a
  sequence is usually half-empty => **fragmentation** (interview topic).
- `num_gpu_blocks` = how many 16-token pages fit the reserved cache.
  e.g. 30081 blocks × 16 = 481,296 tokens = `kv_cache_size_tokens`.

> ⚠️ **Common gotcha — "12 KB/token" is NOT the token's text size.**
> Don't confuse two unrelated numbers:
> - **Text size:** a token ≈ ~4 characters ≈ ~4 bytes, and 16 tokens ≈ ~64 bytes.
>   That estimates the *text/input* footprint (pricing / tokenizer guesses).
> - **KV size:** in the cache each token stores not its characters but its
>   per-layer *attention vectors* — K & V values at every layer, for every KV head.
>   Text bytes and model-internal vector bytes are different storage media, so you
>   cannot multiply "16 tokens × 4 B/token" to get the block size.
> - Ground-truth check: if KV were ~4 B/token, the whole cache for 481,296 tokens
>   would be ~2 MB — but vLLM actually reserves **~5.5 GiB** for it. `481296 × 12 KB`
>   ≈ 5.5 GiB; `481296 × 4 B` ≈ 2 MB (off by ~1000×). Only the 12 KB/token figure is
>   consistent with the measured reservation.

### Why per-token KV is so big (the intuition)

One token's characters are a handful of bytes, but the cache stores the token's
**internal vectors across the whole model**, which is far larger. Unpack it:

A KV cache holds, for **every layer**, a *key* vector (K) and a *value* vector (V),
per **KV head**. Per-token state = `layers × kv_heads × head_dim` numbers, doubled
for K + V, at `dtype` bytes each:
- per layer, per head: a 64-dim K vector + a 64-dim V vector = 128 numbers
- × 2 KV heads per layer = 256 numbers / layer
- × 24 layers = 6,144 numbers / token
- × 2 bytes (bf16) = **12,288 B = 12 KB / token**

```text
one token @ decode:
  attention_state (cached, reused later):  24 layers × 2 heads × 2 (K,V) × 64 dims × 2 B = 12 KB
  its "input" text:                        ~4 chars × 1 B                              ~= 4 B
```

So the cache stores decoder linear-algebra state that will be reused later — not the
token's spelling. Text bytes are irrelevant to the KV figure, which is why your
"16 tokens × 4 B" estimate undercounts by ~1000×.

---

## 6. Reading `nvidia-smi` (VRAM vs cache-usage)

`nvidia-smi` = NVIDIA System Management Interface: GPU temp/power/utilization and,
most usefully, **VRAM used/free** per device.

Idle used-memory (~7.27 GB here @ util 0.85) ≈ model weights + **pre-allocated**
KV cache + CUDA context + compiled kernels + CUDA-graph buffers. Because vLLM
pre-allocates the cache at boot, **idle ≈ under-load used** — nvidia-smi barely
moves with traffic. It shows the *physical reservation*, NOT how full the logical
cache is. The metric that actually rises with load is:

```
vllm:kv_cache_usage_perc   (0 = empty, 1 = 100% of the block pool consumed)
```

Do not use nvidia-smi alone to judge "memory pressure" on vLLM; watch
`kv_cache_usage_perc`.

---

## 7. KV-cache arithmetic (where the numbers come from)

**Formula for KV bytes needed per stored token:**

```
KV bytes/token = 2 (K & V)  ×  L  ×  H_kv  ×  D_head  ×  dtype_bytes
```

| Term | Symbol | Where it comes from | 0.5B value |
|---|---|---|---|
| both projections | 2 | constant (key + value caches) | 2 |
| layers | `num_hidden_layers` | model `config.json` | 24 |
| KV(GQA) heads | `num_key_value_heads` | model `config.json` | 2 |
| head dim | `hidden_size / num_attention_heads` | computed from config (896/14) | 64 |
| bytes/val | dtype | serving dtype (`bf16`) | 2 |

⇒ `2 × 24 × 2 × 64 × 2 = 12,288 B ≈ 12 KB / token`.

*(This is KV/attention state, not the token's text bytes — see the gotcha and
intuition in §5 why "16 tokens × ~4 B" massively undercounts a block.)*

Cross-checks that validate it (repo observed):
- Total cache ≈ `kv_cache_size_tokens` × 12 KB:
  `481296 × 12288 B ≈ 5.5 GiB` ≈ vLLM's reported "Available KV cache memory ~5.1–5.5 GiB".
- One block = 16 tokens × 12 KB = 192 KB; `num_gpu_blocks` × 16 = `kv_cache_size_tokens`.

**Why `H_kv=2` and not 14 attention heads:** Qwen2 uses **GQA** (grouped-query
attention) — 14 *query* heads share just 2 *key/value* head pairs, slashing cache.
It is a config-file design choice, not an arbitrary constant.

**Interview takeaways — know the LEVERS, not magic numbers:**
- Halve `dtype_bytes` (fp16→fp8) ⇒ KV halves.
- More layers / more KV heads / larger head-dim ⇒ more KV per token.
- GQA (fewer KV heads than Q heads) ⇒ much less cache than MHA.
- Concrete 24 / 2 / 64 / 12 KB are *this model's* worked example — for another
  model reread `config.json` and recompute. Engineers derive; they don't memorize.

---

## 8. Mental map of a working setup (all-in-one)

```
server start:
  vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --gpu-memory-utilization 0.85 \   # what fraction of VRAM vLLM may reserve (ROI: weights+cache+graphs)
    --max-model-len 2048 \            # longest single sequence allowed
    [--max-num-seqs 32]               # optional hard cap on concurrent sequences

a user request:
  POST /v1/chat/completions {model, messages, max_tokens, ignore_eos?}

tokenizer -> token ids            # text -> numbers
sequence scheduled (respecting --max-num-seqs & cache)   # each POST = 1 sequence
decode token-by-token, writing KV into PagedAttention BLOCKS (16-token pages)
stop at EOS or --max-tokens       # ignore_eos only skips the EOS stop
```

### What each observability signal tells you
| Signal | Source | Meaning |
|---|---|---|
| nvidia-smi used-MB | `nvidia-smi` | physical VRAM reserved (flat once warm) |
| `num_gpu_blocks` | `/metrics` | how many 16-token pages were pre-allocated |
| `kv_cache_size_tokens` | `/metrics` | total token capacity of the KV pool |
| `kv_cache_usage_perc` | `/metrics` | how much of the pool is currently consumed (moves with load) |
| throughput tok/s / p50-p99 | bench (`loadtest/`) | how fast / how consistent under concurrency |
