# INFRA-CONCEPTS.md — the mental model of this project

A deep, self-contained guide to understanding **everything** in this repo: what
each part is, how it works, why it exists, and the concepts you need to reason
about LLM serving on a small GPU.

Read it top-to-bottom for the big picture, or jump to a section:

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [The mental model: layers of the system](#2-the-mental-model-layers-of-the-system)
3. [The two servers (baseline vs vLLM)](#3-the-two-servers-baseline-vs-vllm)
4. [The data path of one request](#4-the-data-path-of-one-request)
5. [Tokens, sequences, blocks, KV cache](#5-tokens-sequences-blocks-kv-cache)
6. [The three capacity knobs (don't confuse them)](#6-the-three-capacity-knobs)
7. [KV-cache arithmetic (where the numbers come from)](#7-kv-cache-arithmetic)
8. [GPU memory bookkeeping: util, pre-allocation, OOM](#8-gpu-memory-bookkeeping)
9. [Measuring serving: throughput, latency, TTFT, ITL](#9-measuring-serving)
10. [Quantization (why it exists, what it changes)](#10-quantization)
11. [The vLLM config flags we actually use](#11-the-vllm-config-flags-we-actually-use)
12. [The WSL2 environment gotchas](#12-the-wsl2-environment-gotchas)
13. [Reading `nvidia-smi` and `/metrics`](#13-reading-nvidia-smi-and-metrics)
14. [The repo map (every file, what it is)](#14-the-repo-map)
15. [The measured results, summarized](#15-the-measured-results)

All concrete numbers use `Qwen/Qwen2.5-0.5B-Instruct` on an RTX 3070 Laptop
(8 GB) under WSL2. Raw data lives in `loadtest/README.md`.

---

## 1. The project in one paragraph

This repo answers one question with measurements: **why do production LLM servers
use a specialized serving engine (vLLM) instead of a hand-written Python loop over
Hugging Face `transformers`?** We build both — a naive FastAPI + PyTorch baseline
(`v1-baseline/`) and a vLLM deployment (`serve/`) — run the *same* model and
workload through each, and measure throughput, latency, and GPU/KV-cache memory.
The result: the naive server is pinned near ~20 tokens/sec at any load, while vLLM
scales to ~635 tokens/sec under concurrency because it **continuously batches**
many sequences and stores their attention state in **paged** KV-cache blocks. We
also push the hardware to its memory wall with a quantized 7B model to show how
weights and KV cache compete for VRAM.

The point is not "vLLM is faster" but the *mechanisms*: batching, paging,
KV budgeting, and how configuration flags move you around the memory/throughput
trade-off.

---

## 2. The mental model: layers of the system

Think of the project as five layers, each with a clear job:

```
┌──────────────────────────────────────────────────────────────────┐
│ 5. CLIENTS / HARNESS   loadtest/bench.py, run_load.sh,           │
│                        vllm bench serve, curl                     │
│    send HTTP requests, measure tok/s, latency, TTFT, ITL          │
├──────────────────────────────────────────────────────────────────┤
│ 4. SERVING ENGINE      vLLM (serve/)  — the "brain" of serving    │
│    continuous batching · PagedAttention KV cache · scheduler      │
├──────────────────────────────────────────────────────────────────┤
│ 3. API SURFACE         OpenAI-compatible HTTP (vLLM) /           │
│                        /generate (baseline)                       │
├──────────────────────────────────────────────────────────────────┤
│ 2. MODEL RUNTIME       torch kernels, attention, sampling,        │
│                        tokenizer                                  │
├──────────────────────────────────────────────────────────────────┤
│ 1. HARDWARE            RTX 3070 Laptop, 8 GB VRAM, CUDA (WSL2)    │
└──────────────────────────────────────────────────────────────────┘
```

- **Layer 5 (clients)** is where *you* operate. It only speaks HTTP; it doesn't
  know or care how the server batches internally.
- **Layer 4 (engine)** is the star. vLLM manages *which sequences run on the GPU
  at each decode step* and *where their KV cache lives*. This is the layer the
  whole project is about.
- **Layers 1–2** are the physical and numerical substrate. The binding constraint
  is **8 GB of VRAM**, shared by model weights, KV cache, activations, and CUDA
  graphs.

**The core tension** the whole project explores: *more concurrent requests ⇒ more
throughput, but each request's growing KV cache needs memory; weights also need
memory; 8 GB is finite.* Every config flag and every experiment is some answer to
"how do we spend the VRAM budget?"

---

## 3. The two servers (baseline vs vLLM)

Both serve the **same model** (`Qwen/Qwen2.5-0.5B-Instruct`) but embody opposite
design philosophies.

### Baseline — `v1-baseline/main-torch.py` (naive, the "before")

A hand-written FastAPI app:
- Loads the model with `transformers.AutoModelForCausalLM.from_pretrained(...)`
  and `AutoTokenizer` (`.to("cuda")`, fp16).
- Exposes `GET /health` and `POST /generate`.
- On `/generate`, it: applies the chat template → tokenizes → calls
  `model.generate(...)` → decodes the new tokens → returns text + `latency_ms` +
  `tokens_generated`.
- **One request is processed at a time.** `model.generate` runs a Python loop that
  produces one token per step for a *single* sequence (static, non-paged KV). Under
  concurrency, extra requests just queue in the web server.
- Binds port **8001**.

**Why it's slow:** no batching (GPU under-fed between tokens), no paged cache, and
per-token Python overhead. ~20 tok/s for 0.5B here.

### vLLM — `serve/vllm.serve.sh` (production-style, the "after")

vLLM is a **standalone serving engine + OpenAI-compatible HTTP server**. You don't
write the server; you *launch* it:
```bash
./serve/vllm.serve.sh        # wraps: vllm serve Qwen/Qwen2.5-0.5B-Instruct ...
```
- Exposes `/health`, `/v1/models`, `/v1/chat/completions`, `/metrics`, etc.
- Internally runs **continuous batching** and **PagedAttention** (see §4–5).
- Binds port **8000**.

**Why it's fast:** at every decode step it packs tokens from *many* live sequences
into one GPU batch, and stores each sequence's KV cache in small reusable pages.

> Key conceptual point: **there is no application code of yours running the model
> under vLLM.** The vLLM package *is* the application. Your repo only supplies the
> launch configuration (`serve/vllm.serve.sh`, `serve/vllm.env`) and the client
> harness (`loadtest/`).

---

## 4. The data path of one request

### Through vLLM (`POST /v1/chat/completions`)
```
text prompt + params {model, messages, max_tokens, temperature, top_p, ignore_eos?}
   │
   ▼  tokenizer            text → token IDs
   │
   ▼  scheduler            admit as a "sequence" (respecting max-num-seqs & free KV blocks)
   │                       if no room → QUEUE, don't crash (admission control)
   ▼  prefill              process the prompt in one pass; allocate KV blocks for it
   │                       → this is what determines TTFT (time-to-first-token)
   ▼  decode loop          each step: batch tokens from ALL live sequences,
   │                       compute next token, write new K/V into PagedAttention blocks
   │                       → steady-state speed is ITL (inter-token latency)
   ▼  stop                 when EOS emitted, or max_tokens reached
   │                       (ignore_eos=true disables the EOS stop)
   ▼
streamed chunks (SSE) or one JSON response
```

### Through the baseline (`POST /generate`)
```
prompt → chat template → tokenize → model.generate() [single sequence, per-token Python loop]
       → decode → return {completion, tokens_generated, latency_ms}
```
No batching, no pages, no scheduler — one sequence at a time.

---

## 5. Tokens, sequences, blocks, KV cache

These four words describe the *units* of serving. Get them straight.

### Token
The discrete unit the tokenizer emits (BPE). Deterministic, countable — "~4 chars"
is only a sizing *estimate* for text/cost, not an internal unit. A model has a
fixed vocabulary (Qwen2.5 ≈ 151k tokens).

### Sequence
One generation run: prompt + everything generated, the atomic unit the scheduler
tracks. **One HTTP `POST` = one sequence.** N concurrent requests ⇒ N resident
sequences (bounded by `--max-num-seqs` and available cache).

### KV cache
During attention, each token produces a **key** vector and a **value** vector at
**every layer**. The engine caches them so it doesn't recompute the prefix for
every new token. This cache is what makes generation linear instead of quadratic —
and it's what eats VRAM. (`K`/`V` = the query/key/value of transformer attention.)

### Block (a.k.a. PagedAttention page)
vLLM's fixed-size allocation unit for KV cache: the K & V for a run of **16 tokens**,
across all layers at once. A sequence's KV spans many **non-contiguous** blocks —
exactly like OS virtual-memory pages. Benefits: no giant contiguous allocation, low
fragmentation, easy sharing. The **last** block of a sequence is usually partially
empty ⇒ internal fragmentation (some wasted tail block).

Numbers for 0.5B:
- 1 block = 16 tokens × 12 KB/token = **192 KB**
- `num_gpu_blocks` = how many pages fit the reserved cache (e.g. 30,081)
- `kv_cache_size_tokens` = blocks × 16 (e.g. 481,296)

> **Gotcha:** the "12 KB/token" of KV is *not* the token's text size (~4 bytes).
> See §7 for why they differ by ~1000×.

---

## 6. The three capacity knobs

The most common confusion in LLM serving. These are three different limits:

| Knob | Set where | Controls | Analogy |
|---|---|---|---|
| `--max-model-len` | server startup | longest **one sequence** may be (prompt + generated) | size of one container |
| `--max-num-seqs` | server startup | how many **sequences run at once** | containers side-by-side |
| `--max-tokens` | per request JSON | how many NEW tokens **this caller** may generate | how full one container gets |

- A request's total context (= prompt tokens **+** its `max_tokens`) must be
  ≤ `--max-model-len`, or it's rejected:
  `max_tokens=5000 cannot be greater than max_model_len=4096` → **HTTP 400**.
- `--max-model-len` also sizes CUDA-graph capture; raising it re-budgets the same
  reserved pool and yields *fewer* blocks (Phase 2.2 §B).
- `--max-num-seqs` caps simultaneous sequences to keep memory/latency bounded.

---

## 7. KV-cache arithmetic

**Formula — bytes of KV cache per stored token:**

```
KV bytes/token = 2 (K & V) × L (layers) × H_kv (KV heads) × D_head × dtype_bytes
```

For `Qwen/Qwen2.5-0.5B-Instruct`:

| Term | Symbol | Source | 0.5B value |
|---|---|---|---|
| key + value | 2 | constant | 2 |
| layers | `num_hidden_layers` | model `config.json` | 24 |
| KV heads (GQA) | `num_key_value_heads` | model `config.json` | 2 |
| head dim | `hidden_size / num_attention_heads` | 896 / 14 | 64 |
| bytes per value | dtype | bf16 | 2 |

⇒ `2 × 24 × 2 × 64 × 2 = 12,288 B ≈ 12 KB / token`.

**Validation against measured reality:**
- 481,296 tokens × 12 KB ≈ **5.5 GiB** ≈ vLLM's reported "Available KV cache memory".
- 1 block = 16 × 12 KB = 192 KB; 30,081 blocks × 16 = 481,296 tokens. ✓

### Why so big? (the intuition)
A token's *characters* are ~4 bytes, but the cache doesn't store characters — it
stores that token's **attention vectors across the whole model**:
- per layer, per KV head: a 64-dim K vector + a 64-dim V vector = 128 numbers
- × 2 KV heads = 256 numbers/layer
- × 24 layers = 6,144 numbers/token
- × 2 bytes (bf16) = **12 KB**

> **Common gotcha:** "16 tokens × 4 bytes = 64 bytes per block" is wrong by ~1000×,
> because it confuses the token's *text* size with its *KV state* size. Proof: the
> whole 481,296-token cache would be ~2 MB at 4 B/token — but vLLM really reserves
> **~5.5 GiB**. Only 12 KB/token matches the measured reservation.

### GQA (grouped-query attention)
Qwen2 has 14 *query* heads but only **2** *key/value* heads — multiple query heads
share each KV head. That's why `H_kv = 2`, not 14, which slashes cache ~7× versus
plain multi-head attention. It's a model-design choice in `config.json`.

### Know the levers, not the magic numbers
- fp16→fp8 halves `dtype_bytes` ⇒ KV halves.
- More layers / KV heads / head-dim ⇒ more KV per token.
- GQA ⇒ much less cache than MHA.
- The 24/2/64/12 KB figures are *this model's* example. For another model, reread
  `config.json` and recompute. Engineers derive; they don't memorize.

---

## 8. GPU memory bookkeeping

### `--gpu-memory-utilization` (default 0.85 here)
The fraction of VRAM vLLM may reserve in total. That budget must cover:
**weights + KV cache + activation buffers + CUDA-graph capture memory**. vLLM then
computes how many KV blocks fit in whatever is left after weights/activations.

### Pre-allocation (the key behavior)
vLLM **pre-allocates the KV pool at startup**, sized by the util budget. So:
- `nvidia-smi` shows ~flat usage under load (~7.27 GB for 0.5B @ 0.85).
- Idle ≈ under-load *physical* usage.
- What rises with load is **logical** occupancy: `vllm:kv_cache_usage_perc`.

### Why OOM is (mostly) unreachable on a small model
vLLM's scheduler is **admission-controlled**: if admitted sequences would exceed
the pre-allocated block pool, it **queues** new requests rather than oversubscribe.
On 0.5B, weights + activations fit easily even in 45% of the card, so no
"unmeetable allocation" ever occurs — you get *latency growth*, not a crash
(Phase 2.3). A genuine startup abort requires weights that dominate the budget —
which we get with a quantized 7B (Phase 2.4), though even there the abort is
conditional on **process state** (whether `torch.compile` actually compiles or
loads a cached artifact), not a fixed config rule.

### The two failure modes that *are* real
1. **Startup abort** (weight-driven): weights leave too little for the required
   max-length cache, e.g. 7B-AWQ (`0.44 GiB needed > 0.2 GiB available`) — vLLM
   refuses to start and tells you the max supportable length. **Important:** this is
   *conditional*, not a fixed config rule. vLLM sizes KV as
   `budget − (total_consumed + transient_peak_headroom)`, and the second term is the
   **peak *torch* allocation seen during profiling** — which is much larger on a
   **cold** run because `torch.compile`/Inductor actually compiles kernels (and runs
   the traced graph to validate/autotune) than on a **warm** run that loads the
   Ahead-Of-Time (AOT) artifact. We reproduced both: 7B-AWQ `max_model_len=8192`
   **aborts cold** (peak 1.04 GiB → 0.2 GiB cache) but **boots warm** (peak 0.26 GiB
   → 0.98 GiB cache, 18,400 tokens). So "does it fit?" depends on *state*.
   Crucially, this is **not** CUDA-graph capture (which happens later, ~equal in
   both runs) and **not** steady-state serving activations (warmup times were
   identical).
2. **Config rejection**: `max_tokens > max_model_len` → clean HTTP 400.

### The VRAM budget has four line items (not two)
`weights + KV cache + activation buffers + CUDA-graph memory`. Most discussions
fixate on weights-vs-KV; the **transient peak** (dominated by the `torch.compile`
step on a cold run) is the term that makes failures non-deterministic. Any "it
should/shouldn't OOM" claim must be reproduced in a known state before you trust it.

### Memory levers (the "fix" toolkit)
- `GPU_UTIL` — right-size the pool (lower ⇒ smaller cache, less idle VRAM).
- `MAX_MODEL_LEN` — per-sequence headroom vs block count.
- `MAX_NUM_SEQS` — hard cap on concurrent sequences (bounds queueing/latency).
- Quantization — shrink weights (§10).

### AOT vs JIT: what happens at startup (why cold ≠ warm)
This is the concept behind the conditional 7B abort, so it's worth nailing:

- **JIT = Just-In-Time** — compile when first needed. **AOT = Ahead-Of-Time** —
  compile once up front and save the result for later runs.
- A GPU runs small programs called **kernels** (matmul, attention, norm, ...).
  PyTorch "eager" mode launches them one op at a time. `torch.compile` traces the
  model (Dynamo), then optimizes and **generates GPU kernels** (Inductor), fusing
  ops for speed.
- **Cold start:** `torch.compile` *compiles* — and to validate/autotune it **runs
  the traced graph on the GPU**, creating a large, short-lived allocation peak
  (~1.0 GiB here). It then **saves the result** (the "AOT artifact" under
  `~/.cache/vllm/torch_compile_cache`). In our logs: `saved AOT compiled function`,
  `torch.compile took 22 s`.
- **Warm start:** it finds that artifact and **loads it directly** (`Directly load
  AOT compilation`, `~1 s`) — no re-trace, no compile-time GPU run, so no big peak
  (~0.26 GiB).
- vLLM sizes the KV cache from free memory **after** this step and reserves the
  observed peak as headroom → cold gets less KV (0.2 vs 0.98 GiB).
- **This is separate from CUDA-graph capture**, which happens *later* (recording
  the decode step for replay) and costs ~0.4 GiB in both runs. Don't conflate the
  two: the cold run aborts *before* graph capture even starts.

> Analogy: `torch.compile` is a compiler that must *run parts of the program on the
> GPU to test them*, using temporary GPU memory. Compile once (cold) and the
> temporary space was claimed; reuse the saved binary (warm) and it never is.

---

## 9. Measuring serving

### The metrics (and what they mean)
- **Output throughput (tok/s)** — generated tokens per second, aggregated across
  all in-flight requests. The headline capacity number.
- **E2E latency** — total request time (prefill + all decode).
- **TTFT (time-to-first-token)** — send → first token. Dominated by prefill +
  queueing. The latency users feel *first* (and what streaming UIs hang on).
- **ITL / TPOT (inter-token latency / time-per-output-token)** — gap between
  successive tokens. Governs the "feel" of streaming after the first token.
- **p50 / p95 / p99** — percentiles; tail latency matters for SLOs.

### The ideal signature of good serving
> Under rising concurrency: **throughput rises, ITL stays roughly flat, TTFT rises
> moderately.** Batching keeps the token loop fed (flat ITL); the queue deepens
> (rising TTFT). If ITL climbs sharply with load, you're past saturation.

This is exactly what we measure (Phase 3): 121→395→635 tok/s at conc 1/4/8, ITL
~8–9.5 ms, TTFT 26→86 ms.

### The tools in this repo
- **`loadtest/bench.py`** — our async client. `--mode baseline|vllm`, `--model`,
  `--max-tokens`, `--concurrency`, `--requests`; `--stream` adds TTFT/ITL via SSE.
  Prints a human line **and** a `JSON\t...` line for tables.
- **`loadtest/vram_watch.py`** — samples `kv_cache_usage_perc` + `nvidia-smi`
  during a burst; `--ignore-eos` keeps sequences resident so cache fill is visible.
- **`loadtest/run_load.sh`** — runs the standard set end to end.
- **`vllm bench serve`** — the official benchmark: canonical TTFT/ITL/TPOT/E2E,
  throughput, goodput, and Poisson `--request-rate` (arrival-rate) load.

### Why two measurement styles
- **Fixed concurrency** (our client) — hold N requests in flight; good for clean
  scaling curves.
- **Arrival rate** (official `--request-rate`) — Poisson arrivals; closer to real
  traffic.

---

## 10. Quantization

**Problem:** a bf16 model needs 2 bytes per parameter. A 3B model ⇒ ~6 GB of
weights, which on an 8 GB card leaves negative room for KV cache (`max-model-len`
can't be satisfied) → vLLM won't boot.

**Solution:** store weights in fewer bits. **AWQ int4** ≈ 4 bits/weight ⇒ ~4×
smaller weights. Effects we measured:
- 3B: bf16 6 GB (won't boot) → **AWQ 1.95 GiB, 3.99 GiB KV cache** (boots fine).
- 7B: **AWQ 5.29 GiB** — *fits* an 8 GB card, but leaves thin cache headroom
  (0.2 GiB cold / 0.98 GiB warm), so it serves ~1.8–2.25× concurrency and *may*
  abort at `max-model-len 8192` depending on whether `torch.compile` compiled or
  loaded the AOT cache (conditional, not fixed — §8).

**Mechanism in vLLM 0.28:** AWQ/GPTQ load via bundled **Marlin int4** CUDA ops —
no extra Python package, no nvcc/JIT. The launcher exposes `QUANTIZATION=` (blank =
auto-detect from the checkpoint's `config.json`).

**Trade-off / gotcha:** quantization buys memory, not free speed or quality.
Throughput and answer quality change and must be re-measured (a coherent sample
response from 3B-AWQ is recorded in `loadtest/README.md`).

---

## 11. The vLLM config flags we actually use

Set via env overrides in `serve/vllm.serve.sh`:

| Env var → flag | Meaning | Why it matters here |
|---|---|---|
| `MODEL` | model id to serve | default `Qwen/Qwen2.5-0.5B-Instruct` (canonical) |
| `HOST` / `PORT` | bind address | `0.0.0.0:8000` |
| `GPU_UTIL` → `--gpu-memory-utilization` | VRAM budget fraction | 0.85 default; lower to shrink cache (OOM study) |
| `MAX_MODEL_LEN` → `--max-model-len` | longest single sequence | 2048 default; raise ⇒ fewer blocks/headroom |
| `MAX_NUM_SEQS` → `--max-num-seqs` | max concurrent sequences | optional admission cap / recovery lever |
| `QUANTIZATION` → `--quantization` | weight format | blank = auto-detect; force `awq_marlin` if needed |

Plus fixed environment variables (the WSL2 workarounds) — see §12.

---

## 12. The WSL2 environment gotchas

vLLM 0.28 does **not** boot cleanly in this WSL2 + 8 GB environment by default.
`serve/vllm.serve.sh` and `serve/vllm.env` encode the fixes; each is real:

1. **`VLLM_WSL2_ENABLE_PIN_MEMORY=1`** — vLLM disables pinned host memory (and thus
   CUDA UVA, used by its buffer allocator) when it detects WSL2, then fails with
   `RuntimeError: UVA is not available`. This re-enables it (kernel ≥ 4.19.121).

2. **`VLLM_USE_FLASHINFER_SAMPLER=0`** — FlashInfer's top-k/top-p sampler is
   JIT-compiled with nvcc. We have no matching system CUDA toolkit (only the
   driver + a *mismatched* pip nvcc), so the build fails with a CCCL/libcudacxx
   header error. Disabling it makes vLLM use its native Triton sampler (equivalent,
   no recompile).

3. **`CUDA_HOME` → pip-installed nvcc** (`.venv-vllm/.../nvidia/cu13`) — lets tools
   that probe `$CUDA_HOME/bin/nvcc` find a compiler.

4. **`PATH` includes `.venv-vllm/bin`** — so subprocess JIT builds find tools like
   `ninja`.

> This is real-world "infra" work: the job is making the stack *boot* on unusual
> hardware, and documenting *why* each workaround exists (that's `serve/vllm.env`).

Also relevant: **venvs are not relocatable** — Python bakes absolute paths into
`bin/activate` and every console-script shebang, so *never rename/move a venv*;
recreate it instead.

---

## 13. Reading `nvidia-smi` and `/metrics`

| Signal | Source | What it tells you |
|---|---|---|
| used VRAM (MB) | `nvidia-smi` | physical reservation (≈flat once warm) |
| `num_gpu_blocks` | `/metrics` | pages pre-allocated for KV |
| `kv_cache_size_tokens` | `/metrics` | total token capacity of the KV pool |
| `kv_cache_usage_perc` | `/metrics` | fraction of the pool consumed **now** (moves with load) |
| `block_size` | `/metrics` | tokens per page (16) |
| throughput / p50–p99 / TTFT / ITL | `loadtest/`, `vllm bench` | serving performance |

**Rule:** don't use `nvidia-smi` alone to judge vLLM memory pressure — it shows the
pre-allocation, not fullness. Watch `vllm:kv_cache_usage_perc`.

Useful probes:
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics | grep -E 'kv_cache|cache_config_info'
```

---

## 14. The repo map

```
README.md                     Project overview (the public write-up).
AGENTS.md                     Notes for coding agents (commands, conventions, gotchas).

v1-baseline/
  main-torch.py               Naive FastAPI + transformers server (:8001). The "before".
  requirements-torch.txt      Baseline deps (torch/transformers/fastapi).
  README.md                   How to run the baseline.

serve/
  vllm.serve.sh               vLLM launcher; all WSL2 env workarounds live here.
  vllm.env                    Rationale for every env workaround (sourceable).
  requirements.txt            vLLM stack (vllm is the only direct dep).

loadtest/
  bench.py                    Async client: tok/s, e2e latency; --stream → TTFT/ITL.
  vram_watch.py               KV occupancy + nvidia-smi sampler during load.
  run_load.sh                 Standard set: non-stream sweep, stream sweep, official bench.
  README.md                   Measured results + method for every phase.

docs/
  PLAN.md                     Phased roadmap + decision record.
  INFRA-CONCEPTS.md           This file — the mental model.
  TRITON-ARCHITECTURE.md      NVIDIA Triton design/integration (architected, not deployed).

deploy/triton/                Reference Docker stack for a Triton front-end (needs Docker).
```

The **venvs** (`.venv-vllm`, `.venv-torch`) live at the repo root and are
git-ignored — they hold the installed dependencies, not project code.

---

## 15. The measured results

### Phase 2.1 — baseline vs vLLM throughput
| Engine | Conc N | out tok/s | p50 | p95 |
|---|---|---|---|---|
| baseline (torch, static KV) | 1 | ~20 | ~5.0 s | ~6.6 s |
| vLLM | 1 | 110 | 0.90 s | 1.16 s |
| vLLM | 2 | 200 | 0.91 s | 0.99 s |
| vLLM | 4 | 351 | 1.03 s | 1.29 s |
| vLLM | 8 | 388 | 1.00 s | 5.42 s |

Baseline flat regardless of load; vLLM scales with concurrency. Saturation begins
by N=8 (p95 blows up).

### Phase 2.2 — KV/VRAM vs concurrency and max-model-len
- vLLM pre-allocates; `nvidia-smi` ~flat (~7.27 GB). Logical occupancy rises:
  peak `kv_cache_usage_perc` ≈ 0.1% / 0.4% / 0.9% at conc 1/4/8 (0.5B is tiny, so
  <1% — the pool is heavily over-provisioned).
- Raising `max-model-len` 2048→8192 at fixed util: `num_gpu_blocks` 30,081→27,839;
  `kv_cache_size_tokens` 481,296→445,424; VRAM ~unchanged. Same budget, re-partitioned.

### Phase 2.3 — admission control (the honest OOM finding)
- Cache shrank to 1.9 GiB at util 0.45. 48×1500-tok and 128×3500-tok bursts:
  **server stayed healthy**, 0 errors — vLLM *queues*, no allocator OOM.
- Reproducible failure is config: `max_tokens > max_model_len` → HTTP 400.
- Conclusion: a true CUDA OOM on 0.5B is unreachable; needs the quantized phase.

### Phase 2.4 — quantized weight-vs-KV wall
| Model | Weights | KV cache | tokens | Verdict |
|---|---|---|---|---|
| 0.5B bf16 | ~1 GB | ~5.5 GiB | 481,296 | fits comfortably |
| 3B bf16 | ~6 GB | negative | — | won't boot |
| 3B-AWQ int4 | 1.95 GiB | 3.99 GiB | 116,240 | fits well |
| 7B-AWQ int4 @2048 | 5.29 GiB | 0.2 GiB (cold) | 3,696 | boots; ~1.8× concurrency |
| 7B-AWQ int4 @8192, **cold** | 5.29 GiB | 0.2 GiB | needs 0.44 | **startup abort** |
| 7B-AWQ int4 @8192, **warm** | 5.29 GiB | 0.98 GiB | 18,400 | **boots**; 2.25× concurrency |

Note: the @8192 row is **state-dependent** — **cold** (no AOT cache: Inductor compiles
kernels, raising the peak torch allocation to ~1.0 GiB) aborts; **warm** (loads the
AOT artifact, peak ~0.26 GiB) boots. Weights-vs-KV is real, but the compile transient
moves the exact boundary. Reproduced by clearing `~/.cache/vllm/torch_compile_cache`.

### Phase 3 — TTFT/ITL (0.5B, max_tokens=128)
| Conc | out tok/s | TTFT p50 | TTFT p95 | ITL p50 |
|---|---|---|---|---|
| 1 | 121 | 26 ms | 40 ms | 8.1 ms |
| 4 | 395 | 43 ms | 76 ms | 9.3 ms |
| 8 | 635 | 86 ms | 89 ms | 9.5 ms |

Official `vllm bench serve` (conc 8): 820 tok/s; TTFT p50 64.5 ms / p99 205 ms;
ITL p50 8.7 ms / p99 15.3 ms. Poisson `--request-rate 8`: 828 tok/s, TTFT p50 45 ms.

---

*For measured detail see `loadtest/README.md`; for the roadmap and decisions see
`PLAN.md`; for the Triton design see `TRITON-ARCHITECTURE.md`.*
