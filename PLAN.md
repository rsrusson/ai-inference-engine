# Plan — AI Infrastructure Engineer portfolio project: LLM serving with vLLM

## Big picture & framing for your job hunt

The portfolio story: *"I can operate the whole lifecycle that makes inference fast and memory-safe — from hand-written PyTorch serving, through understanding why naive serving wastes GPU memory and throughput, up to a production-grade vLLM deployment that demonstrates continuous batching and PagedAttention, and shows how sequence-length & KV-cache budgeting avoids OOM on an 8 GB GPU, backed by measured load tests."*

- `main-torch.py` is the **naive but working baseline** (single-request, static KV, in-process). It is deliberately kept so Phase 2 can A/B it against vLLM — but it is **deprioritized**, not the centerpiece.
- vLLM (via `.venv-vllm` + `./vllm.serve.sh`) is the serving centerpiece on top.

## Target hardware reality (anchors every number in the plan)

- 1× RTX 3070 Laptop, **8 GB VRAM**, driver 596.08 / CUDA 13.2.
- 16 GB RAM, 16 CPUs, ~945 GB disk — single node. Load-testing happens on-box (WSL2); a reachable URL is available via the VSCode/`0.0.0.0` bind for a host-PC browser if desired.
- We run under **WSL2** (`6.18.x-microsoft-standard-WSL2`). See `vllm.env` for the two WSL2/missing-toolchain flags required to boot vLLM here.

## Model decision (current = canonical, do not reference 3B as a target)

**Permanent serving model: `Qwen/Qwen2.5-0.5B-Instruct`.** Weights already cached in `~/.cache/huggingface`.

Why:
- This 8 GB card must hold: model weights **+** PagedAttention KV cache **+** activation buffers **+** CUDA-graph capture memory. In practice only ~7.0 GB is usable after driver reservation.
- `Qwen2.5-3B` in bf16 (~6 GB weights) boots but leaves **negative KV-cache space** → vLLM aborts (`No available memory for the cache blocks` / `Available KV cache memory: -0.45 GiB`). 3B is NOT deployable here without quantization, so it is not part of the core plan.
- `1.5B` fits (yields ~2.7 GiB KV cache) but we pin **0.5B** everywhere to leave generous KV + sequence-length headroom across concurrent requests on shared VRAM.
- A **quantized** larger model (e.g. `7B-Instruct-AWQ`) is the only path to a bigger model on this card; it is an optional appendix/"expert flex", never the default.

Rule for docs/config: any Model field that launches the engine should say `Qwen/Qwen2.5-0.5B-Instruct` unless a phase is *explicitly* experimenting with another tag.

## Phased plan (each phase yields a portfolio artifact)

### Phase 0 — Env prep
- Isolated `.venv-vllm` (Python 3.10) so vLLM deps don't fight the baseline's `.venv-torch`. ✅ done
- vLLM installed (CUDA wheel) and verified importable on the GPU. ✅ done
- Baseline runtime `.venv-torch` rebuilt at its final path (venvs aren't relocatable; don't rename it).
- Repo hygiene: `.gitignore`, `requirements.txt` (vLLM), `requirements-torch.txt` (baseline), `vllm.env` (env rationale), `vllm.serve.sh` (launcher).

### Phase 1 — Serve Qwen2.5-0.5B-Instruct with vLLM (functional milestone) ✅ served
Launch (as captured in `vllm.serve.sh`):
```bash
./vllm.serve.sh
# = vllm serve Qwen/Qwen2.5-0.5B-Instruct \
#     --host 0.0.0.0 --port 8000 \
#     --gpu-memory-utilization 0.85 --max-model-len 2048 \
#   (with WSL2 pin-memory + flashinfer-sampler env overrides)
```
- Verify `/v1/models`, `/health`, and a sample `POST /v1/chat/completions`. ✅ done
- Learn: OpenAI-compatible API surface, `gpu_memory_utilization` vs `max-model-len`, and that vLLM continuously batches server-side before you even push concurrency.

### Phase 2 — Pedagogical core: continuous batching + PagedAttention + OOM
1. **Throughput scaling (`Phase 2.1`)**: run an identical `temperature=1.0` prompt through (a) baseline `main-torch.py` (single-seq, static KV) and (b) vLLM under concurrency N. Show vLLM's aggregate tok/s climbs with N while the baseline stays flat. Record a small table under `loadtest/`.
2. **VRAM vs concurrency (Phase 2.2 — done)**: poll `nvidia-smi`, record VRAM vs concurrency and vs `max-model-len`. Measured: vLLM pre-allocates the KV pool by `gpu_memory_utilization`, so total GPU is ~flat (~7.27 GB) while PagedAttention block occupancy (`vllm:kv_cache_usage_perc`) rises ~linearly with concurrent long sequences (~0.1%→0.9% at conc 1→8); raising `max-model-len` 2048→8192 shrank `num_gpu_blocks`/cache tokens (30081→27839; 481296→445424) at fixed util. Data + tool (`loadtest/vram_watch.py`) in `loadtest/README.md`.
3. **Force & fix OOM (Phase 2.3 — done, honest finding)**: Option B (0.5B core). Verified KV arithmetic (`2×layers×kv_heads×head_dim×dtype` = 12 KB/token for 0.5B) and that shrinking `gpu_memory_utilization` (0.45) reduces the cache to 1.9 GiB / 165,824 tokens. **Result:** vLLM does NOT allocator-OOM on 0.5B — its scheduler is admission-controlled and *queues* rather than oversubscribes under heavy concurrent load (48×/128× bursts stayed healthy), so an actual GPU OOM on permanent 0.5B is unreachable. The reproducible misconfiguration error is `max_tokens > max_model_len` → HTTP 400. Recovery levers proven: `GPU_UTIL`, `MAX_MODEL_LEN`, and `MAX_NUM_SEQS` (added to `vllm.serve.sh`). A real OOM requires the quantized-larger-model appendix. Full write-up in `loadtest/README.md`.
4. **Extra credit**: KV fragmentation (block size, partial-sequence waste), `--max-seq-len-to-capture`, scheduler knobs.

### Phase 3 — Load-testing instrumentation
Thin async concurrency client (`loadtest/bench.py`) driving `/v1/chat/completions`, plus a results doc/table in `loadtest/`. (Official `benchmark_serving.py` and hey/wrk/locust are optional reinforcements, not required.)

### Phase 4 — (optional stretch) Triton angle
Optionally put the vLLM OpenAI-compatible service behind NVIDIA Triton (HTTP/gRPC) for enterprise framing. Optional; may become its own follow-up project.

### Phase 5 — Portfolio write-up
One README tying measurements to infra-interview concepts:
- Motivation (naive static KV → why batched/Paged serving).
- Baseline vs vLLM numbers + graphs (`matplotlib` optional).
- OOM case study + KV math.
- Architecture diagram (client → WSL2 vLLM :8000; baseline :8001).
- Repo hygiene: `v1-baseline/` (main-torch), `serve/` (launcher/docs), `loadtest/`.

## Deliverables
- Working vLLM `/v1/chat/completions` reachable endpoint.
- Cleaned repo: baseline + serve + loadtest.
- Measured baseline-vs-vLLM numbers + KV/OOM notes.
- README/portfolio narrative covering continuous batching, PagedAttention, KV budgeting, max-seq/len config.
