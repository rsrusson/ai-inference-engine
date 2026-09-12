# TRITON-ARCHITECTURE.md — Where NVIDIA Triton Inference Server fits

Design/reference document for **Phase 4 (optional stretch)** of `PLAN.md`.
This phase is **architected and documented, not deployed** on the dev box — see
[§7 Why not deployed here](#7-why-this-is-not-deployed-on-the-dev-box). A runnable
reference stack is provided under `deploy/triton/` for a Docker-capable host.

> Context: this repo serves `Qwen/Qwen2.5-0.5B-Instruct` with vLLM on an 8 GB
> RTX 3070 Laptop under WSL2. Read `INFRA-CONCEPTS.md` for the full serving/memory
> mental model and `loadtest/README.md` for the measured results.

---

## 1. First, the name collision (common point of confusion)

Two unrelated projects are both called "Triton":

| | **OpenAI Triton** | **NVIDIA Triton Inference Server** |
|---|---|---|
| What | a GPU **kernel compiler** (Python DSL → CUDA) | a **model-serving server** (HTTP/gRPC) |
| Used for | writing fast custom kernels; used *by* PyTorch/vLLM internally | hosting models behind a stable API; a **deployment** layer |
| In this repo | yes — the `triton` pip package vLLM depends on | **not installed** (and not the same thing) |

When someone says "put vLLM behind Triton," they mean the **NVIDIA Inference
Server**. Installing the `triton` Python package does *not* give you it.

---

## 2. Where Triton sits relative to vLLM

vLLM is an *engine + an OpenAI-compatible HTTP server*. NVIDIA Triton is a
*serving platform*. They overlap at the HTTP layer and differ in scope:

```
                         plain vLLM (this repo)                Triton Inference Server
                         ----------------------                -----------------------
client ──HTTP──▶ /v1/chat/completions ──▶ vLLM engine   client ─HTTP/gRPC─▶ Triton
                 (OpenAI schema)                                        │
                                                                        ├─ model repo (versioned dirs)
                                                                        ├─ backends (python, tensorrtllm,
                                                                        │            vllm, onnx, ...)
                                                                        ├─ dynamic batcher / scheduler
                                                                        ├─ ensembles (pre/post-process chains)
                                                                        └─ metrics, model control, KServe API
```

- **vLLM** answers one question well: *fast LLM generation with continuous
  batching and PagedAttention.*
- **Triton** answers a broader operational question: *host many models/backends
  behind one server with versioning, multi-framework support, ensembles, and
  gRPC.* For a single LLM on one GPU, Triton's value is mostly the *operational
  surface and multi-model orchestration* — not raw LLM speed.

---

## 3. Integration modes (the interesting design space)

There are two realistic ways to combine them.

### Mode 1 — vLLM as a Triton **backend** (Triton serves the model itself)
Triton's model repository contains a model whose backend is the **vLLM Python
backend** (`--backend python` + the `vllm`/`vllm_backend` model). Triton starts
and owns the vLLM engine; clients hit Triton (HTTP/gRPC/KServe), and Triton
routes into vLLM's engine.

- **Pros:** single entry point; Triton metrics/model-control; can co-host
  embeddings/rerankers/other models next to the LLM; gRPC + KServe surfaces.
- **Cons:** heavier; Triton's own batching can double up with vLLM's continuous
  batching (usually you let vLLM batch and keep Triton's batcher permissive);
  version coupling between Triton and the vLLM backend.
- **Best when:** you're building a multi-model platform and want one control
  plane.

### Mode 2 — Triton as a **front door** in front of the vLLM OpenAI server
Keep vLLM running exactly as in this repo (`./serve/vllm.serve.sh`, `/v1/chat/completions`),
and put a Triton model in front that acts as a **proxy / orchestrator**:
either an **HTTP proxy model** or a **Business-Logic Scripting (BLS)** Python
model that calls vLLM with `requests`/`aiohttp` and returns the result.

- **Pros:** no coupling to vLLM internals; you keep the measured, tuned vLLM
  setup unchanged; easy to add auth/rate-limit/translation in the proxy; lets
  you A/B "Triton front vs direct."
- **Cons:** an extra network hop and process; the proxy must be written/tested;
  you now operate two servers.
- **Best when:** vLLM is a fixed dependency you don't want to disturb, or you
  want Triton's request features without rehosting the model.

```
Mode 1:  client ─▶ Triton ─▶ [vLLM python backend] ─▶ GPU
Mode 2:  client ─▶ Triton ─▶ [proxy/BLS model] ─HTTP─▶ vLLM (this repo) ─▶ GPU
```

For this project, **Mode 2** is the lower-risk integration and the one whose
"before/after" story (direct vLLM vs Triton-fronted vLLM) is easiest to measure
with the Phase 3 tooling.

---

## 4. Triton model repository layout (reference)

Triton discovers models by directory convention:

```
model_repository/
  llm_proxy/                     # model name
    1/                           # version (integer = version number)
      model.py                   # Python backend (Mode 2: BLS/proxy)
      config.pbtxt?              # optional; auto-generated by --backend python
  # Mode 1 alternative: a vLLM-backend model directory instead of llm_proxy
```

Key `config.pbtxt` concerns: `max_batch_size`, `input`/`output` names+types
(or the newer dynamic request/response API), and `instance_group`. For an LLM,
set `max_batch_size: 0` and use Triton's **decoupled / streaming** API so tokens
can be returned incrementally.

---

## 5. Endpoints and APIs Triton exposes
- **HTTP (KServe v2)**: `POST /v2/models/<model>/infer`, plus model metadata,
  readiness/health (`/v2/health/ready`), and model lifecycle (`/v2/repository/...`).
- **gRPC**: the same inference/metadata surface (lower overhead, streaming).
- **Metrics**: a Prometheus endpoint (`/metrics`) with per-model inference
  counts, latency histograms, queue time, and GPU stats.
- **Model control**: load/unload models at runtime.
- (Optional) **OpenAI-compatible frontend** exists in newer Triton; otherwise a
  proxy model (Mode 2) provides the OpenAI shape.

This breadth — multiple protocols, model lifecycle, ensembles, metrics — is what
"enterprise serving platform" means, and it is exactly what plain vLLM does not
try to be.

---

## 6. When to choose which (decision guidance)

| Situation | Choose |
|---|---|
| One LLM, one GPU, fastest path, OpenAI API | **vLLM alone** (this repo) |
| Many models/backends, multi-framework, one control plane | **Triton** (Mode 1) |
| Fixed vLLM deployment + need gRPC/KServe/ensembles/auth in front | **Triton front (Mode 2)** |
| TensorRT-LLM on NVIDIA stack, enterprise ops | Triton + TensorRT-LLM backend |
| Learning the memory/batching mechanics | vLLM directly (this repo, Phases 2–3) |

Triton is *not* a drop-in speed upgrade for a single vLLM LLM. Its win is
**operational**: protocol coverage, multi-model orchestration, lifecycle, metrics.

---

## 7. Why this is NOT deployed on the dev box

Decision record (truthful; see the repo's "don't fake results" rule):

1. **No Docker.** Triton Inference Server's primary distribution is the
   `nvcr.io/nvidia/tritonserver` **container image**. This WSL2 dev box has no
   Docker daemon/socket and no `docker` binary. (The registry is reachable, but
   we cannot run containers.)
2. **No native server available.** `tritonserver` is not pip-installable; only the
   `tritonclient` *client* is. A native WSL2 build is heavy and fragile — the
   same class of environment pain that already made vLLM boot require dedicated
   workarounds (`vllm.env`).
3. **GPU budget.** The 8 GB card already fully commits to vLLM when serving
   (Phases 2–3). Running Triton and a vLLM engine concurrently would contend for
   the same VRAM; there is no second GPU.
4. **Scope.** Phase 4 was always marked optional. A half-installed server would
   be a worse engineering artifact than a clear, correct architecture + a runnable
   reference stack for a Docker host.

Also note: the `triton` pip package present in `.venv-vllm` is **OpenAI Triton**
(the kernel compiler), unrelated to NVIDIA Triton Inference Server (§1).

---

## 8. Run it on a Docker-capable host (reference)

The `deploy/triton/` directory contains a reference stack (Mode 2 front proxy):
a `docker-compose.yml`, a Triton model repository with a Python-backend proxy
model, and a README with commands. It is **not executed by this repo's dev
workflow**; treat it as a template to run where Docker is available, then point
the Phase 3 load tools at the Triton endpoint to compare against direct vLLM.

---

## 9. Summary / key takeaways
- Remember the **name collision**: Triton is a *platform*, vLLM an *engine*.
- Choose **Mode 1 vs Mode 2**: use Mode 2 when vLLM is a fixed, tuned dependency.
- Triton's value is **operational** (multi-model, protocols, lifecycle, metrics),
  not a magic throughput boost for a single LLM.
- The deployment constraint is real: container-first distribution → needs Docker,
  which the WSL2 dev box lacks.
