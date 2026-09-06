import logging
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI(
    title="AI Inference Baseline",
    version="0.1.0",
    description="FastAPI + PyTorch + Transformers model serving baseline.",
)

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
device: torch.device = torch.device("cpu")
_MODEL_LOCK = False


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for generation.")
    max_tokens: int = Field(128, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability.")


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    completion: str
    latency_ms: float
    tokens_generated: int
    device: str
    gpu_available: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    device: str
    gpu_available: bool


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("CUDA available, using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("MPS available, using Metal backend.")
        return torch.device("mps")
    logger.info("No GPU found, falling back to CPU.")
    return torch.device("cpu")


def _load_model(model_name: str) -> None:
    global model, tokenizer, device, _MODEL_LOCK
    if model is not None:
        return

    _MODEL_LOCK = True
    logger.info("Loading model '%s'...", model_name)
    start = time.perf_counter()

    device = _select_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    ).to(device)
    model.eval()

    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2fs on %s.", elapsed, device)
    _MODEL_LOCK = False


@app.on_event("startup")
async def startup() -> None:
    _load_model(DEFAULT_MODEL)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    gpu_available = torch.cuda.is_available()
    return HealthResponse(
        status="ready" if model is not None else "loading",
        model=DEFAULT_MODEL,
        model_loaded=model is not None,
        device=str(device),
        gpu_available=(gpu_available or torch.backends.mps.is_available()),
    )


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest) -> GenerateResponse:
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please retry.")

    start = time.perf_counter()

    messages = [{"role": "user", "content": req.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)

    latency_ms = (time.perf_counter() - start) * 1000.0

    return GenerateResponse(
        model=DEFAULT_MODEL,
        prompt=req.prompt,
        completion=completion,
        latency_ms=round(latency_ms, 2),
        tokens_generated=len(generated),
        device=str(device),
        gpu_available=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
