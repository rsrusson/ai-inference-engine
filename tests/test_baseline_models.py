"""Unit tests for the baseline server's API models (no GPU, no model load).

The baseline module is imported with torch/transformers/uvicorn stubbed in
conftest.py, so we can validate the Pydantic request/response contracts cheaply.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_generate_request_defaults(baseline):
    req = baseline.GenerateRequest(prompt="hello")
    assert req.max_tokens == 128
    assert req.temperature == 0.7
    assert req.top_p == 0.9


def test_generate_request_requires_nonempty_prompt(baseline):
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="")


def test_generate_request_max_tokens_bounds(baseline):
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="x", max_tokens=4097)
    assert baseline.GenerateRequest(prompt="x", max_tokens=4096).max_tokens == 4096


def test_generate_request_temperature_bounds(baseline):
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="x", temperature=-0.1)
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="x", temperature=2.1)


def test_generate_request_top_p_bounds(baseline):
    with pytest.raises(ValidationError):
        baseline.GenerateRequest(prompt="x", top_p=1.1)
    assert baseline.GenerateRequest(prompt="x", top_p=0.0).top_p == 0.0


def test_health_response_shape(baseline):
    h = baseline.HealthResponse(
        status="ready", model="m", model_loaded=True, device="cuda",
        gpu_available=True,
    )
    assert h.model_dump() == {
        "status": "ready", "model": "m", "model_loaded": True,
        "device": "cuda", "gpu_available": True,
    }


def test_generate_response_shape(baseline):
    r = baseline.GenerateResponse(
        model="m", prompt="p", completion="c", latency_ms=1.5,
        tokens_generated=3, device="cpu", gpu_available=False,
    )
    assert r.tokens_generated == 3
    assert r.latency_ms == 1.5


def test_default_model_is_canonical(baseline):
    assert baseline.DEFAULT_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"
