"""HTTP service around the official APUS-OpenJev reference runtime (openjet_runtime).

POST /decide    {state, instructions, criteria:[{id,description}], primitive?, effort?}
POST /generate  {text, effort?, max_new_tokens?}
GET  /health
"""

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("JEV_MODEL_DIR", "models/APUS-OpenJev-v1/9B-3000")).resolve()
sys.path.insert(0, str(MODEL_DIR))  # the runtime ships inside the model directory
from openjet_runtime import OpenJet  # noqa: E402

app = FastAPI(title="APUS-OpenJev decision service")
runtime: OpenJet | None = None
lock = threading.Lock()  # reference runtime is single-request


class Candidate(BaseModel):
    id: str
    description: str


class DecideRequest(BaseModel):
    state: str
    instructions: str
    criteria: list[Candidate] = Field(min_length=2, max_length=16)
    primitive: Literal["choice", "noul", "score_level"] = "choice"
    effort: Literal["low", "high"] = "high"
    id: str | None = None
    group_id: str | None = None


class GenerateRequest(BaseModel):
    text: str
    effort: Literal["low", "high"] = "high"
    max_new_tokens: int = Field(64, ge=1, le=512)


@app.get("/health")
def health():
    return {
        "ok": runtime is not None,
        "model_dir": str(MODEL_DIR),
        "depths": runtime.depths if runtime else None,
    }


@app.post("/decide")
def decide(req: DecideRequest):
    rid = req.id or uuid.uuid4().hex[:12]
    record = {
        "id": rid,
        "group_id": req.group_id or rid,
        "primitive": req.primitive,
        "state": req.state,
        "instructions": req.instructions,
        "criteria": [c.model_dump() for c in req.criteria],
    }
    t0 = time.perf_counter()
    try:
        with lock:
            result = runtime.decide(record, req.effort)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["id"] = rid
    return result


@app.post("/generate")
def generate(req: GenerateRequest):
    t0 = time.perf_counter()
    try:
        with lock:
            result = runtime.generate_text(req.text, req.effort, req.max_new_tokens)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


def main():
    global runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18310)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()
    t0 = time.time()
    runtime = OpenJet.from_pretrained(MODEL_DIR, args.device, args.dtype)
    print(f"loaded {MODEL_DIR} in {time.time() - t0:.1f}s depths={runtime.depths}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
