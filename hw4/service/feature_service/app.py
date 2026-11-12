"""feature-service: return feature vectors from Redis for a given client_id + as_of date."""

from __future__ import annotations

import json
import os

import redis as redis_lib
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

REDIS_HOST = os.environ.get("HW4_REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("HW4_REDIS_PORT", "6379"))
TIMESTAMP_SUFFIX = "T00:00:00"

app = FastAPI()
_redis: redis_lib.Redis | None = None


def get_redis() -> redis_lib.Redis:
    global _redis
    if _redis is None:
        _redis = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features(
    client_id: str = Query(...),
    as_of: str = Query(...),
):
    r = get_redis()

    # Normalize as_of: if DATE format (YYYY-MM-DD), append T00:00:00
    if len(as_of) == 10:
        as_of_ts = as_of + TIMESTAMP_SUFFIX
    else:
        as_of_ts = as_of

    # Check if client exists at all
    client_exists = r.sismember("hw4:clients", client_id)
    if not client_exists:
        return _err("client_not_found", f"client {client_id!r} not found in feature store", 404)

    redis_key = f"hw4:features:{client_id}:{as_of_ts}"
    raw = r.get(redis_key)
    if raw is None:
        return _err("feature_not_found", f"no features for {client_id!r} at {as_of_ts}", 404)

    payload = json.loads(raw)
    features = {k: v for k, v in payload.items() if k not in ("client_id", "as_of")}

    return JSONResponse({
        "client_id": client_id,
        "as_of": as_of,
        "features": features,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
