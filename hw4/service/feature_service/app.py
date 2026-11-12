"""feature-service: return feature vectors from Redis for a given client_id + as_of date."""

from __future__ import annotations

import json
import os
from datetime import datetime

import redis as redis_lib
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

REDIS_HOST = os.environ.get("HW4_REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("HW4_REDIS_PORT", "6379"))
TIMESTAMP_SUFFIX = "T00:00:00"

app = FastAPI()
_redis: redis_lib.Redis | None = None


def get_redis() -> redis_lib.Redis:
    global _redis
    if _redis is None:
        _redis = redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=3, socket_timeout=3,
        )
    return _redis


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(RequestValidationError)
def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _err("invalid_request", "client_id and as_of query parameters are required", 400)


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return _err("features_unavailable", f"feature store failure: {type(exc).__name__}", 503)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def get_features(
    client_id: str = Query(...),
    as_of: str = Query(...),
):
    if not client_id.strip():
        return _err("invalid_request", "client_id must be non-empty", 400)
    try:
        datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        return _err("invalid_request", "as_of must be YYYY-MM-DD", 400)
    as_of_ts = as_of + TIMESTAMP_SUFFIX
    redis_key = f"hw4:features:{client_id}:{as_of_ts}"

    try:
        r = get_redis()
        client_exists = r.sismember("hw4:clients", client_id)
        raw = r.get(redis_key) if client_exists else None
    except redis_lib.RedisError as exc:
        return _err("features_unavailable", f"redis unavailable: {type(exc).__name__}", 503)

    if not client_exists:
        return _err("client_not_found", f"client {client_id!r} not found in feature store", 404)
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
