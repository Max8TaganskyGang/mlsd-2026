"""gateway: public API on port 8080. Orchestrates feature-service and model-service."""

from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import datetime
from typing import Any

import httpx
import psycopg2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

PG_DSN = os.environ.get(
    "HW4_POSTGRES_DSN",
    "postgresql://airflow:airflow@postgres/airflow",
)
FEATURE_SERVICE = os.environ.get("HW4_FEATURE_SERVICE_URL", "http://feature-service:8081")
MODEL_SERVICE = os.environ.get("HW4_MODEL_SERVICE_URL", "http://model-service:8082")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI()


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

def _get_active_model() -> dict | None:
    try:
        conn = psycopg2.connect(PG_DSN)
        cur = conn.cursor()
        cur.execute(
            "SELECT model_name, model_version, activation_id, state_id "
            "FROM hw4.active_model ORDER BY activated_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "model_name": row[0],
            "model_version": str(row[1]),
            "activation_id": row[2],
            "state_id": row[3],
        }
    except Exception:
        return None


def _activate_model(model_name: str, model_version: str, activation_id: str) -> dict:
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    # Check idempotency
    cur.execute(
        "SELECT model_name, model_version, activation_id, state_id FROM hw4.active_model WHERE activation_id = %s",
        (activation_id,),
    )
    existing = cur.fetchone()
    if existing:
        conn.close()
        return {
            "model_name": existing[0],
            "model_version": str(existing[1]),
            "activation_id": existing[2],
            "state_id": existing[3],
        }
    state_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO hw4.active_model (model_name, model_version, activation_id, state_id, activated_at)
        VALUES (%s, %s, %s, %s, NOW())
        """,
        (model_name, model_version, activation_id, state_id),
    )
    conn.close()
    return {
        "model_name": model_name,
        "model_version": model_version,
        "activation_id": activation_id,
        "state_id": state_id,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    active = _get_active_model()
    if active is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({
        "status": "ok",
        "model_name": active["model_name"],
        "model_version": active["model_version"],
    })


@app.get("/model")
def get_model():
    active = _get_active_model()
    if active is None:
        return _err("not_ready", "no active model", 503)
    return JSONResponse(active)


class ActivateRequest(BaseModel):
    model_name: str
    model_version: str
    activation_id: str


@app.post("/admin/activate")
def admin_activate(req: ActivateRequest):
    result = _activate_model(req.model_name, req.model_version, req.activation_id)
    return JSONResponse(result)


class PredictRequest(BaseModel):
    client_id: Any = None
    as_of: Any = None

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, v):
        if not isinstance(v, str) or not v:
            raise ValueError("client_id must be a non-empty string")
        return v

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, v):
        if not isinstance(v, str):
            raise ValueError("as_of must be a string")
        if not DATE_RE.match(v):
            raise ValueError("as_of must be YYYY-MM-DD")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("as_of is not a valid date")
        return v


def _do_predict(client_id: str, as_of: str) -> JSONResponse:
    active = _get_active_model()
    if active is None:
        return _err("not_ready", "no active model configured", 503)

    model_name = active["model_name"]
    model_version = active["model_version"]
    feature_key = f"hw4:features:{client_id}:{as_of}"

    # Get features from feature-service
    try:
        resp = httpx.get(
            f"{FEATURE_SERVICE}/features",
            params={"client_id": client_id, "as_of": as_of},
            timeout=10.0,
        )
    except Exception:
        return _err("features_unavailable", "feature-service unreachable", 503)

    if resp.status_code == 404:
        body = resp.json()
        err_code = body.get("error", {}).get("code", "feature_not_found")
        err_msg = body.get("error", {}).get("message", "not found")
        return JSONResponse({"error": {"code": err_code, "message": err_msg}}, status_code=404)

    if resp.status_code != 200:
        return _err("features_unavailable", f"feature-service returned {resp.status_code}", 503)

    features = resp.json().get("features", {})

    # Call model-service
    try:
        mresp = httpx.post(
            f"{MODEL_SERVICE}/predict",
            json={"model_name": model_name, "model_version": model_version, "features": features},
            timeout=10.0,
        )
    except Exception:
        return _err("model_unavailable", "model-service unreachable", 503)

    if mresp.status_code == 404:
        return _err("model_not_found", f"model {model_name} v{model_version} not found", 404)
    if mresp.status_code == 409:
        body = mresp.json()
        return JSONResponse(body, status_code=409)
    if mresp.status_code != 200:
        return _err("model_unavailable", f"model-service returned {mresp.status_code}", 503)

    probability = mresp.json().get("probability")

    return JSONResponse({
        "client_id": client_id,
        "as_of": as_of,
        "probability": probability,
        "model_name": model_name,
        "model_version": model_version,
        "feature_key": feature_key,
    })


@app.post("/predict")
def predict(body: dict):
    # Manual validation for proper invalid_request errors
    client_id = body.get("client_id")
    as_of = body.get("as_of")

    if client_id is None or as_of is None:
        return _err("invalid_request", "client_id and as_of are required", 400)
    if not isinstance(client_id, str):
        return _err("invalid_request", "client_id must be a string", 400)
    if not isinstance(as_of, str):
        return _err("invalid_request", "as_of must be a string", 400)
    if not DATE_RE.match(as_of):
        return _err("invalid_request", "as_of must be YYYY-MM-DD", 400)
    try:
        datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        return _err("invalid_request", "as_of is not a valid date", 400)

    return _do_predict(client_id, as_of)


class BatchItem(BaseModel):
    client_id: Any = None
    as_of: Any = None


def _predict_one(item: Any) -> dict:
    import json as _json
    if not isinstance(item, dict):
        return {"error": {"code": "invalid_request", "message": "item must be an object"}}
    client_id = item.get("client_id")
    as_of = item.get("as_of")
    if client_id is None or as_of is None:
        return {"error": {"code": "invalid_request", "message": "client_id and as_of required"}}
    if not isinstance(client_id, str) or not isinstance(as_of, str):
        return {"error": {"code": "invalid_request", "message": "invalid field types"}}
    resp = _do_predict(client_id, as_of)
    return _json.loads(resp.body)


@app.post("/predict/batch")
def predict_batch(body: dict):
    items = body.get("items")
    if not isinstance(items, list):
        return _err("invalid_request", "items must be a list", 400)
    return JSONResponse({"results": [_predict_one(item) for item in items]})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
