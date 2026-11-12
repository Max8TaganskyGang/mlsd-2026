"""gateway: public API on port 8080. Orchestrates feature-service and model-service."""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any

import httpx
import psycopg2
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PG_DSN = os.environ.get(
    "HW4_POSTGRES_DSN",
    "postgresql://airflow:airflow@postgres/airflow",
)
FEATURE_SERVICE = os.environ.get("HW4_FEATURE_SERVICE_URL", "http://feature-service:8081")
MODEL_SERVICE = os.environ.get("HW4_MODEL_SERVICE_URL", "http://model-service:8082")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DECLARED_CODES = {
    "invalid_request", "client_not_found", "feature_not_found", "model_not_found",
    "incompatible_model", "features_unavailable", "model_unavailable", "not_ready",
}

app = FastAPI()


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(ApiError(code, message, status).body(), status_code=status)


@app.exception_handler(ApiError)
def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(exc.body(), status_code=exc.status)


@app.exception_handler(RequestValidationError)
def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _err("invalid_request", "malformed request body", 400)


@app.exception_handler(StarletteHTTPException)
def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        return _err("invalid_request", "unknown endpoint", 404)
    return _err("invalid_request", str(exc.detail), exc.status_code)


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return _err("not_ready", f"internal failure: {type(exc).__name__}", 503)


def _json_object(body: Any) -> dict:
    if not isinstance(body, dict):
        raise ApiError("invalid_request", "body must be a JSON object", 400)
    return body


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

def _row_to_model(row) -> dict:
    return {
        "model_name": row[0],
        "model_version": str(row[1]),
        "activation_id": row[2],
        "state_id": row[3],
    }


def _get_active_model() -> dict | None:
    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT model_name, model_version, activation_id, state_id "
                "FROM hw4.active_model ORDER BY activated_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    return _row_to_model(row) if row else None


def _require_active_model() -> dict:
    active = _get_active_model()
    if active is None:
        raise ApiError("not_ready", "no active model configured", 503)
    return active


def _activate_model(model_name: str, model_version: str, activation_id: str) -> dict:
    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
    except Exception:
        raise ApiError("not_ready", "state store unavailable", 503)
    try:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT model_name, model_version, activation_id, state_id "
            "FROM hw4.active_model WHERE activation_id = %s",
            (activation_id,),
        )
        existing = cur.fetchone()
        if existing:
            return _row_to_model(existing)
        state_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO hw4.active_model "
            "(model_name, model_version, activation_id, state_id, activated_at) "
            "VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (activation_id) DO NOTHING",
            (model_name, model_version, activation_id, state_id),
        )
        cur.execute(
            "SELECT model_name, model_version, activation_id, state_id "
            "FROM hw4.active_model WHERE activation_id = %s",
            (activation_id,),
        )
        return _row_to_model(cur.fetchone())
    finally:
        conn.close()


def _ensure_model_loadable(model_name: str, model_version: str, timeout: float = 10.0) -> None:
    try:
        resp = httpx.post(
            f"{MODEL_SERVICE}/load",
            json={"model_name": model_name, "model_version": model_version},
            timeout=httpx.Timeout(timeout, connect=2.0),
        )
    except Exception:
        raise ApiError("model_unavailable", "model-service unreachable", 503)
    if resp.status_code != 200:
        raise _upstream_error(resp, "model_unavailable")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _validate_item(item: Any) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise ApiError("invalid_request", "request must be a JSON object", 400)
    client_id = item.get("client_id")
    as_of = item.get("as_of")
    if client_id is None or as_of is None:
        raise ApiError("invalid_request", "client_id and as_of are required", 400)
    if not isinstance(client_id, str) or not client_id.strip():
        raise ApiError("invalid_request", "client_id must be a non-empty string", 400)
    if not isinstance(as_of, str) or not DATE_RE.match(as_of):
        raise ApiError("invalid_request", "as_of must be a YYYY-MM-DD string", 400)
    try:
        datetime.strptime(as_of, "%Y-%m-%d")
    except ValueError:
        raise ApiError("invalid_request", "as_of is not a valid date", 400)
    return client_id, as_of


def _upstream_error(resp: httpx.Response, fallback_code: str) -> ApiError:
    try:
        error = resp.json().get("error", {})
        code, message = error.get("code"), error.get("message", "")
    except Exception:
        code, message = None, ""
    if code in DECLARED_CODES:
        return ApiError(code, message or code, resp.status_code if resp.status_code >= 400 else 503)
    return ApiError(fallback_code, f"upstream returned {resp.status_code}", 503)


def _fetch_features(client_id: str, as_of: str) -> dict:
    try:
        resp = httpx.get(
            f"{FEATURE_SERVICE}/features",
            params={"client_id": client_id, "as_of": as_of},
            timeout=httpx.Timeout(4.0, connect=2.0),
        )
    except Exception:
        raise ApiError("features_unavailable", "feature-service unreachable", 503)
    if resp.status_code != 200:
        raise _upstream_error(resp, "features_unavailable")
    try:
        features = resp.json()["features"]
    except Exception:
        raise ApiError("features_unavailable", "feature-service returned invalid body", 503)
    if not isinstance(features, dict):
        raise ApiError("features_unavailable", "feature-service returned invalid body", 503)
    return features


def _score(model_name: str, model_version: str, features: dict) -> float:
    try:
        resp = httpx.post(
            f"{MODEL_SERVICE}/predict",
            json={"model_name": model_name, "model_version": model_version, "features": features},
            timeout=httpx.Timeout(8.0, connect=2.0),
        )
    except Exception:
        raise ApiError("model_unavailable", "model-service unreachable", 503)
    if resp.status_code != 200:
        raise _upstream_error(resp, "model_unavailable")
    try:
        probability = resp.json()["probability"]
    except Exception:
        raise ApiError("model_unavailable", "model-service returned invalid body", 503)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) \
            or not 0.0 <= float(probability) <= 1.0:
        raise ApiError("model_unavailable", "model-service returned invalid probability", 503)
    return float(probability)


def _predict(client_id: str, as_of: str, active: dict) -> dict:
    features = _fetch_features(client_id, as_of)
    probability = _score(active["model_name"], active["model_version"], features)
    return {
        "client_id": client_id,
        "as_of": as_of,
        "probability": probability,
        "model_name": active["model_name"],
        "model_version": active["model_version"],
        "feature_key": f"hw4:features:{client_id}:{as_of}",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    active = _get_active_model()
    if active is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    try:
        _ensure_model_loadable(active["model_name"], active["model_version"], timeout=5.0)
    except ApiError:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({
        "status": "ok",
        "model_name": active["model_name"],
        "model_version": active["model_version"],
    })


@app.get("/model")
def get_model():
    return JSONResponse(_require_active_model())


@app.post("/admin/activate")
def admin_activate(payload: Any = Body(None)):
    body = _json_object(payload)
    values = {}
    for field in ("model_name", "model_version", "activation_id"):
        value = body.get(field)
        if field == "model_version" and isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise ApiError("invalid_request", f"{field} must be a non-empty string", 400)
        values[field] = value
    _ensure_model_loadable(values["model_name"], values["model_version"])
    return JSONResponse(
        _activate_model(values["model_name"], values["model_version"], values["activation_id"])
    )


@app.post("/predict")
def predict(payload: Any = Body(None)):
    client_id, as_of = _validate_item(_json_object(payload))
    active = _require_active_model()
    return JSONResponse(_predict(client_id, as_of, active))


@app.post("/predict/batch")
def predict_batch(payload: Any = Body(None)):
    items = _json_object(payload).get("items")
    if not isinstance(items, list):
        raise ApiError("invalid_request", "items must be a list", 400)
    active = _require_active_model()
    model_error: ApiError | None = None
    if items:
        try:
            _ensure_model_loadable(active["model_name"], active["model_version"])
        except ApiError as exc:
            model_error = exc
    results = []
    for index, item in enumerate(items):
        try:
            client_id, as_of = _validate_item(item)
            if model_error is not None:
                _fetch_features(client_id, as_of)
                raise model_error
            result = {"index": index, "ok": True, **_predict(client_id, as_of, active)}
        except ApiError as exc:
            result = {"index": index, "ok": False}
            if isinstance(item, dict):
                for field in ("client_id", "as_of"):
                    if field in item:
                        result[field] = item[field]
            result.update(exc.body())
        results.append(result)
    return JSONResponse({"results": results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
