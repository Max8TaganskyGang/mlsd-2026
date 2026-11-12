"""model-service: load a model from MLflow and apply it to a feature vector."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

import joblib
import mlflow
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.exceptions import MlflowException

sys.path.insert(0, "/opt/hw4")
from hw4pipeline.common import FEATURE_NAMES  # noqa: E402,F401

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

app = FastAPI()
mlflow.set_tracking_uri(MLFLOW_URI)
_model_cache: dict[tuple[str, str], tuple[Any, list[str]]] = {}
_cache_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(ApiError)
def _api_error(_: Request, exc: ApiError) -> JSONResponse:
    return _err(exc.code, exc.message, exc.status)


@app.exception_handler(RequestValidationError)
def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _err("invalid_request", "malformed request body", 400)


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return _err("model_unavailable", f"model failure: {type(exc).__name__}", 503)


def _resolve_run_id(model_name: str, model_version: str) -> str:
    client = mlflow.tracking.MlflowClient()
    try:
        return client.get_model_version(model_name, model_version).run_id
    except MlflowException as exc:
        if exc.error_code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise ApiError(
                "model_not_found",
                f"model {model_name!r} version {model_version!r} not found",
                404,
            )
        raise ApiError("model_unavailable", f"MLflow error: {exc.error_code}", 503)
    except Exception as exc:
        raise ApiError("model_unavailable", f"MLflow unreachable: {type(exc).__name__}", 503)


def _load_model(model_name: str, model_version: str) -> tuple[Any, list[str]]:
    key = (model_name, model_version)
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    run_id = _resolve_run_id(model_name, model_version)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="bundle", dst_path=tmpdir
                )
            )
            pipeline = joblib.load(bundle_dir / "pipeline.joblib")
            schema_raw = json.loads((bundle_dir / "schema.json").read_bytes())
    except Exception as exc:
        raise ApiError("model_unavailable", f"cannot load model artifacts: {type(exc).__name__}", 503)
    try:
        feature_order = [f["name"] for f in schema_raw["features"]]
    except Exception:
        raise ApiError("incompatible_model", "model schema.json is malformed", 409)
    with _cache_lock:
        _model_cache[key] = (pipeline, feature_order)
    return pipeline, feature_order


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Any = Body(None)):
    if not isinstance(payload, dict):
        raise ApiError("invalid_request", "body must be a JSON object", 400)
    model_name = payload.get("model_name")
    model_version = payload.get("model_version")
    features = payload.get("features")
    if isinstance(model_version, int) and not isinstance(model_version, bool):
        model_version = str(model_version)
    if not isinstance(model_name, str) or not model_name \
            or not isinstance(model_version, str) or not model_version:
        raise ApiError("invalid_request", "model_name and model_version must be strings", 400)
    if not isinstance(features, dict):
        raise ApiError("invalid_request", "features must be a JSON object", 400)

    pipeline, feature_order = _load_model(model_name, model_version)

    if set(features) != set(feature_order):
        raise ApiError(
            "incompatible_model",
            f"feature schema mismatch: expected {sorted(feature_order)}, got {sorted(features)}",
            409,
        )
    try:
        x = np.array([[float(features[name]) for name in feature_order]])
    except (TypeError, ValueError) as exc:
        raise ApiError("incompatible_model", f"cannot convert features to numbers: {exc}", 409)

    try:
        classes = list(pipeline.classes_)
        prob = float(pipeline.predict_proba(x)[0, classes.index(1)])
    except Exception as exc:
        raise ApiError("model_unavailable", f"model failed to predict: {type(exc).__name__}", 503)
    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ApiError("model_unavailable", "model produced invalid probability", 503)
    return JSONResponse({"probability": prob})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
