"""model-service: load a model from MLflow and apply it to a feature vector."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, "/opt/hw4")
from hw4pipeline.common import FEATURE_NAMES

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

app = FastAPI()
mlflow.set_tracking_uri(MLFLOW_URI)
_model_cache: dict[tuple[str, str], Any] = {}


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _load_model(model_name: str, model_version: str):
    key = (model_name, model_version)
    if key in _model_cache:
        return _model_cache[key]

    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version(model_name, model_version)
    except Exception:
        return None

    run_id = mv.run_id
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="bundle", dst_path=tmpdir
            )
            bundle_dir = Path(artifact_path)
            pipeline = joblib.load(bundle_dir / "pipeline.joblib")
            schema_raw = json.loads((bundle_dir / "schema.json").read_bytes())
            feature_order = [f["name"] for f in schema_raw["features"]]
            _model_cache[key] = (pipeline, feature_order)
            return _model_cache[key]
    except Exception:
        return None


class PredictRequest(BaseModel):
    model_name: str
    model_version: str
    features: dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    result = _load_model(req.model_name, req.model_version)
    if result is None:
        return _err("model_not_found", f"model {req.model_name!r} version {req.model_version!r} not found", 404)

    pipeline, feature_order = result

    # Verify schema compatibility
    incoming_keys = set(req.features.keys())
    expected_keys = set(feature_order)
    if incoming_keys != expected_keys:
        return _err(
            "incompatible_model",
            f"feature schema mismatch: expected {sorted(expected_keys)}, got {sorted(incoming_keys)}",
            409,
        )

    # Build feature vector in model's expected order
    try:
        x = np.array([[float(req.features[name]) for name in feature_order]])
    except (ValueError, KeyError) as e:
        return _err("incompatible_model", f"cannot convert features to float: {e}", 409)

    classes = list(pipeline.classes_)
    pos_idx = classes.index(1)
    prob = float(pipeline.predict_proba(x)[0, pos_idx])

    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        return _err("model_unavailable", "model produced invalid probability", 503)

    return JSONResponse({"probability": prob})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
