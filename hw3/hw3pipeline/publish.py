"""publish: hash every result file, write result.json, then swap current.json."""

from __future__ import annotations

from .common import CONTRACT_VERSION, SPLITS, StageError, dumps, sha256_bytes
from .storage import BUCKET, RunStore
from .tracking import Tracker

REQUIRED = (
    "input/input.csv.gz",
    "input/request.json",
    "examples/manifest.json",
    "selection/experiments.json",
    "selection/validation_predictions.parquet",
    "selection/validation_metrics.json",
    "selection/selected.json",
    "selection/bundle/manifest.json",
    "selection/bundle/schema.json",
    "selection/bundle/requirements.lock",
    "selection/bundle/pipeline.joblib",
    "evaluation/test_predictions.parquet",
    "evaluation/test_metrics.json",
)


def run(execution_id: str) -> None:
    store = RunStore(execution_id)
    request = store.get_json("input/request.json")
    selected = store.get_json("selection/selected.json")
    keys = [key for key in store.list() if key not in {"result.json", "error.json"}]
    missing = [key for key in REQUIRED if key not in keys]
    missing += [
        f"examples/{split}/"
        for split in SPLITS
        if not any(key.startswith(f"examples/{split}/") for key in keys)
    ]
    if missing:
        raise StageError("INVALID_REQUEST", "result is incomplete: " + ", ".join(missing))

    tracker = Tracker(request)
    parent_id = tracker.parent(create=False)
    result = {
        "contract_version": CONTRACT_VERSION,
        "execution_id": execution_id,
        "submission_sha": request["submission_sha"],
        "dataset_version": request["dataset_version"],
        "input_sha256": request["input_sha256"],
        "model_id": selected["model_id"],
        "selected_experiment_id": selected["experiment_id"],
        "parent_mlflow_run_id": parent_id,
        "status": "ready",
        "files": [{"path": key, "sha256": sha256_bytes(store.get_bytes(key))} for key in keys],
    }
    body = store.put_json("result.json", result)
    pointer = {
        "execution_id": execution_id,
        "result_uri": f"s3://{BUCKET}/{store.key('result.json')}",
        "sha256": sha256_bytes(body),
    }
    store.s3.put_object(Bucket=BUCKET, Key="current.json", Body=dumps(pointer))
    tracker.log(
        parent_id,
        tags={"published": "true", "result_uri": pointer["result_uri"], "model_id": selected["model_id"]},
    )
    tracker.finished(parent_id)
    print(f"published {pointer['result_uri']} with {len(keys)} files")
