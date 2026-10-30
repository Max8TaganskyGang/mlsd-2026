"""MLflow runs: one parent per execution, one child per candidate, resumed on retry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from mlflow.utils.time import get_current_time_millis

from .common import CONTRACT_VERSION, SEED

EXPERIMENT = "hw3"
IDENTITY = ("contract_version", "execution_id", "submission_sha", "dataset_version", "input_sha256")


class Tracker:
    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        experiment = self.client.get_experiment_by_name(EXPERIMENT)
        self.experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else self.client.create_experiment(EXPERIMENT)
        )

    def _identity_tags(self) -> dict[str, str]:
        return {key: str(self.request[key]) for key in IDENTITY} | {"seed": str(SEED)}

    def _find(self, filters: dict[str, str]) -> list:
        query = " and ".join(f"tags.`{key}` = '{value}'" for key, value in filters.items())
        return self.client.search_runs([self.experiment_id], filter_string=query, max_results=50)

    def parent(self, *, create: bool) -> str:
        runs = self._find(
            {
                "run_kind": "parent",
                "contract_version": CONTRACT_VERSION,
                "execution_id": self.request["execution_id"],
                "submission_sha": self.request["submission_sha"],
            }
        )
        if len(runs) > 1:
            raise RuntimeError("more than one MLflow parent run for this execution")
        if runs:
            return runs[0].info.run_id
        if not create:
            raise RuntimeError("MLflow parent run of this execution is missing")
        tags = self._identity_tags() | {
            "run_kind": "parent",
            "mlflow.runName": self.request["execution_id"],
        }
        return self.client.create_run(self.experiment_id, tags=tags).info.run_id

    def child(self, parent_id: str, config: dict[str, Any]) -> str:
        identifier = config["experiment_id"]
        runs = self._find({"mlflow.parentRunId": parent_id, "experiment_id": identifier})
        if len(runs) > 1:
            raise RuntimeError(f"more than one MLflow child run for {identifier}")
        if runs:
            return runs[0].info.run_id
        tags = self._identity_tags() | {
            "run_kind": "candidate",
            "experiment_id": identifier,
            "family": config["family"],
            "mlflow.parentRunId": parent_id,
            "parent_run_id": parent_id,
            "mlflow.runName": identifier,
        }
        return self.client.create_run(self.experiment_id, tags=tags).info.run_id

    def running(self, run_id: str) -> None:
        self.client.update_run(run_id, status="RUNNING")

    def finished(self, run_id: str) -> None:
        self.client.set_terminated(run_id, status="FINISHED")

    def failed(self, run_id: str) -> None:
        self.client.set_terminated(run_id, status="FAILED")

    def log(
        self,
        run_id: str,
        *,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float | None] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        now = get_current_time_millis()
        self.client.log_batch(
            run_id,
            params=[Param(key, str(value)) for key, value in (params or {}).items()],
            metrics=[
                Metric(key, float(value), now, 0)
                for key, value in (metrics or {}).items()
                if value is not None
            ],
            tags=[RunTag(key, str(value)) for key, value in (tags or {}).items()],
        )

    def artifact(self, run_id: str, path: Path, artifact_path: str | None = None) -> None:
        self.client.log_artifact(run_id, str(path), artifact_path)

    def artifacts(self, run_id: str, directory: Path, artifact_path: str) -> None:
        self.client.log_artifacts(run_id, str(directory), artifact_path)


def request_params(request: dict[str, Any]) -> dict[str, Any]:
    params = {
        "contract_version": request["contract_version"],
        "dataset_version": request["dataset_version"],
        "input_path": request["input_path"],
        "input_sha256": request["input_sha256"],
        "observed_from": request["observed_from"],
        "observed_until": request["observed_until"],
        "history_days": request["history_days"],
        "horizon_days": request["horizon_days"],
        "seed": SEED,
    }
    for item in request["snapshots"]:
        params[f"as_of.{item['split']}"] = item["as_of"]
    params["snapshots_json"] = json.dumps(request["snapshots"], sort_keys=True)
    return params
