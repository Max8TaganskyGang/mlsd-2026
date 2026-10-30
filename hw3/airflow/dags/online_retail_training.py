"""HW3: temporal repeat-purchase training pipeline.

Trigger with dag_run_id=<execution_id> and conf={"request_path": "/input/<request>.json"}.
Every task runs one stage of the hw3pipeline package in a fresh Python process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

PROJECT = os.environ.get("HW3_PROJECT_DIR", "/opt/hw3")
STAGES = ["stage_input", "build_examples", "train_select", "evaluate", "publish"]
MAX_LOG_LINES = 4000


def run_stage(stage: str, **context) -> None:
    dag_run = context["dag_run"]
    request_path = str((dag_run.conf or {}).get("request_path") or "")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=PROJECT,
        HW3_EXECUTION_ID=dag_run.run_id,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="2026",
    )
    try:
        with open(request_path, encoding="utf-8") as stream:
            request = json.load(stream)
        env.update(
            HW3_SUBMISSION_SHA=str(request.get("submission_sha", "")),
            HW3_DATASET_VERSION=str(request.get("dataset_version", "")),
            HW3_INPUT_SHA256=str(request.get("input_sha256", "")),
        )
    except (OSError, ValueError, AttributeError):
        pass  # stage_input reports INVALID_REQUEST itself

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hw3pipeline",
            stage,
            "--execution-id",
            dag_run.run_id,
            "--request-path",
            request_path,
        ],
        cwd=PROJECT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = 0
    for line in process.stdout:
        lines += 1
        if lines <= MAX_LOG_LINES:
            print(line.rstrip())
    if lines > MAX_LOG_LINES:
        print(f"... {lines - MAX_LOG_LINES} more lines suppressed")
    if process.wait():
        raise AirflowFailException(f"{stage} failed with exit code {process.returncode}")


with DAG(
    dag_id="online_retail_training",
    description="HW3 temporal examples, candidate selection, evaluation and bundle publication",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"retries": 0, "owner": "hw3"},
    tags=["hw3"],
) as dag:
    tasks = [
        PythonOperator(task_id=stage, python_callable=run_stage, op_kwargs={"stage": stage})
        for stage in STAGES
    ]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
