"""evaluate: score test with the frozen bundle (separate process) and the constant."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import metrics
from .common import CONTRACT_VERSION, PROJECT_ROOT, StageError, dumps
from .dataset import prediction_rows, read_examples, write_predictions
from .storage import RunStore
from .tracking import Tracker


def apply_bundle(bundle: Path, examples: Path, output: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "hw3pipeline.apply_bundle", str(bundle), str(examples), str(output)],
        cwd=PROJECT_ROOT,
        env=os.environ | {"PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise StageError("INVALID_BUNDLE", (result.stderr or result.stdout).strip()[-500:])
    return json.loads(output.read_text())


def run(execution_id: str) -> None:
    store = RunStore(execution_id)
    request = store.get_json("input/request.json")
    selected = store.get_json("selection/selected.json")
    experiments = store.get_json("selection/experiments.json")
    constant = next(
        item["experiment_id"] for item in experiments["candidates"] if item["family"] == "constant"
    )
    chosen = selected["experiment_id"]
    p_train = float(selected["p_train"])

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        bundle = work / "bundle"
        store.download_prefix("selection/bundle/", bundle)
        store.download_prefix("examples/test/", work / "test")
        test = read_examples(work / "test")
        applied = apply_bundle(bundle, work / "test", work / "applied.json")
        if applied["model_id"] != selected["model_id"]:
            raise StageError("INVALID_BUNDLE", "bundle model_id differs from selected.json")
        loaded = {(row["customer_id"], row["as_of"]): row["probability"] for row in applied["predictions"]}
        if set(loaded) != {(row["customer_id"], row["as_of"]) for row in test}:
            raise StageError("INVALID_BUNDLE", "bundle predictions do not cover test keys")
        chosen_probabilities = [loaded[(row["customer_id"], row["as_of"])] for row in test]
        rows = prediction_rows(chosen, "test", test, chosen_probabilities)
        if chosen != constant:
            rows += prediction_rows(constant, "test", test, [p_train] * len(test))

        records = []
        for identifier, role in ((chosen, "selected"), (constant, "baseline")):
            own = [row for row in rows if row["experiment_id"] == identifier]
            for group, members in (
                ("all", own),
                ("order_count_eq_1", [row for row in own if row["in_slice"]]),
            ):
                records.append(
                    metrics.record(
                        identifier,
                        role,
                        "test",
                        group,
                        [row["label"] for row in members],
                        [row["probability"] for row in members],
                    )
                )
        evaluation = work / "evaluation"
        evaluation.mkdir()
        write_predictions(rows, evaluation / "test_predictions.parquet")
        (evaluation / "test_metrics.json").write_bytes(
            dumps(
                {
                    "contract_version": CONTRACT_VERSION,
                    "execution_id": execution_id,
                    "model_id": selected["model_id"],
                    "records": records,
                }
            )
        )
        store.delete_prefix("evaluation/")
        for name in ("test_predictions.parquet", "test_metrics.json"):
            store.put_file("evaluation/" + name, evaluation / name)

        tracker = Tracker(request)
        parent_id = tracker.parent(create=False)
        tracker.running(parent_id)
        tracker.log(
            parent_id,
            metrics={
                f"test.{item['role']}.{item['group']}.{name}": item[name]
                for item in records
                for name in ("log_loss", "roc_auc")
            }
            | {f"test.{item['role']}.{item['group']}.n": item["n"] for item in records},
        )
        for name in ("test_predictions.parquet", "test_metrics.json"):
            tracker.artifact(parent_id, evaluation / name)
        tracker.finished(parent_id)
    for item in records:
        print(
            f"test {item['role']}/{item['group']}: n={item['n']} log_loss={item['log_loss']} "
            f"roc_auc={item['roc_auc']}"
        )
