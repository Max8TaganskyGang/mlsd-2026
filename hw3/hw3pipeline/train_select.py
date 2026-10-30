"""train_select: fit candidates on train, compare on validation, freeze the winner."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from . import metrics
from .common import (
    CANDIDATES_PATH,
    CONTRACT_VERSION,
    FEATURE_SOURCE_TYPES,
    FEATURES,
    LOCK_PATH,
    SEED,
    StageError,
    dumps,
    require_full_calendar,
    sha256_bytes,
    sha256_file,
    strict_loads,
)
from .dataset import features, labels, prediction_rows, read_examples, write_predictions
from .models import InvalidCandidate, build_pipeline, resolved_params, validate_candidates
from .storage import RunStore
from .tracking import Tracker, request_params

SELECTION_RULE = "validation_log_loss_v1"
FAMILY_RANK = {"constant": 0, "logistic": 1, "tree": 2}
TIE = 1e-12
IDENTITY = ("contract_version", "execution_id", "submission_sha", "dataset_version", "input_sha256")


def positive_probability(pipeline, frame) -> list[float]:
    if frame.empty:
        return []
    column = list(pipeline.classes_).index(1)
    values = pipeline.predict_proba(frame)[:, column]
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result):
        raise StageError("INVALID_BUNDLE", "model produced a probability outside [0, 1]")
    return result


def choose(scores: dict[str, float], families: dict[str, str]) -> str:
    best = min(scores.values())
    tied = [key for key, value in scores.items() if value <= best + TIE]
    return min(tied, key=lambda key: (FAMILY_RANK[families[key]], key))


def schema_document() -> dict:
    return {
        "schema_version": "hw3-features-v1",
        "features": [
            {"name": name, "source_type": source, "model_type": "float64"}
            for name, source in zip(FEATURES, FEATURE_SOURCE_TYPES, strict=True)
        ],
        "positive_class": 1,
    }


def write_bundle(pipeline, family: str, experiment_id: str, directory: Path) -> str:
    directory.mkdir(parents=True)
    joblib.dump(pipeline, directory / "pipeline.joblib")
    shutil.copyfile(LOCK_PATH, directory / "requirements.lock")
    (directory / "schema.json").write_bytes(dumps(schema_document()))
    files = sorted(
        (
            {"path": name, "sha256": sha256_file(directory / name)}
            for name in ("pipeline.joblib", "requirements.lock", "schema.json")
        ),
        key=lambda item: item["path"],
    )
    model_id = sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "family": family,
        "experiment_id": experiment_id,
        "files": files,
    }
    (directory / "manifest.json").write_bytes(dumps(manifest))
    return model_id


def run(execution_id: str) -> None:
    store = RunStore(execution_id)
    request = store.get_json("input/request.json")
    require_full_calendar(request)
    try:
        declared = strict_loads(CANDIDATES_PATH.read_bytes())
        candidates = validate_candidates(declared)
    except (OSError, ValueError, InvalidCandidate) as error:
        raise StageError("INVALID_REQUEST", f"config/candidates.json: {error}") from error

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        store.download_prefix("examples/train/", work / "train")
        store.download_prefix("examples/validation/", work / "validation")
        train, validation = read_examples(work / "train"), read_examples(work / "validation")
        y_train = labels(train)
        if len(train) == 0:
            raise StageError("EMPTY_TRAIN", "train examples are empty")
        if len(set(y_train.tolist())) < 2:
            raise StageError("TRAIN_SINGLE_CLASS", "train examples contain a single class")
        if len(validation) == 0:
            raise StageError("EMPTY_VALIDATION", "validation examples are empty")
        x_train, x_validation = features(train), features(validation)
        y_validation = labels(validation).tolist()
        p_train = int(y_train.sum()) / len(y_train)

        tracker = Tracker(request)
        parent_id = tracker.parent(create=True)
        tracker.running(parent_id)
        tracker.log(
            parent_id,
            params=request_params(request) | {"selection_rule": SELECTION_RULE},
            metrics={"train.n": len(train), "train.p": p_train, "validation.n": len(validation)},
        )

        fitted, entries, records, scores, all_rows = {}, [], [], {}, []
        for config in candidates:
            identifier, family = config["experiment_id"], config["family"]
            run_id = tracker.child(parent_id, config)
            tracker.running(run_id)
            np.random.seed(SEED)
            pipeline = build_pipeline(config)
            with threadpool_limits(limits=1):
                pipeline.fit(x_train, y_train)
                probabilities = positive_probability(pipeline, x_validation)
            rows = prediction_rows(identifier, "validation", validation, probabilities)
            record = metrics.record(
                identifier, "candidate", "validation", "all", y_validation, probabilities
            )
            resolved = resolved_params(pipeline)
            child_dir = work / "children" / identifier
            child_dir.mkdir(parents=True)
            (child_dir / "config.json").write_bytes(dumps(config))
            (child_dir / "resolved_params.json").write_bytes(dumps(resolved))
            (child_dir / "validation_metrics.json").write_bytes(dumps(record))
            write_predictions(rows, child_dir / "validation_predictions.parquet")
            tracker.log(
                run_id,
                params={
                    "candidate_json": json.dumps(config, sort_keys=True),
                    "seed": SEED,
                    "family": family,
                    "as_of.train": train[0]["as_of"],
                    "as_of.validation": validation[0]["as_of"],
                },
                metrics={
                    "validation.all.log_loss": record["log_loss"],
                    "validation.all.roc_auc": record["roc_auc"],
                    "validation.all.n": record["n"],
                    "validation.all.n_positive": record["n_positive"],
                },
            )
            for name in (
                "config.json",
                "resolved_params.json",
                "validation_metrics.json",
                "validation_predictions.parquet",
            ):
                tracker.artifact(run_id, child_dir / name)
            fitted[identifier] = pipeline
            scores[identifier] = record["log_loss"]
            records.append(record)
            all_rows += rows
            entries.append(
                {
                    "experiment_id": identifier,
                    "family": family,
                    "config": config,
                    "resolved_params": resolved,
                    "mlflow_run_id": run_id,
                    "status": "FINISHED",
                }
            )
            print(f"candidate {identifier}: validation log_loss={record['log_loss']:.6f}")

        families = {item["experiment_id"]: item["family"] for item in candidates}
        winner = choose(scores, families)
        winner_entry = next(item for item in entries if item["experiment_id"] == winner)
        bundle_dir = work / "selection" / "bundle"
        model_id = write_bundle(fitted[winner], families[winner], winner, bundle_dir)
        tracker.artifacts(winner_entry["mlflow_run_id"], bundle_dir, "bundle")
        tracker.log(winner_entry["mlflow_run_id"], tags={"selected": "true", "model_id": model_id})
        for entry in entries:
            tracker.finished(entry["mlflow_run_id"])

        identity = {key: request[key] for key in IDENTITY}
        selection = work / "selection"
        (selection / "experiments.json").write_bytes(dumps(identity | {"candidates": entries}))
        write_predictions(all_rows, selection / "validation_predictions.parquet")
        (selection / "validation_metrics.json").write_bytes(
            dumps(
                {
                    "contract_version": CONTRACT_VERSION,
                    "execution_id": execution_id,
                    "records": records,
                }
            )
        )
        (selection / "selected.json").write_bytes(
            dumps(
                identity
                | {
                    "experiment_id": winner,
                    "family": families[winner],
                    "mlflow_run_id": winner_entry["mlflow_run_id"],
                    "validation_log_loss": scores[winner],
                    "p_train": p_train,
                    "model_id": model_id,
                    "selection_rule": SELECTION_RULE,
                    "tie_tolerance": TIE,
                    "tie_order": ["constant", "logistic", "tree", "experiment_id"],
                }
            )
        )

        store.delete_prefix("selection/")
        for path in sorted(selection.rglob("*")):
            if path.is_file():
                store.put_file("selection/" + path.relative_to(selection).as_posix(), path)
        request_copy = work / "request.json"
        request_copy.write_bytes(store.get_bytes("input/request.json"))
        tracker.artifact(parent_id, request_copy)
        for name in ("experiments.json", "selected.json", "validation_metrics.json"):
            tracker.artifact(parent_id, selection / name)
        tracker.log(
            parent_id,
            metrics={"selected.validation.log_loss": scores[winner]},
            tags={"selected_experiment_id": winner, "model_id": model_id},
        )
        tracker.finished(parent_id)
        print(f"selected {winner} ({families[winner]}), model_id={model_id}")
