"""Shared constants, stage errors, request validation and JSON helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

CONTRACT_VERSION = "hw3-temporal-v1"
SEED = 2026
SPLITS = ("train", "validation", "test")
FEATURES = ("order_count", "total_amount", "average_order_amount")
FEATURE_SOURCE_TYPES = ("int64", "decimal(20,6)", "decimal(20,2)")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
INPUT_ROOT = Path("/input")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "config" / "candidates.json"
LOCK_PATH = PROJECT_ROOT / "requirements.lock"

EXECUTION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SHA256 = re.compile(r"[0-9a-f]{64}")
SUBMISSION_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class StageError(Exception):
    """Understandable stage refusal with a contract error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_loads(data: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value: str) -> Any:
        raise ValueError("non-finite JSON number: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise StageError("INVALID_REQUEST", f"{field} must be a string timestamp")
    try:
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError as error:
        raise StageError("INVALID_REQUEST", f"{field} is not YYYY-MM-DDTHH:MM:SS") from error
    if parsed.strftime(TIMESTAMP_FORMAT) != value:
        raise StageError("INVALID_REQUEST", f"{field} is not canonical")
    return parsed


def fmt(value: datetime) -> str:
    return value.strftime(TIMESTAMP_FORMAT)


def finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("non-finite metric")
    return value


def validate_request(request: Any, execution_id: str) -> dict[str, Any]:
    """Check the request shape; returns the request unchanged."""
    if not isinstance(request, dict):
        raise StageError("INVALID_REQUEST", "request must be a JSON object")
    required = {
        "contract_version",
        "execution_id",
        "submission_sha",
        "dataset_version",
        "input_path",
        "input_sha256",
        "observed_from",
        "observed_until",
        "history_days",
        "horizon_days",
        "snapshots",
    }
    missing = sorted(required - request.keys())
    if missing:
        raise StageError("INVALID_REQUEST", "missing fields: " + ", ".join(missing))
    if request["contract_version"] != CONTRACT_VERSION:
        raise StageError("INVALID_REQUEST", "unsupported contract_version")
    if not isinstance(request["execution_id"], str) or not EXECUTION_ID.fullmatch(
        request["execution_id"]
    ):
        raise StageError("INVALID_REQUEST", "execution_id must match [A-Za-z0-9_-]{1,64}")
    if request["execution_id"] != execution_id:
        raise StageError("INVALID_REQUEST", "execution_id differs from dag_run_id")
    if not isinstance(request["submission_sha"], str) or not SUBMISSION_SHA.fullmatch(
        request["submission_sha"]
    ):
        raise StageError("INVALID_REQUEST", "submission_sha must be a full git SHA")
    if not isinstance(request["dataset_version"], str) or not request["dataset_version"]:
        raise StageError("INVALID_REQUEST", "dataset_version must be a non-empty string")
    if not isinstance(request["input_sha256"], str) or not SHA256.fullmatch(
        request["input_sha256"]
    ):
        raise StageError("INVALID_REQUEST", "input_sha256 must be a lowercase SHA-256")
    input_path = request["input_path"]
    if (
        not isinstance(input_path, str)
        or not input_path
        or PurePosixPath(input_path).is_absolute()
        or ".." in PurePosixPath(input_path).parts
    ):
        raise StageError("INVALID_REQUEST", "input_path must be relative to /input")
    if request["history_days"] != 90 or request["horizon_days"] != 30:
        raise StageError("INVALID_REQUEST", "history_days=90 and horizon_days=30 are fixed")
    observed_from = parse_timestamp(request["observed_from"], "observed_from")
    observed_until = parse_timestamp(request["observed_until"], "observed_until")
    if observed_from >= observed_until:
        raise StageError("INVALID_REQUEST", "observed_from must precede observed_until")
    snapshots = request["snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise StageError("INVALID_REQUEST", "snapshots must be a non-empty list")
    seen: set[str] = set()
    for item in snapshots:
        if not isinstance(item, dict) or set(item) != {"as_of", "split"}:
            raise StageError("INVALID_REQUEST", "snapshot must contain as_of and split")
        if item["split"] not in SPLITS or item["split"] in seen:
            raise StageError("INVALID_REQUEST", "snapshot splits must be unique train/validation/test")
        seen.add(item["split"])
        parse_timestamp(item["as_of"], "as_of")
    if len({item["as_of"] for item in snapshots}) != len(snapshots):
        raise StageError("INVALID_REQUEST", "one date cannot belong to two splits")
    return request


def windows(request: dict[str, Any]) -> list[dict[str, Any]]:
    """History and label windows for every snapshot of the request."""
    history = timedelta(days=request["history_days"])
    horizon = timedelta(days=request["horizon_days"])
    result = []
    for item in request["snapshots"]:
        as_of = parse_timestamp(item["as_of"], "as_of")
        result.append(
            {
                "split": item["split"],
                "as_of": fmt(as_of),
                "history_from": fmt(as_of - history),
                "label_until": fmt(as_of + horizon),
            }
        )
    return result


def require_full_calendar(request: dict[str, Any]) -> None:
    """The full DAG needs exactly train/validation/test with matured earlier labels."""
    splits = [item["split"] for item in request["snapshots"]]
    if sorted(splits) != sorted(SPLITS):
        raise StageError("INVALID_REQUEST", "full DAG requires exactly train, validation and test")
    by_split = {item["split"]: item for item in windows(request)}
    if not (
        by_split["train"]["label_until"] <= by_split["validation"]["as_of"]
        and by_split["validation"]["label_until"] <= by_split["test"]["as_of"]
    ):
        raise StageError(
            "INVALID_REQUEST", "labels of an earlier split must mature before the next split"
        )
