"""Shared constants and helpers for hw4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "hw4-serving-v1"
SCHEMA_VERSION = "hw4-features-v1"
SEED = 2026

FEATURE_NAMES = (
    "order_count_90d",
    "total_amount_90d",
    "average_order_amount_90d",
    "days_since_last_order",
    "distinct_stock_codes_90d",
    "return_rate_90d",
    "order_count_30d",
    "total_amount_30d",
)

FEATURE_SOURCE_TYPES = (
    "int64",
    "decimal(20,6)",
    "decimal(20,2)",
    "int64",
    "int64",
    "decimal(10,6)",
    "int64",
    "decimal(20,6)",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def schema_document() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "features": [
            {"name": name, "source_type": src, "model_type": "float64"}
            for name, src in zip(FEATURE_NAMES, FEATURE_SOURCE_TYPES, strict=True)
        ],
        "positive_class": 1,
    }
