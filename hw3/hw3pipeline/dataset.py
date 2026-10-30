"""Example tables -> ordered keys, labels and the float64 model input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import FEATURES

PREDICTION_SCHEMA = pa.schema(
    [
        ("experiment_id", pa.string()),
        ("customer_id", pa.string()),
        ("as_of", pa.string()),
        ("split", pa.string()),
        ("label", pa.int64()),
        ("probability", pa.float64()),
        ("in_slice", pa.bool_()),
    ]
)


def read_examples(directory: Path) -> list[dict]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        return []
    rows = pq.read_table([str(path) for path in files]).to_pylist()
    return sorted(rows, key=lambda row: (row["as_of"], row["customer_id"]))


def features(rows: list[dict], order: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Exact decimals become float64 exactly once, in the declared feature order."""
    return pd.DataFrame(
        {name: np.array([float(row[name]) for row in rows], dtype=np.float64) for name in order},
        columns=list(order),
    )


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([int(row["label"]) for row in rows], dtype=np.int64)


def prediction_rows(
    experiment_id: str, split: str, rows: list[dict], probabilities: list[float]
) -> list[dict]:
    return [
        {
            "experiment_id": experiment_id,
            "customer_id": row["customer_id"],
            "as_of": row["as_of"],
            "split": split,
            "label": int(row["label"]),
            "probability": float(probability),
            "in_slice": int(row["order_count"]) == 1,
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def write_predictions(rows: list[dict], path: Path) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), path)
    return path
