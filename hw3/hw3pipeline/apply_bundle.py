"""Load a frozen bundle in its own process and score example rows.

Usage: python -m hw3pipeline.apply_bundle BUNDLE_DIR EXAMPLES_DIR OUTPUT_JSON
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib

from .common import CONTRACT_VERSION, sha256_bytes, sha256_file, strict_loads
from .dataset import features, read_examples

BUNDLE_FILES = ("pipeline.joblib", "requirements.lock", "schema.json")


class InvalidBundle(Exception):
    pass


def verify(bundle: Path) -> dict:
    manifest = strict_loads((bundle / "manifest.json").read_bytes())
    files = sorted(
        ({"path": name, "sha256": sha256_file(bundle / name)} for name in BUNDLE_FILES),
        key=lambda item: item["path"],
    )
    model_id = sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )
    if (
        manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("files") != files
        or manifest.get("model_id") != model_id
    ):
        raise InvalidBundle("bundle files do not match manifest.json")
    return manifest


def main(bundle: Path, examples: Path, output: Path) -> int:
    try:
        manifest = verify(bundle)
        schema = strict_loads((bundle / "schema.json").read_bytes())
        order = tuple(item["name"] for item in schema["features"])
        if schema.get("positive_class") != 1 or any(
            item.get("model_type") != "float64" for item in schema["features"]
        ):
            raise InvalidBundle("unsupported schema.json")
        pipeline = joblib.load(bundle / "pipeline.joblib")
    except (OSError, ValueError, KeyError, TypeError, InvalidBundle) as error:
        print(f"INVALID_BUNDLE: {error}", file=sys.stderr)
        return 4
    rows = read_examples(examples)
    probabilities: list[float] = []
    if rows:
        frame = features(rows, order)
        column = list(pipeline.classes_).index(1)
        probabilities = [float(value) for value in pipeline.predict_proba(frame)[:, column]]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        print("INVALID_BUNDLE: probability outside [0, 1]", file=sys.stderr)
        return 4
    output.write_text(
        json.dumps(
            {
                "model_id": manifest["model_id"],
                "predictions": [
                    {"customer_id": row["customer_id"], "as_of": row["as_of"], "probability": value}
                    for row, value in zip(rows, probabilities, strict=True)
                ],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
