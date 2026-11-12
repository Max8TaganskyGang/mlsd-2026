"""feature-build: compute 8 features, write parquet to SeaweedFS, train+register models in MLflow,
set initial active model in Postgres."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import boto3
import joblib
import mlflow
import mlflow.tracking
import numpy as np
import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/opt/hw4")
from hw4pipeline.common import (
    CONTRACT_VERSION,
    FEATURE_NAMES,
    FEATURE_SOURCE_TYPES,
    SEED,
    dumps,
    schema_document,
    sha256_bytes,
    sha256_file,
)

INPUT_DIR = Path(os.environ.get("HW4_INPUT_DIR", "/input"))
BUCKET = os.environ.get("HW4_BUCKET", "hw4")
S3_ENDPOINT = os.environ.get("HW4_S3_ENDPOINT", "http://seaweedfs:8333")
AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "local-hw4")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "local-hw4-only")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
PG_DSN = os.environ.get(
    "HW4_POSTGRES_DSN",
    "postgresql://airflow:airflow@postgres/airflow",
)
LOCK_PATH = Path("/opt/hw4/requirements.lock")
TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"
DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        try:
            s3.create_bucket(Bucket=BUCKET)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), TIMESTAMP_FMT)


def load_history(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    df = pd.DataFrame(rows)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["invoice_at"] = pd.to_datetime(df["invoice_at"], format=TIMESTAMP_FMT)
    return df


def compute_features_for_snapshot(df: pd.DataFrame, as_of_dt: datetime) -> pd.DataFrame:
    """Compute 8 features for all clients with any event strictly before as_of_dt."""
    dt90 = as_of_dt - timedelta(days=90)
    dt30 = as_of_dt - timedelta(days=30)

    # Population: clients with any event strictly before as_of
    eligible = df[df["invoice_at"] < as_of_dt]
    if eligible.empty:
        return pd.DataFrame()
    client_ids = eligible["customer_id"].unique()

    # 90d window: [as_of - 90 days, as_of)
    w90 = df[(df["invoice_at"] >= dt90) & (df["invoice_at"] < as_of_dt)]
    # 30d window: [as_of - 30 days, as_of)
    w30 = df[(df["invoice_at"] >= dt30) & (df["invoice_at"] < as_of_dt)]

    # Purchases: quantity > 0 AND unit_price > 0
    purchases90 = w90[(w90["quantity"] > 0) & (w90["unit_price"] > 0)]
    # Cancellations: invoice_id starts with 'C'
    cancels90 = w90[w90["invoice_id"].str.startswith("C", na=False)]
    purchases30 = w30[(w30["quantity"] > 0) & (w30["unit_price"] > 0)]

    # All purchases before as_of (for days_since_last_order)
    all_purchases_before = df[
        (df["invoice_at"] < as_of_dt)
        & (df["quantity"] > 0)
        & (df["unit_price"] > 0)
    ]

    records = []
    for cid in client_ids:
        p90 = purchases90[purchases90["customer_id"] == cid]
        c90 = cancels90[cancels90["customer_id"] == cid]
        p30 = purchases30[purchases30["customer_id"] == cid]
        past_purchases = all_purchases_before[all_purchases_before["customer_id"] == cid]

        order_count_90d = len(p90)
        total_amount_90d = Decimal(0)
        for _, row in p90.iterrows():
            total_amount_90d += Decimal(str(row["quantity"])) * Decimal(str(row["unit_price"]))
        total_amount_90d = total_amount_90d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)

        if order_count_90d > 0:
            avg = total_amount_90d / Decimal(order_count_90d)
            average_order_amount_90d = avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        else:
            average_order_amount_90d = Decimal("0.00")

        if past_purchases.empty:
            days_since_last_order = -1
        else:
            last_purchase_dt = past_purchases["invoice_at"].max()
            days_since_last_order = (as_of_dt.date() - last_purchase_dt.date()).days

        distinct_stock_codes_90d = p90["stock_code"].nunique()

        denom90 = order_count_90d + len(c90)
        if denom90 == 0:
            return_rate_90d = Decimal("0.000000")
        else:
            return_rate_90d = (Decimal(len(c90)) / Decimal(denom90)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_EVEN
            )

        order_count_30d = len(p30)
        total_amount_30d = Decimal(0)
        for _, row in p30.iterrows():
            total_amount_30d += Decimal(str(row["quantity"])) * Decimal(str(row["unit_price"]))
        total_amount_30d = total_amount_30d.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)

        records.append({
            "client_id": str(cid),
            "as_of": as_of_dt.strftime(TIMESTAMP_FMT),
            "order_count_90d": int(order_count_90d),
            "total_amount_90d": str(total_amount_90d),
            "average_order_amount_90d": str(average_order_amount_90d),
            "days_since_last_order": int(days_since_last_order),
            "distinct_stock_codes_90d": int(distinct_stock_codes_90d),
            "return_rate_90d": str(return_rate_90d),
            "order_count_30d": int(order_count_30d),
            "total_amount_30d": str(total_amount_30d),
        })

    return pd.DataFrame(records)


def compute_all_features(df: pd.DataFrame, snapshots: list[dict]) -> pd.DataFrame:
    frames = []
    for snap in snapshots:
        as_of_dt = _parse_ts(snap["as_of"])
        frame = compute_features_for_snapshot(df, as_of_dt)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["client_id", "as_of"] + list(FEATURE_NAMES))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# MLflow model registration
# ---------------------------------------------------------------------------

def build_bundle(pipeline, family: str, model_name: str, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, bundle_dir / "pipeline.joblib")
    if LOCK_PATH.exists():
        import shutil
        shutil.copyfile(LOCK_PATH, bundle_dir / "requirements.lock")
        bundle_files = ("pipeline.joblib", "requirements.lock", "schema.json")
    else:
        bundle_files = ("pipeline.joblib", "schema.json")

    (bundle_dir / "schema.json").write_bytes(dumps(schema_document()))

    files = sorted(
        ({"path": name, "sha256": sha256_file(bundle_dir / name)} for name in bundle_files),
        key=lambda item: item["path"],
    )
    model_id = sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "family": family,
        "experiment_id": model_name,
        "files": files,
    }
    (bundle_dir / "manifest.json").write_bytes(dumps(manifest))


def train_and_register(features_df: pd.DataFrame, model_name: str, pipeline: Pipeline) -> str:
    """Train pipeline on features_df, register in MLflow, return version string."""
    # Build float features matrix for training
    float_cols = []
    for name in FEATURE_NAMES:
        col = features_df[name].apply(lambda x: float(x) if x != -1 else -1.0)
        float_cols.append(col)
    X = np.column_stack([c.values for c in float_cols])

    # Synthetic labels: clients with order_count_90d > median get label=1
    median_orders = np.median(X[:, 0])
    y = (X[:, 0] > median_orders).astype(int)
    if len(set(y)) < 2:
        y = np.array([i % 2 for i in range(len(y))])

    np.random.seed(SEED)
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        family = "logistic" if "logistic" in model_name.lower() else "boosting"
        build_bundle(pipeline, family, model_name, bundle_dir)

        with mlflow.start_run():
            mlflow.log_artifacts(str(bundle_dir), "bundle")
            run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/bundle"
    mv = mlflow.register_model(model_uri, model_name)
    return str(mv.version)


# ---------------------------------------------------------------------------
# Postgres setup
# ---------------------------------------------------------------------------

def setup_postgres() -> None:
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS hw4")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hw4.active_model (
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            activation_id TEXT NOT NULL,
            state_id TEXT NOT NULL,
            activated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT hw4_active_model_activation_id_unique UNIQUE (activation_id)
        )
    """)
    conn.close()


def set_initial_active_model(model_name: str, model_version: str) -> None:
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM hw4.active_model")
    (count,) = cur.fetchone()
    if count == 0:
        state_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO hw4.active_model (model_name, model_version, activation_id, state_id, activated_at)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (model_name, model_version, "initial", state_id),
        )
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("feature-build: starting")

    # Read request.json
    request_path = INPUT_DIR / "request.json"
    request = json.loads(request_path.read_text())
    snapshots = request["snapshots"]

    # Load history
    history_path = INPUT_DIR / "history.csv.gz"
    print(f"feature-build: loading history from {history_path}")
    df = load_history(history_path)
    print(f"feature-build: loaded {len(df)} rows")

    # Compute features
    print("feature-build: computing features")
    features_df = compute_all_features(df, snapshots)
    print(f"feature-build: computed {len(features_df)} feature rows")

    # Write parquet to seaweedfs
    s3 = s3_client()
    ensure_bucket(s3)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "features.parquet"
        table = pa.Table.from_pandas(features_df, preserve_index=False)
        pq.write_table(table, parquet_path)
        with open(parquet_path, "rb") as f:
            s3.put_object(
                Bucket=BUCKET,
                Key="offline/features/features.parquet",
                Body=f.read(),
            )
    print("feature-build: parquet written to seaweedfs")

    # Set up Postgres schema
    setup_postgres()
    print("feature-build: postgres schema ready")

    # Train + register models in MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Wait for mlflow to be reachable
    for attempt in range(30):
        try:
            mlflow.set_experiment("hw4-models")
            break
        except Exception as e:
            if attempt == 29:
                raise
            print(f"feature-build: waiting for mlflow ({attempt+1}/30)...")
            time.sleep(5)

    logistic_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=SEED, max_iter=1000)),
    ])
    boosting_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=SEED, n_estimators=50)),
    ])

    print("feature-build: training hw4-logistic")
    logistic_version = train_and_register(features_df, "hw4-logistic", logistic_pipeline)
    print(f"feature-build: registered hw4-logistic v{logistic_version}")

    print("feature-build: training hw4-boosting")
    boosting_version = train_and_register(features_df, "hw4-boosting", boosting_pipeline)
    print(f"feature-build: registered hw4-boosting v{boosting_version}")

    # Set initial active model
    set_initial_active_model("hw4-logistic", logistic_version)
    print(f"feature-build: active model set to hw4-logistic v{logistic_version}")

    print("feature-build: done")


if __name__ == "__main__":
    main()
