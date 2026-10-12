"""DAG расчёта клиентских признаков Online Retail II."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from airflow.decorators import dag, task

DATASET_PATH = Path(os.environ.get("HW2_DATASET_PATH", "/opt/airflow/hw2/data/sample.csv"))
BUCKET = os.environ.get("HW2_BUCKET", "hw2")
INPUT_KEY = "input/sample.csv"
FEATURES_PREFIX = "features/"
REDIS_PREFIX = "hw2:customer:"
SPARK_JOB = os.environ.get("HW2_SPARK_JOB", "/opt/airflow/spark_jobs/customer_features.py")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "hw2-local")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "hw2-local-only")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))


def s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )


@dag(
    dag_id="online_retail_features",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=["hw2"],
)
def online_retail_features():
    @task
    def get_dataset() -> dict:
        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"input CSV not found: {DATASET_PATH}")
        digest = hashlib.sha256()
        rows = 0
        with DATASET_PATH.open("rb") as stream:
            header = stream.readline()
            digest.update(header)
            if not header.strip():
                raise ValueError("input CSV is empty")
            for line in stream:
                digest.update(line)
                rows += 1
        return {
            "path": str(DATASET_PATH),
            "size": DATASET_PATH.stat().st_size,
            "sha256": digest.hexdigest(),
            "lines": rows,
        }

    @task
    def put_dataset(dataset: dict) -> str:
        from botocore.exceptions import ClientError

        client = s3_client()
        try:
            client.head_bucket(Bucket=BUCKET)
        except ClientError:
            client.create_bucket(Bucket=BUCKET)
        client.upload_file(dataset["path"], BUCKET, INPUT_KEY)
        stored = client.head_object(Bucket=BUCKET, Key=INPUT_KEY)
        if stored["ContentLength"] != dataset["size"]:
            raise RuntimeError("uploaded CSV size differs from the source")
        return f"s3a://{BUCKET}/{INPUT_KEY}"

    @task
    def features_engineering(input_uri: str) -> str:
        client = s3_client()
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=FEATURES_PREFIX
        )
        for page in pages:
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            for start in range(0, len(objects), 1000):
                client.delete_objects(
                    Bucket=BUCKET, Delete={"Objects": objects[start : start + 1000]}
                )

        output_uri = f"s3a://{BUCKET}/{FEATURES_PREFIX}"
        subprocess.run(
            ["python", SPARK_JOB, "--input", input_uri, "--output", output_uri],
            check=True,
        )
        return output_uri

    @task
    def load_features(output_uri: str) -> int:
        import pyarrow.dataset as ds
        import pyarrow.fs as pafs
        import redis

        endpoint = S3_ENDPOINT.removeprefix("https://").removeprefix("http://")
        scheme = "https" if S3_ENDPOINT.startswith("https://") else "http"
        filesystem = pafs.S3FileSystem(
            access_key=S3_ACCESS_KEY,
            secret_key=S3_SECRET_KEY,
            region=S3_REGION,
            endpoint_override=endpoint,
            scheme=scheme,
        )
        path = output_uri.removeprefix("s3a://").rstrip("/")
        dataset = ds.dataset(path, filesystem=filesystem, format="parquet")

        store = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        fresh: set[str] = set()
        pipe = store.pipeline(transaction=False)
        for batch in dataset.to_batches(
            columns=["customer_id", "order_count", "total_amount", "average_order_amount"]
        ):
            for row in batch.to_pylist():
                customer = str(row["customer_id"])
                key = REDIS_PREFIX + customer
                value = {
                    "order_count": int(row["order_count"]),
                    "total_amount": f"{Decimal(str(row['total_amount'])):.6f}",
                    "average_order_amount": f"{Decimal(str(row['average_order_amount'])):.2f}",
                }
                pipe.set(key, json.dumps(value, separators=(",", ":")))
                fresh.add(key)
                if len(pipe) >= 1000:
                    pipe.execute()
        pipe.execute()

        stale = [key for key in store.scan_iter(REDIS_PREFIX + "*", count=1000) if key not in fresh]
        for start in range(0, len(stale), 1000):
            store.delete(*stale[start : start + 1000])
        if not fresh:
            raise RuntimeError("no features were loaded")
        return len(fresh)

    load_features(features_engineering(put_dataset(get_dataset())))


online_retail_features()
