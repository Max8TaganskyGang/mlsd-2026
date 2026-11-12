"""feature-loader: read features.parquet from SeaweedFS and load into Redis."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq
import redis as redis_lib
from botocore.config import Config

BUCKET = os.environ.get("HW4_BUCKET", "hw4")
S3_ENDPOINT = os.environ.get("HW4_S3_ENDPOINT", "http://seaweedfs:8333")
AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "local-hw4")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "local-hw4-only")
REDIS_HOST = os.environ.get("HW4_REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("HW4_REDIS_PORT", "6379"))


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


def main() -> None:
    print("feature-loader: starting")

    # Download parquet from seaweedfs
    s3 = s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key="offline/features/features.parquet")
    parquet_bytes = obj["Body"].read()
    df = pq.read_table(io.BytesIO(parquet_bytes)).to_pandas()
    print(f"feature-loader: loaded {len(df)} rows from parquet")

    # Connect to Redis
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    feature_cols = [
        "order_count_90d",
        "total_amount_90d",
        "average_order_amount_90d",
        "days_since_last_order",
        "distinct_stock_codes_90d",
        "return_rate_90d",
        "order_count_30d",
        "total_amount_30d",
    ]

    pipe = r.pipeline()
    for _, row in df.iterrows():
        client_id = str(row["client_id"])
        as_of = str(row["as_of"])  # "YYYY-MM-DDTHH:MM:SS"

        value = {
            "client_id": client_id,
            "as_of": as_of,
        }
        for col in feature_cols:
            val = row[col]
            # Store int features as int, decimal features as string
            if col in ("order_count_90d", "days_since_last_order", "distinct_stock_codes_90d", "order_count_30d"):
                value[col] = int(val)
            else:
                value[col] = str(val)

        redis_key = f"hw4:features:{client_id}:{as_of}"
        pipe.set(redis_key, json.dumps(value))
        # Also track known clients
        pipe.sadd("hw4:clients", client_id)

    pipe.execute()
    print(f"feature-loader: loaded {len(df)} feature records into Redis")
    print("feature-loader: done")


if __name__ == "__main__":
    main()
