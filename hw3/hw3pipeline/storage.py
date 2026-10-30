"""S3 (SeaweedFS) access for run artifacts under s3://hw3/runs/<execution_id>/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .common import dumps, strict_loads

BUCKET = os.environ.get("HW3_BUCKET", "hw3")


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("HW3_S3_ENDPOINT", "http://seaweedfs:8333"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-hw3"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-hw3-only"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


class RunStore:
    """Objects of one execution; every key is relative to runs/<execution_id>/."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.prefix = f"runs/{execution_id}/"
        self.s3 = client()

    def key(self, relative: str) -> str:
        return self.prefix + relative

    def ensure_bucket(self) -> None:
        ensure_bucket(self.s3)

    def put_bytes(self, relative: str, data: bytes) -> None:
        self.s3.put_object(Bucket=BUCKET, Key=self.key(relative), Body=data)

    def put_json(self, relative: str, value: Any) -> bytes:
        data = dumps(value)
        self.put_bytes(relative, data)
        return data

    def put_file(self, relative: str, path: Path) -> None:
        self.put_bytes(relative, path.read_bytes())

    def get_bytes(self, relative: str) -> bytes:
        return self.s3.get_object(Bucket=BUCKET, Key=self.key(relative))["Body"].read()

    def get_json(self, relative: str) -> Any:
        return strict_loads(self.get_bytes(relative))

    def exists(self, relative: str) -> bool:
        try:
            self.s3.head_object(Bucket=BUCKET, Key=self.key(relative))
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def list(self, relative_prefix: str = "") -> list[str]:
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=self.key(relative_prefix)):
            keys += [item["Key"][len(self.prefix) :] for item in page.get("Contents", [])]
        return sorted(keys)

    def delete(self, relative: str) -> None:
        self.s3.delete_object(Bucket=BUCKET, Key=self.key(relative))

    def delete_prefix(self, relative_prefix: str) -> None:
        """Remove a stage's previous logical result so a retry replaces it."""
        for relative in self.list(relative_prefix):
            self.delete(relative)

    def download(self, relative: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.get_bytes(relative))
        return target

    def download_prefix(self, relative_prefix: str, target: Path) -> list[Path]:
        paths = []
        for relative in self.list(relative_prefix):
            paths.append(self.download(relative, target / relative[len(relative_prefix) :]))
        return paths


def ensure_bucket(s3=None) -> None:
    s3 = s3 or client()
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        try:
            s3.create_bucket(Bucket=BUCKET)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise
