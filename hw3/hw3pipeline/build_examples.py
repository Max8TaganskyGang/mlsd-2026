"""build_examples: Spark builds temporal examples (population, features, labels)."""

from __future__ import annotations

import os
import tempfile
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .common import (
    CONTRACT_VERSION,
    StageError,
    parse_timestamp,
    sha256_file,
    windows,
)
from .stage_input import resolve_input
from .storage import RunStore

APP_NAME = "online_retail_training"

EXAMPLE_SCHEMA = pa.schema(
    [
        ("customer_id", pa.string()),
        ("as_of", pa.string()),
        ("order_count", pa.int64()),
        ("total_amount", pa.decimal128(20, 6)),
        ("average_order_amount", pa.decimal128(20, 2)),
        ("label", pa.int64()),
        ("split", pa.string()),
    ]
)

COLUMN_ALIASES = {
    "invoice_id": ("invoice_id", "Invoice", "invoice"),
    "customer_id": ("customer_id", "Customer ID", "customer"),
    "invoice_at": ("invoice_at", "InvoiceDate", "invoice_date"),
    "quantity": ("quantity", "Quantity"),
    "unit_price": ("unit_price", "Price", "price"),
}


def check_observation(request: dict, plan: list[dict]) -> None:
    observed_from = request["observed_from"]
    observed_until = request["observed_until"]
    for item in plan:
        if item["history_from"] < observed_from or item["label_until"] > observed_until:
            raise StageError(
                "INCOMPLETE_OBSERVATION",
                f"{item['split']} windows [{item['history_from']}, {item['label_until']}) are "
                f"outside observation [{observed_from}, {observed_until})",
            )


def spark_session():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(APP_NAME)
        .master(os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"))
        .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", "airflow-scheduler"))
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "1g"))
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "1g"))
        .config("spark.executor.cores", os.environ.get("SPARK_EXECUTOR_CORES", "2"))
        .config("spark.cores.max", os.environ.get("SPARK_CORES_MAX", "2"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def positive_purchases(spark, path: Path):
    """Positive purchase lines: customer, invoice, timestamp string and exact amount."""
    from pyspark.sql import functions as F

    raw = (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("mode", "PERMISSIVE")
        .option("encoding", "UTF-8")
        .csv(str(path))
    )
    columns = {}
    for name, options in COLUMN_ALIASES.items():
        found = next((option for option in options if option in raw.columns), None)
        if found is None:
            raise StageError("INVALID_REQUEST", f"history has no column {name}")
        columns[name] = found

    def clean(name: str):
        return F.regexp_replace(F.coalesce(F.col(columns[name]), F.lit("")), r"^\s+|\s+$", "")

    quantity = clean("quantity").cast("decimal(38,6)")
    price = clean("unit_price").cast("decimal(20,6)")
    lines = raw.select(
        clean("customer_id").alias("customer_id"),
        clean("invoice_id").alias("invoice_id"),
        clean("invoice_at").alias("invoice_at"),
        quantity.alias("quantity"),
        price.alias("unit_price"),
    )
    return lines.where(
        (F.col("customer_id") != "")
        & ~F.upper(F.col("invoice_id")).startswith("C")
        & F.col("invoice_at").rlike(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        & F.to_timestamp(F.col("invoice_at"), "yyyy-MM-dd'T'HH:mm:ss").isNotNull()
        & (F.col("quantity") == F.floor(F.col("quantity")))
        & (F.col("quantity") > 0)
        & (F.col("unit_price") > 0)
    ).select(
        "customer_id",
        "invoice_id",
        "invoice_at",
        (F.col("quantity").cast("decimal(20,0)") * F.col("unit_price")).alias("amount"),
    )


def snapshot_examples(purchases, item: dict):
    """Customers with history in [t-90d, t), their features and the 30-day label."""
    from pyspark.sql import functions as F

    at = F.col("invoice_at")
    history = (
        purchases.where((at >= item["history_from"]) & (at < item["as_of"]))
        .groupBy("customer_id")
        .agg(
            F.countDistinct("invoice_id").alias("order_count"),
            F.sum("amount").cast("decimal(20,6)").alias("total_amount"),
        )
    )
    future = (
        purchases.where((at >= item["as_of"]) & (at < item["label_until"]))
        .select("customer_id")
        .distinct()
        .withColumn("future", F.lit(1))
    )
    return history.join(future, "customer_id", "left").select(
        F.col("customer_id"),
        F.lit(item["as_of"]).alias("as_of"),
        F.col("order_count").cast("long").alias("order_count"),
        F.col("total_amount"),
        F.bround(F.col("total_amount") / F.col("order_count").cast("decimal(20,0)"), 2)
        .cast("decimal(20,2)")
        .alias("average_order_amount"),
        F.coalesce(F.col("future"), F.lit(0)).cast("long").alias("label"),
        F.lit(item["split"]).alias("split"),
    )


def to_table(rows: list[dict]) -> pa.Table:
    rows = sorted(rows, key=lambda row: (row["as_of"], row["customer_id"]))
    for row in rows:
        if not isinstance(row["total_amount"], Decimal) or row["total_amount"] <= 0:
            raise StageError("INVALID_REQUEST", "total_amount must be a positive decimal")
    return pa.Table.from_pylist(rows, schema=EXAMPLE_SCHEMA)


def run(execution_id: str) -> None:
    store = RunStore(execution_id)
    request = store.get_json("input/request.json")
    plan = windows(request)
    check_observation(request, plan)
    for item in plan:
        parse_timestamp(item["as_of"], "as_of")
    history = resolve_input(request["input_path"])
    if sha256_file(history) != request["input_sha256"]:
        raise StageError("INPUT_HASH_MISMATCH", "history changed after stage_input")

    spark = spark_session()
    try:
        application_id = spark.sparkContext.applicationId
        purchases = positive_purchases(spark, history).persist()
        frames = [snapshot_examples(purchases, item) for item in plan]
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.unionByName(frame)
        collected = [row.asDict() for row in combined.collect()]
        purchases.unpersist()
    finally:
        spark.stop()

    store.delete_prefix("examples/")
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "execution_id": execution_id,
        "input_sha256": request["input_sha256"],
        "spark_application_id": application_id,
        "history_days": request["history_days"],
        "horizon_days": request["horizon_days"],
        "splits": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        for item in plan:
            rows = [row for row in collected if row["split"] == item["split"]]
            table = to_table(rows)
            local = Path(directory) / f"{item['split']}.parquet"
            pq.write_table(table, local)
            relative = f"examples/{item['split']}/part-00000.parquet"
            store.put_file(relative, local)
            manifest["splits"][item["split"]] = {
                "as_of": item["as_of"],
                "history": [item["history_from"], item["as_of"]],
                "label": [item["as_of"], item["label_until"]],
                "path": relative,
                "sha256": sha256_file(local),
                "n": table.num_rows,
                "n_positive": sum(row["label"] for row in rows),
            }
            print(
                f"{item['split']}: as_of={item['as_of']} n={table.num_rows} "
                f"positive={manifest['splits'][item['split']]['n_positive']}"
            )
    store.put_json("examples/manifest.json", manifest)
