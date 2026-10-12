"""Расчёт клиентских признаков Online Retail II в Spark.

Читает CSV из S3, считает признаки по клиентам и перезаписывает Parquet в S3.
"""

import argparse
import os
import socket

from pyspark.sql import SparkSession, functions as F, types as T

APP_NAME = "online_retail_features"
MONEY = T.DecimalType(20, 6)
AVERAGE = T.DecimalType(20, 2)

INPUT_SCHEMA = T.StructType(
    [
        T.StructField(name, T.StringType(), True)
        for name in (
            "invoice_id",
            "stock_code",
            "description",
            "quantity",
            "invoice_at",
            "unit_price",
            "customer_id",
            "country",
        )
    ]
)


def build_session(master: str) -> SparkSession:
    endpoint = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
    driver_host = os.environ.get("SPARK_DRIVER_HOST") or socket.gethostbyname(socket.gethostname())
    return (
        SparkSession.builder.appName(APP_NAME)
        .master(master)
        .config("spark.driver.host", driver_host)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "1g"))
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "1g"))
        .config("spark.executor.cores", os.environ.get("SPARK_EXECUTOR_CORES", "2"))
        .config("spark.cores.max", os.environ.get("SPARK_CORES_MAX", "2"))
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.endpoint.region", os.environ.get("S3_REGION", "us-east-1"))
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("S3_ACCESS_KEY", "hw2-local"))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("S3_SECRET_KEY", "hw2-local-only"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(endpoint.startswith("https")).lower())
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )


def customer_features(raw):
    invoice = F.col("invoice_id")
    customer = F.col("customer_id")
    quantity = F.col("quantity").cast(T.LongType())
    price = F.col("unit_price").cast(MONEY)

    purchases = (
        raw.select(invoice, customer, quantity.alias("quantity"), price.alias("unit_price"))
        .where(customer.isNotNull() & (customer != ""))
        .where(invoice.isNotNull() & ~F.lower(invoice).startswith("c"))
        .where((F.col("quantity") > 0) & (F.col("unit_price") > 0))
        .withColumn(
            "amount",
            F.col("quantity").cast(T.DecimalType(20, 0)) * F.col("unit_price"),
        )
    )

    totals = purchases.groupBy("customer_id").agg(
        F.countDistinct("invoice_id").cast(T.LongType()).alias("order_count"),
        F.sum("amount").cast(MONEY).alias("total_amount"),
    )

    return totals.select(
        F.col("customer_id").cast(T.StringType()).alias("customer_id"),
        F.col("order_count"),
        F.col("total_amount"),
        F.bround(
            F.col("total_amount") / F.col("order_count").cast(T.DecimalType(20, 0)), 2
        ).cast(AVERAGE).alias("average_order_amount"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3a://hw2/input/sample.csv")
    parser.add_argument("--output", default="s3a://hw2/features/")
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"))
    args = parser.parse_args()

    spark = build_session(args.master)
    try:
        raw = (
            spark.read.option("header", "true")
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "FAILFAST")
            .schema(INPUT_SCHEMA)
            .csv(args.input)
        )
        features = customer_features(raw)
        features.write.mode("overwrite").parquet(args.output)
        written = spark.read.parquet(args.output)
        print(f"customers written: {written.count()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
