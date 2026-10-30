#!/bin/sh
# Long-running non-Airflow services of the shared hw3 image.
set -eu

CLASSPATH="${SPARK_CONF_DIR}:${SPARK_HOME}/jars/*"

case "${1:-}" in
  spark-master)
    exec java -Xmx512m -cp "${CLASSPATH}" org.apache.spark.deploy.master.Master \
      --host spark-master --port 7077 --webui-port 8080
    ;;
  spark-worker)
    exec java -Xmx256m -cp "${CLASSPATH}" org.apache.spark.deploy.worker.Worker \
      --host spark-worker \
      --cores "${SPARK_WORKER_CORES:-2}" \
      --memory "${SPARK_WORKER_MEMORY:-2g}" \
      --work-dir /tmp/spark-work \
      --webui-port 8081 \
      spark://spark-master:7077
    ;;
  mlflow)
    exec mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --workers 1 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts \
      --serve-artifacts
    ;;
  *)
    echo "usage: service.sh spark-master|spark-worker|mlflow" >&2
    exit 2
    ;;
esac
