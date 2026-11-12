#!/bin/sh
# Long-running and one-shot services of the hw4 image.
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
  gateway)
    exec python -m service.gateway.app
    ;;
  feature-service)
    exec python -m service.feature_service.app
    ;;
  model-service)
    exec python -m service.model_service.app
    ;;
  feature-build)
    exec python -m offline.feature_build.main
    ;;
  feature-loader)
    exec python -m offline.feature_loader.main
    ;;
  *)
    echo "usage: service.sh spark-master|spark-worker|mlflow|gateway|feature-service|model-service|feature-build|feature-loader" >&2
    exit 2
    ;;
esac
