#!/bin/sh
set -eu

ROLE="${SPARK_ROLE:-}"
if [ -z "$ROLE" ] && [ "$#" -gt 0 ]; then
  exec "$@"
fi

CLASSPATH="${SPARK_HOME}/conf:${SPARK_HOME}/jars/*"

case "$ROLE" in
  master)
    exec java \
      -Xmx"${SPARK_DAEMON_MEMORY:-512m}" \
      -cp "${CLASSPATH}" \
      org.apache.spark.deploy.master.Master \
      --host "${SPARK_MASTER_HOST:-spark-master}" \
      --port "${SPARK_MASTER_PORT:-7077}" \
      --webui-port "${SPARK_MASTER_WEBUI_PORT:-8080}"
    ;;
  worker)
    exec java \
      -Xmx"${SPARK_DAEMON_MEMORY:-256m}" \
      -cp "${CLASSPATH}" \
      org.apache.spark.deploy.worker.Worker \
      --cores "${SPARK_WORKER_CORES:-1}" \
      --memory "${SPARK_WORKER_MEMORY:-1536m}" \
      --webui-port "${SPARK_WORKER_WEBUI_PORT:-8081}" \
      "spark://${SPARK_MASTER_HOST:-spark-master}:${SPARK_MASTER_PORT:-7077}"
    ;;
  *)
    echo "Set SPARK_ROLE=master|worker or pass a command" >&2
    exit 1
    ;;
esac
