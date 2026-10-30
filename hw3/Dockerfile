FROM eclipse-temurin:17-jre-jammy@sha256:8993f1aed8b25fcea7a7047a7949c1866fa558fc6830d938c22c4f13b26be9d7 AS java

FROM apache/airflow:2.10.5-python3.12@sha256:6499a680a93463846d3a6be980e85d601dc97b0d81e82eed9ef5e5cb9da31b79

USER root
COPY --from=java /opt/java/openjdk /opt/java/openjdk
RUN mkdir -p /opt/hw3 /mlflow /tmp/spark-work \
    && chown -R airflow:0 /opt/hw3 /mlflow /tmp/spark-work

ENV JAVA_HOME=/opt/java/openjdk \
    PATH="/opt/java/openjdk/bin:${PATH}" \
    SPARK_HOME=/home/airflow/.local/lib/python3.12/site-packages/pyspark \
    SPARK_CONF_DIR=/opt/hw3/spark/conf \
    SPARK_SCALA_VERSION=2.12 \
    PYSPARK_PYTHON=python3 \
    PYTHONPATH=/opt/hw3 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

USER airflow
COPY --chown=airflow:0 requirements.lock /opt/hw3/requirements.lock
RUN pip install --quiet --no-cache-dir --no-deps --retries 10 --requirement /opt/hw3/requirements.lock \
    && pip check > /dev/null \
    && test -x "${SPARK_HOME}/bin/spark-class" \
    && java -version > /dev/null 2>&1

COPY --chown=airflow:0 spark /opt/hw3/spark
COPY --chown=airflow:0 bin /opt/hw3/bin
COPY --chown=airflow:0 config /opt/hw3/config
COPY --chown=airflow:0 hw3pipeline /opt/hw3/hw3pipeline
COPY --chown=airflow:0 airflow/dags /opt/airflow/dags
RUN chmod 0755 /opt/hw3/bin/service.sh \
    && python -c "import hw3pipeline.train_select, hw3pipeline.build_examples, hw3pipeline.publish"
