FROM eclipse-temurin:17-jre-jammy@sha256:8993f1aed8b25fcea7a7047a7949c1866fa558fc6830d938c22c4f13b26be9d7 AS java

FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

ENV JAVA_HOME=/opt/java/openjdk \
    SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark \
    SPARK_SCALA_VERSION=2.12 \
    PYSPARK_PYTHON=python3 \
    PYTHONUNBUFFERED=1
ENV PATH="${JAVA_HOME}/bin:${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

COPY --from=java /opt/java/openjdk /opt/java/openjdk

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --retries 10 --requirement /tmp/requirements.txt

COPY fetch_jars.py /tmp/fetch_jars.py
RUN python /tmp/fetch_jars.py "${SPARK_HOME}/jars"

COPY jobs /opt/spark-jobs
COPY entrypoint.sh /opt/spark-entrypoint.sh
RUN chmod +x /opt/spark-entrypoint.sh

WORKDIR /opt/spark-jobs
ENTRYPOINT ["/opt/spark-entrypoint.sh"]
