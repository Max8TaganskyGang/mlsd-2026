FROM eclipse-temurin:17-jre-jammy@sha256:8993f1aed8b25fcea7a7047a7949c1866fa558fc6830d938c22c4f13b26be9d7 AS java

FROM apache/airflow:2.10.5-python3.12@sha256:6499a680a93463846d3a6be980e85d601dc97b0d81e82eed9ef5e5cb9da31b79

USER root
COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH="/opt/java/openjdk/bin:${PATH}" \
    PYSPARK_PYTHON=python3 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
USER airflow

COPY constraints-3.12.txt /tmp/constraints-3.12.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --retries 10 \
    --constraint /tmp/constraints-3.12.txt \
    --requirement /tmp/requirements.txt

COPY fetch_jars.py /tmp/fetch_jars.py
RUN python /tmp/fetch_jars.py "$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')/jars"
