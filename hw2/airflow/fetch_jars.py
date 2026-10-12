"""Скачивает закреплённые JAR для s3a в каталог jars Spark и сверяет sha1."""

import hashlib
import sys
import urllib.request
from pathlib import Path

MAVEN = "https://repo1.maven.org/maven2"
JARS = [
    ("org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar",
     "a65839fbf1869f81a1632e09f415e586922e4f80"),
    ("com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar",
     "02deec3a0ad83d13d032b1812421b23d7a961eea"),
]


def main(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path, sha1 in JARS:
        data = urllib.request.urlopen(f"{MAVEN}/{path}", timeout=300).read()
        actual = hashlib.sha1(data).hexdigest()
        if actual != sha1:
            raise SystemExit(f"sha1 mismatch for {path}: {actual}")
        (target / path.rsplit("/", 1)[1]).write_bytes(data)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
