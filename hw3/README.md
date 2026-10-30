# HW3. Временная модель повторной покупки

Условие — [HW3.md](HW3.md), технический контракт — [TECHNICAL.md](TECHNICAL.md),
сценарии самопроверки — [data/SCENARIOS.md](data/SCENARIOS.md).

Для клиента Online Retail II на дату `t` проект оценивает вероятность хотя бы
одной положительной покупки в `[t, t+30 дней)`. Spark строит временные примеры,
Airflow запускает стадии, SeaweedFS (S3) хранит результаты, MLflow связывает
кандидатов, выбор и комплект модели.

## Состав среды

Все сервисы описаны в [compose.yaml](compose.yaml). Наружу порты не публикуются,
сеть одна — `default` проекта.

| Сервис | Образ | Назначение |
| --- | --- | --- |
| `postgres` | `postgres:16.6-bookworm@sha256:…` | метаданные Airflow |
| `airflow-init` | проектный образ | `airflow db migrate`, пользователь `hw3`, bucket `hw3`; завершается |
| `airflow-webserver` | проектный образ | REST API `http://airflow-webserver:8080/api/v1` |
| `airflow-scheduler` | проектный образ | `LocalExecutor`, исполняет задачи и Spark-драйвер |
| `spark-master`, `spark-worker` | проектный образ | Spark 3.5.4 standalone, worker 2 ядра / 2 ГБ |
| `seaweedfs` | `chrislusf/seaweedfs:4.48@sha256:…` | S3 `http://seaweedfs:8333`, bucket `hw3` |
| `mlflow` | проектный образ | `http://mlflow:5000`, SQLite и артефакты в volume `mlflow-data` |

Проектный образ один ([Dockerfile](Dockerfile)): `apache/airflow:2.10.5-python3.12`
по digest, Java 17 из `eclipse-temurin` по digest, зависимости ставятся
`pip install --no-deps` строго из [requirements.lock](requirements.lock). Lock —
полный `pip freeze` окружения, полученный разрешением `pyspark 3.5.4`,
`mlflow 2.22.0`, `scikit-learn 1.5.2`, `scipy 1.14.1`, `joblib 1.4.2` против
официальных constraints Airflow 2.10.5 для Python 3.12. В нём же закреплены
`numpy 1.26.4`, `pandas 2.1.4`, `pyarrow 19.0.0`, `boto3 1.36.3`. Пути не зависят от
архитектуры (`SPARK_HOME` — каталог pip-пакета `pyspark`), сборка работает
классическим builder'ом (`DOCKER_BUILDKIT=0`).

Постоянные данные — в именованных volumes `postgres-data`, `seaweedfs-data`,
`mlflow-data`; пересоздание контейнеров `seaweedfs` и `mlflow` их не теряет.

Локальные учебные учётные данные: Airflow `hw3` / `local-hw3-only`,
S3 `local-hw3` / `local-hw3-only`.

## Запуск

```sh
# полный набор (≈15 МБ), проверяется SHA-256 из контракта
python3 hw3/data/download.py --source \
  https://github.com/fall-out-bug/mlsd-2026-starter/releases/download/hw3-data-v1/online-retail-ii-v1.csv.gz

cd hw3
docker compose up --build --wait
```

`hw3/data/` монтируется в `/input` только для чтения (другой каталог —
через `HW3_INPUT_DIR`). Запрос для полного набора — копия
`data/request.full.template.json` с уникальным `execution_id` и SHA сдачи,
положенная в `hw3/data/`. Запуск DAG через API изнутри сети проекта:

```sh
docker compose exec airflow-webserver curl -s -u hw3:local-hw3-only \
  -H 'Content-Type: application/json' \
  -X POST http://localhost:8080/api/v1/dags/online_retail_training/dagRuns \
  -d '{"dag_run_id": "full_01", "conf": {"request_path": "/input/request.full.json"}}'
```

`dag_run_id` обязан совпадать с `execution_id` запроса. Остановка —
`docker compose down` (с `-v` — вместе с данными).

Проверка курса из корня репозитория: `sh check.sh HW3`. Контроллер читает
`/proc` и рассчитан на Linux; на macOS его удобно запускать в контейнере
с Docker CLI и сокетом Docker Desktop:

```sh
R="$PWD"; docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$R:$R" -w "$R" \
  docker:27-cli sh -c 'apk add -q python3 git && git config --global --add safe.directory "*" && sh check.sh HW3'
```

## DAG `online_retail_training`

`schedule=None`, без catchup, `max_active_runs=1`, `retries=0`. Цепочка:

```
stage_input → build_examples → train_select → evaluate → publish
```

Каждая задача запускает свою стадию пакета [hw3pipeline](hw3pipeline/) отдельным
процессом `python -m hw3pipeline <stage>` и передаёт ей `HW3_EXECUTION_ID`,
`HW3_SUBMISSION_SHA`, `HW3_DATASET_VERSION`, `HW3_INPUT_SHA256`,
`MLFLOW_TRACKING_URI`; один поток численных библиотек (`OMP/OPENBLAS/MKL=1`).
Ненулевой код возврата стадии роняет задачу и весь DAG.

| Стадия | Модуль | Что делает |
| --- | --- | --- |
| `stage_input` | [stage_input.py](hw3pipeline/stage_input.py) | проверяет запрос и `execution_id == dag_run_id`, SHA-256 истории; копирует `input/input.csv.gz` и точные байты `input/request.json` в S3 |
| `build_examples` | [build_examples.py](hw3pipeline/build_examples.py) | Spark-приложение `online_retail_training` на `spark://spark-master:7077`: окна, отбор, decimal-агрегаты, метки; пишет `examples/{split}/part-00000.parquet` и `examples/manifest.json` |
| `train_select` | [train_select.py](hw3pipeline/train_select.py) | обучает кандидатов только на train, сравнивает на validation, замораживает комплект |
| `evaluate` | [evaluate.py](hw3pipeline/evaluate.py) | применяет комплект в отдельном процессе ([apply_bundle.py](hw3pipeline/apply_bundle.py)), считает четыре test-метрики |
| `publish` | [publish.py](hw3pipeline/publish.py) | хеширует все файлы исполнения, пишет `result.json`, затем одним объектом `current.json` |

## Временные примеры

Положительная покупка — непустой `customer_id`, `quantity > 0` (целое),
`unit_price > 0`, `invoice_id` не начинается с `C`/`c`; время сравнивается как
строка `YYYY-MM-DDTHH:MM:SS` без часового пояса. Для снимка `t`:

- история признаков — `[t−90 дней, t)`, в выборку входит каждый клиент
  хотя бы с одной такой покупкой (будущее на состав не влияет);
- `label = 1`, если есть покупка в `[t, t+30 дней)`, иначе `0`; отрицательные
  примеры сохраняются все, без балансировки;
- `order_count` — `countDistinct(invoice_id)`; `total_amount` — сумма
  `quantity × unit_price` в `decimal` (цена — `decimal(20,6)`), без float;
  `average_order_amount = bround(total / order_count, 2)` — half even;
- если окно истории или метки выходит за `[observed_from, observed_until)`,
  стадия отказывает с `INCOMPLETE_OBSERVATION` до запуска Spark.

Spark выполняет всю работу над товарными строками. Драйвер получает только
готовые агрегаты (тысячи строк) и записывает Parquet через pyarrow с точной
схемой контракта: пустая часть тоже даёт читаемый файл, `_SUCCESS` и колонок
разделов нет.

## Кандидаты и выбор

[config/candidates.json](config/candidates.json):

| `experiment_id` | Семейство | Обработка | Настройки |
| --- | --- | --- | --- |
| `constant` | constant | — | `DummyClassifier(strategy="prior")`, даёт `p_train` |
| `logistic_log1p_standard` | logistic | `log1p` → `StandardScaler` | `C=1.0`, `max_iter=1000` |
| `logistic_standard` | logistic | `StandardScaler` | `C=1.0`, `max_iter=1000` |
| `tree_depth3_leaf100` | tree | — | `max_depth=3`, `min_samples_leaf=100` |
| `tree_depth4_leaf50` | tree | — | `max_depth=4`, `min_samples_leaf=50` |

Почему так. Признаки денежные и с тяжёлым правым хвостом, поэтому для
логистической регрессии сравниваются два варианта: обычная стандартизация и
стандартизация после `log1p`, которая сжимает хвост. Масштабирование оценивается
только на train внутри Pipeline. Регуляризация L2 по умолчанию, `C=1`: признаков
три, переобучение не главный риск. Деревья взяты неглубокие и с крупными
листьями, чтобы вероятности в листьях оценивались по десяткам-сотням примеров
и были калиброваннее. Это важно, потому что критерий — log loss, а не
точность. Всё выбрано до оценки на test.

Сборка Pipeline из конфигурации — функция `build_pipeline` в
[models.py](hw3pipeline/models.py). Она проверяет ограничения контракта
(разрешённые классы обработки, `numpy.log1p`, сетку дерева,
`class_weight=None`, `random_state=2026`) и ничего не читает и не обучает.
`resolved_params` — полный `get_params(deep=False)` каждого шага.

Правило выбора `validation_log_loss_v1`: минимальный log loss на validation
(clipping `1e-12`). Значения в пределах `1e-12` от минимума равны, порядок
предпочтения: константа, логистическая регрессия, дерево, затем меньший
`experiment_id`. Переобучения на train+validation нет.

## Комплект модели

`selection/bundle/`: `pipeline.joblib` (обученный Pipeline с обработкой),
`schema.json` (`hw3-features-v1`: `order_count`, `total_amount`,
`average_order_amount` → `float64`, `positive_class=1`), `requirements.lock`
(тот же lock, что у образа) и `manifest.json` с `model_id` — SHA-256
канонического JSON трёх файлов. `evaluate` загружает комплект в отдельном
процессе, сверяет манифест (иначе `INVALID_BUNDLE`), переводит decimal в
float64 один раз в порядке `schema.json` и берёт `predict_proba` класса 1.

Восстановить выбранный комплект: прочитать `s3://hw3/current.json`, перейти по
`result_uri`, скачать `runs/<execution_id>/selection/bundle/` (тот же комплект
лежит в артефактах `bundle/` выбранного child run MLflow) и выполнить
`joblib.load("pipeline.joblib").predict_proba(X)` с тем же lock.

## Результаты и MLflow

```text
s3://hw3/
  current.json                      {execution_id, result_uri, sha256(result.json)}
  runs/<execution_id>/
    input/input.csv.gz, input/request.json
    examples/{train,validation,test}/part-00000.parquet, examples/manifest.json
    selection/experiments.json, validation_predictions.parquet,
              validation_metrics.json, selected.json, bundle/…
    evaluation/test_predictions.parquet, test_metrics.json
    result.json                     status=ready, files=[{path, sha256}]
    error.json                      только при отказе: {stage, error_code, message}
```

MLflow experiment `hw3`: один parent run на `execution_id` (теги
`run_kind=parent` и идентичность запроса, параметры календаря и окон, артефакты
`request.json`, `experiments.json`, `selected.json`, обе таблицы метрик,
test-прогнозы, метрики `test.<role>.<group>.<metric>`). На каждого кандидата —
child run с `config.json`, своими validation-прогнозами и метриками, параметром
`candidate_json` и seed 2026; у выбранного ещё и `bundle/`.

Повтор того же исполнения находит parent и children по тегам и возобновляет их,
поэтому дубликатов не появляется. Каждая стадия перед записью удаляет свой
префикс в S3, так что старые part-файлы исчезают. Новый `execution_id` создаёт
новую историю, а прежние runs остаются. При отказе `result.json` не пишется и
`current.json` не меняется. Коды отказов: `INPUT_HASH_MISMATCH`,
`INVALID_REQUEST`, `INCOMPLETE_OBSERVATION`, `EMPTY_TRAIN`, `TRAIN_SINGLE_CLASS`,
`EMPTY_VALIDATION`, `INVALID_BUNDLE`.

Таблица метрик конкретного запуска — `evaluation/test_metrics.json` и
`selection/validation_metrics.json`; числа в README источником истины не
являются.
