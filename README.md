# MLSD 2026

Один репозиторий на весь курс. Сейчас сданы и поддерживаются HW1 и HW2.

| Работа | Условие | Локальная проверка | Маркер сдачи |
|---|---|---|---|
| HW1: сервис в Docker | [hw1/README.md](hw1/README.md) | `sh check.sh HW1` | `MLSD-HW: HW1` |
| HW2: клиентские признаки | [hw2/README.md](hw2/README.md) | `sh check.sh HW2` | `MLSD-HW: HW2` |

## HW1: health-сервис

Небольшой HTTP-сервис на Python 3.12. Отвечает на `GET /health` JSON-объектом
`{"status":"ok"}`.

### Сборка и запуск

Из корня репозитория:

```sh
docker build --tag hw1 .
docker run --publish 8080:8080 hw1
```

Проверка:

```sh
curl http://127.0.0.1:8080/health
```

Ожидаемый ответ: `{"status":"ok"}`.

## HW2: признаки Online Retail II

Компоненты в `hw2/`:

- **Airflow** (`airflow-webserver`, `airflow-scheduler`, `airflow-init`) — DAG `online_retail_features`;
- **PostgreSQL** — метаданные Airflow;
- **Spark** (`spark-master`, `spark-worker`) — расчёт признаков;
- **SeaweedFS** — S3-совместимое хранилище (`hw2/input/sample.csv`, `hw2/features/`);
- **Redis** — признаки по ключам `hw2:customer:<id>`.

### Запуск среды

```sh
cd hw2
docker compose up --build --wait
```

Веб-интерфейс Airflow: http://127.0.0.1:8080 (логин/пароль `airflow` / `airflow`).
Запустите DAG `online_retail_features` кнопкой Trigger DAG.

Остановка:

```sh
docker compose down
```

### Где лежит результат

- CSV в S3: bucket `hw2`, ключ `input/sample.csv`;
- Parquet в S3: prefix `features/`;
- Redis: `hw2:customer:<customer_id>` → JSON с `order_count`, `total_amount`, `average_order_amount`.

Подробности и версии — в [hw2/TECHNICAL.md](hw2/TECHNICAL.md).

## Локальная проверка курса

Из корня репозитория:

```sh
sh check.sh HW1
sh check.sh HW2
```

`PASS` — локальная проверка прошла. Это ещё не сдача: отправьте коммит с маркером
в `main` после регистрации на https://mlsd.beetles.family
