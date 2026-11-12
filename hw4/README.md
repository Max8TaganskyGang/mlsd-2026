# HW4. Онлайн-фичестор и сервис применения модели

Продолжение HW3. Добавлены онлайн-хранилище признаков (Redis), сервис расчёта признаков и сервисы применения моделей.

## Запуск

```bash
cd hw4
docker compose up --build --wait
```

Переменная окружения `HW4_INPUT_DIR` указывает на каталог с входными данными (по умолчанию `./data`).

## Переменные окружения

| Переменная            | Описание                                 | Значение по умолчанию                    |
|-----------------------|------------------------------------------|------------------------------------------|
| `HW4_INPUT_DIR`       | Путь к каталогу с историей и запросом   | `./data`                                 |
| `HW4_S3_ENDPOINT`     | Endpoint SeaweedFS S3 API                | `http://seaweedfs:8333`                  |
| `HW4_BUCKET`          | Имя бакета                               | `hw4`                                    |
| `MLFLOW_TRACKING_URI` | URI MLflow                               | `http://mlflow:5000`                     |
| `HW4_REDIS_HOST`      | Redis hostname                           | `redis`                                  |
| `HW4_REDIS_PORT`      | Redis port                               | `6379`                                   |
| `HW4_POSTGRES_DSN`    | PostgreSQL DSN                           | `postgresql://airflow:airflow@postgres/airflow` |

## Проверка

```bash
sh check.sh HW4
```

## Сервисы

| Сервис            | Порт  | Описание                             |
|-------------------|-------|--------------------------------------|
| `gateway`         | 8080  | Публичный API                        |
| `feature-service` | 8081  | Выдача признаков из Redis            |
| `model-service`   | 8082  | Применение модели из MLflow          |
| `feature-build`   | —     | Расчёт признаков + обучение моделей  |
| `feature-loader`  | —     | Загрузка признаков в Redis           |

## Эндпоинты gateway (8080)

- `GET /health` — статус и активная модель
- `POST /predict` — прогноз для одного клиента
- `POST /predict/batch` — пакетный прогноз
- `GET /model` — информация об активной модели
- `POST /admin/activate` — смена активной модели
