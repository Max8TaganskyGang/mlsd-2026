# MLSD HW1: health-сервис в Docker

Небольшой HTTP-сервис на Python 3.12. Отвечает на `GET /health` JSON-объектом
`{"status":"ok"}`.

## Требования

- Docker Desktop или Docker Engine 24+
- Git

## Сборка и запуск

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

## Локальная проверка курса

```sh
sh ./check.sh
```

`PASS` значит, что контейнер собрался и `/health` отвечает. Это ещё не сдача:
отправьте коммит `MLSD-HW: HW1` в `main` после подключения репозитория на
https://mlsd.beetles.family
