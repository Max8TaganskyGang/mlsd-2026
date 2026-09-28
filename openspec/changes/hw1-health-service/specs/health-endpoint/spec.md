# Health endpoint

## Requirements

### Requirement: Health check JSON

Сервис MUST отвечать на `GET /health` HTTP 200 и JSON-объектом верхнего
уровня, в котором есть строковое поле `status` со значением `ok`. Другие
поля допустимы.

#### Scenario: Successful health probe

- WHEN клиент отправляет `GET /health` на порт 8080
- THEN ответ имеет код 200 и тело — JSON-объект с `"status": "ok"`

#### Scenario: Response within two seconds

- WHEN контейнер запущен и слушает порт 8080
- THEN `GET /health` завершается успешно не дольше чем за две секунды

### Requirement: Reproducible Docker image

Образ MUST собираться командой `docker build` с контекстом `.` из корня
репозитория. Каждая инструкция `FROM` (кроме `scratch`) MUST указывать
digest `sha256` длиной 64 шестнадцатеричных символа.

#### Scenario: Build from repository root

- WHEN выполняется `docker build --tag hw1 .`
- THEN сборка завершается без ошибки и получается образ `hw1`

#### Scenario: Published port

- WHEN выполняется `docker run --publish 8080:8080 hw1`
- THEN сервис внутри контейнера слушает порт 8080
