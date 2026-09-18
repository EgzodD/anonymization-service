# Anonymization Service

Микросервис анонимизации персональных данных в русскоязычных текстах.
Использует Microsoft Presidio + кастомные распознаватели для русского языка.

**Самостоятельный сервис.** Ядро (обезличивание текста через `/anonymize/text`)
работает автономно и не требует БД. Интеграция с Chatwoot — **опциональная**,
включается флагом `CHATWOOT_ENABLED=true` и добавляет эндпоинты для работы
с базой Chatwoot (conversation/batch) и webhook. По умолчанию флаг выключен.

---

## 1. Клонирование репозитория и модель

```bash
git clone https://gitea.guiaidn.ru/egzoddd/ASKII-Anonimization_users_data_with_text-MODULE.git anonymization-module
cd anonymization-module
```

Публичное зеркало: `https://github.com/EgzodD/anonymization-service.git` — может отставать от gitea.

### Модель распознавания ФИО (обязательно)

Модель PERSON (дообученный ruBERT, ~114 МБ) **не хранится в git** — это внутренний
актив проекта. Без неё сервис не стартует, а Docker-образ не соберётся.

Получите архив модели у владельца проекта и распакуйте так, чтобы файлы лежали
прямо в `models/person_ruBERT/`:

```
models/person_ruBERT/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Если архив доступен по ссылке (`.tar.gz` с этими файлами в корне):

```bash
PERSON_MODEL_URL=<ссылка> ./scripts/fetch_person_model.sh
```

Проверка: `ls models/person_ruBERT/config.json` — файл должен существовать.

Текущая версия модели — **v3 от 18.09.2026** (контрольная сумма `model.safetensors`
начинается с `6f26ae39`). Архив предыдущей версии модели с этим кодом тоже работает,
но качество ниже (см. ниже).

### Качество распознавания ФИО

Замер сервиса целиком, один раз, по заранее записанному протоколу
(`data/eval_v2/PROTOCOL.md`, поправка 3; результаты — `data/eval_v2/results/acceptance_stage3.json`).
Наборы на обучении и настройке не использовались.

| Набор | Строгий F1 по ФИО | Имён скрыто полностью | Ложное ФИО в текстах без ПДн |
|---|---|---|---|
| test_v2 — синтетические обращения, 424 имени | 0,826 (было 0,718) | 96,7 % (было 92,9 %) | 1,3 % (было 6,2 %) |
| pii_benchmark (redmadrobot, независимая разметка) | 0,637 (было 0,324) | 81,6 % (было 72,6 %) | 4,1 % (было 20,4 %) |
| factRuEval-2016 — новости, 1 345 персон | 0,843 (было 0,521) | 90,9 % (было 92,6 %) | 8,2 % (было 29,3 %) |

Задержка модели — 3,5 мс, сервиса целиком — около 13–16 мс на обращение (CPU).

**Известное ограничение.** Одиночная фамилия без имени в длинном тексте
(«Мураками сделал…») скрывается хуже, чем у прежней модели: на новостях доля утечек
ФИО выросла с 3,1 до 7,9 %. На обращениях (test_v2) разницы нет. Строгий F1 на
pii_benchmark занижен: в его разметке инициалы не входят в имя, а сервис их скрывает.

---

## 2. Настройка переменных окружения

```bash
cp .env.example .env
```

Сервис работает в двух режимах. Что заполнять в `.env` — зависит от режима.

### Режим A. Автономный (по умолчанию) — обезличивание текста

Подключение к БД и Chatwoot не нужно. Достаточно:

```env
CHATWOOT_ENABLED=false          # можно не указывать — это значение по умолчанию
API_KEY=<длинный_случайный_ключ>  # защита эндпоинтов; пусто = без аутентификации (только для разработки)
PERSON_MODEL_DIR=/app/models/person_ruBERT  # модель ФИО; ОБЯЗАТЕЛЬНА — без неё сервис не стартует
```

`DATABASE_URL` и `CHATWOOT_WEBHOOK_SECRET` в этом режиме не требуются.

### Режим B. С интеграцией Chatwoot

Добавляются параметры БД. `DATABASE_URL` обязателен — без него сервис не
стартует (fail-fast).

```env
CHATWOOT_ENABLED=true
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
CHATWOOT_WEBHOOK_SECRET=<секрет_из_chatwoot>   # желателен на проде; пусто = проверка подписи выключена
API_KEY=<ключ>
PERSON_MODEL_DIR=/app/models/person_ruBERT
```

| Параметр | Описание | Пример |
|----------|----------|--------|
| USER | Имя пользователя PostgreSQL | `chatwoot` |
| PASSWORD | Пароль | `secret123` |
| HOST | Хост БД | `localhost` / `192.168.1.100` / `chatwoot-db` |
| PORT | Порт PostgreSQL | `5432` |
| DB_NAME | Имя базы данных | `chatwoot_production` |

Пример для БД в Docker-сети:
```env
DATABASE_URL=postgresql://chatwoot:secret123@chatwoot-db:5432/chatwoot_production
```

### Режим C. Обезличивание документов (PDF / Word)

Опциональный адаптер: приём файлов `.docx` и `.pdf`. Включается флагом и не зависит
от Chatwoot — можно комбинировать с любым режимом выше.

```env
DOCUMENT_ENABLED=true           # по умолчанию false — эндпоинт /anonymize/document выключен
# Необязательные лимиты (значения по умолчанию):
DOCUMENT_MAX_BYTES=20971520     # 20 МБ на файл
DOCUMENT_MAX_PDF_PAGES=100      # предел страниц PDF
DOCUMENT_PDF_DPI=150            # DPI растеризации PDF
```

При `DOCUMENT_ENABLED=false` библиотеки для документов не загружаются, ядро остаётся лёгким.

---

## 3. Запуск через Docker

### Вариант A: Автономный режим (по умолчанию)

БД не нужна. `.env` — как в Режиме A (шаг 2), модель — в `models/person_ruBERT/`
(шаг 1). Поднимается один контейнер:

```bash
docker compose build               # первая сборка качает ~1 ГБ: torch (CPU), transformers, spaCy ru_core_news_lg
docker compose up -d
docker compose logs -f anonymizer  # дождаться «Application startup complete» (10–25 с), затем Ctrl+C
```

Затем — проверка из раздела 4.

- Ошибка `permission denied ... docker.sock` — пользователь не в группе `docker`,
  выполняйте команды через `sudo`.
- Остановить: `docker compose down`. Пересобрать после изменения кода:
  `docker compose up -d --build`.

> ⚠️ `docker-compose.yml` публикует порт на всех интерфейсах (`0.0.0.0:8000`).
> С пустым `API_KEY` любой в локальной сети сможет вызывать API без
> аутентификации, а при включённой интеграции с Chatwoot — получать разговоры
> из базы. Задайте `API_KEY` в `.env`; если сервис нужен только на этой машине,
> замените в compose порт на `"127.0.0.1:8000:8000"`.

Сервис `db` в `docker-compose.yml` по умолчанию закомментирован — он нужен
только для интеграции с Chatwoot (Вариант C).

### Вариант B: Подключение к существующей БД Chatwoot (продакшн)

`.env` — как в Режиме B (`CHATWOOT_ENABLED=true` + `DATABASE_URL`). Сервис
подключается к базе Chatwoot напрямую:

```bash
docker compose up --build -d
```

Если БД Chatwoot работает в отдельной Docker-сети, подключите сервис к ней.
Откройте `docker-compose.yml` и добавьте:

```yaml
services:
  anonymizer:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    networks:
      - chatwoot_network

networks:
  chatwoot_network:
    external: true
```

Имя сети (`chatwoot_network`) замените на реальное.
Узнать его можно командой:

```bash
docker network ls
```

### Вариант C: Локальная разработка с тестовой БД

Для тестирования интеграции без настоящей БД Chatwoot. Поднимается свой
PostgreSQL с тестовыми данными (3 контакта, 3 разговора, 6 сообщений).
Раскомментируйте сервис `db` и `depends_on` в `docker-compose.yml`. Порт
Postgres наружу не публикуется — доступ только внутри Docker-сети:

```yaml
services:
  db:
    image: postgres:16-alpine
    env_file: .env
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  anonymizer:
    build: .
    ports:
      - "8000:8000"
    environment:
      CHATWOOT_ENABLED: "true"
      DATABASE_URL: postgresql://user:password@db:5432/conversations_db
    depends_on:
      - db

volumes:
  pgdata:
```

```bash
docker compose up --build -d
```

---

## 4. Проверка работоспособности

После запуска сервис доступен на `http://localhost:8000`.

Проверка статуса:
```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:
```json
{
  "status": "ok",
  "analyzer_ready": true,
  "person_model_loaded": true,
  "db_connected": null,
  "chatwoot_enabled": false,
  "supported_entities": ["ADDRESS", "AGE", "CREDIT_CARD", "CRYPTO", "DATE_OF_BIRTH", "DATE_TIME",
                         "EMAIL", "EMAIL_ADDRESS", "IBAN_CODE", "ID", "INN", "IN_VOTER", "IP_ADDRESS",
                         "MEDICAL_LICENSE", "ORGANIZATION", "PASSPORT", "PERSON", "PHONE_NUMBER",
                         "SNILS", "URL"]
}
```

`supported_entities` — все типы, которые знает движок, включая встроенные в Presidio.
Собственный плейсхолдер есть только у типов из раздела 8; остальные (например
IP-адрес, URL, IBAN, дата и время) тоже маскируются, но общим `<PII>`.

`db_connected` равен `null` в автономном режиме (БД не используется). При
`CHATWOOT_ENABLED=true` поле показывает `true`/`false` по состоянию базы, а
`status` становится `degraded`, если база недоступна.

`person_model_loaded` показывает, загружена ли модель распознавания ФИО. **В проде
должно быть `true`.** Если `false` — ФИО не распознаются и уйдут в ответ открытым
текстом; `status` при этом `degraded`. Обычно сервис в таком состоянии просто не
стартует, но если он был запущен с `ALLOW_NO_PERSON_MODEL=true` (режим для тестов
и CI) — это единственный способ увидеть проблему снаружи.

Swagger UI (интерактивная документация API):
```
http://localhost:8000/docs
```

### Проверка, что работает актуальная версия

Два вызова. Если `API_KEY` задан, добавьте к ним `-H "X-API-Key: <ключ>"`.

```bash
# 1. Модель ФИО загружена
curl -s http://localhost:8000/health
# ждём: "status": "ok" и "person_model_loaded": true

# 2. Обезличивание работает, исходные значения по умолчанию не отдаются
curl -s -X POST http://localhost:8000/anonymize/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Иван Петров, ИНН 7707083893"}'
# ждём: "anonymized": "<PERSON>, ИНН <INN>", "mapping": {} и пустые "value"
```

Если в ответе `mapping` заполнен без `"return_mapping": true`, а в `/health` нет поля
`person_model_loaded` — отвечает устаревшая сборка сервиса.

### Визуальная проверка — веб-стенд

Для демонстрации удобнее страница с подсветкой плейсхолдеров, восстановлением
текста, кастомным параметром и загрузкой `.docx`/`.pdf`. Запускается локально
(нужен venv из раздела 9), к Docker-контейнеру не обращается:

```bash
.venv/bin/python tests/web-тест/server.py   # затем http://localhost:8080
```

Готовые тестовые документы — в `tests/web-тест/тестовые_файлы/`. Подробнее —
`tests/web-тест/README.md`.

---

## 5. Интеграция с Chatwoot (webhook)

Сервис интегрируется с Chatwoot через вебхуки — **без изменений в основном проекте**.
Работает по той же схеме, что и SpringQwenWebhook (AI-бот).

### Как подключить

1. Откройте Chatwoot: **Settings -> Integrations -> Webhooks -> Add new webhook**
2. Укажите URL: `http://anonymizer:8000/webhook` (или `http://localhost:8000/webhook`)
3. Выберите событие: `message_created`
4. Сохраните

Теперь при каждом новом сообщении Chatwoot автоматически отправляет
его на наш сервис. Сервис анонимизирует текст и данные отправителя.

### Как это работает в архитектуре

```
Пользователь пишет сообщение в чат
          |
          v
     +-----------+
     |  Chatwoot  |  (основной проект, не трогаем)
     +-----------+
          |
          | webhook: message_created
          |
          v
+-------------------------+                    +------------------------+
|  Anonymization Service  |                    |  SpringQwenWebhook     |
|  POST /webhook          |                    |  (AI-бот)              |
|                         |                    |                        |
|  1. Получает сообщение  |   анонимизирован-  |  Получает чистый текст |
|  2. Находит ПДн         |-- ный текст -------->  без ПДн              |
|  3. Заменяет на         |                    |  Генерирует ответ      |
|     плейсхолдеры        |                    |  Отправляет в Chatwoot |
+-------------------------+                    +------------------------+
          |
          | Также можно вызывать
          | напрямую по HTTP
          |
          v
   Любой другой сервис
   POST /anonymize/text
   POST /anonymize/conversation
```

Сервис **только читает** данные — ничего не записывает и не изменяет в БД.

### Пример webhook payload от Chatwoot

```json
{
  "event": "message_created",
  "id": 100,
  "content": "Здравствуйте, меня зовут Иван Петров, тел +79991234567",
  "message_type": "incoming",
  "conversation": {"id": 1, "status": "open"},
  "sender": {
    "id": 10,
    "name": "Иван Петров",
    "email": "ivan@mail.ru",
    "phone_number": "+79991234567"
  }
}
```

### Ответ сервиса

```json
{
  "event": "message_created",
  "message_id": 100,
  "conversation_id": 1,
  "original_content": "Здравствуйте, меня зовут Иван Петров, тел +79991234567",
  "anonymized_content": "Здравствуйте, меня зовут <PERSON>, тел <PHONE>",
  "sender_anonymized": {
    "id": 10,
    "name": "<PERSON>",
    "email": "<EMAIL>",
    "phone_number": "<PHONE>"
  },
  "entities_found": [...],
  "total_entities": 4
}
```

---

## 6. Использование API напрямую

Все эндпоинты ниже защищены ключом: если в `.env` задан `API_KEY`, передавайте
заголовок `X-API-Key`. Без ключа сервис ответит `401`. При пустом `API_KEY`
аутентификация выключена и заголовок можно не указывать (только для разработки).

### 6.1 Анонимизация произвольного текста

Не требует БД. Принимает текст (до 50 000 символов), возвращает обезличенный.

```bash
curl -X POST http://localhost:8000/anonymize/text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <ключ>" \
  -d '{"text": "Меня зовут Иван Петров, тел +7 999 123 45 67, email ivan@mail.ru"}'
```

Ответ по умолчанию — **без исходных значений**: `value` пустые, `mapping` пустой.
Это сделано намеренно: `mapping` позволяет восстановить ПДн, поэтому отдаётся
только по явному запросу.

```json
{
  "original": "Меня зовут Иван Петров, тел +7 999 123 45 67, email ivan@mail.ru",
  "anonymized": "Меня зовут <PERSON>, тел <PHONE>, email <EMAIL>",
  "entities_found": [
    {"entity_type": "PERSON",        "start": 11, "end": 22, "score": 0.99, "value": ""},
    {"entity_type": "PHONE_NUMBER",  "start": 28, "end": 44, "score": 0.9,  "value": ""},
    {"entity_type": "EMAIL_ADDRESS", "start": 52, "end": 64, "score": 1.0,  "value": ""}
  ],
  "mapping": {}
}
```

Необязательные параметры запроса:

| Параметр | Что делает |
|----------|-----------|
| `"return_mapping": true` | вернуть `mapping` и значения в `entities_found` — нужно для восстановления текста (6.2). Выдача пишется в лог для аудита (только количество, без значений) |
| `"disable_entities": ["INN", ...]` | не маскировать перечисленные типы. Может только **сужать** политику: неизвестные и запрещённые типы игнорируются, данные остаются скрытыми |

Ответ с `"return_mapping": true`:

```json
{
  "original": "Меня зовут Иван Петров, тел +7 999 123 45 67, email ivan@mail.ru",
  "anonymized": "Меня зовут <PERSON>, тел <PHONE>, email <EMAIL>",
  "entities_found": [
    {"entity_type": "PERSON",        "start": 11, "end": 22, "score": 0.99, "value": "Иван Петров"},
    {"entity_type": "PHONE_NUMBER",  "start": 28, "end": 44, "score": 0.9,  "value": "+7 999 123 45 67"},
    {"entity_type": "EMAIL_ADDRESS", "start": 52, "end": 64, "score": 1.0,  "value": "ivan@mail.ru"}
  ],
  "mapping": {
    "<PERSON>": "Иван Петров",
    "<PHONE>": "+7 999 123 45 67",
    "<EMAIL>": "ivan@mail.ru"
  }
}
```

Несколько значений одного типа получают номера: `<PHONE>`, `<PHONE_2>`, `<PHONE_3>` —
каждое восстанавливается отдельно.

### 6.2 Восстановление исходного текста

Основной сценарий с внешней LLM: обезличить текст → отправить модели → в её ответе
вернуть исходные значения по `mapping` из шага 6.1.

```bash
curl -X POST http://localhost:8000/deanonymize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <ключ>" \
  -d '{"text": "Перезвоните <PERSON> по номеру <PHONE>.",
       "mapping": {"<PERSON>": "Иван Петров", "<PHONE>": "+7 999 123 45 67"}}'
```

Ответ:
```json
{
  "deanonymized": "Перезвоните Иван Петров по номеру +7 999 123 45 67.",
  "replaced": 2
}
```

`replaced` — сколько плейсхолдеров заменено. `mapping` храните на своей стороне
как ПДн: сервис его не сохраняет и не пишет в лог.

### 6.3 Анонимизация разговора из БД

Подтягивает из БД conversation + все messages + contact и обезличивает.

```bash
curl -X POST http://localhost:8000/anonymize/conversation \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <ключ>" \
  -d '{"conversation_id": 1}'
```

Ответ содержит:
- анонимизированные поля conversation (identifier, additional_attributes, custom_attributes)
- анонимизированный contact (name, email, phone, attributes)
- список всех messages с анонимизированным content

### 6.4 Пакетная анонимизация

Несколько разговоров за один запрос:

```bash
curl -X POST http://localhost:8000/anonymize/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <ключ>" \
  -d '{"conversation_ids": [1, 2, 3]}'
```

### 6.5 Обезличивание документа (PDF / Word)

Требует `DOCUMENT_ENABLED=true`. Принимает файл `.docx` или `.pdf`, возвращает
обезличенную версию тем же типом (multipart-загрузка).

```bash
curl -X POST http://localhost:8000/anonymize/document \
  -H "X-API-Key: <ключ>" \
  -F "file=@/path/to/document.docx" \
  -o anonymized_document.docx
```

- **Word (.docx):** обезличивается тело, таблицы и колонтитулы; очищаются свойства
  документа (автор и т.п.). Вёрстка сохраняется.
- **PDF:** страницы растеризуются, области с ПДн закрашиваются, файл собирается заново
  — на выходе **нет текстового слоя**, скопировать ПДн из результата нельзя.
- Сводка о найденном (без значений ПДн) — в заголовке ответа `X-Anonymization-Summary`.
- `mapping` для документов не возвращается (это ключ де-анонимизации целого файла).
- **Ограничение:** ПДн ищутся только в тексте. Данные на встроенных картинках
  (сканы, фото документов) не распознаются и остаются в результате как есть — для
  них нужен OCR. Пример — `tests/web-тест/тестовые_файлы/с_картинками.*`.
- Необязательный form-параметр `disable_entities` (список типов через запятую) —
  как в `/anonymize/text`, только сужение.

---

## 7. Вызов из другого сервиса (Python)

```python
import httpx

ANONYMIZER_URL = "http://anonymizer:8000"  # имя контейнера в Docker-сети
HEADERS = {"X-API-Key": "<ключ>"}           # значение API_KEY из .env сервиса

# Текст на лету
response = httpx.post(f"{ANONYMIZER_URL}/anonymize/text", headers=HEADERS, json={
    "text": "Клиент Иван Петров, тел +79991234567"
})
clean = response.json()["anonymized"]
# "Клиент <PERSON>, тел <PHONE>"

# Целый разговор из БД
response = httpx.post(f"{ANONYMIZER_URL}/anonymize/conversation", headers=HEADERS, json={
    "conversation_id": 42
})
data = response.json()
for msg in data["messages"]:
    print(msg["anonymized_content"])
```

---

## 8. Что анонимизируется

| Тип ПДн | Плейсхолдер | Пример |
|---------|-------------|--------|
| ФИО | `<PERSON>` | Иван Петров -> `<PERSON>` |
| Телефон | `<PHONE>` | +7 999 123 45 67 -> `<PHONE>` |
| Email | `<EMAIL>` | ivan@mail.ru -> `<EMAIL>` |
| ИНН | `<INN>` | 772012345678 -> `<INN>` |
| СНИЛС | `<SNILS>` | 123-456-789 00 -> `<SNILS>` |
| Паспорт | `<PASSPORT>` | 45 15 678901 -> `<PASSPORT>` |
| Дата рождения | `<DATE_OF_BIRTH>` | 15.03.1990 -> `<DATE_OF_BIRTH>` |
| Банковская карта | `<CREDIT_CARD>` | 4276 1234 5678 9012 -> `<CREDIT_CARD>` |
| Адрес | `<ADDRESS>` | г. Москва, ул. Ленина, д. 5, кв. 12 -> `<ADDRESS>` |
| Прочее: IP, URL, IBAN, криптокошелёк, дата и время | `<PII>` | 192.168.1.10 -> `<PII>` |

Несколько значений одного типа нумеруются: `<PHONE>`, `<PHONE_2>`, …

### Форматы на входе

| Формат | Эндпоинт | Требует |
|--------|----------|---------|
| Произвольный текст | `/anonymize/text` | — |
| Разговор/пакет из БД (JSON) | `/anonymize/conversation`, `/anonymize/batch` | `CHATWOOT_ENABLED=true` |
| Word `.docx`, PDF | `/anonymize/document` | `DOCUMENT_ENABLED=true` |

### Что НЕ анонимизируется

**LOCATION** — отдельные упоминания городов и стран («живу в Москве») остаются в
тексте как есть: сами по себе они не идентифицируют человека. Адрес с улицей и
домом — это уже ПДн, он маскируется как `<ADDRESS>`.

### Какие поля обрабатываются из БД

| Таблица | Поля |
|---------|------|
| conversations | identifier, additional_attributes, custom_attributes |
| messages | content, content_attributes |
| contacts | name, email, phone, additional_attributes, custom_attributes |

---

## 9. Локальный запуск и тесты (без Docker)

### Установка

Нужен Python 3.12. Команды выполняются из корня репозитория, модель — в
`models/person_ruBERT/` (шаг 1).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
# модель ФИО: CPU-сборка torch, версии — как в Dockerfile
.venv/bin/pip install "transformers==5.12.1" "torch==2.12.1" \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
.venv/bin/python -m spacy download ru_core_news_lg
.venv/bin/pip install ruff pytest-cov     # для линтера и покрытия (scripts/check.sh)
```

Запускайте именно `.venv/bin/python` / `.venv/bin/...`: в системном Python этих
зависимостей нет, и сервис упадёт на импорте.

### Запуск сервиса

```bash
PERSON_MODEL_DIR="$PWD/models/person_ruBERT" .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Путь к модели задан явно, потому что в `.env` для Docker указан путь внутри
контейнера (`/app/...`), которого на хосте нет. Для документов добавьте
`DOCUMENT_ENABLED=true` перед командой. Проверка — раздел 4.

### Тесты

```bash
# как в CI — без модели ФИО (тесты с маркером requires_model пропускаются)
ALLOW_NO_PERSON_MODEL=true .venv/bin/pytest -m "not requires_model" -q

# полный набор, включая гейт утечек LeakRate — нужна модель
PERSON_MODEL_DIR="$PWD/models/person_ruBERT" .venv/bin/pytest -q

# линтер + тесты + покрытие одной командой (то же, что CI)
./scripts/check.sh

# интерактивное меню по категориям: security, privacy, speed, ...
./scripts/test_menu.py
```

Что покрыто, текущие цифры прогона и почему часть тестов помечена `xfail` —
в `tests/README.md` и `tests/МЕТОДИКА_ТЕСТИРОВАНИЯ.md`.

---

## 10. Структура проекта

```
anonymization-module/
├── app/                           # код сервиса — единственное, что попадает в Docker-образ (+ модель)
│   ├── main.py                    # FastAPI: /health, /anonymize/text, /deanonymize
│   ├── auth.py                    # Проверка API-ключа (заголовок X-API-Key)
│   ├── config.py                  # Настройки из .env (PERSON_MODEL_DIR, API_KEY, флаги режимов)
│   ├── models.py                  # Pydantic-схемы ядра
│   ├── anonymizer.py              # Логика обезличивания: Presidio, разрешение пересечений, mapping
│   ├── custom_recognizers.py      # Regex-распознаватели русских ПДн (ИНН, СНИЛС, паспорт, адрес, ...)
│   ├── person_transformer_recognizer.py  # Распознавание ФИО дообученной моделью ruBERT
│   └── integrations/
│       ├── chatwoot/          # Опциональный адаптер Chatwoot (за CHATWOOT_ENABLED)
│       │   ├── database.py    #   SQLAlchemy-модели + ленивое подключение к БД
│       │   ├── schemas.py     #   Pydantic-схемы Chatwoot
│       │   ├── service.py     #   Обход conversation → messages → contacts
│       │   └── router.py      #   Эндпоинты /anonymize/conversation, /batch, /webhook
│       └── documents/         # Опциональный адаптер документов (за DOCUMENT_ENABLED)
│           ├── docx_handler.py #   Обезличивание Word (.docx) — тело/таблицы/колонтитулы
│           ├── pdf_handler.py  #   Обезличивание PDF растеризацией (без AGPL)
│           ├── metadata.py     #   Очистка метаданных документа
│           └── router.py       #   Эндпоинт /anonymize/document
├── models/                        # НЕ в git — модель кладётся вручную (шаг 1)
│   └── person_ruBERT/
├── tests/
│   ├── README.md                  # Как устроены тесты, цифры прогона, xfail
│   ├── МЕТОДИКА_ТЕСТИРОВАНИЯ.md   # Методики тестирования на примерах из набора
│   ├── conftest.py                # Фикстуры; пропуск requires_model без модели
│   ├── test_*.py                  # Ядро: распознавание, API, обратимость, грязные форматы, LeakRate, скорость
│   ├── integrations/
│   │   ├── chatwoot/              # Тесты адаптера Chatwoot
│   │   └── documents/             # Тесты адаптера PDF / Word
│   └── web-тест/                  # Веб-стенд для ручной проверки + генератор тестовых документов
├── scripts/
│   ├── check.sh                   # Линтер + тесты + покрытие (локальная замена CI)
│   ├── test_menu.py               # Меню запуска тестов по категориям
│   ├── demo_examples.py           # Показ «вход → выход» на примерах
│   └── fetch_person_model.sh      # Загрузка модели по ссылке (PERSON_MODEL_URL)
├── data/
│   ├── ru_training_data.csv       # Датасет для проверки качества распознавания
│   └── training/                  # Обучение и оценка модели PERSON: наборы train/dev/test, ноутбуки, скрипты метрик
├── .github/workflows/             # CI: ci.yml (на каждый push), nightly.yml (по расписанию)
├── Dockerfile                     # Образ сервиса
├── docker-compose.yml             # Запуск контейнера (и БД для режима Chatwoot)
├── init.sql                       # Тестовые данные для локальной БД
├── requirements.txt               # Зависимости сервиса
├── requirements-dev.txt           # + зависимости для тестов
├── pyproject.toml                 # Настройки pytest (маркеры) и ruff
├── .env.example                   # Шаблон переменных окружения
└── .dockerignore
```
