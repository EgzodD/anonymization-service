"""
Микросервис анонимизации персональных данных.
Использует Microsoft Presidio с кастомными распознавателями для русского языка.
Работает как самостоятельный сервис (standalone); интеграция с БД Chatwoot —
опциональная, включается флагом CHATWOOT_ENABLED (см. app/integrations/chatwoot).
"""

import logging

from fastapi import Depends, FastAPI

from app import __version__
from app.anonymizer import (
    EXCLUDED_ENTITIES,
    analyzer_engine,
    anonymize_text,
    deanonymize_text,
)
from app.auth import require_api_key
from app.config import settings
from app.models import (
    AnonymizeTextRequest,
    AnonymizeTextResponse,
    DeanonymizeRequest,
    DeanonymizeResponse,
    HealthResponse,
)
from app.person_transformer_recognizer import person_model_available, person_model_name

logger = logging.getLogger(__name__)

# ─── Модель PERSON обязательна ─────────────────────────────────────────────
# Fail fast: без дообученной модели ФИО не распознаются вообще и уходят в выдачу
# как есть. Раньше в этом случае молча включался фолбэк на стоковую Natasha —
# сервис выглядел исправным, а на деле давал утечки (0.94%, 3 из 318). Лучше не
# стартовать совсем, чем работать дырявым и рапортовать "ok".
if not person_model_available():
    if not settings.allow_no_person_model:
        raise RuntimeError(
            "Модель PERSON не найдена: PERSON_MODEL_DIR не задан или в нём нет "
            "config.json. Сервис не может работать без неё — ФИО не будут "
            "обнаружены и попадут в ответ в открытом виде. Укажите PERSON_MODEL_DIR "
            "(загрузить модель: scripts/fetch_person_model.sh). Только для тестов и "
            "CI запуск без модели разрешается флагом ALLOW_NO_PERSON_MODEL=true."
        )
    logger.warning(
        "ЗАПУСК БЕЗ МОДЕЛИ PERSON (ALLOW_NO_PERSON_MODEL=true). ФИО НЕ распознаются. "
        "Это режим для тестов и CI — в проде он недопустим."
    )

_CORE_DESCRIPTION = (
    "Микросервис анонимизации персональных данных в русскоязычных текстах.\n\n"
    "Использует Microsoft Presidio + кастомные распознаватели для русского языка.\n\n"
    "Работает как самостоятельный (standalone) сервис: принимает произвольный текст "
    "и возвращает анонимизированный результат. `mapping` (ключ деобезличивания) "
    "выдаётся только при `return_mapping=true` и логируется (аудит).\n\n"
    "Обратная операция — `/deanonymize`: по тексту с плейсхолдерами и `mapping` "
    "восстанавливает исходные значения (сценарий «безопасная LLM в контуре»).\n\n"
    "**Поддерживаемые типы ПДн:**\n"
    "- PERSON — ФИО\n"
    "- PHONE_NUMBER — номера телефонов (+7, 8)\n"
    "- EMAIL_ADDRESS — email-адреса\n"
    "- INN — ИНН (10/12 цифр)\n"
    "- SNILS — СНИЛС\n"
    "- PASSPORT — серия и номер паспорта\n"
    "- DATE_OF_BIRTH — дата рождения\n"
    "- CREDIT_CARD — номера карт\n"
    "- ADDRESS — адрес (город/улица/дом/квартира)\n\n"
    "**Не скрывается:** отдельное упоминание города/локации без улицы "
    "(«офис в Москве») — адресом не считается\n\n"
    "**Кастомные параметры запроса:**\n"
    "- `disable_entities` — список типов, которые НЕ скрывать в этом запросе "
    "(например `[\"DATE_OF_BIRTH\"]`). Доступен в `/anonymize/text`"
)

_CHATWOOT_DESCRIPTION_SUFFIX = (
    ", `/anonymize/conversation`, `/anonymize/batch`.\n"
    "  - Работает только на **сужение**: перечисленные типы пропускаются, "
    "включить запрещённые (LOCATION/NRP) или отсутствующие в политике типы нельзя.\n"
    "  - Неизвестные названия игнорируются (данные останутся скрытыми), "
    "каждое применение пишется в лог для аудита.\n"
    "  - Если значение ловится несколькими распознавателями (напр. дата — "
    "`DATE_OF_BIRTH` и `DATE_TIME`), чтобы не скрывать его, нужно отключить "
    "**все** такие типы.\n\n"
    "**Интеграция с Chatwoot включена (CHATWOOT_ENABLED=true).** Дополнительно доступны:\n"
    "- `/anonymize/conversation` — анонимизация conversation с подтягиванием "
    "messages.content/content_attributes, contacts.name/email/phone/additional_attributes/"
    "custom_attributes, conversations.identifier/additional_attributes/custom_attributes\n"
    "- `/anonymize/batch` — пакетная анонимизация нескольких conversations\n"
    "- `/webhook` — приём webhook от Chatwoot (message_created и др.), с проверкой "
    "HMAC-подписи при заданном CHATWOOT_WEBHOOK_SECRET\n\n"
    "**Аутентификация:** X-API-Key header (если задан API_KEY в env)"
)

_STANDALONE_DESCRIPTION_SUFFIX = (
    ".\n"
    "  - Работает только на **сужение**: перечисленные типы пропускаются, "
    "включить запрещённые (LOCATION/NRP) или отсутствующие в политике типы нельзя.\n"
    "  - Неизвестные названия игнорируются (данные останутся скрытыми), "
    "каждое применение пишется в лог для аудита.\n"
    "  - Если значение ловится несколькими распознавателями (напр. дата — "
    "`DATE_OF_BIRTH` и `DATE_TIME`), чтобы не скрывать его, нужно отключить "
    "**все** такие типы.\n\n"
    "**Интеграция с Chatwoot отключена** (CHATWOOT_ENABLED=false, по умолчанию). "
    "Чтобы включить эндпоинты `/anonymize/conversation`, `/anonymize/batch`, `/webhook` — "
    "задайте CHATWOOT_ENABLED=true и DATABASE_URL в env.\n\n"
    "**Аутентификация:** X-API-Key header (если задан API_KEY в env)"
)

_description = _CORE_DESCRIPTION + (
    _CHATWOOT_DESCRIPTION_SUFFIX if settings.chatwoot_enabled else _STANDALONE_DESCRIPTION_SUFFIX
)

app = FastAPI(
    title="Anonymization Service",
    description=_description,
    # Версия модуля из app/__init__.py. До 1.0.0 здесь стояло "2.0.0" из шаблона
    # первого коммита — число ни разу не менялось за 114 коммитов и ничего не
    # значило. Теперь номер один на всё: git-тег, образ, OpenAPI, /health.
    version=__version__,
)

# ─── Опциональная интеграция с Chatwoot ────────────────────────────────────
# Импорт роутера (а с ним и sqlalchemy-моделей) происходит ТОЛЬКО если
# интеграция включена — при chatwoot_enabled=False ядро не тянет за собой
# ORM Chatwoot и не создаёт никаких соединений с БД.
if settings.chatwoot_enabled:
    # Fail fast: если интеграция включена без DATABASE_URL — падаем сразу при
    # старте приложения понятной ошибкой, а не тихо где-то в рантайме на первом
    # обращении к БД (см. также защитную проверку в database.get_engine()).
    if not settings.database_url:
        raise RuntimeError(
            "CHATWOOT_ENABLED=true, но DATABASE_URL не задан. "
            "Укажите DATABASE_URL в .env для работы интеграции с Chatwoot."
        )

    from app.integrations.chatwoot.router import router as chatwoot_router

    app.include_router(chatwoot_router)
    logger.info("Интеграция с Chatwoot включена (CHATWOOT_ENABLED=true)")
else:
    logger.info("Интеграция с Chatwoot отключена (CHATWOOT_ENABLED=false) — сервис работает standalone")

# ─── Опциональный адаптер документов (PDF/Word) ────────────────────────────
# Импорт роутера (а с ним python-docx/pdfplumber/pypdfium2) — только при флаге,
# чтобы standalone-ядро не тянуло эти библиотеки.
if settings.document_enabled:
    from app.integrations.documents.router import router as documents_router

    app.include_router(documents_router)
    logger.info("Адаптер документов включён (DOCUMENT_ENABLED=true): POST /anonymize/document")
else:
    logger.info("Адаптер документов отключён (DOCUMENT_ENABLED=false)")


# ─── Эндпоинты ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Проверка состояния сервиса и списка поддерживаемых типов сущностей.

    Ядро не зависит от БД: подключение к Chatwoot проверяется, только если
    интеграция включена (CHATWOOT_ENABLED=true). Иначе `db_connected` = null.
    """
    db_ok = None
    if settings.chatwoot_enabled:
        db_ok = False
        try:
            # sqlalchemy импортируется здесь, а не на уровне модуля: в standalone
            # ядро не должно тянуть БД-стек в память вообще.
            from sqlalchemy import text as sql_text

            from app.integrations.chatwoot.database import get_session_local

            db = get_session_local()()
            try:
                db.execute(sql_text("SELECT 1"))
                db_ok = True
            finally:
                db.close()
        except Exception:
            db_ok = False

    supported = analyzer_engine.get_supported_entities(language="ru")
    filtered = [e for e in supported if e not in EXCLUDED_ENTITIES]

    analyzer_ready = True
    person_ready = person_model_available()
    status = (
        "ok"
        if analyzer_ready and person_ready and (not settings.chatwoot_enabled or db_ok)
        else "degraded"
    )

    return HealthResponse(
        status=status,
        version=__version__,
        analyzer_ready=analyzer_ready,
        person_model_loaded=person_ready,
        person_model=person_model_name(),
        db_connected=db_ok,
        chatwoot_enabled=settings.chatwoot_enabled,
        supported_entities=sorted(filtered),
    )


@app.post(
    "/anonymize/text",
    response_model=AnonymizeTextResponse,
    tags=["Anonymization"],
    dependencies=[Depends(require_api_key)],
)
def anonymize_free_text(request: AnonymizeTextRequest):
    """Анонимизация произвольного текста. Возвращает анонимизированный текст.

    `mapping` (ключ деобезличивания) и исходные значения в `entities_found`
    выдаются ТОЛЬКО при `return_mapping=true` — иначе опускаются (безопасно по
    умолчанию), каждая выдача пишется в лог для аудита. `disable_entities` —
    сужение политики (см. описание поля).
    """
    result = anonymize_text(request.text, disable_entities=request.disable_entities)
    if request.return_mapping:
        # Аудит выдачи ключа деобезличивания: только счётчик, без значений ПДн.
        logger.info(
            "Выдан mapping деобезличивания: %d ключ(ей), %d сущностей",
            len(result["mapping"]), len(result["entities_found"]),
        )
    else:
        # де-анонимизирующие данные не выдаём: ни mapping, ни значения
        result["mapping"] = {}
        for e in result["entities_found"]:
            e["value"] = ""
    return AnonymizeTextResponse(**result)


@app.post(
    "/deanonymize",
    response_model=DeanonymizeResponse,
    tags=["Anonymization"],
    dependencies=[Depends(require_api_key)],
)
def deanonymize(request: DeanonymizeRequest):
    """Восстановление исходного текста: замена плейсхолдеров на значения из `mapping`.

    Сценарий: анонимизировать текст → отдать версию с `<PERSON>`/`<PHONE>` внешней
    LLM → восстановить её ответ по `mapping`. `mapping` — ключ деобезличивания,
    поэтому передаётся вызывающим (сервис его не хранит); значения не логируются.
    """
    result = deanonymize_text(request.text, request.mapping)
    return DeanonymizeResponse(**result)
