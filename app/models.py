"""
Ядровые Pydantic-модели сервиса (не зависят от Chatwoot).
Модели, специфичные для интеграции с Chatwoot, см. app/integrations/chatwoot/schemas.py.
"""

from pydantic import BaseModel, Field

# --- Request ---

# Общее описание кастомного параметра для всех запросов анонимизации.
# Публичное имя (без ведущего "_"), т.к. используется и в ядре, и в схемах интеграций.
DISABLE_ENTITIES_DESC = (
    "Необязательный список типов сущностей, которые НЕ нужно скрывать в этом запросе "
    "(например [\"DATE_OF_BIRTH\"]). Параметр может только СУЖАТЬ анонимизацию: "
    "перечисленные типы будут пропущены. Включить типы вне политики сервиса "
    "(в т.ч. LOCATION/NRP) через него нельзя. Неизвестные названия игнорируются "
    "(данные останутся скрытыми). Каждое применение логируется."
)


RETURN_MAPPING_DESC = (
    "Возвращать ли mapping (ключ деобезличивания: плейсхолдер→исходное значение) "
    "и исходные значения в entities_found. По умолчанию FALSE — де-анонимизирующие "
    "данные НЕ выдаются (безопасно). Установите true, только если вам нужен реверс "
    "(сценарий «LLM в контуре» через /deanonymize). Каждая выдача mapping "
    "логируется для аудита."
)


class AnonymizeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Текст для анонимизации")
    disable_entities: list[str] | None = Field(default=None, description=DISABLE_ENTITIES_DESC)
    return_mapping: bool = Field(default=False, description=RETURN_MAPPING_DESC)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Меня зовут Иван Петров, мой телефон +7 999 123 45 67, email ivan@mail.ru",
                    "disable_entities": ["DATE_OF_BIRTH"],
                    "return_mapping": False,
                }
            ]
        }
    }


# --- Response ---

class EntityFound(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    value: str


class AnonymizeTextResponse(BaseModel):
    original: str
    anonymized: str
    entities_found: list[EntityFound]
    mapping: dict[str, str]


class DeanonymizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000,
                      description="Текст с плейсхолдерами (<PERSON>, <PHONE>, ...)")
    mapping: dict[str, str] = Field(
        ..., description="Карта плейсхолдер→исходное значение из ответа анонимизации")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Перезвоните <PERSON> по номеру <PHONE>.",
                    "mapping": {"<PERSON>": "Иван Петров", "<PHONE>": "+7 999 123 45 67"},
                }
            ]
        }
    }


class DeanonymizeResponse(BaseModel):
    deanonymized: str
    replaced: int  # сколько плейсхолдеров восстановлено


class HealthResponse(BaseModel):
    status: str
    # Версия модуля (SemVer) — по работающему контейнеру всегда видно, что запущено.
    # Источник: app/__init__.py:__version__.
    version: str
    analyzer_ready: bool
    # false = модель PERSON не загружена, ФИО НЕ распознаются. Вынесено в health
    # осознанно: снаружи это единственный способ отличить рабочий сервис от того,
    # который пропускает ФИО. В проде значение всегда должно быть true.
    person_model_loaded: bool
    # Имя дообученной модели ФИО: PERSON_MODEL_NAME, иначе имя папки из
    # PERSON_MODEL_DIR. Нужно, чтобы отличить прогон на разных моделях при
    # одной и той же версии кода (цифры качества зависят от модели не меньше).
    # Отдаём только имя, не путь: раскрывать раскладку файловой системы наружу
    # незачем. null = модель не подключена.
    person_model: str | None = None
    db_connected: bool | None = None  # null = интеграция с Chatwoot выключена, БД не проверялась
    chatwoot_enabled: bool
    supported_entities: list[str]
