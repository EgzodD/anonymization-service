"""
Модуль анонимизации текста с Microsoft Presidio.
Настроен для русского языка с кастомными распознавателями.
"""

import logging

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import OperatorConfig

from app.custom_recognizers import ALL_RU_RECOGNIZERS
from app.person_filters import filter_person_results

logger = logging.getLogger(__name__)

# Типы сущностей, которые НЕ надо скрывать (LOCATION по запросу)
EXCLUDED_ENTITIES = {"LOCATION", "NRP"}

# Операторы анонимизации для каждого типа
OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "INN": OperatorConfig("replace", {"new_value": "<INN>"}),
    "SNILS": OperatorConfig("replace", {"new_value": "<SNILS>"}),
    "PASSPORT": OperatorConfig("replace", {"new_value": "<PASSPORT>"}),
    "DATE_OF_BIRTH": OperatorConfig("replace", {"new_value": "<DATE_OF_BIRTH>"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
    "ADDRESS": OperatorConfig("replace", {"new_value": "<ADDRESS>"}),
    "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
}


def _resolve_overlaps(results):
    """Оставляет непересекающиеся спаны — как presidio делает для текста.

    Одно значение (например число-ИНН) может ловиться сразу несколькими
    распознавателями (ИНН + телефон + паспорт). В анонимизированном ТЕКСТЕ
    presidio оставляет один спан (высший score), а вот mapping раньше строился
    по всем «сырым» результатам — и плейсхолдер затирался чужим значением
    (<PHONE> получал значение ИНН). Здесь берём тот же непересекающийся набор:
    сортировка по score убыв., при равенстве — длиннее; пересекающиеся с уже
    выбранными отбрасываем.
    """
    chosen = []
    for r in sorted(results, key=lambda x: (-x.score, -(x.end - x.start))):
        if any(not (r.end <= c.start or r.start >= c.end) for c in chosen):
            continue
        chosen.append(r)
    return chosen


def _build_analyzer() -> AnalyzerEngine:
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "ru", "model_name": "ru_core_news_lg"},
        ],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["ru"],
    )
    for recognizer in ALL_RU_RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
        if hasattr(recognizer, "load"):
            recognizer.load()
    # Снимаем встроенный presidio PhoneRecognizer: он метит любое правдоподобное
    # 10+-значное число как телефон (номер заказа/трека/договора → <PHONE>,
    # перемаскирование). Российские телефоны покрывают наши ru_phone_* паттерны
    # (+7/8, мобильный 8/7-9XX и голый 9XXXXXXXXX).
    analyzer.registry.recognizers = [
        r for r in analyzer.registry.recognizers if r.name != "PhoneRecognizer"
    ]
    return analyzer


analyzer_engine = _build_analyzer()
anonymizer_engine = AnonymizerEngine()
deanonymizer_engine = DeanonymizeEngine()

# Политика: полный набор типов, которые сервис вообще имеет право скрывать
# (всё поддерживаемое минус безусловно исключённое). Кастомный параметр запроса
# может только СУЖАТЬ этот набор (отключать отдельные типы), но не расширять его.
POLICY_ENTITIES = frozenset(
    e for e in analyzer_engine.get_supported_entities(language="ru")
    if e not in EXCLUDED_ENTITIES
)


def resolve_disabled_entities(disable_entities) -> frozenset:
    """
    Приводит клиентский список отключаемых типов к безопасному множеству.

    Правила безопасности:
    - можно только СУЖАТЬ: отключить разрешено лишь типы из POLICY_ENTITIES;
    - неизвестные/запрещённые названия молча игнорируются (безопасное поведение —
      данные останутся скрытыми), но пишутся в лог как предупреждение;
    - факт отключения логируется (что именно отключили) для аудита.
    """
    if not disable_entities:
        return frozenset()

    requested = {str(x).strip().upper() for x in disable_entities if str(x).strip()}
    valid = frozenset(requested & POLICY_ENTITIES)
    ignored = requested - POLICY_ENTITIES

    if valid:
        logger.info("Кастомный параметр: отключены типы сущностей: %s", sorted(valid))
    if ignored:
        logger.warning(
            "Кастомный параметр: проигнорированы недопустимые типы (нет в политике): %s",
            sorted(ignored),
        )
    return valid


def analyze_text(text: str, disable_entities=None) -> list:
    disabled = resolve_disabled_entities(disable_entities)
    results = analyzer_engine.analyze(
        text=text,
        language="ru",
    )
    # Убираем LOCATION/исключённые типы, а также отключённые запросом типы
    results = [
        r for r in results
        if r.entity_type not in EXCLUDED_ENTITIES and r.entity_type not in disabled
    ]
    # Ложные PERSON («гражданине», «г . Москва», «Тульской») — после ОБОИХ источников
    # имён (модель ruBERT и встроенный spaCy), до разрешения пересечений: тогда
    # на месте выброшенного ложного имени может остаться, например, адрес.
    return filter_person_results(text, results)


def anonymize_text(text: str, disable_entities=None) -> dict:
    if not text or not text.strip():
        return {
            "original": text,
            "anonymized": text,
            "entities_found": [],
            "mapping": {},
        }

    results = analyze_text(text, disable_entities=disable_entities)

    if not results:
        return {
            "original": text,
            "anonymized": text,
            "entities_found": [],
            "mapping": {},
        }

    # Единый источник истины для ТЕКСТА, mapping и entities_found — один
    # непересекающийся набор `resolved`, обходим его слева направо и собираем
    # всё за один проход. Так исключены две проблемы сразу:
    #   1) рассинхрон текста и mapping (раньше текст строил presidio по сырому
    #      `results`, а mapping — по отдельному resolve; на неоднозначных «голых»
    #      числах в тексте стоял <PASSPORT>, а в mapping ключ был <PHONE>);
    #   2) схлопывание mapping: несколько значений одного типа (три телефона)
    #      писались в один ключ <PHONE>, и все, кроме последнего, терялись —
    #      деобезличить их было нельзя. Теперь повторы нумеруются: первый остаётся
    #      <PHONE> (обратная совместимость), далее <PHONE_2>, <PHONE_3> …, и текст
    #      с mapping совпадают ключ-в-ключ.
    resolved = sorted(_resolve_overlaps(results), key=lambda x: x.start)

    parts = []
    mapping = {}
    entities_found = []
    seen = {}
    cursor = 0
    for r in resolved:
        if r.entity_type in EXCLUDED_ENTITIES:
            continue
        op = OPERATORS.get(r.entity_type, OPERATORS["DEFAULT"])
        base = op.params.get("new_value", f"<{r.entity_type}>")
        n = seen.get(base, 0) + 1
        seen[base] = n
        placeholder = base if n == 1 else f"{base[:-1]}_{n}>"

        parts.append(text[cursor:r.start])
        parts.append(placeholder)
        cursor = r.end

        original_value = text[r.start : r.end]
        mapping[placeholder] = original_value
        entities_found.append(
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 2),
                "value": original_value,
            }
        )
    parts.append(text[cursor:])

    return {
        "original": text,
        "anonymized": "".join(parts),
        "entities_found": entities_found,
        "mapping": mapping,
    }


def deanonymize_text(text: str, mapping: dict) -> dict:
    """Восстанавливает исходные значения: заменяет плейсхолдеры на значения mapping.

    Сценарий: анонимизировать → отдать текст с <PERSON>/<PHONE> внешней LLM →
    восстановить её ответ по mapping. Плейсхолдеры у нас уникальны (<PHONE>,
    <PHONE_2>, …) и значения не содержат «<…>», поэтому обратная замена
    однозначна. Длинные ключи заменяем первыми — на случай, если один плейсхолдер
    является префиксом другого.

    mapping — ключ деобезличивания: значения (реальные ПДн) НЕ логируем, только
    счётчик замен.
    """
    if not text or not mapping:
        return {"deanonymized": text, "replaced": 0}
    result = text
    replaced = 0
    for placeholder in sorted(mapping, key=len, reverse=True):
        occurrences = result.count(placeholder)
        if occurrences:
            result = result.replace(placeholder, mapping[placeholder])
            replaced += occurrences
    return {"deanonymized": result, "replaced": replaced}


def anonymize_json(data, all_entities: list | None = None, disable_entities=None) -> tuple:
    """Рекурсивно анонимизирует все строковые значения в dict/list (для jsonb-полей)."""
    if all_entities is None:
        all_entities = []

    if data is None:
        return None, all_entities

    if isinstance(data, str):
        result = anonymize_text(data, disable_entities=disable_entities)
        all_entities.extend(result["entities_found"])
        return result["anonymized"], all_entities

    if isinstance(data, dict):
        anonymized_dict = {}
        for key, value in data.items():
            anonymized_dict[key], all_entities = anonymize_json(
                value, all_entities, disable_entities=disable_entities
            )
        return anonymized_dict, all_entities

    if isinstance(data, list):
        anonymized_list = []
        for item in data:
            anon_item, all_entities = anonymize_json(
                item, all_entities, disable_entities=disable_entities
            )
            anonymized_list.append(anon_item)
        return anonymized_list, all_entities

    return data, all_entities
