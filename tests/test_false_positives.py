"""
Ложные срабатывания на негативных примерах (precision-сторож).

Набор `test_negatives.jsonl` — предложения БЕЗ ПДн, но с числами и словами,
похожими на ПДн (номер заказа, артикул, трек, сумма, счёт-фактура). Ни одно из
них скрывать не нужно: любой плейсхолдер здесь = ложное срабатывание
(over-masking).

Категория: unit (качество распознавания). Не требует модели PERSON — без неё
модельные FP просто исчезают, потолок только снижается.

СТАТУС: действующий гейт, ложных срабатываний 0/20 (17.09.2026).

История: 7/20 → 5/20 после дообучения (27.07) → 1/20 после правок распознавателей
(коммит 01d0d4d сняли встроенный presidio-телефон и context-free паспорт) → 0/20.
Последний случай — модель метила «Штрихкод» как ФИО — закрыт фильтром ложных
PERSON (app/person_filters.py, правило F3: одно словарное нарицательное слово).
Пометка xfail снята сразу, как тест начал проходить: оставленная пометка
превращает гейт в молчащий — возврат ошибки прогон бы не заметил.
"""
import json
import os

import pytest

from app.anonymizer import anonymize_text

pytestmark = pytest.mark.unit

NEG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training",
                        "test", "test_negatives.jsonl")


def _load():
    with open(NEG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_no_false_positives_on_negatives():
    """На предложениях без ПДн не должно быть ни одного плейсхолдера."""
    offenders = []
    for ex in _load():
        res = anonymize_text(ex["text"])
        types = [e["entity_type"] for e in res["entities_found"]]
        if types:
            # значение не печатаем — только типы (сами тексты негативов не ПДн,
            # но держим единый стиль: в отчёт об ошибке идут только типы + индекс)
            offenders.append((types, res["anonymized"]))
    assert not offenders, (
        f"ложных срабатываний: {len(offenders)} из негативов; "
        f"типы: {[t for t, _ in offenders]}"
    )
