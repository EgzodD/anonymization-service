"""
Синтезатор обучающих данных, версия 2: инварианты разметки и проверок.

Главное свойство синтезатора — разметка точная по построению: значения ПДн ставит наш
код, поэтому позиции известны. Версия 2 добавила два преобразования ПОСЛЕ подстановки
(опечатки и нижний регистр), и они способны сдвинуть позиции. Эти тесты фиксируют, что
не сдвигают: иначе обучающий набор молча поедет, а увидим мы это только по качеству
модели через несколько дней обучения.

Языковая модель здесь не нужна: проверяется только наш код.
"""
import importlib.util
import os
import random

import pytest

pytestmark = [pytest.mark.unit]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "synth_llm", os.path.join(ROOT, "data", "synth", "synth_llm.py"))
SL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SL)

TEMPLATES = [
    "Здравствуйте! Меня зовут Иван Иванович Петров, мой телефон [ТЕЛЕФОН], живу на [УЛИЦА].",
    "Прошу вернуть деньги за заказ [ЗАКАЗ]. Договор с [ОРГАНИЗАЦИЯ]. Анна Сергеевна Петрова, [EMAIL]",
    "Петрову Ивану Ивановичу отказали. Паспорт [ПАСПОРТ], ИНН [ИНН], адрес [АДРЕС]. Жду ответа.",
    "Олег Олегович Сидоров и Мария Павловна Сидорова оформили полис. Карта [КАРТА], "
    "дата рождения [ДАТА_РОЖДЕНИЯ].",
    "Офис на [УЛИЦА] закрыт, звоните на [ТЕЛЕФОН] — отвечает А. С. Петрова.",
]


def values(text, items):
    return [text[x["start"]:x["stop"]] for x in items]


@pytest.mark.parametrize("n", range(60))
def test_опечатки_не_сдвигают_разметку(n):
    """Порча идёт только вне значений, а позиции пересчитываются по длинам отрезков."""
    rng = random.Random(1000 + n)
    text, spans, negs, errors = SL.fill(TEMPLATES[n % len(TEMPLATES)], rng)
    assert not errors
    was_pii, was_neg = values(text, spans), values(text, negs)
    dirty, spans2, negs2, made = SL.corrupt(text, spans, negs, rng)
    assert made, "порча не внесена — тест ничего не проверил"
    assert dirty != text
    assert values(dirty, spans2) == was_pii
    assert values(dirty, negs2) == was_neg


@pytest.mark.parametrize("n", range(20))
def test_нижний_регистр_сохраняет_позиции(n):
    """lower() не меняет длину строки, поэтому спаны остаются валидными и после порчи."""
    rng = random.Random(2000 + n)
    text, spans, negs, _ = SL.fill(TEMPLATES[n % len(TEMPLATES)], rng)
    was_pii = [v.lower() for v in values(text, spans)]
    low = text.lower()
    assert len(low) == len(text)
    dirty, spans2, _negs2, _ = SL.corrupt(low, spans, negs, rng)
    assert values(dirty, spans2) == was_pii


def test_негативы_не_попадают_в_разметку_пдн():
    """[УЛИЦА]/[ОРГАНИЗАЦИЯ]/[ЗАКАЗ] — не ПДн: их позиции идут отдельным списком."""
    rng = random.Random(7)
    text, spans, negs, errors = SL.fill(
        "Офис на [УЛИЦА], договор с [ОРГАНИЗАЦИЯ], заказ [ЗАКАЗ]. Петров Иван Иванович", rng)
    assert not errors
    assert len(negs) == 3 and [s["type"] for s in spans] == ["PERSON"]
    for neg in negs:
        assert not any(s["start"] < neg["stop"] and neg["start"] < s["stop"] for s in spans)
    assert all(text[n["start"]:n["stop"]] for n in negs)


def test_значения_негативов_не_требуют_падежа():
    """Формы негативов подходят к любому предлогу: сокращение, кавычки, номер после №."""
    rng = random.Random(11)
    for _ in range(50):
        assert SL.neg_value(rng, "УЛИЦА").startswith(("ул. ", "пр. ", "пер. ", "бул. "))
        assert "«" in SL.neg_value(rng, "ОРГАНИЗАЦИЯ")
        assert SL.neg_value(rng, "ЗАКАЗ").startswith("№ ")


def test_почти_дубли_ловятся():
    """Пилот ловил только полные дубли; пересказ теми же словами должен отсекаться."""
    dup = SL.NearDup(0.6)
    base = "Добрый день, не приходит код подтверждения на телефон, прошу разобраться скорее"
    sim, sh = dup.worst(base)
    assert sim == 0.0
    dup.add(sh)
    again, _ = dup.worst(base + " ещё раз")
    assert again >= 0.6
    other, _ = dup.worst("Хочу расторгнуть договор и забрать оригиналы документов из офиса")
    assert other < 0.6


def test_сценарии_разные_внутри_запроса():
    """Причина однообразия в пилоте — один сценарий на все тексты запроса."""
    rng = random.Random(5)
    scens = [SL.scenario(rng, 0.25) for _ in range(40)]
    assert len({(s["channel"], s["style"], s["length"]) for s in scens}) > 20
    assert any(not s["pii"] for s in scens), "нет текстов только с негативами"
    assert any(s["neg"] for s in scens if s["pii"]), "нет негативов внутри текстов с ПДн"


def test_испорченные_слова_помечаются():
    """Нужно отличать ложное срабатывание сервиса на опечатке от ПДн, вписанных моделью.

    Испорченные слова («дувх», «эень») сервис принимает за ФИО — это его ошибка, а не
    неразмеченные ПДн, и такой текст из набора выбрасывать нельзя: он как раз и нужен.
    """
    rng = random.Random(3)
    text, spans, negs, _ = SL.fill(TEMPLATES[0], rng)
    dirty, spans2, _negs2, marks = SL.corrupt(text, spans, negs, rng, rate=2.0)
    assert marks, "порча есть, а помеченных слов нет"
    for start, stop in marks:
        assert 0 <= start < stop <= len(dirty)
        assert dirty[start:stop].strip(), "пометка на пустом месте"
    # помеченные слова действительно изменились
    assert any(dirty[a:b] not in text for a, b in marks)
    # значения ПДн не помечены: порча их не касается
    for sp in spans2:
        assert not any(a < sp["stop"] and sp["start"] < b for a, b in marks)


def test_метки_в_любом_регистре():
    """Модель писала [Телефон] и [Имя] — регистр метки приводим сами."""
    rng = random.Random(13)
    text, spans, _negs, errors = SL.fill("Мой [Телефон] и почта [email]. Петров Иван", rng)
    assert not errors, errors
    assert {s["type"] for s in spans} == {"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"}
    assert "[" not in text


def test_метка_человека_без_падежа_в_брак():
    """Падеж неизвестен -> подстановка даст неверную грамматику, текст бракуем."""
    _text, _spans, _negs, errors = SL.fill("Здравствуйте, [ИМЯ], помогите", random.Random(2))
    assert errors and "без падежа" in errors[0]


def test_сценарий_с_человеком_распознаётся():
    """На этой проверке держится состав набора по форматам имени (только имя, только фамилия)."""
    assert SL.wants_person({"pii": ["ФАМИЛИЯ", "ПАСПОРТ"]})
    assert SL.wants_person({"pii": ["два человека: ФИО и ФИО"]})
    assert not SL.wants_person({"pii": ["ТЕЛЕФОН", "АДРЕС"]})
    assert not SL.wants_person({"pii": []})
