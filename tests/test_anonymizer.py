"""
Тесты модуля анонимизации — проверка распознавания русских ПДн.

Категория: модульные (unit) — каждый тип ПДн проверяется изолированно.
Запуск только этой категории: pytest -m unit (или через scripts/test_menu.py).
"""

import csv
import os

import pytest

from app.anonymizer import analyze_text, anonymize_text

pytestmark = pytest.mark.unit


class TestPersonRecognition:
    def test_recognizes_full_name(self):
        result = anonymize_text("Обращение от Ивана Петрова по вопросу кредита")
        assert "<PERSON>" in result["anonymized"]

    def test_recognizes_full_name_with_patronymic(self):
        result = anonymize_text("Клиент: Наталья Владимировна Кузнецова")
        assert "<PERSON>" in result["anonymized"]
        assert "Наталья" not in result["anonymized"]


class TestPhoneRecognition:
    def test_recognizes_plus7_phone(self):
        result = anonymize_text("Телефон: +7 999 123 45 67")
        assert "<PHONE>" in result["anonymized"]
        assert "999" not in result["anonymized"]

    def test_recognizes_8_phone(self):
        result = anonymize_text("Звоните 8 (495) 555-12-34")
        assert "<PHONE>" in result["anonymized"]

    def test_recognizes_compact_phone(self):
        result = anonymize_text("Номер: +79161234567")
        assert "<PHONE>" in result["anonymized"]


class TestEmailRecognition:
    def test_recognizes_email(self):
        result = anonymize_text("Почта: ivan.petrov@yandex.ru")
        assert "<EMAIL>" in result["anonymized"]
        assert "ivan.petrov" not in result["anonymized"]


class TestInnRecognition:
    def test_recognizes_inn_12_with_context(self):
        # 772012345670 — валидный ИНН (12 цифр, проходит алгоритм ФНС)
        result = anonymize_text("ИНН: 772012345670")
        assert "<INN>" in result["anonymized"]

    def test_recognizes_inn_10_with_context(self):
        # 7701234560 — валидный ИНН (10 цифр, проходит алгоритм ФНС)
        result = anonymize_text("ИНН клиента 7701234560")
        assert "<INN>" in result["anonymized"]


class TestSnilsRecognition:
    def test_recognizes_snils_dashes(self):
        # 123-456-789 64 — валидный СНИЛС (проходит алгоритм ПФР)
        result = anonymize_text("СНИЛС 123-456-789 64")
        assert "<SNILS>" in result["anonymized"]

    def test_recognizes_snils_spaces(self):
        # 123 456 789 64 — валидный СНИЛС (пробельный формат)
        result = anonymize_text("страховое свидетельство 123 456 789 64")
        assert "<SNILS>" in result["anonymized"]


class TestPassportRecognition:
    def test_recognizes_passport_with_context(self):
        result = anonymize_text("Паспорт серия 4515 номер 123456")
        entities = [e["entity_type"] for e in result["entities_found"]]
        assert "PASSPORT" in entities


class TestDateOfBirthRecognition:
    def test_recognizes_dob_dot_format(self):
        result = anonymize_text("Дата рождения: 15.03.1990")
        assert "<DATE_OF_BIRTH>" in result["anonymized"]

    def test_recognizes_dob_slash_format(self):
        result = anonymize_text("Родился 25/12/1985")
        # Может детектиться как DATE_OF_BIRTH или DATE_TIME — оба варианта корректны
        assert "25/12/1985" not in result["anonymized"]


class TestCreditCardRecognition:
    def test_recognizes_card_spaces(self):
        result = anonymize_text("Номер карты 4276 1234 5678 9012")
        assert "<CREDIT_CARD>" in result["anonymized"]

    def test_recognizes_card_dashes(self):
        result = anonymize_text("Карта: 4276-5500-1234-7890")
        assert "<CREDIT_CARD>" in result["anonymized"]


class TestAddressPolicy:
    """Политика: адрес (улица/дом/квартира, с городом) СКРЫВАЕТСЯ; город сам по
    себе (без улицы) — не адрес и остаётся."""

    def test_full_address_hidden(self):
        result = anonymize_text("Адрес: г. Москва, ул. Ленина, д. 5, кв. 12")
        assert "<ADDRESS>" in result["anonymized"]
        assert "Ленина" not in result["anonymized"]

    def test_address_with_city_hidden(self):
        result = anonymize_text(
            "Санкт-Петербург, Невский проспект, дом 18, квартира 47")
        assert "<ADDRESS>" in result["anonymized"]
        assert "Невский" not in result["anonymized"]

    def test_bare_city_preserved(self):
        # Город САМ ПО СЕБЕ (без улицы/дома) — не адрес, не маскируется.
        result = anonymize_text("Офис расположен в Санкт-Петербурге")
        assert "Санкт-Петербург" in result["anonymized"]


class TestEdgeCases:
    def test_empty_string(self):
        result = anonymize_text("")
        assert result["anonymized"] == ""
        assert result["entities_found"] == []

    def test_whitespace_only(self):
        result = anonymize_text("   ")
        assert result["entities_found"] == []

    def test_no_pii(self):
        result = anonymize_text("Сегодня хорошая погода в парке")
        assert result["anonymized"] == "Сегодня хорошая погода в парке"

    def test_multiple_entities(self):
        text = "Иван Петров, +79991234567, ivan@mail.ru"
        result = anonymize_text(text)
        assert "<PHONE>" in result["anonymized"]
        assert "<EMAIL>" in result["anonymized"]

    def test_mapping_returned(self):
        result = anonymize_text("Телефон: +7 999 123 45 67")
        assert len(result["mapping"]) > 0


class TestAnonymizeJson:
    def test_anonymizes_flat_dict(self):
        from app.anonymizer import anonymize_json
        data = {"name": "Иван Петров", "city": "Москва"}
        result, entities = anonymize_json(data)
        assert "<PERSON>" in result["name"]
        assert "Москв" in result["city"]  # LOCATION не скрывается

    def test_anonymizes_nested_dict(self):
        from app.anonymizer import anonymize_json
        data = {"info": {"phone": "Телефон: +79991234567", "note": "обычный текст"}}
        result, entities = anonymize_json(data)
        assert "<PHONE>" in result["info"]["phone"]
        assert result["info"]["note"] == "обычный текст"

    def test_anonymizes_list(self):
        from app.anonymizer import anonymize_json
        data = ["Иван Петров", "обычный текст"]
        result, entities = anonymize_json(data)
        assert "<PERSON>" in result[0]

    def test_handles_none(self):
        from app.anonymizer import anonymize_json
        result, entities = anonymize_json(None)
        assert result is None

    def test_handles_numbers(self):
        from app.anonymizer import anonymize_json
        data = {"count": 42, "active": True}
        result, entities = anonymize_json(data)
        assert result["count"] == 42
        assert result["active"] is True


class TestDatasetCoverage:
    """Проверяем покрытие на CSV-датасете."""

    @pytest.fixture
    def dataset(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "ru_training_data.csv"
        )
        rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def test_dataset_recall(self, dataset):
        """Recall >= 0.5 на датасете — хотя бы половина сущностей распознана."""
        total = 0
        found = 0

        seen_texts = {}
        for row in dataset:
            text = row["text"]
            expected_type = row["entity_type"]
            if text not in seen_texts:
                seen_texts[text] = analyze_text(text)

            results = seen_texts[text]
            detected_types = {r.entity_type for r in results}
            total += 1
            if expected_type in detected_types:
                found += 1

        recall = found / total if total > 0 else 0
        assert recall >= 0.95, f"Recall={recall:.2f} < 0.95 (found {found}/{total})"


# Границы маски ФИО (18.09.2026): точка инициала входит в маску, двойная фамилия —
# одна маска. Раньше «В.Н» маскировалось без точки, а «Некрасов-Буров» — двумя
# масками с дефисом между ними.
@pytest.mark.requires_model
@pytest.mark.parametrize("text,expected", [
    ("Прошу связаться с Тетериной В.Н. по заявке", "Прошу связаться с <PERSON> по заявке"),
    ("Влас Некрасов-Буров, ваша посылка доставлена", "<PERSON>, ваша посылка доставлена"),
])
def test_person_mask_boundaries(text, expected):
    assert anonymize_text(text)["anonymized"] == expected


# Адреса в разговорной записи (18.09.2026, по разбору утечек на внешних наборах:
# целиком открытыми оставались 32–36 % адресов). Примеры — собственные.
ADDRESS_LEAK_CASES = [
    ("без маркера", "Жду мастера, Рябиновая 47", "Рябиновая 47"),
    ("город без маркера", "Везите в Тверь, Заречная 12 кв 5", "Заречная 12"),
    ("пр. и номер без «д.»", "Живу на пр. Строителей 91 кв. 14", "Строителей 91"),
    ("ш без точки", "Адрес: ш Каширское 23", "Каширское 23"),
    ("аллея", "доставка на алл. Липовая 7", "Липовая 7"),
    ("числовое название", "Курьер, наб. 50 лет Октября 3", "50 лет Октября 3"),
    ("порядковое без маркера", "Сургут, 1-я Парковая 9", "1-я Парковая 9"),
    ("строчными с маркером", "адрес: пр. речная 101", "речная 101"),
    ("строчными полностью", "курск, садовая 14, кв 88", "садовая 14"),
    ("корпус слитно", "мастера на Казачья 193к2", "Казачья 193к2"),
    ("подъезд", "ул. 8 Марта 84 кв. 256 подъезд 6", "8 Марта 84"),
]


@pytest.mark.parametrize("desc,text,value", ADDRESS_LEAK_CASES, ids=[c[0] for c in ADDRESS_LEAK_CASES])
def test_address_not_leaked(desc, text, value):
    out = anonymize_text(text)["anonymized"]
    assert value not in out, f"{desc}: адрес остался в тексте"


# Не адрес: числа рядом со словами с заглавной — возраст, даты, количества.
@pytest.mark.parametrize("text", [
    "Смирнова 3 раза звонила", "было 3 Июня 2020", "в 2 Корпуса 5 человек", "Статья 25 закона",
    "Глава 12 договора", "Первая 5 минут бесплатно", "Проживаю в Москве",
])
def test_not_address(text):
    assert "<ADDRESS>" not in anonymize_text(text)["anonymized"]


def test_address_does_not_swallow_words_before_marker():
    """Регистронезависимый шаблон «Название + маркер» маскировал «Офис на ул.» и оставлял
    название улицы открытым."""
    out = anonymize_text("Офис на ул. Ленина работает до шести вечера")["anonymized"]
    assert out.startswith("Офис на ") and "Ленина" not in out


@pytest.mark.requires_model
def test_name_before_address_both_masked():
    """Имя перед адресом через запятую не принимается за город: иначе маски пересекались
    и терялись обе."""
    out = anonymize_text("Срочно! Татьяна Николаева, наб. Центральная 83а кв. 112")["anonymized"]
    assert out == "Срочно! <PERSON>, <ADDRESS>"


# Адрес без маркера не должен срабатывать на фамилии перед датой или в нумерованном
# списке: маска адреса наезжала на ФИО, и имя открывалось (найдено 18.09.2026).
@pytest.mark.requires_model
@pytest.mark.parametrize("text,name", [
    ("Министр Геннадий Воронов 14 июня провёл совещание", "Геннадий"),
    ("Ответственные лица: 1 ) Самойлов 2 ) Кречетова", "Самойлов"),
])
def test_bare_address_does_not_break_names(text, name):
    out = anonymize_text(text)["anonymized"]
    assert "<ADDRESS>" not in out and name not in out


@pytest.mark.parametrize("text,value", [
    ("привезите на Вишнёвую улицу, д 3, кв 17", "Вишнёвую улицу"),
    ("адрес: тула, ленинское шоссе 40, кв 9", "ленинское шоссе 40"),
])
def test_address_reverse_forms(text, value):
    assert value not in anonymize_text(text)["anonymized"]
