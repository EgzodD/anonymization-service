"""
Независимый тестовый набор v2 для оценки обнаружения ПДн в русскоязычных текстах.

Зачем v2. Старый held-out (data/training/test) сгенерирован тем же генератором,
что и обучающие данные: 128 из 132 тестовых примеров с ПДн (97 %) построены по
шаблонам фраз, которые модель видела при обучении. Такой тест измеряет
запоминание шаблонов, а не обобщение. v2 строится так, чтобы с обучением не
пересекалось ничего, кроме самого языка:

  * шаблоны фраз написаны заново, вручную, независимо от generate_dataset.py и
    augment_negatives.py;
  * имена, отчества и фамилии — из словарей, не пересекающихся с обучающими;
    фамилии включают морфологические типы, которых в обучении не было вовсе
    (-ский/-цкий, -енко, -ых/-их, несклоняемые женские -ук/-ян);
  * для негативов — города, улицы, организации и существительные не из
    обучающих списков;
  * склонение фамилий и отчеств — по явным правилам русской грамматики, а не
    автоматическим анализатором: pymorphy склоняет «Черных» как прилагательное
    («чёрных»), и такие ошибки генератора наказывали бы модели несправедливо.

Эталонная разметка точна по построению: спан = позиция вставленного значения.
Проверки целостности и непересечения с обучением выполняются здесь же; при
любом нарушении скрипт завершается с ошибкой и файл набора не пишется.

Набор замораживается коммитом ДО первого прогона моделей (см. PROTOCOL.md).

Запуск (из корня репозитория):
    .venv/bin/python data/eval_v2/build_test_v2.py
"""
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter

SEED = 20260915
INSTANCES_PII = 4        # экземпляров на каждый шаблон с ПДн
INSTANCES_NEG_SLOT = 4   # экземпляров на шаблон-негатив со слотом
LOWER_SHARE = 0.15       # доля ФИО, записанных строчными (небрежный ввод)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DIR = os.path.join(ROOT, "data", "training")
OUT_PATH = os.path.join(HERE, "test_v2.jsonl")
REPORT_PATH = os.path.join(HERE, "build_report.json")

CASES = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

# ══════════════════════════════════════════════════════════════════════════
#  СЛОВАРИ (не пересекаются с generate_dataset.py / augment_negatives.py)
# ══════════════════════════════════════════════════════════════════════════
MALE_NAMES = [
    "Тимофей", "Леонид", "Глеб", "Всеволод", "Станислав", "Вадим", "Эдуард",
    "Ярослав", "Аркадий", "Валерий", "Геннадий", "Филипп", "Руслан", "Тарас",
    "Матвей", "Ростислав", "Богдан", "Марк", "Остап", "Святослав",
]
FEMALE_NAMES = [
    "Василиса", "Злата", "Алла", "Варвара", "Регина", "Лариса", "Жанна", "Инна",
    "Римма", "Милана", "Ульяна", "Таисия", "Олеся", "Софья", "Зоя", "Ксения",
    "Валентина", "Нина", "Эльвира", "Ангелина",
]
# отчества — пары (муж., жен.)
PATRONYMICS = [
    ("Геннадьевич", "Геннадьевна"), ("Леонидович", "Леонидовна"),
    ("Станиславович", "Станиславовна"), ("Вадимович", "Вадимовна"),
    ("Валерьевич", "Валерьевна"), ("Эдуардович", "Эдуардовна"),
    ("Аркадьевич", "Аркадьевна"), ("Тимофеевич", "Тимофеевна"),
    ("Ярославович", "Ярославовна"), ("Глебович", "Глебовна"),
    ("Константинович", "Константиновна"), ("Вячеславович", "Вячеславовна"),
    ("Германович", "Германовна"), ("Ефимович", "Ефимовна"),
    ("Филиппович", "Филипповна"),
]
# фамилии: (мужская форма, класс склонения)
SURNAMES = [
    ("Белоусов", "ov"), ("Горбунов", "ov"), ("Широков", "ov"), ("Субботин", "ov"),
    ("Колесников", "ov"), ("Прохоров", "ov"), ("Шестаков", "ov"), ("Рябинин", "ov"),
    ("Ковальский", "sky"), ("Вишневский", "sky"), ("Загорский", "sky"),
    ("Белецкий", "sky"), ("Островский", "sky"),
    ("Гончаренко", "fixed"), ("Мельниченко", "fixed"), ("Бондаренко", "fixed"),
    ("Черных", "fixed"), ("Седых", "fixed"), ("Долгих", "fixed"),
    ("Ткачук", "cons"), ("Шевчук", "cons"), ("Акопян", "cons"),
    ("Мартиросян", "cons"), ("Полищук", "cons"),
]
CITIES = [
    "Тверь", "Курск", "Смоленск", "Архангельск", "Петрозаводск", "Кострома",
    "Липецк", "Киров", "Пенза", "Мурманск", "Вологда", "Чебоксары", "Саранск",
    "Таганрог", "Серпухов", "Ногинск", "Электросталь", "Пушкино",
]
STREETS = [
    "Тверская", "Пролетарская", "Кооперативная", "Ломоносова", "Чкалова",
    "Некрасова", "Фрунзе", "Космонавтов", "Строителей", "Энтузиастов",
]
ORGS = [
    "Континент", "Эталон", "Ладога", "Созвездие", "Витязь", "Кристалл",
    "Магистраль", "Аврора", "Меридиан", "Радуга",
]
NOUNS = [
    "Спецификация", "Претензия", "Доверенность", "Выписка", "Справка",
    "Протокол", "Смета", "Квота", "Лимит", "Пароль", "Логин", "Баланс",
    "Маршрут", "Посылка", "Возврат",
]

# ══════════════════════════════════════════════════════════════════════════
#  СКЛОНЕНИЕ — явные правила
# ══════════════════════════════════════════════════════════════════════════
def decline_surname(base, cls, gender, case):
    i = CASES.index(case)
    if cls == "fixed":
        return base
    if cls == "cons":
        if gender == "femn":
            return base                          # женские -ук/-ян не склоняются
        return base + ["", "а", "у", "а", "ом", "е"][i]
    if cls == "ov":
        if gender == "masc":
            return base + ["", "а", "у", "а", "ым", "е"][i]
        return base + ["а", "ой", "ой", "у", "ой", "ой"][i]
    if cls == "sky":
        stem = base[:-2]                         # Ковальск-ий
        if gender == "masc":
            return stem + ["ий", "ого", "ому", "ого", "им", "ом"][i]
        return stem + ["ая", "ой", "ой", "ую", "ой", "ой"][i]
    raise ValueError(cls)


def decline_patronymic(word, gender, case):
    i = CASES.index(case)
    if gender == "masc":                         # -ович / -евич
        return word + ["", "а", "у", "а", "ем", "е"][i]
    stem = word[:-1]                             # -овн-а / -евн-а
    return stem + ["а", "ы", "е", "у", "ой", "е"][i]


def decline_first_name(word, gender, case):
    i = CASES.index(case)
    if gender == "masc":
        if word.endswith("ий"):                  # Аркадий, Валерий, Геннадий
            return word[:-1] + ["й", "я", "ю", "я", "ем", "и"][i]
        if word.endswith("ей"):                  # Тимофей, Матвей
            return word[:-1] + ["й", "я", "ю", "я", "ем", "е"][i]
        return word + ["", "а", "у", "а", "ом", "е"][i]
    if word.endswith("ия"):                      # Таисия, Ксения
        return word[:-1] + ["я", "и", "и", "ю", "ей", "и"][i]
    if word.endswith("я"):                       # Софья, Зоя, Олеся
        return word[:-1] + ["я", "и", "е", "ю", "ей", "е"][i]
    return word[:-1] + ["а", "ы", "е", "у", "ой", "е"][i]   # -а


# ══════════════════════════════════════════════════════════════════════════
#  ФОРМАТЫ ФИО
#  seen = формат встречался в обучающих данных (generate_dataset.gen_person)
# ══════════════════════════════════════════════════════════════════════════
FORMATS = {
    "ИОФ": True,    # Всеволод Глебович Белоусов
    "ФИО": True,    # Белоусов Всеволод Глебович
    "ИФ": True,     # Всеволод Белоусов
    "ФИ": True,     # Белоусов Всеволод
    "ИО": True,     # Всеволод Глебович
    "ИнФ": True,    # В.Г. Белоусов
    "ФИн": False,   # Белоусов В.Г.
    "И": False,     # Всеволод
    "Ф": False,     # Белоусов
}
GROUPS = {
    "full": ["ИОФ", "ФИО", "ИФ", "ФИ", "ИнФ", "ФИн", "Ф"],
    "any": ["ИОФ", "ФИО", "ИФ", "ФИ", "ИО", "ИнФ", "ФИн", "И", "Ф"],
    "address": ["ИО", "И"],
    "short": ["И", "ИФ"],
    "ini": ["ИнФ", "ФИн"],
}


def make_person(rng, case, group, exclude=()):
    while True:
        gender = rng.choice(["masc", "femn"])
        first = rng.choice(MALE_NAMES if gender == "masc" else FEMALE_NAMES)
        pm, pf = rng.choice(PATRONYMICS)
        patr = pm if gender == "masc" else pf
        surname, cls = rng.choice(SURNAMES)
        if (first, surname) not in exclude:
            break
    fmt = rng.choice(GROUPS[group])
    f = decline_first_name(first, gender, case)
    p = decline_patronymic(patr, gender, case)
    s = decline_surname(surname, cls, gender, case)
    dot = rng.choice([".", ". "])                # «В.Г.» или «В. Г.»
    ini = f"{first[0]}{dot}{patr[0]}."
    text = {
        "ИОФ": f"{f} {p} {s}", "ФИО": f"{s} {f} {p}", "ИФ": f"{f} {s}",
        "ФИ": f"{s} {f}", "ИО": f"{f} {p}", "ИнФ": f"{ini} {s}",
        "ФИн": f"{s} {ini}", "И": f, "Ф": s,
    }[fmt]
    lower = rng.random() < LOWER_SHARE
    if lower:
        text = text.lower()
    meta = {
        "gender": gender, "case": case, "format": fmt,
        "format_seen_in_training": FORMATS[fmt], "lower": lower,
        "surname_class": cls if ("Ф" in fmt) else None,
        "lex": {"first": first, "patronymic": patr, "surname": surname},
    }
    return text, gender, meta, (first, surname)


# ══════════════════════════════════════════════════════════════════════════
#  ДРУГИЕ ПДн — генераторы написаны заново
# ══════════════════════════════════════════════════════════════════════════
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]


def gen_phone(rng):
    d = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))  # noqa: E731
    op = "9" + d(2)
    return rng.choice([
        f"+7 {op} {d(3)}-{d(2)}-{d(2)}", f"8 ({op}) {d(3)} {d(2)} {d(2)}",
        f"+7{op}{d(7)}", f"8-{op}-{d(3)}-{d(2)}-{d(2)}",
        f"+7 (4{d(2)}) {d(3)}-{d(2)}-{d(2)}", f"8 {op} {d(3)} {d(4)}",
    ])


def gen_email(rng):
    login = rng.choice([
        "g.goncharenko", "office.north", "v.akopyan", "zayavki2026", "dispatcher24",
        "buh.kontinent", "t.shevchuk", "priemnaya", "sales-team", "r.belecky",
    ])
    domain = rng.choice(["rambler.ru", "ya.ru", "outlook.com", "internet.ru",
                         "hotmail.com", "proton.me"])
    return f"{login}@{domain}"


def gen_inn(rng):
    if rng.random() < 0.5:
        b = [rng.randint(0, 9) for _ in range(9)]
        c = sum(x * y for x, y in zip([2, 4, 10, 3, 5, 9, 4, 6, 8], b)) % 11 % 10
        return "".join(map(str, b + [c]))
    b = [rng.randint(0, 9) for _ in range(10)]
    c1 = sum(x * y for x, y in zip([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], b)) % 11 % 10
    c2 = sum(x * y for x, y in zip([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], b + [c1])) % 11 % 10
    return "".join(map(str, b + [c1, c2]))


def snils_checksum(digits9):
    total = sum(int(x) * w for x, w in zip(digits9, range(9, 0, -1)))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    r = total % 101
    return 0 if r in (100, 101) else r


def gen_snils(rng):
    s = "".join(str(rng.randint(0, 9)) for _ in range(9))
    sep = rng.choice(["-", " "])
    return f"{s[:3]}{sep}{s[3:6]}{sep}{s[6:]} {snils_checksum(s):02d}"


def gen_passport(rng):
    ser = f"{rng.randint(1, 99):02d}{rng.randint(0, 99):02d}"
    num = f"{rng.randint(100000, 999999)}"
    return rng.choice([f"{ser[:2]} {ser[2:]} {num}", f"{ser} {num}"])


def gen_dob(rng):
    y, m, dd = rng.randint(1950, 2006), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"{dd:02d}.{m:02d}.{y}", f"{dd} {MONTHS_GEN[m - 1]} {y}"])


def luhn_ok(num):
    s = 0
    for i, ch in enumerate(reversed(num)):
        v = int(ch)
        if i % 2 == 1:
            v = v * 2 - 9 if v * 2 > 9 else v * 2
        s += v
    return s % 10 == 0


def gen_card(rng):
    while True:
        n = rng.choice(["2200", "2202", "4276", "4377", "5469", "5213"]) + \
            "".join(str(rng.randint(0, 9)) for _ in range(12))
        if luhn_ok(n):
            break
    return rng.choice([" ".join(n[i:i + 4] for i in range(0, 16, 4)), n])


VALUE_GEN = {
    "phone": ("PHONE_NUMBER", gen_phone), "email": ("EMAIL_ADDRESS", gen_email),
    "inn": ("INN", gen_inn), "snils": ("SNILS", gen_snils),
    "passport": ("PASSPORT", gen_passport), "dob": ("DATE_OF_BIRTH", gen_dob),
    "card": ("CREDIT_CARD", gen_card),
}

# ══════════════════════════════════════════════════════════════════════════
#  ШАБЛОНЫ С ПДн
#  {P:падеж:группа} — ФИО; {P2:...} — второе лицо; {g:муж|жен} — согласование
#  с полом P; {phone} {email} {inn} {snils} {passport} {dob} {card} — значения
#  (суффикс 2 — второе значение того же типа: {phone2}, {inn2}).
# ══════════════════════════════════════════════════════════════════════════
PII_TEMPLATES = [
    # разговорный чат
    "добрый вечер, пишет {P:nomn:any}, у меня не проходит оплата",
    "Алло, соедините с {P:ablt:full}, пожалуйста.",
    "Это снова {P:nomn:any}. Номер для обратной связи: {phone}.",
    "Коллеги, {P:nomn:full} {g:просил|просила} перенести встречу на четверг.",
    "Передайте {P:datv:full}, что пропуск готов.",
    "{P:nomn:address}, ваш заказ уже в пути.",
    "Спасибо, {P:nomn:short}! Всё получилось.",
    "Подскажите, {P:nomn:full} сегодня на месте?",
    "От кого пришло письмо? От {P:gent:full}.",
    "Скиньте, пожалуйста, договор {P:gent:full} на {email}.",
    "Кто {g:оформлял|оформляла} возврат? Кажется, {P:nomn:full}.",
    "У {P:gent:full} сменился номер, теперь {phone}.",
    "Напомните {P:datv:address} про оплату до пятницы.",
    "Мы с {P:ablt:full} договорились созвониться после обеда.",
    "Всё ок, счёт {g:подписал|подписала} {P:nomn:full}.",
    # заметки оператора / CRM
    "Входящий звонок. Абонент: {P:nomn:full}. Тел.: {phone}.",
    "Карточка клиента обновлена: {P:nomn:full}, e-mail {email}.",
    "Ответственный за обращение — {P:nomn:full}.",
    "Требуется перезвонить {P:datv:full} по номеру {phone} до 18:00.",
    "Жалоба зарегистрирована со слов {P:gent:full}.",
    "Статус: ожидает документы от {P:gent:full}.",
    "Оператор {P:nomn:full} {g:закрыл|закрыла} тикет без решения.",
    "Возврат средств одобрен для {P:gent:full}, карта {card}.",
    "Повторное обращение {P:gent:full}; предыдущий номер заявки утерян.",
    "Сверка данных: {P:nomn:full}, дата рождения {dob}, ИНН {inn}.",
    "Смена владельца счёта: с {P:gent:full} на {P2:accs:full}.",
    "Эскалация на старшего смены {P:accs:full}.",
    "Курьер {P:nomn:full} не {g:смог|смогла} дозвониться до получателя.",
    "Проверка СНИЛС {snils} выполнена, владелец — {P:nomn:full}.",
    "Уведомление отправлено на {email} ({P:nomn:full}).",
    # официальные заявления и письма
    "Я, {P:nomn:full}, {g:ознакомлен|ознакомлена} с условиями договора.",
    "Прошу выдать справку на имя {P:gent:full}, паспорт {passport}.",
    "Доверенность выдана {P:datv:full} сроком на один год.",
    "Настоящим подтверждаю, что {P:nomn:full} работает в отделе продаж с 2019 года.",
    "Генеральному директору от {P:gent:full}. Заявление.",
    "В связи с переездом прошу направлять корреспонденцию {P:datv:full} на {email}.",
    "Акт приёма-передачи подписан со стороны заказчика {P:ablt:full}.",
    "{g:Гражданин|Гражданка} {P:nomn:full}, {dob} года рождения, "
    "{g:обратился|обратилась} в отделение.",
    "Сведения о заявителе: {P:nomn:full}; ИНН {inn}; СНИЛС {snils}.",
    "Прошу перечислить компенсацию на карту {card}, держатель {P:nomn:full}.",
    "Сообщаем, что {P:nomn:full} {g:назначен|назначена} руководителем проекта.",
    "Характеристика на {P:accs:full} прилагается.",
    "Выписка из приказа: предоставить отпуск {P:datv:full} с 1 по 14 июля.",
    "{g:Протокол вёл|Протокол вела} {P:nomn:full}.",
    "Основание: заявление {P:gent:full} от прошлой недели.",
    # электронная почта
    "Здравствуйте, {P:nomn:address}!\nВысылаю скан паспорта {passport}.\n"
    "С уважением, {P2:nomn:full}",
    "Добрый день! Прошу связаться со мной по номеру {phone}.\n{P:nomn:full}",
    "Коллеги, в копии {P:nomn:full} — {g:он|она} курирует поставки.",
    "Отправитель: {P:nomn:full} <{email}>",
    "Спасибо за быстрый ответ.\n--\n{P:nomn:full}\nтел. {phone}",
    # медицина
    "{g:Пациент|Пациентка} {P:nomn:full}, {dob} г. р., {g:записан|записана} "
    "к терапевту на 9:30.",
    "Талон к врачу выдан на имя {P:gent:full}.",
    "Результаты анализов {P:gent:full} готовы, приходите в понедельник.",
    "Полис {P:gent:full} продлён, контактный телефон {phone}.",
    "Направление на обследование {g:получил|получила} {P:nomn:full}.",
    # банк и финансы
    "Перевод от {P:gent:full} на сумму 15 000 рублей зачислен.",
    "Кредитный договор оформлен на {P:accs:full}, ИНН заёмщика {inn}.",
    "Держатель карты {card}: {P:nomn:full}.",
    "По счёту {P:gent:full} заблокированы операции до выяснения.",
    "Поручителем выступает {P:nomn:full}, паспорт {passport}.",
    # кадры и образование
    "В резюме указаны контакты кандидата: {P:nomn:full}, {phone}, {email}.",
    "Собеседование с {P:ablt:full} перенесено на завтра.",
    "Приказ о зачислении {g:студента|студентки} {P:gent:full} подписан.",
    "Научный руководитель — {P:nomn:full}.",
    "Благодарность объявлена {P:datv:full} за досрочное выполнение плана.",
    "{g:Табель заполнил|Табель заполнила} {P:nomn:full}.",
    "На вакансию {g:откликнулся|откликнулась} {P:nomn:full}, дата рождения {dob}.",
    # доставка и торговля
    "Посылку {g:получил|получила} {P:nomn:full}, подпись сверена.",
    "Заказ оформлен на {P:accs:full}, доставка по звонку на {phone}.",
    "Получатель — {P:nomn:full}, при себе иметь паспорт {passport}.",
    "Возврат товара от {P:gent:full} принят на склад.",
    "Курьеру передать заказ лично {P:datv:full}.",
    # право и жильё
    "Собственником квартиры является {P:nomn:full}.",
    "Иск подан {P:ablt:full} в районный суд.",
    "Свидетелем по делу проходит {P:nomn:full}.",
    "Договор аренды заключён между {P:ablt:full} и {P2:ablt:full}.",
    "Показания счётчиков {g:передал|передала} {P:nomn:full}.",
    "{g:Житель|Жительница} {P:nomn:full} просит проверить протечку.",
    # позиция в предложении и пунктуация
    "«{P:nomn:full}» — так подписано последнее письмо.",
    "Звонок {g:принял дежурный|приняла дежурная} ({P:nomn:full}).",
    "{g:Ответ подготовил|Ответ подготовила}: {P:nomn:ini}",
    "{P:nomn:full} — ответственный исполнитель.",
    "Кому: {P:datv:full}\nТема: продление договора",
    "Подпись: ___________ /{P:nomn:ini}/",
    "Согласовано: {P:nomn:ini}, {P2:nomn:ini}.",
    "По словам {P:gent:full}, оплата прошла ещё вчера.",
    "Вопрос к {P:datv:short}: где лежат акты?",
    "Для {P:gent:full} забронирован номер на двое суток.",
    "Если {P:nomn:full} не выйдет на связь, звоните на {phone}.",
    "Ранее клиент представился как {P:nomn:full}, но в паспорте другая фамилия.",
    # плотные ПДн
    "Все три реквизита — {inn}, {snils} и {passport} — принадлежат {P:datv:full}.",
    "Проверьте, пожалуйста, совпадает ли {email} с почтой {P:gent:full}.",
    "Номер {phone} принадлежит {P:datv:full}?",
    "Дата рождения {P:gent:full}: {dob}.",
    "Сотрудник {P:nomn:full} (табельный № 4471) {g:уволен|уволена} "
    "по собственному желанию.",
    "Оплата картой {card} прошла успешно, спасибо, {P:nomn:address}.",
    "Контакты для экстренной связи: {P:nomn:full}, {phone}; {P2:nomn:full}, {phone2}.",
    "Обновите ИНН в профиле {P:gent:full}: был указан {inn}, верный — {inn2}.",
    "На встрече присутствовали {P:nomn:full}, {P2:nomn:full} и представитель поставщика.",
    "Благодарим {P:accs:full} за участие в опросе!",
]

# ══════════════════════════════════════════════════════════════════════════
#  НЕГАТИВЫ — ПДн нет; слоты {city} {street} {org} {noun}
# ══════════════════════════════════════════════════════════════════════════
NEG_TEMPLATES = [
    "Филиал в г. {city} закрыт на санитарный день.",
    "Рейс до города {city} задерживается на два часа.",
    "Склад г. {city} принимает возвраты только по будням.",
    "Ваша посылка прибыла в сортировочный центр (г. {city}).",
    "Цены в городе {city} могут отличаться от указанных на сайте.",
    "Сервисный центр переехал на ул. {street}, д. 3.",
    "Самовывоз: улица {street}, вход со двора.",
    "Пробка на улице {street} в сторону центра.",
    "Остановка «Улица {street}» временно перенесена.",
    "Поставка идёт по рамочному соглашению с «{org}».",
    "АО «{org}» подтвердило получение платежа.",
    "Магазин «{org}» работает до 22:00.",
    "Партнёрская программа «{org}» продлена до конца года.",
    "{noun}: статус не изменился.",
    "{noun} — во вложении.",
    "Раздел «{noun}» обновлён.",
    "Поле «{noun}» заполнено неверно.",
    "Вкладка «{noun}» не открывается.",
]
NEG_FIXED = [
    "Любовь к порядку у нас в крови — спасибо за терпение.",
    "Роза ветров на схеме склада указана неверно.",
    "Надеемся на понимание и скорейшее решение вопроса.",
    "Проспект Энтузиастов перекрыт до 18:00.",
    "Горячая линия работает круглосуточно.",
    "Тариф «Максимум» подключён с первого числа.",
    "Акция «Золотая осень» завершилась.",
    "Кнопка «Отправить» неактивна.",
    "Серпухов, Киров и Пушкино вошли в зону доставки.",
    "Добрый день! Уточните номер заказа, пожалуйста.",
    "Вчера на Тверской открылся новый офис.",
    "Встреча в переговорной «Байкал» в 15:00.",
    "Номер заказа 4815162342 — проверьте статус в личном кабинете.",
    "Счёт на оплату № 2031 выставлен, срок оплаты — пять рабочих дней.",
    "Оценка сервиса: 5 из 5, спасибо за обратную связь!",
    "Приложение обновлено до версии 12.4.1.",
]

SLOT_RE = re.compile(r"\{([^{}]+)\}")


def build_pii(template, rng):
    genders = {}
    used = set()
    text, spans, pos = "", [], 0
    # пол P нужен для {g:...}, которые могут стоять ДО слота P, — выбираем заранее
    persons = {}
    for key in ("P", "P2"):
        m = re.search(r"\{" + key + r":(\w+):(\w+)\}", template)
        if m:
            t, g, meta, ident = make_person(rng, m.group(1), m.group(2), exclude=used)
            used.add(ident)
            persons[key] = (t, meta)
            genders[key] = g
    for m in SLOT_RE.finditer(template):
        text += template[pos:m.start()]
        slot = m.group(1)
        if slot.startswith("g:"):
            male, female = slot[2:].split("|")
            text += male if genders["P"] == "masc" else female
        elif slot.split(":")[0] in persons:
            val, meta = persons[slot.split(":")[0]]
            start = len(text)
            text += val
            spans.append({"start": start, "stop": len(text), "type": "PERSON", "meta": meta})
        else:
            base = slot.rstrip("0123456789")
            etype, fn = VALUE_GEN[base]
            val = fn(rng)
            start = len(text)
            text += val
            spans.append({"start": start, "stop": len(text), "type": etype, "meta": {}})
        pos = m.end()
    text += template[pos:]
    return text, spans


def build_neg(template, rng):
    pools = {"city": CITIES, "street": STREETS, "org": ORGS, "noun": NOUNS}
    return SLOT_RE.sub(lambda m: rng.choice(pools[m.group(1)]), template)


# ══════════════════════════════════════════════════════════════════════════
#  ПРОВЕРКИ
# ══════════════════════════════════════════════════════════════════════════
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


WORD = re.compile(r"[а-яёa-z0-9]+", re.I)


def frame_of_example(ex):
    t, out, pos = ex["text"], [], 0
    for s in sorted(ex["spans"], key=lambda s: s["start"]):
        out.append(t[pos:s["start"]])
        out.append(" <X> ")
        pos = s["stop"]
    out.append(t[pos:])
    return norm_frame("".join(out))


def norm_frame(s):
    return " ".join(s.lower().replace("ё", "е").split())


def frames_of_template(tpl):
    """Все варианты рамки шаблона: слоты -> <X>, {g:..} в обоих родах."""
    variants = set()
    for gi in (0, 1):
        def sub(m, gi=gi):
            slot = m.group(1)
            if slot.startswith("g:"):
                return slot[2:].split("|")[gi]
            return " <X> "
        variants.add(norm_frame(SLOT_RE.sub(sub, tpl)))
    return variants


def jaccard(a, b):
    A = set(WORD.findall(a.replace("<x>", ""))) - {"x"}
    B = set(WORD.findall(b.replace("<x>", ""))) - {"x"}
    return len(A & B) / len(A | B) if A | B else 0.0


def main():
    rng = random.Random(SEED)
    failures = []

    # ── обучающая сторона ────────────────────────────────────────────────
    gen = load_module("generate_dataset", os.path.join(TRAIN_DIR, "generate_dataset.py"))
    aug = load_module("augment_negatives", os.path.join(TRAIN_DIR, "augment_negatives.py"))
    train_files = ["train/train.jsonl", "dev/dev.jsonl", "test/test.jsonl",
                   "train/train_negatives.jsonl", "dev/dev_negatives.jsonl"]
    train_rows = []
    for rel in train_files:
        train_rows += read_jsonl(os.path.join(TRAIN_DIR, rel))
    train_frames = {frame_of_example(r) for r in train_rows}
    src_templates = list(gen.TEMPLATES)
    for name in ("NEG_TEMPLATES_TOPO", "NEG_TEMPLATES_NOUN", "NEG_TEMPLATES_STREET",
                 "NEG_TEMPLATES_ORG", "PERSON_TOPO_TEMPLATES"):
        src_templates += list(getattr(aug, name))
    for t in src_templates:
        train_frames |= frames_of_template(t)

    # ── 1. словари не пересекаются с обучающими ───────────────────────────
    def low(xs):
        return {x.lower().replace("ё", "е") for x in xs}

    train_lex = {
        "first": low(gen.MALE_NAMES + gen.FEMALE_NAMES),
        "patronymic": low(gen.MALE_PATR + gen.FEMALE_PATR),
        "surname": low(gen.SURNAMES_M + [s + "а" for s in gen.SURNAMES_M]),
        "city": low(aug.TOPONYMS), "street": low(aug.STREETS),
        "org": low(aug.ORGS), "noun": low(aug.NOUNS),
    }
    test_lex = {
        "first": low(MALE_NAMES + FEMALE_NAMES),
        "patronymic": low([p for pair in PATRONYMICS for p in pair]),
        "surname": low([s for s, _ in SURNAMES]),
        "city": low(CITIES), "street": low(STREETS), "org": low(ORGS), "noun": low(NOUNS),
    }
    lex_overlap = {}
    for k in test_lex:
        inter = sorted(test_lex[k] & train_lex[k])
        lex_overlap[k] = inter
        if inter:
            failures.append(f"словарь «{k}» пересекается с обучающим: {inter}")
    # имя из теста не должно совпадать ни с одним обучающим именем/фамилией/отчеством
    all_train_person = train_lex["first"] | train_lex["patronymic"] | train_lex["surname"]
    all_test_person = test_lex["first"] | test_lex["patronymic"] | test_lex["surname"]
    cross = sorted(all_test_person & all_train_person)
    if cross:
        failures.append(f"элементы ФИО теста встречаются в обучающих словарях: {cross}")

    # эмпирически: слова внутри PERSON-спанов обучающих файлов
    train_person_words = set()
    for r in train_rows:
        for s in r["spans"]:
            if s["type"] == "PERSON":
                for w in WORD.findall(r["text"][s["start"]:s["stop"]].lower()):
                    if len(w) > 1:
                        train_person_words.add(w.replace("ё", "е"))

    # ── 2. сборка набора ─────────────────────────────────────────────────
    examples, n = [], 0
    for ti, tpl in enumerate(PII_TEMPLATES, 1):
        for _ in range(INSTANCES_PII):
            text, spans = build_pii(tpl, rng)
            n += 1
            examples.append({"id": f"v2-{n:04d}", "template_id": f"P{ti:03d}",
                             "subset": "pii", "text": text, "spans": spans})
    for ti, tpl in enumerate(NEG_TEMPLATES, 1):
        seen_txt = set()
        for _ in range(INSTANCES_NEG_SLOT):
            text = build_neg(tpl, rng)
            if text in seen_txt:
                continue
            seen_txt.add(text)
            n += 1
            examples.append({"id": f"v2-{n:04d}", "template_id": f"N{ti:03d}",
                             "subset": "neg", "text": text, "spans": []})
    for fi, text in enumerate(NEG_FIXED, 1):
        n += 1
        examples.append({"id": f"v2-{n:04d}", "template_id": f"F{fi:03d}",
                         "subset": "neg", "text": text, "spans": []})

    # ── 3. целостность разметки ───────────────────────────────────────────
    for ex in examples:
        prev_end = -1
        for s in sorted(ex["spans"], key=lambda s: s["start"]):
            val = ex["text"][s["start"]:s["stop"]]
            if not val or val != val.strip():
                failures.append(f"{ex['id']}: пустой спан или пробел на краю: {val!r}")
            if s["start"] < prev_end:
                failures.append(f"{ex['id']}: пересекающиеся спаны")
            prev_end = s["stop"]
        if "{" in ex["text"] or "}" in ex["text"]:
            failures.append(f"{ex['id']}: незаполненный слот: {ex['text']!r}")
        leaked_words = []
        for s in ex["spans"]:
            if s["type"] == "PERSON":
                for w in WORD.findall(ex["text"][s["start"]:s["stop"]].lower()):
                    w = w.replace("ё", "е")
                    if len(w) > 1 and w in train_person_words:
                        leaked_words.append(w)
        if leaked_words:
            failures.append(f"{ex['id']}: слова ФИО встречались в обучающих спанах: {leaked_words}")

    # в негативах нет ни одного элемента ФИО из словарей (ни тестовых, ни обучающих)
    all_names = all_test_person | all_train_person
    for ex in examples:
        if ex["subset"] == "neg":
            hit = [w for w in WORD.findall(ex["text"].lower().replace("ё", "е")) if w in all_names]
            if hit:
                failures.append(f"{ex['id']}: в негативе слово из словаря ФИО: {hit}")

    # ── 4. шаблоны не совпадают с обучающими ──────────────────────────────
    exact, sims = [], []
    for kind, tpls in (("P", PII_TEMPLATES), ("N", NEG_TEMPLATES)):
        for ti, tpl in enumerate(tpls, 1):
            fr = frames_of_template(tpl)
            if fr & train_frames:
                exact.append(f"{kind}{ti:03d}")
            best = max((jaccard(a, b), b) for a in fr for b in train_frames)
            sims.append({"template_id": f"{kind}{ti:03d}", "max_jaccard": round(best[0], 3),
                         "closest_training_frame": best[1]})
    for fi, text in enumerate(NEG_FIXED, 1):
        fr = norm_frame(text)
        if fr in train_frames:
            exact.append(f"F{fi:03d}")
        best = max((jaccard(fr, b), b) for b in train_frames)
        sims.append({"template_id": f"F{fi:03d}", "max_jaccard": round(best[0], 3),
                     "closest_training_frame": best[1]})
    if exact:
        failures.append(f"шаблоны совпадают с обучающими рамками: {exact}")
    if len({t for t in PII_TEMPLATES}) != len(PII_TEMPLATES):
        failures.append("в PII_TEMPLATES есть дубликаты")

    # ── 5. отчёт ─────────────────────────────────────────────────────────
    pii = [e for e in examples if e["subset"] == "pii"]
    ents = [s for e in pii for s in e["spans"]]
    persons = [s for s in ents if s["type"] == "PERSON"]
    report = {
        "seed": SEED,
        "examples_total": len(examples),
        "examples_pii": len(pii),
        "examples_neg": len(examples) - len(pii),
        "templates": {"pii": len(PII_TEMPLATES), "neg_slot": len(NEG_TEMPLATES),
                      "neg_fixed": len(NEG_FIXED)},
        "entities_by_type": dict(Counter(s["type"] for s in ents)),
        "person_by_format": dict(Counter(s["meta"]["format"] for s in persons)),
        "person_by_case": dict(Counter(s["meta"]["case"] for s in persons)),
        "person_by_surname_class": dict(Counter(str(s["meta"]["surname_class"]) for s in persons)),
        "person_lowercase": sum(s["meta"]["lower"] for s in persons),
        "person_format_unseen_in_training": sum(not s["meta"]["format_seen_in_training"]
                                                for s in persons),
        "checks": {
            "lexicon_overlap_with_training": lex_overlap,
            "exact_template_matches_with_training": exact,
            "max_jaccard_over_templates": max(x["max_jaccard"] for x in sims),
            "templates_with_jaccard_ge_0_5": [x for x in sims if x["max_jaccard"] >= 0.5],
            "training_files_compared": train_files,
            "training_template_sources": ["generate_dataset.TEMPLATES",
                                          "augment_negatives.*_TEMPLATES"],
        },
        "failures": failures,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if failures:
        print("ПРОВЕРКИ НЕ ПРОЙДЕНЫ — набор не записан:")
        for x in failures[:40]:
            print("  -", x)
        sys.exit(1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()
    report["sha256_test_v2"] = sha
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"},
                     ensure_ascii=False, indent=2))
    print("max Jaccard с обучающими рамками:", report["checks"]["max_jaccard_over_templates"])
    print("sha256:", sha)


if __name__ == "__main__":
    main()
