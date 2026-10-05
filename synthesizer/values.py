"""
Подстановка персональных данных и трудных негативов: разметка точная по построению.

Языковая модель пишет только текст с метками и условными людьми. Значения и их позиции
ставит этот модуль, поэтому разметку невозможно получить сдвинутой — в отличие от
разметки, которую рисует сама модель.

Люди. Модель пишет условных людей (Иван Иванович Петров, Анна Сергеевна Петрова, Олег
Олегович Сидоров, Мария Павловна Сидорова) и склоняет их сама — склоняет она хорошо.
Метки с падежом ([ФИО:дат:ж]) она ставила неверно («Здравствуйте, [ФИО:род:ж]»), поэтому
от них отказались. Код находит каждую форму условного имени, по ней определяет падеж и
род и подставляет случайного человека из лексикона в том же падеже и формате; один
условный человек в тексте — одно и то же подставленное имя.

Номера и адреса — по меткам: [ТЕЛЕФОН] [EMAIL] [ИНН] [СНИЛС] [ПАСПОРТ] [КАРТА] [АДРЕС]
[ДАТА_РОЖДЕНИЯ]. Значения берутся из обучающих лексиконов data/training_v3/generate_v3.py
(проверены на непересечение с отложенными наборами) и генераторов build_test_v2.py
(контрольные суммы, «грязные» форматы).

Трудные негативы — [УЛИЦА] [ОРГАНИЗАЦИЯ] [ЗАКАЗ] — персональными данными НЕ являются;
их позиции возвращаются отдельным списком, в разметку ПДн они не попадают.
"""
import importlib.util
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "gen_v3", os.path.join(ROOT, "data", "training_v3", "generate_v3.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
B = G.B                                  # build_test_v2: генераторы номеров

CASE = {"им": "nomn", "род": "gent", "дат": "datv", "вин": "accs", "твор": "ablt", "пред": "loct"}
GENDER = {"м": "masc", "ж": "femn"}
# Метку модель пишет и не заглавными ([Имя], [Телефон]) — регистр приводим сами
TAG_RE = re.compile(r"\[([А-ЯЁа-яёA-Za-z_]+)(?::([а-яё]+))?(?::([мж]))?\]")
PERSON_TAGS = {"ФИО": "full", "ИМЯ": "first", "ФАМИЛИЯ": "surname"}
OTHER_TAGS = {"ТЕЛЕФОН": "PHONE_NUMBER", "EMAIL": "EMAIL_ADDRESS", "ИНН": "INN", "СНИЛС": "SNILS",
              "ПАСПОРТ": "PASSPORT", "КАРТА": "CREDIT_CARD", "АДРЕС": "ADDRESS",
              "ДАТА_РОЖДЕНИЯ": "DATE_OF_BIRTH"}

# Трудные негативы — подставляет код, в разметку ПДн НЕ попадают (см. шапку файла).
NEG_TAGS = ("УЛИЦА", "ОРГАНИЗАЦИЯ", "ЗАКАЗ")
# Улицы, названные по фамилии: именно на них сервис ставит ложное ФИО. Остальные
# («Садовая», «Мира») для негатива слабые — берутся реже, ради правдоподобия текста.
STREETS_BY_PERSON = """Гагарина Пушкина Кирова Лермонтова Чапаева Жукова Толстого Чехова Шевченко
Королёва Циолковского Гончарова Есенина Островского Шолохова Менделеева Павлова
Пархоменко Щорса""".split()

# ── значения ПДн ────────────────────────────────────────────────────────────
def person(rng, kind, case, gender):
    gender = gender or rng.choice(["masc", "femn"])
    first = rng.choice(G.MALE if gender == "masc" else G.FEMALE)
    patr = rng.choice(G.PATR)[0 if gender == "masc" else 1]
    cls = rng.choices(list(G.SURN), weights=[45, 12, 18, 18, 7])[0]
    sur = rng.choice(G.SURN[cls])
    f, p, s = (G.decline_first(first, gender, case), G.decline_patr(patr, gender, case),
               G.decline_surname(sur, cls, gender, case))
    if kind == "first":
        return f
    if kind == "surname":
        return s
    ini = f"{first[0]}.{patr[0]}."
    return rng.choice([f"{f} {p} {s}", f"{s} {f} {p}", f"{f} {s}", f"{s} {f}", f"{s} {ini}", f"{ini} {s}"])


def address(rng):
    city, street = rng.choice(G.CITIES), rng.choice(G.STREETS)
    house, apt = rng.randint(1, 200), rng.randint(1, 300)
    return rng.choice([f"г. {city}, ул. {street}, д. {house}, кв. {apt}", f"{city}, {street} {house}, кв {apt}",
                       f"ул. {street}, {house}-{apt}", f"пр. {street} {house}, кв. {apt}"])


def value(rng, tag):
    if tag == "ТЕЛЕФОН":
        return B.gen_phone(rng)
    if tag == "EMAIL":
        return B.gen_email(rng)
    if tag == "ИНН":
        return B.gen_inn(rng)
    if tag == "СНИЛС":
        return B.gen_snils(rng)
    if tag == "ПАСПОРТ":
        return B.gen_passport(rng)
    if tag == "КАРТА":
        return B.gen_card(rng)
    if tag == "ДАТА_РОЖДЕНИЯ":
        return B.gen_dob(rng)
    return address(rng)


def neg_value(rng, tag):
    """Трудный негатив: похоже на ПДн, но ПДн не является.

    Формы выбраны так, чтобы подходить к любому предлогу: сокращение («ул. Гагарина»),
    название в кавычках, номер после знака №. Падеж вокруг метки модель не угадывает,
    а эти формы в падеже не нуждаются.
    """
    if tag == "УЛИЦА":                      # чаще — улица, названная по фамилии
        kind = rng.choices(["ул.", "пр.", "пер.", "бул."], weights=[70, 15, 10, 5])[0]
        pool = STREETS_BY_PERSON if rng.random() < 0.8 else G.STREETS
        return f"{kind} {rng.choice(pool)}"
    if tag == "ОРГАНИЗАЦИЯ":                # компания по фамилии, известный бренд или название-имя
        r = rng.random()
        if r < 0.45:
            cls = rng.choices(list(G.SURN), weights=[45, 12, 18, 18, 7])[0]
            sur = rng.choice(G.SURN[cls])
            form = rng.choices(["ООО", "АО", "ЗАО"], weights=[70, 20, 10])[0]
            return f"{form} " + rng.choice([f"«{sur} и партнёры»", f"«{sur}»", f"«{sur}-Сервис»",
                                            f"«{sur} и сыновья»"])
        if r < 0.8:
            return rng.choices(["ООО", "АО", "ЗАО"], weights=[70, 20, 10])[0] + f" «{rng.choice(G.ORGS)}»"
        return rng.choice(["магазин", "сервис", "тариф", "программа"]) + f" «{rng.choice(G.BRANDS)}»"
    letter = rng.choice("ABCDEFKMNP")       # номер заказа: цифры не должны стать паспортом или ИНН
    return rng.choice([f"№ {rng.randint(100000, 99999999)}", f"№ {letter}-{rng.randint(10000, 99999)}",
                       f"№ 2026/{rng.randint(1000, 9999)}", f"№ {rng.randint(10, 99)}-{rng.randint(100, 999)}/26"])


# Условные люди: (имя, отчество, фамилия, класс фамилии, род). Все формы строятся теми же
# функциями склонения, что и подставляемые имена, — падеж определяется точно.
DUMMIES = [("Иван", "Иванович", "Петров", "ov", "masc"), ("Анна", "Сергеевна", "Петров", "ov", "femn"),
           ("Олег", "Олегович", "Сидоров", "ov", "masc"), ("Мария", "Павловна", "Сидоров", "ov", "femn")]


def _dummy_forms():
    """Словоформа -> все варианты прочтения [(человек, часть, падеж)].

    Формы неоднозначны: «Петрова» — и «Анна Петрова» (им.), и «Ивана Петрова» (род.);
    выбор делается по соседним словам и по остальному тексту (_person_runs).
    """
    forms = {}
    for i, (f, p, s, cls, g) in enumerate(DUMMIES):
        for c in G.CASES:
            for part, w in (("f", G.decline_first(f, g, c)), ("p", G.decline_patr(p, g, c)),
                            ("s", G.decline_surname(s, cls, g, c))):
                if (i, part, c) not in forms.setdefault(w, []):
                    forms[w].append((i, part, c))
        for part, w in (("fi", f[0] + "."), ("pi", p[0] + ".")):
            forms.setdefault(w, []).append((i, part, None))
    return forms


_FORMS = _dummy_forms()
# Слово с заглавной, целиком заглавными, инициал или слово строчными — последнее потому,
# что в разговорном стиле модель пишет «анна, помоги», и такое имя тоже надо подставить.
# Подбор идёт по словарю форм условных людей, а среди них нарицательных слов нет.
# Инициал стоит первым и без замыкающей проверки: в «И.И.» второй инициал идёт сразу за
# точкой, и общая проверка «дальше не буква» его отбрасывала — склеенные инициалы
# оставались неподставленными.
_WORD_OR_INI = re.compile(
    r"(?<![\w-])(?:[А-ЯЁа-яё]\.|(?:[А-ЯЁ][а-яё]+|[А-ЯЁ]{2,}|[а-яё]{3,})(?![\w-]))")


def _person_runs(text):
    """Цепочки форм условного имени: [(start, end, [(часть, падеж, слово)...], человек)]."""
    toks = []
    for m in _WORD_OR_INI.finditer(text):
        w = m.group(0)
        cands = _FORMS.get(w) or _FORMS.get(w.capitalize())
        if cands:
            toks.append((m.start(), m.end(), w, cands))
    groups, cur = [], []
    for t in toks:                               # соседние через пробелы — одна цепочка
        if cur and text[cur[-1][1]:t[0]].strip() == "":
            cur.append(t)
        else:
            if cur:
                groups.append(cur)
            cur = [t]
    if cur:
        groups.append(cur)
    # люди, названные однозначно (имя или отчество — у них один человек)
    certain = {c[0][0] for g in groups for t in g for c in [t[3]] if len({x[0] for x in c}) == 1
               and c[0][1] in ("f", "p")}
    runs = []
    for g in groups:
        common = set.intersection(*[{x[0] for x in t[3]} for t in g])
        if not common:                           # разные люди подряд — берём по одному слову
            for t in g:
                runs.append(_pick([t], {x[0] for x in t[3]}, certain))
            continue
        runs.append(_pick(g, common, certain))
    return [r for r in runs if r and any(part in ("f", "p", "s") for part, _, _ in r[2])]


def _pick(g, candidates, certain):
    who = next((w for w in sorted(candidates) if w in certain), min(candidates))
    parts = []
    for _start, _end, w, cands in g:
        opts = [x for x in cands if x[0] == who]
        part, case = opts[0][1], opts[0][2]
        parts.append((part, case, w))
    # падеж цепочки — общий для её слов: у первого слова может быть несколько вариантов
    return [g[0][0], g[-1][1], parts, who]


def _replacement(rng, parts, gender):
    """Случайный человек из лексикона в тех же падежах и формате, что условный."""
    first = rng.choice(G.MALE if gender == "masc" else G.FEMALE)
    patr = rng.choice(G.PATR)[0 if gender == "masc" else 1]
    cls = rng.choices(list(G.SURN), weights=[45, 12, 18, 18, 7])[0]
    sur = rng.choice(G.SURN[cls])
    case = next((c for _, c, _ in parts if c), "nomn")
    out = []
    for part, c, word in parts:
        c = c or case
        if part == "f":
            val = G.decline_first(first, gender, c)
        elif part == "p":
            val = G.decline_patr(patr, gender, c)
        elif part == "s":
            val = G.decline_surname(sur, cls, gender, c)
        else:
            val = (first if part == "fi" else patr)[0] + "."
        if word.isupper() and len(word) > 2:
            val = val.upper()
        elif word.islower():                # «анна» -> «юлия», а не «Юлия»: регистр как у модели
            val = val.lower()
        out.append(val)
    return " ".join(out)


def fill(template, rng):
    """Условные имена и метки -> значения. Возвращает (текст, спаны, негативы, ошибки)."""
    people = {}
    for start, end, parts, who in sorted(_person_runs(template), reverse=True):
        gender = DUMMIES[who][4]
        rng_person = people.setdefault(who, random.Random(rng.random()))
        state = rng_person.getstate()
        val = _replacement(rng_person, parts, gender)
        rng_person.setstate(state)          # один и тот же человек — одно и то же имя
        template = template[:start] + "\x00P" + val + "\x00" + template[end:]
    return _fill_tags(template, rng)


def _fill_tags(template, rng):
    """Метки -> значения. Возвращает (текст, спаны, негативы, ошибки)."""
    out, spans, negs, errors, pos = "", [], [], [], 0
    rx = re.compile(r"\x00P([^\x00]*)\x00|" + TAG_RE.pattern)
    for m in rx.finditer(template):
        if m.group(0).startswith("\x00"):
            out += template[pos:m.start()]
            spans.append({"start": len(out), "stop": len(out) + len(m.group(1)), "type": "PERSON"})
            out += m.group(1)
            pos = m.end()
            continue
        out += template[pos:m.start()]
        tag, case, gender = m.group(2).upper(), m.group(3), m.group(4)   # группы метки в шаблоне
        if tag in PERSON_TAGS:
            if case not in CASE:
                # падеж неизвестен — любая подстановка даст неверную грамматику вокруг
                # имени, а это учит модель ФИО неправильным согласованиям
                errors.append(f"метка человека без падежа: {m.group(0)}")
                case = "им"
            val, typ = person(rng, PERSON_TAGS[tag], CASE[case], GENDER.get(gender)), "PERSON"
        elif tag in OTHER_TAGS:
            val, typ = value(rng, tag), OTHER_TAGS[tag]
            if isinstance(val, tuple):
                val = val[0]
        elif tag in NEG_TAGS:
            val, typ = neg_value(rng, tag), None
            negs.append({"start": len(out), "stop": len(out) + len(val), "kind": tag})
        else:
            errors.append(f"неизвестная метка {m.group(0)}")
            val, typ = m.group(0), None
        if typ:
            spans.append({"start": len(out), "stop": len(out) + len(val), "type": typ})
        out += val
        pos = m.end()
    out += template[pos:]
    return out, spans, negs, errors
