"""
Синтезатор обучающих данных: локальная языковая модель пишет текст с метками,
наш код подставляет ПДн и строит разметку (задача 3а в БУДУЩИЕ_ПЛАНЫ.md).

Модель — через Ollama (по умолчанию qwen3:8b, Apache-2.0), всё локально и бесплатно:
ни один текст не покидает машину.

Люди. Модель пишет только условных людей (Иван Иванович Петров, Анна Сергеевна Петрова,
Олег Олегович Сидоров, Мария Павловна Сидорова) и склоняет их сама — склоняет она хорошо.
Метки с падежом ([ФИО:дат:ж]) модель ставила неверно («Здравствуйте, [ФИО:род:ж]»), поэтому
от них отказались. Код находит каждую форму условного имени, по ней — падеж и род, и
подставляет случайного человека из лексикона в том же падеже и формате; один условный
человек в тексте — одно и то же подставленное имя.

Номера и адреса — метки: [ТЕЛЕФОН] [EMAIL] [ИНН] [СНИЛС] [ПАСПОРТ] [КАРТА] [АДРЕС]
[ДАТА_РОЖДЕНИЯ]. Метки [ФИО:падеж:род] по-прежнему разбираются, если модель их поставит.
Значения подставляются из обучающих лексиконов data/training_v3/generate_v3.py (они
проверены на непересечение с отложенными наборами), позиции известны по построению —
разметка точная.

Что добавлено в версии 2 (по итогам пилота на 50 текстах, 21.09.2026):

  * Своё задание на каждый текст. В пилоте один сценарий выдавался на все пять текстов
    запроса, и модель писала пять похожих обращений. Теперь вид, стиль, длина, набор ПДн
    и негативы выбираются для каждого текста отдельно и перечисляются в задании списком.

  * Опечатки делает код, а не модель. Стиль «с опечатками» модель в пилоте игнорировала —
    писала чисто. Теперь порчу вносит corrupt(): пропуск и удвоение буквы, перестановка
    соседних, соседняя клавиша, «ё» -> «е», потерянная запятая, строчная после точки.
    Порча идёт только ВНЕ подставленных значений, поэтому разметка остаётся точной.
    Отдельно — перевод всего текста в нижний регистр (длины не меняются, спаны те же):
    это целевой случай, на котором модель ФИО слабее всего (MultiCoNER — 22 % скрыто
    не полностью).

  * Трудные негативы ставит код: метки [УЛИЦА] [ОРГАНИЗАЦИЯ] [ЗАКАЗ] -> «ул. Гагарина»,
    «ООО «Коваль и партнёры»», «№ 4815162342». Это НЕ персональные данные; их позиции
    пишутся в negative_spans. Если сервис находит на них PERSON — это его ложное
    срабатывание: текст в брак НЕ уходит (он правильный), а случай попадает в отчёт,
    поле person_fp_on_negatives. Маски других типов на негативах не считаются ошибкой:
    «ул. Гагарина» сервис по политике закрывает как адрес, и это ожидаемо.

  * Проверка на почти-дубли: Жаккар по тройкам слов против всех принятых текстов
    (обратный индекс по тройкам, порог --dup-threshold). Пилот ловил только полные дубли.

Проверки каждого текста (результат — в поле checks, брак — reject=True):
  unknown_tag      — метка не из списка;
  leftover         — после подстановки остались скобки/метки;
  duplicate        — такой же текст (с точностью до меток) уже был;
  near_duplicate   — Жаккар с принятым текстом выше порога;
  extra_pii        — сервис нашёл ПДн ВНЕ подставленных значений и вне негативов: модель
                     могла вписать имя или номер сама — это неразмеченные ПДн, текст
                     уходит в брак (или это ложное срабатывание сервиса — видно при
                     ручной проверке);
  no_tags          — текст без меток там, где сценарий их требовал;
  no_person        — сценарий просил человека, а в тексте его нет.

Синтетика — для обучения и поиска ошибок, НЕ для приёмки: проверка на синтетике того же
происхождения завышает оценку (вывод статьи, раздел 2).

Запуск:
    .venv/bin/python data/synth/synth_llm.py --n 50 --out data/synth/pilot_v2.jsonl
    .venv/bin/python data/synth/review.py data/synth/pilot_v2.jsonl > pilot_v2_review.txt
"""
import argparse
import difflib
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")

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

DOMAINS = ["банк", "мобильная связь и интернет", "доставка и интернет-магазин", "ЖКХ и управляющая компания",
           "медицинская клиника", "госуслуги и МФЦ", "страхование", "автосервис", "HR и подбор персонала",
           "аренда жилья", "авиакомпания и билеты", "онлайн-школа"]
CHANNELS = ["сообщение в чате поддержки", "электронное письмо", "расшифровка телефонного звонка",
            "заявка через форму на сайте", "сообщение в мессенджере"]
STYLES = ["официальный", "разговорный", "раздражённый, клиент недоволен",
          "краткий, почти без знаков препинания", "вежливый и подробный",
          "сбивчивый, с повторами и уточнениями"]
LENGTHS = ["одно предложение", "два-три предложения", "четыре-шесть предложений"]
PII_SETS = [["ФИО"], ["ФИО", "ТЕЛЕФОН"], ["ИМЯ", "EMAIL"], ["ФАМИЛИЯ", "ПАСПОРТ"], ["ФИО", "АДРЕС"],
            ["ФИО", "ИНН", "СНИЛС"], ["ИМЯ", "ФАМИЛИЯ", "ТЕЛЕФОН"], ["ФИО", "КАРТА"],
            ["ФИО", "ДАТА_РОЖДЕНИЯ", "АДРЕС"], ["два человека: ФИО и ФИО"]]

# Трудные негативы — подставляет код, в разметку ПДн НЕ попадают (см. шапку файла).
NEG_TAGS = ("УЛИЦА", "ОРГАНИЗАЦИЯ", "ЗАКАЗ")
# Улицы, названные по фамилии: именно на них сервис ставит ложное ФИО. Остальные
# («Садовая», «Мира») для негатива слабые — берутся реже, ради правдоподобия текста.
STREETS_BY_PERSON = """Гагарина Пушкина Кирова Лермонтова Чапаева Жукова Толстого Чехова Шевченко
Королёва Циолковского Гончарова Есенина Островского Шолохова Менделеева Павлова
Пархоменко Щорса""".split()
NEG_HINT = {"УЛИЦА": "упомяни улицу — поставь метку [УЛИЦА] («живу на [УЛИЦА]», «офис на [УЛИЦА]»)",
            "ОРГАНИЗАЦИЯ": "упомяни компанию — поставь метку [ОРГАНИЗАЦИЯ] («договор с [ОРГАНИЗАЦИЯ]»)",
            "ЗАКАЗ": "упомяни номер заказа или договора — поставь метку [ЗАКАЗ] («заказ [ЗАКАЗ]»)"}

PROMPT = """Ты помогаешь готовить обучающие данные для системы обезличивания текстов.
Напиши {k} разных реалистичных текста на русском языке. Сфера у всех одна: {domain}.
Автор — клиент (не оператор), если вид текста это допускает.

Требования к каждому тексту:
{specs}

Правила:
1. Людей называй ТОЛЬКО условными именами и пиши их СЛОВАМИ, склоняя по правилам
   русского языка: мужчина — Иван Иванович Петров, женщина — Анна Сергеевна Петрова,
   второй человек в тексте — Олег Олегович Сидоров или Мария Павловна Сидорова.
   Можно писать полностью, только имя, только фамилию, «Петрову Анне», «А. С. Петрова»,
   «Петров И. И.». Никаких других имён и фамилий не используй.
   Метки для людей НЕ существуют: [ИМЯ], [ФИО], [Имя] писать нельзя — только слова.
2. Телефоны, почту, номера документов, адреса НЕ пиши — вместо них ставь метки:
   [ТЕЛЕФОН] [EMAIL] [ИНН] [СНИЛС] [ПАСПОРТ] [КАРТА] [АДРЕС] [ДАТА_РОЖДЕНИЯ].
   Других меток нет. Конкретные даты и числа не пиши: сроки указывай словами —
   «на прошлой неделе», «в прошлый вторник», «второй месяц».
3. Метки [УЛИЦА] [ОРГАНИЗАЦИЯ] [ЗАКАЗ] — это не персональные данные, значения подставит
   наш код. Ставь их только там, где этого просит требование, и пиши вокруг предлоги так,
   чтобы фраза читалась.

В ответе — только готовые тексты обращений. Условия и их формулировки не повторяй,
заголовки не пиши. Тексты должны отличаться друг от друга: разное начало, разное
построение фраз, разная лексика; у каждого свой повод обращения."""

SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
          "required": ["texts"]}


def scenario(rng, neg_share):
    """Сценарий одного текста. Пустой набор ПДн — текст только из трудных негативов."""
    only_neg = rng.random() < neg_share
    neg = list(rng.sample(NEG_TAGS, rng.randint(2, 3) if only_neg else
                          rng.choices([0, 1, 2], weights=[55, 35, 10])[0]))
    return {"channel": rng.choice(CHANNELS), "style": rng.choice(STYLES), "length": rng.choice(LENGTHS),
            "pii": [] if only_neg else rng.choice(PII_SETS), "neg": neg,
            "lower": rng.random() < 0.12, "typos": 0}
def wants_person(sc):
    """Требует ли сценарий человека в тексте.

    Модель иногда пишет обращение вообще без имени. Такой текст сам по себе верный, но
    сдвигает состав набора: реже всего она упоминает человека там, где просят только
    фамилию или только имя, а это как раз форматы, на которых модель ФИО слабее всего.
    """
    return bool(set(sc["pii"]) & set(PERSON_TAGS)) or sc["pii"] == ["два человека: ФИО и ФИО"]


def spec_line(i, sc):
    """Задание на один текст — строка нумерованного списка в запросе."""
    parts = [f"вид: {sc['channel']}", f"стиль: {sc['style']}", f"длина: {sc['length']}"]
    pii = sc["pii"]
    if not pii:
        parts.append("персональных данных и людей в тексте НЕТ")
    elif pii == ["два человека: ФИО и ФИО"]:
        parts.append("упомяни двух разных людей (Петров/Петрова и Сидоров/Сидорова)")
    else:
        person = [p for p in pii if p in PERSON_TAGS]
        other = [f"[{p}]" for p in pii if p in OTHER_TAGS]
        if person:
            parts.append("упомяни условного человека" + (" только по имени" if person == ["ИМЯ"] else
                         " только по фамилии" if person == ["ФАМИЛИЯ"] else ""))
        if other:
            parts.append("используй метки " + ", ".join(other))
    parts += [NEG_HINT[t] for t in sc["neg"]]
    return f"Текст {i} — " + "; ".join(parts) + "."


def ask(model, prompt, timeout=900):
    body = {"model": model, "prompt": prompt, "stream": False, "format": SCHEMA, "think": False,
            "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 900}}
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return json.loads(d["response"]).get("texts", []), {
        "tokens": d.get("eval_count"), "seconds": round(time.time() - t0, 1)}


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
# слово с заглавной (или целиком заглавными) либо инициал; не часть слова через дефис («Иван-чай»)
_WORD_OR_INI = re.compile(r"(?<![\w-])(?:[А-ЯЁ][а-яё]+|[А-ЯЁ]{2,}|[А-ЯЁ]\.)(?![\w-])")


def _person_runs(text):
    """Цепочки форм условного имени: [(start, end, [(часть, падеж, слово)...], человек)]."""
    toks = []
    for m in _WORD_OR_INI.finditer(text):
        w = m.group(0)
        cands = _FORMS.get(w.capitalize() if w.isupper() and len(w) > 2 else w)
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


# ── опечатки и почти-дубли ──────────────────────────────────────────────────
# соседние клавиши русской раскладки — самая частая опечатка при быстром наборе
_KEY = {"й": "цф", "ц": "йув", "у": "цка", "к": "уев", "е": "кнп", "н": "егр", "г": "ншо", "ш": "гщл",
        "щ": "шзд", "з": "щхж", "х": "зэж", "ф": "йыв", "ы": "фвс", "в": "ыам", "а": "впр", "п": "аро",
        "р": "пол", "о": "рлд", "л": "одж", "д": "лжэ", "ж": "дэх", "э": "жх", "я": "чс", "ч": "ясм",
        "с": "чми", "м": "сит", "и": "мтб", "т": "ибю", "ь": "тбю", "б": "ью", "ю": "б"}


def _apply(op, seg, rng):
    """Одна порча. None — к этому отрезку операция не применима."""
    letters = [i for i, c in enumerate(seg) if c.isalpha()]
    if op == "comma":
        pos = [i for i, c in enumerate(seg) if c == ","]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + seg[i + 1:]
    if op == "yo":
        pos = [i for i, c in enumerate(seg) if c in "ёЁ"]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + ("е" if seg[i] == "ё" else "Е") + seg[i + 1:]
    if op == "case":                       # строчная буква после точки
        pos = [m.end() - 1 for m in re.finditer(r"[.!?]\s+[А-ЯЁ]", seg)]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + seg[i].lower() + seg[i + 1:]
    if not letters:
        return None
    if op == "drop":
        i = rng.choice(letters)
        return seg[:i] + seg[i + 1:]
    if op == "double":
        i = rng.choice(letters)
        return seg[:i] + seg[i] * 2 + seg[i + 1:]
    if op == "swap":
        pairs = [i for i in letters if i + 1 < len(seg) and seg[i + 1].isalpha()]
        if not pairs:
            return None
        i = rng.choice(pairs)
        return seg[:i] + seg[i + 1] + seg[i] + seg[i + 2:]
    near = [i for i in letters if seg[i].lower() in _KEY]       # op == "key"
    if not near:
        return None
    i = rng.choice(near)
    c = rng.choice(_KEY[seg[i].lower()])
    return seg[:i] + (c.upper() if seg[i].isupper() else c) + seg[i + 1:]


def _damaged(orig, new):
    """Диапазоны изменённых слов в испорченном отрезке.

    Нужны, чтобы отличать два случая: сервис нашёл «ПДн» в слове, которое испортили мы
    (это его ложное срабатывание на опечатке — ценный обучающий пример, текст годный),
    и сервис нашёл ПДн, которые вписала сама модель (неразмеченные ПДн — текст в брак).
    """
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=orig, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        a, b = j1, max(j2, j1 + 1)
        while a > 0 and (new[a - 1].isalnum() or new[a - 1] == "-"):
            a -= 1
        while b < len(new) and (new[b].isalnum() or new[b] == "-"):
            b += 1
        out.append((a, min(b, len(new))))
    return out


def _typos(seg, rng, n):
    """n порч в одном отрезке; возвращает (отрезок, сколько внесено)."""
    done = 0
    for _ in range(n):
        ops = ["drop", "swap", "double", "key", "yo", "comma", "case"]
        rng.shuffle(ops)
        for op in ops:
            new = _apply(op, seg, rng)
            if new is not None:
                seg, done = new, done + 1
                break
    return seg, done


def corrupt(text, spans, negs, rng, rate=1.0):
    """Опечатки ВНЕ подставленных значений -> (текст, спаны, негативы, испорченные слова).

    Значения ПДн и негативов не трогаем: их позиции должны остаться точными, а «грязные»
    форматы номеров и так делают генераторы build_test_v2. Позиции пересчитываются по
    длинам отрезков, поэтому разметка остаётся верной при любой порче.
    """
    items = sorted([("p", dict(x)) for x in spans] + [("n", dict(x)) for x in negs],
                   key=lambda it: it[1]["start"])
    pieces, pos = [], 0
    for _kind, it in items:
        pieces.append(["free", text[pos:it["start"]]])
        pieces.append(["keep", text[it["start"]:it["stop"]]])
        pos = it["stop"]
    pieces.append(["free", text[pos:]])
    for p in pieces:
        if p[0] == "free" and len(p[1]) > 8:
            orig = p[1]
            p[1], _k = _typos(orig, rng, max(1, round(len(orig) / 55 * rate)))
            p.append(_damaged(orig, p[1]))
    out, k, marks = "", 0, []
    for piece in pieces:
        kind, txt = piece[0], piece[1]
        if kind == "keep":
            items[k][1]["start"], items[k][1]["stop"] = len(out), len(out) + len(txt)
            k += 1
        else:
            marks += [(len(out) + a, len(out) + b) for a, b in (piece[2] if len(piece) > 2 else ())]
        out += txt
    return (out, [it for kind, it in items if kind == "p"],
            [it for kind, it in items if kind == "n"], marks)


_WORDS = re.compile(r"\w+")


class NearDup:
    """Почти-дубли: Жаккар по тройкам слов. Обратный индекс по тройкам — чтобы не
    сравнивать каждый новый текст со всеми принятыми (на тысячах текстов это заметно)."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.index, self.sets = {}, []

    @staticmethod
    def shingles(text):
        w = _WORDS.findall(text.lower())
        return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)} or set(w)

    def worst(self, text):
        """(худшее сходство с принятыми, набор тройек) — набор нужен для add()."""
        sh = self.shingles(text)
        hits = {}
        for g in sh:
            for j in self.index.get(g, ()):
                hits[j] = hits.get(j, 0) + 1
        best = 0.0
        for j, inter in hits.items():
            union = len(sh) + len(self.sets[j]) - inter
            if union:
                best = max(best, inter / union)
        return round(best, 3), sh

    def add(self, sh):
        j = len(self.sets)
        self.sets.append(sh)
        for g in sh:
            self.index.setdefault(g, []).append(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-request", type=int, default=3,
                    help="текстов за запрос; чем больше, тем хуже модель держит список требований")
    ap.add_argument("--out", default=os.path.join(HERE, "pilot_v2.jsonl"))
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--neg-share", type=float, default=0.25,
                    help="доля текстов только из трудных негативов, без ПДн")
    ap.add_argument("--typo-share", type=float, default=0.3, help="доля текстов с опечатками")
    ap.add_argument("--dup-threshold", type=float, default=0.6,
                    help="Жаккар по тройкам слов, выше которого текст считается почти-дублем")
    a = ap.parse_args()

    model_dir = os.path.join(ROOT, "models", "person_ruBERT")
    if os.path.isdir(model_dir):                 # с моделью ложные ФИО на негативах видны честно
        os.environ.setdefault("PERSON_MODEL_DIR", model_dir)
    else:
        os.environ.setdefault("ALLOW_NO_PERSON_MODEL", "true")
    from app.anonymizer import analyze_text

    rng = random.Random(a.seed)
    dup = NearDup(a.dup_threshold)
    rows, seen, stat = [], set(), {"requests": 0, "tokens": 0, "seconds": 0.0}
    while len(rows) < a.n:
        domain = rng.choice(DOMAINS)
        scens = [scenario(rng, a.neg_share) for _ in range(a.per_request)]
        prompt = PROMPT.format(k=a.per_request, domain=domain,
                               specs="\n".join(spec_line(i, sc) for i, sc in enumerate(scens, 1)))
        try:
            texts, info = ask(a.model, prompt)
        except Exception as exc:  # noqa: BLE001 — генерация фоновая: фиксируем и идём дальше
            print("ошибка запроса:", exc)
            continue
        stat["requests"] += 1
        stat["tokens"] += info["tokens"] or 0
        stat["seconds"] += info["seconds"]
        for tpl, sc in zip(texts, scens):
            sc = dict(sc, domain=domain)
            text, spans, negs, errors = fill(tpl, rng)
            if sc["lower"]:                      # длины не меняются — спаны остаются верными
                text = text.lower()
            marks = []
            if rng.random() < a.typo_share:
                text, spans, negs, marks = corrupt(text, spans, negs, rng)
                sc["typos"] = len(marks)
            key = " ".join(TAG_RE.sub("<X>", tpl).lower().split())
            sim, sh = dup.worst(text)
            checks = {"unknown_tag": errors, "leftover": bool(re.search(r"[\[\]{}]", text)),
                      "duplicate": key in seen, "near_duplicate": sim if sim >= a.dup_threshold else 0,
                      "no_tags": bool(sc["pii"]) and not spans,
                      "no_person": wants_person(sc) and not any(x["type"] == "PERSON" for x in spans)}
            seen.add(key)
            found = [r for r in analyze_text(text)
                     if not any(r.start < x["stop"] and x["start"] < r.end for x in spans)]
            # на негативах ошибкой считаем только ложное ФИО: «ул. Гагарина» сервис по
            # политике закрывает как адрес, и это ожидаемо, а не брак текста
            on_neg = [any(r.start < x["stop"] and x["start"] < r.end for x in negs) for r in found]
            on_typo = [any(r.start < b and x < r.end for x, b in marks) for r in found]
            # Всё, что вне spans, персональными данными НЕ является — набор синтетический.
            # В брак текст уходит только если ПДн вписала сама модель: тогда в обучающих
            # данных окажется неразмеченное имя или номер. Срабатывания на негативах и на
            # испорченных нами словах — ложные срабатывания сервиса, текст при этом годный.
            checks["extra_pii"] = [f"{r.entity_type}: {text[r.start:r.end]}"
                                   for r, neg, typo in zip(found, on_neg, on_typo)
                                   if not neg and not typo]
            person_fp = [text[r.start:r.end] for r, neg in zip(found, on_neg)
                         if neg and r.entity_type == "PERSON"]
            typo_fp = [text[r.start:r.end] for r, neg, typo in zip(found, on_neg, on_typo)
                       if typo and not neg and r.entity_type == "PERSON"]
            reject = bool(errors or checks["leftover"] or checks["duplicate"] or checks["near_duplicate"]
                          or checks["no_tags"] or checks["no_person"] or checks["extra_pii"])
            if not reject:
                dup.add(sh)
            rows.append({"text": text, "spans": spans, "negative_spans": negs, "template": tpl,
                         "scenario": sc, "model": a.model, "checks": checks, "similarity": sim,
                         "person_fp_on_negatives": person_fp,
                         "person_fp_on_typos": typo_fp, "reject": reject})
        ok = sum(not r["reject"] for r in rows)
        print(f"запросов {stat['requests']}, текстов {len(rows)} (годных {ok}), "
              f"{stat['tokens'] / max(stat['seconds'], 1e-9):.1f} ток/с", flush=True)
    rows = rows[:a.n]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rej = sum(r["reject"] for r in rows)
    why = {k: sum(bool(r["checks"][k]) for r in rows) for k in rows[0]["checks"]}
    good = [r for r in rows if not r["reject"]]
    neg_spans = sum(len(r["negative_spans"]) for r in rows)
    rep = {"model": a.model, "texts": len(rows), "rejected": rej, "reject_share": round(rej / len(rows), 3),
           "reasons": why, "requests": stat["requests"], "tokens": stat["tokens"],
           "minutes": round(stat["seconds"] / 60, 1),
           "tokens_per_sec": round(stat["tokens"] / max(stat["seconds"], 1e-9), 1),
           "sec_per_text": round(stat["seconds"] / max(len(rows), 1), 1),
           "scenarios": {"negative_only": sum(not r["scenario"]["pii"] for r in rows),
                         "with_negatives": sum(bool(r["negative_spans"]) for r in rows),
                         "with_typos": sum(bool(r["scenario"]["typos"]) for r in rows),
                         "lowercase": sum(r["scenario"]["lower"] for r in rows),
                         "domains": len({r["scenario"]["domain"] for r in rows})},
           "typos": {"words": sum(r["scenario"]["typos"] for r in rows),
                     "person_fp": sum(len(r["person_fp_on_typos"]) for r in rows),
                     "examples": sorted({x for r in rows for x in r["person_fp_on_typos"]})[:10]},
           "negatives": {"spans": neg_spans,
                         "person_fp": sum(len(r["person_fp_on_negatives"]) for r in rows),
                         "examples": sorted({x for r in rows for x in r["person_fp_on_negatives"]})[:10]},
           "similarity": {"max_accepted": max((r["similarity"] for r in good), default=0.0),
                          "mean_accepted": round(sum(r["similarity"] for r in good) / max(len(good), 1), 3)}}
    with open(a.out.replace(".jsonl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
