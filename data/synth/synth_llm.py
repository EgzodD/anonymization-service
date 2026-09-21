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

Проверки каждого текста (результат — в поле checks, брак — reject=True):
  unknown_tag      — метка не из списка;
  leftover         — после подстановки остались скобки/метки;
  duplicate        — такой же текст (с точностью до меток) уже был;
  extra_pii        — сервис нашёл ПДн ВНЕ подставленных значений: модель могла вписать
                     имя или номер сама — это неразмеченные ПДн, текст уходит в брак
                     (или это ложное срабатывание сервиса — видно при ручной проверке);
  no_tags          — текст без меток там, где сценарий их требовал.

Синтетика — для обучения и поиска ошибок, НЕ для приёмки (вывод статьи, раздел 2).

Запуск пилота:
    .venv/bin/python data/synth/synth_llm.py --n 50 --out data/synth/pilot_qwen3.jsonl
"""
import argparse
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
TAG_RE = re.compile(r"\[([А-ЯЁA-Z_]+)(?::([а-яё]+))?(?::([мж]))?\]")
PERSON_TAGS = {"ФИО": "full", "ИМЯ": "first", "ФАМИЛИЯ": "surname"}
OTHER_TAGS = {"ТЕЛЕФОН": "PHONE_NUMBER", "EMAIL": "EMAIL_ADDRESS", "ИНН": "INN", "СНИЛС": "SNILS",
              "ПАСПОРТ": "PASSPORT", "КАРТА": "CREDIT_CARD", "АДРЕС": "ADDRESS",
              "ДАТА_РОЖДЕНИЯ": "DATE_OF_BIRTH"}

DOMAINS = ["банк", "мобильная связь и интернет", "доставка и интернет-магазин", "ЖКХ и управляющая компания",
           "медицинская клиника", "госуслуги и МФЦ", "страхование", "автосервис", "HR и подбор персонала",
           "аренда жилья", "авиакомпания и билеты", "онлайн-школа"]
CHANNELS = ["сообщение в чате поддержки", "электронное письмо", "расшифровка телефонного звонка",
            "заявка через форму на сайте", "сообщение в мессенджере"]
STYLES = ["официальный", "разговорный", "раздражённый, клиент недоволен", "краткий, почти без знаков препинания",
          "с опечатками и строчными буквами", "вежливый и подробный"]
PII_SETS = [["ФИО"], ["ФИО", "ТЕЛЕФОН"], ["ИМЯ", "EMAIL"], ["ФАМИЛИЯ", "ПАСПОРТ"], ["ФИО", "АДРЕС"],
            ["ФИО", "ИНН", "СНИЛС"], ["ИМЯ", "ФАМИЛИЯ", "ТЕЛЕФОН"], ["ФИО", "КАРТА"],
            ["ФИО", "ДАТА_РОЖДЕНИЯ", "АДРЕС"], ["два человека: ФИО и ФИО"], []]

PROMPT = """Ты помогаешь готовить обучающие данные для системы обезличивания текстов.
Напиши {k} разных реалистичных текста на русском языке.
Сфера: {domain}. Вид текста: {channel}. Стиль: {style}.
Автор — клиент (не оператор), если вид текста это допускает.

Правила для персональных данных:
1. Людей называй ТОЛЬКО условными именами и склоняй их по правилам русского языка:
   мужчина — Иван Иванович Петров, женщина — Анна Сергеевна Петрова, второй человек в
   тексте — Олег Олегович Сидоров или Мария Павловна Сидорова. Можно писать полностью,
   только имя, только фамилию, «Петрову Анне», «А. С. Петрова», «Петров И. И.».
   Никаких других имён и фамилий не используй.
2. Телефоны, почту, номера документов, адреса НЕ пиши — вместо них ставь метки:
   [ТЕЛЕФОН] [EMAIL] [ИНН] [СНИЛС] [ПАСПОРТ] [КАРТА] [АДРЕС] [ДАТА_РОЖДЕНИЯ].
{pii_hint}
Тексты должны отличаться друг от друга по построению и лексике. Длина — от одного до четырёх предложений."""

SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
          "required": ["texts"]}


def pii_hint(pii):
    if not pii:
        return ("3. В этих текстах НЕ должно быть ни людей, ни меток. Но пусть в них будут похожие "
                "вещи: улицы и города, названные в честь людей, названия компаний, номера заказов и "
                "договоров, должности.")
    if pii == ["два человека: ФИО и ФИО"]:
        return "3. В каждом тексте упомяни двух разных людей (Петров/Петрова и Сидоров/Сидорова)."
    person = [p for p in pii if p in PERSON_TAGS]
    other = [f"[{p}]" for p in pii if p in OTHER_TAGS]
    parts = []
    if person:
        parts.append("упомяни условного человека" + (" только по имени" if person == ["ИМЯ"] else
                     " только по фамилии" if person == ["ФАМИЛИЯ"] else ""))
    if other:
        parts.append("используй метки " + ", ".join(other))
    return "3. В каждом тексте " + " и ".join(parts) + "."


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
    """Условные имена и метки -> значения. Возвращает (текст, спаны, ошибки)."""
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
    """Метки -> значения. Возвращает (текст, спаны, ошибки)."""
    out, spans, errors, pos = "", [], [], 0
    rx = re.compile(r"\x00P([^\x00]*)\x00|" + TAG_RE.pattern)
    for m in rx.finditer(template):
        if m.group(0).startswith("\x00"):
            out += template[pos:m.start()]
            spans.append({"start": len(out), "stop": len(out) + len(m.group(1)), "type": "PERSON"})
            out += m.group(1)
            pos = m.end()
            continue
        out += template[pos:m.start()]
        tag, case, gender = m.group(2), m.group(3), m.group(4)     # группы метки в общем шаблоне
        if tag in PERSON_TAGS:
            if case not in CASE:
                errors.append(f"падеж «{case}» в {m.group(0)}")
                case = "им"
            val, typ = person(rng, PERSON_TAGS[tag], CASE[case], GENDER.get(gender)), "PERSON"
        elif tag in OTHER_TAGS:
            val, typ = value(rng, tag), OTHER_TAGS[tag]
            if isinstance(val, tuple):
                val = val[0]
        else:
            errors.append(f"неизвестная метка {m.group(0)}")
            val, typ = m.group(0), None
        if typ:
            spans.append({"start": len(out), "stop": len(out) + len(val), "type": typ})
        out += val
        pos = m.end()
    out += template[pos:]
    return out, spans, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-request", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "pilot_qwen3.jsonl"))
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    os.environ.setdefault("ALLOW_NO_PERSON_MODEL", "true")
    from app.anonymizer import analyze_text

    rng = random.Random(a.seed)
    rows, seen, stat = [], set(), {"requests": 0, "tokens": 0, "seconds": 0.0}
    while len(rows) < a.n:
        sc = {"domain": rng.choice(DOMAINS), "channel": rng.choice(CHANNELS), "style": rng.choice(STYLES),
              "pii": rng.choice(PII_SETS)}
        prompt = PROMPT.format(k=a.per_request, pii_hint=pii_hint(sc["pii"]), **{k: sc[k] for k in ("domain", "channel", "style")})
        try:
            texts, info = ask(a.model, prompt)
        except Exception as exc:  # noqa: BLE001 — пилот: фиксируем и идём дальше
            print("ошибка запроса:", exc)
            continue
        stat["requests"] += 1
        stat["tokens"] += info["tokens"] or 0
        stat["seconds"] += info["seconds"]
        for tpl in texts:
            text, spans, errors = fill(tpl, rng)
            key = " ".join(TAG_RE.sub("<X>", tpl).lower().split())
            checks = {"unknown_tag": errors, "leftover": bool(re.search(r"[\[\]{}]", text)),
                      "duplicate": key in seen, "no_tags": bool(sc["pii"]) and not spans}
            seen.add(key)
            found = [r for r in analyze_text(text)
                     if not any(r.start < s["stop"] and s["start"] < r.end for s in spans)]
            checks["extra_pii"] = [f"{r.entity_type}: {text[r.start:r.end]}" for r in found]
            reject = bool(errors or checks["leftover"] or checks["duplicate"] or checks["no_tags"]
                          or checks["extra_pii"])
            rows.append({"text": text, "spans": spans, "template": tpl, "scenario": sc, "model": a.model,
                         "checks": checks, "reject": reject})
        print(f"запросов {stat['requests']}, текстов {len(rows)}, "
              f"{stat['tokens'] / max(stat['seconds'], 1e-9):.1f} ток/с", flush=True)
    rows = rows[:a.n]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rej = sum(r["reject"] for r in rows)
    why = {k: sum(bool(r["checks"][k]) for r in rows) for k in rows[0]["checks"]}
    rep = {"model": a.model, "texts": len(rows), "rejected": rej, "reject_share": round(rej / len(rows), 3),
           "reasons": why, "requests": stat["requests"], "tokens": stat["tokens"],
           "minutes": round(stat["seconds"] / 60, 1),
           "tokens_per_sec": round(stat["tokens"] / max(stat["seconds"], 1e-9), 1),
           "sec_per_text": round(stat["seconds"] / max(len(rows), 1), 1)}
    with open(a.out.replace(".jsonl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
