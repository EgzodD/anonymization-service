"""
Третий внешний набор: персоны из factRuEval-2016 (решение по пункту 0.5 плана).

Зачем нужен. На этапе 3 модель ФИО может обучаться на redmadrobot-rnd/pii_train.
Тогда pii_benchmark из того же источника перестаёт быть «чужим» набором, и
обобщение проверять не на чем. Модель в сервисе отвечает ТОЛЬКО за ФИО (паспорта,
СНИЛС, телефоны ищут правила), поэтому третьему набору достаточно разметки персон.

Источник: https://github.com/dialogue-evaluation/factRuEval-2016, лицензия MIT,
коммит dfe96237fce79fde13b2021d1938587e4ea3042e. Берётся только testset: 132
новостных и аналитических текста, ручная разметка соревнования «Диалог-2016».
Проверено до сборки: символьные смещения всех 59 382 токенов совпадают с текстом.
Русских данных в ai4privacy нет (проверено по распределению языков, 17.09.2026).

Согласования (все зафиксированы ДО любого прогона систем на наборе):

1. ПЕРСОНА = части имени. Упоминание Person в источнике состоит из спанов name,
   surname, patronymic, nickname. Эталон PERSON — части name/surname/patronymic,
   соседние (разделённые только пробелами и точками) склеены в одну сущность,
   как в load_bench.py. Должность (job) в персону не входит.
2. ТОЛЬКО ПРОЗВИЩЕ. Упоминание, где нет ни имени, ни фамилии («Пьеро»), —
   спорный случай для обезличивания. Оно сохраняется с типом OTHER_PERSON_NICK:
   в метрики не входит, а абзац с ним не считается «чистым».
3. ЕДИНИЦА — абзац (строка текста). Сервис получает обращения, а не статьи;
   кластер для бутстрепа — документ (template_id), абзацы одного текста зависимы.

Домен — новости, не обращения клиентов. Это намеренно: набор проверяет перенос
на чужой домен, а не качество на своём.

Запуск:
    .venv/bin/python data/eval_v2/load_ext.py [путь к клону factRuEval-2016]
Без пути репозиторий клонируется во временный каталог на зафиксированном коммите.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ext.jsonl")
REPORT = os.path.join(HERE, "ext_report.json")

REPO = "https://github.com/dialogue-evaluation/factRuEval-2016.git"
COMMIT = "dfe96237fce79fde13b2021d1938587e4ea3042e"
NAME_PARTS = {"name", "surname", "patronymic"}
GLUE = re.compile(r"^[\s.]*$")


def fetch(dst):
    subprocess.run(["git", "clone", "-q", REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", COMMIT], check=True)


def read_doc(base):
    text = open(base + ".txt", encoding="utf-8").read()
    spans = {}
    for line in open(base + ".spans", encoding="utf-8"):
        p = line.split("#")[0].split()
        if len(p) >= 4:
            spans[p[0]] = (p[1], int(p[2]), int(p[2]) + int(p[3]))
    persons = []
    for line in open(base + ".objects", encoding="utf-8"):
        p = line.split("#")[0].split()
        if len(p) >= 3 and p[1] == "Person":
            persons.append([spans[s] for s in p[2:] if s in spans])
    return text, persons


def person_spans(text, parts):
    """Части одного упоминания -> сущности PERSON (соседние части склеены)."""
    names = sorted((s, e) for t, s, e in parts if t in NAME_PARTS)
    if not names:
        return [], [(s, e) for _, s, e in parts]          # только прозвище
    out = []
    for s, e in names:
        if out and GLUE.match(text[out[-1][1]:s]):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out, []


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = None
    if src is None:
        tmp = tempfile.mkdtemp(prefix="factrueval_")
        src = os.path.join(tmp, "repo")
        fetch(src)
    head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == COMMIT, f"ожидался коммит {COMMIT}, получен {head}"

    rows, types, cross, docs = [], Counter(), 0, 0
    for base in sorted(glob.glob(os.path.join(src, "testset", "book_*.txt"))):
        base = base[:-4]
        doc = os.path.basename(base)
        docs += 1
        text, persons = read_doc(base)
        ents = set()
        for parts in persons:
            pers, nick = person_spans(text, parts)
            ents |= {(s, e, "PERSON") for s, e in pers}
            ents |= {(s, e, "OTHER_PERSON_NICK") for s, e in nick}
        pos = 0
        for n, para in enumerate(text.split("\n")):
            a, b = pos, pos + len(para)
            pos = b + 1
            inside = sorted((s - a, e - a, t) for s, e, t in ents if a <= s and e <= b)
            cross += sum(1 for s, e, _ in ents if s < b and e > a and not (a <= s and e <= b))
            if not para.strip():
                continue
            # вложенные дубли одного типа (одно имя в двух упоминаниях) — один раз
            spans = []
            for s, e, t in inside:
                if any(t == u["type"] and u["start"] <= s and e <= u["stop"] for u in spans):
                    continue
                spans.append({"start": s, "stop": e, "type": t, "meta": {}})
            for sp in spans:
                assert para[sp["start"]:sp["stop"]].strip(), f"пустой спан: {doc}"
                types[sp["type"]] += 1
            rows.append({"id": f"ext-{doc}-{n:03d}", "template_id": f"ext-{doc}",
                         "subset": "pii" if spans else "neg",
                         "text": para, "spans": spans})
    assert cross == 0, f"сущностей через границу абзаца: {cross}"

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    rep = {
        "source": f"dialogue-evaluation/factRuEval-2016 testset (MIT), commit {COMMIT}",
        "documents": docs, "paragraphs": len(rows),
        "paragraphs_with_person": sum(1 for r in rows if any(s["type"] == "PERSON" for s in r["spans"])),
        "paragraphs_clean": sum(1 for r in rows if r["subset"] == "neg"),
        "entities_by_type": dict(types.most_common()),
        "sha256": sha,
        "note": "Только для финальной приёмки (правило П1). OTHER_PERSON_NICK в метрики не входит.",
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
