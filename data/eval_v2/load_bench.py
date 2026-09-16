"""
Конвертер стороннего бенчмарка redmadrobot-rnd/pii_benchmark в наш формат.

Зачем нужен. Наш набор v2 и оцениваемая система сделаны одним человеком —
это записано в ограничениях как слабое место. Сторонний набор снимает претензию:
разметка выполнена независимо, тексты частично взяты из реальных продакшн-логов
(значения ПДн в них заменены синтетическими), лицензия MIT.

Источник: https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark
2841 предложение, 21 тип сущностей, разметка BIO по токенам.

Два согласования, без которых сравнение было бы некорректным (оба обратимы
и зафиксированы здесь явно):

1. ИМЯ ПО ЧАСТЯМ. В источнике имя размечено раздельно: FIRST_NAME, LAST_NAME,
   MIDDLE_NAME — три спана. Наши системы (и spaCy, и Natasha, и модели ruBERT)
   выдают имя одним спаном. Поэтому соседние части имени, разделённые только
   пробелами и точками, склеиваются в одну сущность PERSON. Без этого строгое
   совпадение границ не считалось бы ни у одной системы.

2. СИМВОЛЬНЫЕ ПОЗИЦИИ. В источнике разметка токенная, а поле text не равно
   склейке токенов через пробел. Позиции восстанавливаются выравниванием
   токенов по тексту; примеры, где выравнивание не удалось, отбрасываются
   (их 2 из 2841) — молча терять разметку нельзя.

Запуск:
    .venv/bin/python data/eval_v2/load_bench.py
"""
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bench.jsonl")
REPORT = os.path.join(HERE, "bench_report.json")

# соответствие типов: их -> наши. Типы, которых наш сервис не поддерживает,
# сохраняются с префиксом OTHER_ и в метрики не идут.
TYPE_MAP = {
    "FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "MIDDLE_NAME": "PERSON",
    "PHONE": "PHONE_NUMBER", "EMAIL": "EMAIL_ADDRESS",
    "INN": "INN", "SNILS": "SNILS", "PASSPORT": "PASSPORT",
    "CREDIT_CARD": "CREDIT_CARD",
}
PERSON_PARTS = {"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"}
GLUE = re.compile(r"^[\s.]*$")      # чем разрешено разделять части имени


def spans_from_bio(text, tokens, tags):
    """Токенная разметка BIO -> символьные спаны. None, если не выровнялось."""
    offsets, pos = [], 0
    for tok in tokens:
        i = text.find(tok, pos)
        if i < 0:
            i = text.lower().find(tok.lower(), pos)      # различия регистра
            if i < 0:
                return None
        offsets.append((i, i + len(tok)))
        pos = i + len(tok)
    spans, cur = [], None
    for (s, e), tag in zip(offsets, tags):
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        pref, typ = tag[:1], tag[2:]
        if pref == "B" or cur is None or cur["src"] != typ:
            if cur:
                spans.append(cur)
            cur = {"start": s, "stop": e, "src": typ}
        else:
            cur["stop"] = e
    if cur:
        spans.append(cur)
    return spans


def merge_person(text, spans):
    """Склеивает соседние части имени в одну сущность PERSON."""
    out, i = [], 0
    spans = sorted(spans, key=lambda s: s["start"])
    while i < len(spans):
        s = spans[i]
        if s["src"] in PERSON_PARTS:
            start, stop = s["start"], s["stop"]
            j = i + 1
            while (j < len(spans) and spans[j]["src"] in PERSON_PARTS
                   and GLUE.match(text[stop:spans[j]["start"]])):
                stop = spans[j]["stop"]
                j += 1
            out.append({"start": start, "stop": stop, "type": "PERSON", "meta": {}})
            i = j
        else:
            out.append({"start": s["start"], "stop": s["stop"],
                        "type": TYPE_MAP.get(s["src"], "OTHER_" + s["src"]), "meta": {}})
            i += 1
    return out


def main():
    from datasets import load_dataset
    ds = load_dataset("redmadrobot-rnd/pii_benchmark")["test"]
    rows, dropped, merged_cnt = [], 0, 0
    types = Counter()
    for n, ex in enumerate(ds, 1):
        tokens, tags = json.loads(ex["tokens"]), json.loads(ex["ner_tags"])
        raw = spans_from_bio(ex["text"], tokens, tags)
        if raw is None:
            dropped += 1
            continue
        parts = sum(1 for s in raw if s["src"] in PERSON_PARTS)
        spans = merge_person(ex["text"], raw)
        persons = sum(1 for s in spans if s["type"] == "PERSON")
        merged_cnt += parts - persons
        for s in spans:
            types[s["type"]] += 1
        # проверка целостности: спан обязан совпадать с текстом
        for s in spans:
            assert ex["text"][s["start"]:s["stop"]].strip(), f"пустой спан в примере {n}"
        rows.append({"id": f"bench-{n:05d}", "template_id": f"bench-{n:05d}",
                     "subset": "pii" if any(not s["type"].startswith("OTHER_") for s in spans) else "neg",
                     "text": ex["text"], "spans": spans})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {
        "source": "redmadrobot-rnd/pii_benchmark (MIT)",
        "examples_in_source": len(ds), "examples_written": len(rows),
        "dropped_alignment_failed": dropped,
        "person_parts_merged": merged_cnt,
        "entities_by_type": dict(types.most_common()),
        "note": "Типы с префиксом OTHER_ наш сервис не поддерживает, в метрики не входят.",
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
