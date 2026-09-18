"""
Три новых внешних набора в формате оценки (PROTOCOL.md, поправка 4).

  hive.jsonl — hivetrace/pii-bench (Apache-2.0): 1 810 текстов поддержки и
               мессенджеров, ручная разметка двумя экспертами, символьные позиции.
  alro.jsonl — alrosait/pii-synthetic-ru (MIT): 4 500 текстов, ФИО и адреса.
               Разметка — строки без позиций; позиция находится поиском, размечаются
               ВСЕ вхождения строки (повтор имени — тоже ПДн). Не найденная строка —
               пример отбрасывается и учитывается в отчёте.
  mcr.jsonl  — MultiCoNER ru test (CC BY 4.0): случайные 4 000 предложений,
               сид 20260918. Токены склеиваются через пробел; персоны (PER) -> PERSON,
               остальные типы -> OTHER_*. Текст целиком строчный.

Метаданные исходных наборов (домен, форма имени, краевой случай, категория
негатива) сохраняются в поле meta примера — для разбора ошибок по категориям.

Эти наборы НЕЛЬЗЯ добавлять в обучающие корпуса (hivetrace об этом прямо просит;
для остальных — правило П1).

Запуск:
    .venv/bin/python data/eval_v2/load_external2.py
"""
import hashlib
import json
import os
import random
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "external2_report.json")
MC_URL = ("https://huggingface.co/datasets/tomaarsen/MultiCoNER/resolve/"
          "refs%2Fconvert%2Fparquet/ru/test/0000.parquet")
MC_LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-CORP", "I-CORP", "B-GRP", "I-GRP",
             "B-PROD", "I-PROD", "B-CW", "I-CW"]
MC_SEED, MC_SIZE = 20260918, 4000

HIVE_TYPES = {"NAME": "PERSON", "PHONE_NUMBER": "PHONE_NUMBER", "EMAIL": "EMAIL_ADDRESS",
              "ADDRESS": "ADDRESS", "BANK_CARD_NUMBER": "CREDIT_CARD", "INN": "INN",
              "SNILS": "SNILS", "PASSPORT_NUMBER": "PASSPORT"}


def write(name, rows):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def subset(spans):
    return "pii" if any(not s["type"].startswith("OTHER_") for s in spans) else "neg"


def build_hive():
    from datasets import load_dataset
    rows, types = [], Counter()
    ds = load_dataset("hivetrace/pii-bench")
    for split in ("domain", "entity"):
        for ex in ds[split]:
            spans = []
            for e in ex["entities"]:
                assert ex["text"][e["start"]:e["end"]] == e["text"], ex["id"]
                t = HIVE_TYPES.get(e["type"], "OTHER_" + e["type"])
                spans.append({"start": e["start"], "stop": e["end"], "type": t, "meta": {}})
                types[t] += 1
            rid = f"hive-{split}-{ex['id']}"
            rows.append({"id": rid, "template_id": rid, "subset": subset(spans), "text": ex["text"],
                         "spans": spans, "meta": {"split": split, "domain": ex["domain"]}})
    return rows, {"examples": len(rows), "entities_by_type": dict(types.most_common())}


def build_alro():
    from datasets import load_dataset
    rows, types, dropped = [], Counter(), 0
    for ex in load_dataset("alrosait/pii-synthetic-ru")["train"]:
        text, spans, ok = ex["text"], [], True
        for e in ex["entities"] or []:
            hits = [m.start() for m in re.finditer(re.escape(e["text"]), text)]
            if not hits:
                ok = False
                break
            t = "PERSON" if e["type"] == "NAME" else "ADDRESS" if e["type"] == "ADDRESS" \
                else "OTHER_" + e["type"]
            for h in hits:
                spans.append({"start": h, "stop": h + len(e["text"]), "type": t, "meta": {}})
        if not ok:
            dropped += 1
            continue
        uniq = {(s["start"], s["stop"], s["type"]): s for s in spans}
        spans = sorted(uniq.values(), key=lambda s: s["start"])
        for s in spans:
            types[s["type"]] += 1
        meta = {k: ex[k] for k in ("domain", "entity_type", "name_form", "addr_form", "style",
                                   "neg_category", "edge_case", "has_typo")}
        rows.append({"id": f"alro-{ex['id']}", "template_id": f"alro-{ex['id']}",
                     "subset": subset(spans), "text": text, "spans": spans, "meta": meta})
    return rows, {"examples": len(rows), "dropped_not_found": dropped,
                  "entities_by_type": dict(types.most_common())}


def build_mcr():
    import io
    import urllib.request

    import pandas as pd
    data = urllib.request.urlopen(MC_URL, timeout=300).read()
    df = pd.read_parquet(io.BytesIO(data))
    idx = sorted(random.Random(MC_SEED).sample(range(len(df)), MC_SIZE))
    rows, types = [], Counter()
    for i in idx:
        toks, tags = list(df["tokens"][i]), [MC_LABELS[t] for t in df["ner_tags"][i]]
        text, offs = "", []
        for t in toks:
            if text:
                text += " "
            offs.append((len(text), len(text) + len(t)))
            text += t
        spans, cur = [], None
        for (s, e), tag in zip(offs, tags):
            if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["src"] != tag[2:])):
                if cur:
                    spans.append(cur)
                cur = {"start": s, "stop": e, "src": tag[2:]}
            elif tag.startswith("I-"):
                cur["stop"] = e
            else:
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
        spans = [{"start": s["start"], "stop": s["stop"],
                  "type": "PERSON" if s["src"] == "PER" else "OTHER_" + s["src"], "meta": {}}
                 for s in spans]
        for s in spans:
            types[s["type"]] += 1
        rid = f"mcr-{int(df['id'][i]):06d}"
        rows.append({"id": rid, "template_id": rid, "subset": subset(spans), "text": text,
                     "spans": spans, "meta": {}})
    return rows, {"source_size": len(df), "examples": len(rows), "seed": MC_SEED,
                  "entities_by_type": dict(types.most_common())}


def main():
    rep = {}
    for name, fn in (("hive", build_hive), ("alro", build_alro), ("mcr", build_mcr)):
        rows, info = fn()
        info["sha256"] = write(f"{name}.jsonl", rows)
        info["with_person"] = sum(1 for r in rows if any(s["type"] == "PERSON" for s in r["spans"]))
        info["clean"] = sum(1 for r in rows if not r["spans"])
        rep[name] = info
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
