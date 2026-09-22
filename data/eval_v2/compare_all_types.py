"""
Сравнение с конкурентами по всем типам ПДн (PROTOCOL.md, поправка 6).

Метрика не зависит от названий типов у разных систем: значение ПДн эталона считается
СКРЫТЫМ ПОЛНОСТЬЮ, если каждое его слово закрыто любой маской системы. Цена — доля
текстов без разметки, в которых система поставила хоть одну маску.

Системы:
  ours — наш сервис, текущий код (все найденные сущности);
  S6   — redmadrobot rubert-base-pii-ner, все метки;
  S7   — PIIDetector (alrosait), метки NAME и ADDRESS.

Запуск:
    .venv/bin/python data/eval_v2/compare_all_types.py [--run]
    --run — заново получить предсказания (иначе берутся сохранённые results/preds_all_*.jsonl)
Результат: results/all_types_paired.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from run_eval import load_dataset  # noqa: E402

DATASETS = ("hive", "bench", "v2", "alro")
TYPES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "INN", "SNILS", "PASSPORT", "CREDIT_CARD",
         "DATE_OF_BIRTH", "ADDRESS"]
S7_TYPES = {"PERSON", "ADDRESS"}                 # что PIIDetector вообще умеет искать
WORD = re.compile(r"\w+")
N_ITER, SEED = 2000, 2026


def systems():
    os.environ.setdefault("PERSON_MODEL_DIR", os.path.join(ROOT, "models", "person_ruBERT"))
    from app.anonymizer import anonymize_text

    def ours(text):
        return [(e["start"], e["end"]) for e in anonymize_text(text)["entities_found"]]

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    name = "redmadrobot-rnd/rubert-base-pii-ner"
    pipe = pipeline("token-classification", model=AutoModelForTokenClassification.from_pretrained(name),
                    tokenizer=AutoTokenizer.from_pretrained(name), aggregation_strategy="first")

    def s6(text):
        return [(int(e["start"]), int(e["end"])) for e in pipe(text)
                if (e.get("entity_group") or "O") != "O"]

    import spacy
    from huggingface_hub import snapshot_download
    nlp = spacy.load(snapshot_download("alrosait/spacy_ru_core_news_lg_pii"))

    def s7(text):
        return [(e.start_char, e.end_char) for e in nlp(text).ents]
    return {"ours": ours, "S6": s6, "S7": s7}


def run_all():
    sysm = systems()
    for ds in DATASETS:
        rows = load_dataset(ds)
        for code, fn in sysm.items():
            path = os.path.join(RES, f"preds_all_{code}_{ds}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    t0 = time.perf_counter()
                    spans = fn(r["text"])
                    f.write(json.dumps({"id": r["id"], "spans": spans,
                                        "ms": round((time.perf_counter() - t0) * 1000, 2)}) + "\n")
            print("готово", code, ds, flush=True)


def covered(text, span, pred):
    mask = np.zeros(len(text) + 1, dtype=bool)
    for a, b in pred:
        mask[a:b] = True
    s, e = span
    words = [(s + m.start(), s + m.end()) for m in WORD.finditer(text[s:e])]
    return bool(words) and all(mask[a:b].all() for a, b in words)


def vectors(ds, rows, code):
    preds = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, f"preds_all_{code}_{ds}.jsonl"),
                                                       encoding="utf-8"))}
    out = []
    for r in rows:
        pred = [tuple(x) for x in preds[r["id"]]["spans"]]
        v = {}
        for t in TYPES:
            gold = [(s["start"], s["stop"]) for s in r["spans"] if s["type"] == t]
            v[t] = (len(gold), sum(covered(r["text"], g, pred) for g in gold))
        v["clean"] = (1, int(bool(pred))) if not r["spans"] else (0, 0)
        out.append(v)
    return out


def rate(vs, key, idx=None):
    items = vs if idx is None else [vs[i] for i in idx]
    n = sum(x[key][0] for x in items)
    k = sum(x[key][1] for x in items)
    return k / n if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_all()
    report = {}
    for ds in DATASETS:
        rows = load_dataset(ds)
        V = {c: vectors(ds, rows, c) for c in ("ours", "S6", "S7")}
        idx = np.random.default_rng(SEED).integers(0, len(rows), size=(N_ITER, len(rows)))
        res = {}
        for key in TYPES + ["clean"]:
            n = sum(x[key][0] for x in V["ours"])
            if not n:
                continue
            entry = {"n": n, "values": {c: round(rate(V[c], key), 4) for c in V}, "pairs": {}}
            for o in ("S6", "S7"):
                if key != "clean" and o == "S7" and key not in S7_TYPES:
                    entry["pairs"][o] = {"verdict": "unsupported"}
                    continue
                d = np.array([rate(V["ours"], key, i) - rate(V[o], key, i) for i in idx
                              if rate(V["ours"], key, i) is not None])
                lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
                if key == "clean":                      # для лишних масок меньше — лучше
                    lo, hi = -hi, -lo
                verdict = "win" if lo > 0 else "loss" if hi < 0 else "tie"
                entry["pairs"][o] = {"diff": round(rate(V["ours"], key) - rate(V[o], key), 4),
                                     "ci95": [round(lo, 4), round(hi, 4)] if key != "clean" else [round(-hi, 4), round(-lo, 4)],
                                     "verdict": verdict}
            res[key] = entry
        report[ds] = res
        print(f"== {ds}")
        for k, e in res.items():
            print(f"   {k:14} n={e['n']:5} ours {e['values']['ours']:.3f} S6 {e['values']['S6']:.3f} "
                  f"S7 {e['values']['S7']:.3f}  ", {o: p["verdict"] for o, p in e["pairs"].items()})
    with open(os.path.join(RES, "all_types_paired.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
