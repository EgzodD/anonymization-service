"""
Прогон систем распознавания ФИО по протоколу PROTOCOL.md.

Каждая система запускается ОТДЕЛЬНЫМ процессом (модели тяжёлые, по 1–1,5 ГБ):
    .venv/bin/python data/eval_v2/run_eval.py --system S1 --dataset v2
    ...
    .venv/bin/python data/eval_v2/run_eval.py --aggregate

Предсказания пишутся в results/preds_<система>_<набор>.jsonl — их можно проверить
вручную. Метрики и доверительные интервалы считает шаг --aggregate.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import metrics as M  # noqa: E402

PROD_MODEL = os.path.join(ROOT, "models", "person_ruBERT")
BASE_MODEL = os.path.join(ROOT, "models", "person_ruBERT_baseline")

DATASETS = {
    "v2": os.path.join(HERE, "test_v2.jsonl"),
    "old": os.path.join(ROOT, "data", "training", "test", "test.jsonl"),
}
SYSTEMS = {
    "S1": "spaCy ru_core_news_lg (NER-бэкенд Presidio)",
    "S2": "Natasha (slovnet NER)",
    "S3": "rubert-tiny2, дообученная на новостях (базовая линия)",
    "S4": "rubert-tiny2, дообученная под домен обращений (наша)",
    "S5": "сервис целиком (модель + spaCy + regex-правила)",
    # те же модели с наивной склейкой подслов (aggregation_strategy="simple"):
    # показывают вклад декодирования, см. поправку 1 в PROTOCOL.md
    "S3simple": "то же, что S3, но склейка подслов simple",
    "S4simple": "то же, что S4, но склейка подслов simple",
}


def load_dataset(name):
    rows = []
    with open(DATASETS[name], encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "id": r.get("id", f"{name}-{i:04d}"),
                "template_id": r.get("template_id", f"{name}-{i:04d}"),
                "subset": r.get("subset", "pii" if r["spans"] else "neg"),
                "text": r["text"],
                "spans": r["spans"],
            })
    return rows


# ── системы ───────────────────────────────────────────────────────────────
def make_system(code):
    if code == "S1":
        import spacy
        nlp = spacy.load("ru_core_news_lg")
        def run(text):
            return [(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ == "PER"]
    elif code == "S2":
        from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
        seg, tagger = Segmenter(), NewsNERTagger(NewsEmbedding())
        def run(text):
            d = Doc(text)
            d.segment(seg)
            d.tag_ner(tagger)
            return [(s.start, s.stop) for s in d.spans if s.type == "PER"]
    elif code.startswith(("S3", "S4")):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        model_dir = BASE_MODEL if code.startswith("S3") else PROD_MODEL
        # По умолчанию словоуровневая склейка "first": метка слова = метка его
        # первого подтокена — ровно так модель обучалась. Стратегия "simple"
        # рвёт слово там, где метка меняется внутри него, и порождает обрывки
        # («март» + «иросян»), что искажает оценку самой модели.
        agg = "simple" if code.endswith("simple") else "first"
        pipe = pipeline("token-classification",
                        model=AutoModelForTokenClassification.from_pretrained(model_dir),
                        tokenizer=AutoTokenizer.from_pretrained(model_dir),
                        aggregation_strategy=agg)
        def run(text):
            out = []
            for ent in pipe(text):
                group = (ent.get("entity_group") or ent.get("entity") or "")
                if group.rsplit("-", 1)[-1].upper() in ("PER", "PERSON"):
                    out.append((int(ent["start"]), int(ent["end"])))
            return out
    elif code == "S5":
        os.environ["PERSON_MODEL_DIR"] = PROD_MODEL
        from app.anonymizer import anonymize_text
        def run(text):
            res = anonymize_text(text)
            spans = [(e["start"], e["end"]) for e in res["entities_found"]
                     if e["entity_type"] == "PERSON"]
            return spans, res["anonymized"]
    else:
        raise SystemExit(f"неизвестная система {code}")
    return run


def cmd_run(code, dataset):
    os.makedirs(RESULTS, exist_ok=True)
    rows = load_dataset(dataset)
    run = make_system(code)
    run(rows[0]["text"])                       # прогрев
    out_path = os.path.join(RESULTS, f"preds_{code}_{dataset}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            t0 = time.perf_counter()
            res = run(r["text"])
            ms = (time.perf_counter() - t0) * 1000
            spans, anon = (res if isinstance(res, tuple) else (res, None))
            rec = {"id": r["id"], "spans": [[int(a), int(b)] for a, b in spans],
                   "ms": round(ms, 2)}
            if anon is not None:
                rec["anonymized"] = anon
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{code}/{dataset}: {len(rows)} примеров -> {os.path.basename(out_path)}")


# ── агрегация ─────────────────────────────────────────────────────────────
def per_template_counts(rows, preds):
    acc = {}
    for r in rows:
        gold = [(s["start"], s["stop"]) for s in r["spans"] if s["type"] == "PERSON"]
        p = preds.get(r["id"], {}).get("spans", [])
        c = M.example_counts(r["text"], gold, [tuple(x) for x in p], r["subset"] == "neg")
        vec = np.array([c[k] for k in M.COUNT_FIELDS], dtype=float)
        acc[r["template_id"]] = acc.get(r["template_id"], 0) + vec
    return acc


def totals_of(acc):
    return dict(zip(M.COUNT_FIELDS, sum(acc.values())))


METRIC_FNS = {
    "strict_f1": lambda t: M.summarize(t)["strict"]["f1"],
    "strict_precision": lambda t: M.summarize(t)["strict"]["precision"],
    "strict_recall": lambda t: M.summarize(t)["strict"]["recall"],
    "full_masked_rate": lambda t: M.summarize(t)["full_masked_rate"],
    "overlap_f1": lambda t: M.summarize(t)["overlap"]["f1"],
    "neg_fp_rate": lambda t: M.summarize(t)["neg_fp_rate"],
}
PRIMARY = ["strict_f1", "full_masked_rate"]


def subgroup_table(rows, preds, key_fn):
    """Полнота по подгруппам сущностей (падеж, формат, регистр, класс фамилии)."""
    agg = {}
    for r in rows:
        pred = M.normalize(r["text"], [tuple(x) for x in preds.get(r["id"], {}).get("spans", [])])
        for s in r["spans"]:
            if s["type"] != "PERSON":
                continue
            key = key_fn(s)
            if key is None:
                continue
            g = M.normalize(r["text"], [(s["start"], s["stop"])])
            covered = M.count_coverage(r["text"], g, pred)
            strict = M.count_strict(g, pred)[0]
            a = agg.setdefault(key, [0, 0, 0])
            a[0] += 1
            a[1] += covered
            a[2] += strict
    return {k: {"n": v[0], "full_masked_rate": v[1] / v[0], "strict_recall": v[2] / v[0]}
            for k, v in sorted(agg.items())}


def cmd_aggregate(n_iter):
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"),
              "env": {"python": platform.python_version(),
                      "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                               cwd=ROOT, capture_output=True,
                                               text=True).stdout.strip()},
              "datasets": {}, "systems": SYSTEMS}
    for name, path in DATASETS.items():
        report["env"][f"sha256_{name}"] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    for pkg in ("transformers", "torch", "spacy", "natasha"):
        try:
            report["env"][pkg] = __import__(pkg).__version__
        except Exception:                                   # noqa: BLE001
            report["env"][pkg] = "n/a"

    cache = {}
    for dataset in DATASETS:
        rows = load_dataset(dataset)
        block = {"examples": len(rows),
                 "person_entities": sum(1 for r in rows for s in r["spans"]
                                        if s["type"] == "PERSON"),
                 "systems": {}}
        for code in SYSTEMS:
            pp = os.path.join(RESULTS, f"preds_{code}_{dataset}.jsonl")
            if not os.path.exists(pp):
                continue
            preds = {}
            with open(pp, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    preds[r["id"]] = r
            acc = per_template_counts(rows, preds)
            cache[(dataset, code)] = acc
            res = M.summarize(totals_of(acc))
            res["latency_ms_median"] = float(np.median([r["ms"] for r in preds.values()]))
            res["ci95"] = {}
            for m in PRIMARY:
                pt, lo, hi, _ = M.bootstrap(acc, METRIC_FNS[m], n_iter=n_iter)
                res["ci95"][m] = {"point": pt, "lo": lo, "hi": hi}
            if dataset == "v2":
                res["subgroups"] = {
                    "case": subgroup_table(rows, preds, lambda s: s["meta"].get("case")),
                    "format": subgroup_table(rows, preds, lambda s: s["meta"].get("format")),
                    "lowercase": subgroup_table(rows, preds, lambda s: str(s["meta"].get("lower"))),
                    "surname_class": subgroup_table(
                        rows, preds, lambda s: s["meta"].get("surname_class")),
                    "format_seen_in_training": subgroup_table(
                        rows, preds, lambda s: str(s["meta"].get("format_seen_in_training"))),
                }
            block["systems"][code] = res
        report["datasets"][dataset] = block

    # парные сравнения на одних и тех же ресэмплах
    comparisons = {}
    for dataset in DATASETS:
        for a, b in (("S4", "S3"), ("S4", "S1"), ("S4", "S2"), ("S5", "S4")):
            if (dataset, a) not in cache or (dataset, b) not in cache:
                continue
            for m in PRIMARY:
                pt, lo, hi, samples = M.bootstrap(cache[(dataset, a)], METRIC_FNS[m],
                                                  n_iter=n_iter,
                                                  paired_with=cache[(dataset, b)])
                comparisons[f"{dataset}:{a}-{b}:{m}"] = {
                    "diff": pt, "ci95": [lo, hi], "p_value": M.two_sided_p(samples),
                    "significant": bool(lo > 0 or hi < 0)}
    report["paired_comparisons"] = comparisons

    # H2: завышение из-за совпадения шаблонов
    if ("old", "S4") in cache and ("v2", "S4") in cache:
        h2 = {}
        for m in PRIMARY:
            gains = {}
            for ds in ("old", "v2"):
                pt, _, _, _ = M.bootstrap(cache[(ds, "S4")], METRIC_FNS[m], n_iter=200,
                                          paired_with=cache[(ds, "S3")])
                gains[ds] = pt
            h2[m] = {"gain_old": gains["old"], "gain_v2": gains["v2"],
                     "inflation": gains["old"] - gains["v2"]}
        report["h2_template_overlap_inflation"] = h2

    out = os.path.join(RESULTS, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("записано:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS))
    ap.add_argument("--dataset", choices=list(DATASETS), default="v2")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n-iter", type=int, default=10000)
    a = ap.parse_args()
    if a.aggregate:
        cmd_aggregate(a.n_iter)
    elif a.system:
        cmd_run(a.system, a.dataset)
    else:
        ap.error("нужно --system или --aggregate")
