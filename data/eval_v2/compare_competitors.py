"""
Парное сравнение нашего сервиса со сторонними системами по всем метрикам ФИО.

Для каждой пары «наш сервис — другая система» на одном наборе считается разность
метрик и её 95 % доверительный интервал (парный бутстрап по примерам, 2000 итераций,
сид 2026). Вердикт по правилу, записанному в отчёте:
  win   — интервал разности целиком на стороне «нам лучше»;
  loss  — интервал целиком на стороне «нам хуже»;
  tie   — интервал содержит ноль.
Для «ложное ФИО в текстах без ПДн» лучше меньшее значение, для остальных — большее.

Наши версии: S5-v3 — замер после приёмки (слепой); S5-r7 — текущий код с правилами
от 18.09 (на hive — не слепой, см. поправку 5 протокола).

Запуск: .venv/bin/python data/eval_v2/compare_competitors.py -> results/competitors_paired.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_external2 as A  # noqa: E402
from run_eval import load_dataset  # noqa: E402

METRICS = ["precision", "recall", "f1", "f1_overlap", "full_masked", "false_person_rate"]
LOWER_IS_BETTER = {"false_person_rate"}
PAIRS = [("S5-v3", "S6"), ("S5-v3", "S7"), ("S5-v3", "S5-prev"),
         ("S5-r7", "S6"), ("S5-r7", "S7"), ("S5-r7", "S5-v3")]
# на наборах приёмки (test_v2, pii_benchmark, factRuEval) — только текущая версия против конкурентов
PAIRS_LATEST = [("S5-r7", "S6"), ("S5-r7", "S7")]
N_ITER, SEED = 2000, 2026


def vectors(ds, rows, system):
    preds = A.load_preds(system, ds)
    return np.array([A.counts(r, [tuple(x) for x in preds[r["id"]]["spans"]]) for r in rows])


def verdict(lo, hi, metric):
    if metric in LOWER_IS_BETTER:
        lo, hi = -hi, -lo
    return "win" if lo > 0 else "loss" if hi < 0 else "tie"


def compare(ds, rows, label, pairs=PAIRS):
    systems = {s for p in pairs for s in p}
    vec = {s: vectors(ds, rows, s) for s in systems}
    idx = np.random.default_rng(SEED).integers(0, len(rows), size=(N_ITER, len(rows)))
    boots = {s: [A.summ(vec[s][i].sum(0)) for i in idx] for s in systems}
    point = {s: A.summ(vec[s].sum(0)) for s in systems}
    ci = {s: {m: [round(float(x), 4) for x in np.percentile([b[m] for b in boots[s]], [2.5, 97.5])]
              for m in METRICS} for s in systems}
    out = {"dataset": label, "texts": len(rows),
           "systems": {s: {m: round(point[s][m], 4) for m in METRICS} for s in systems},
           "systems_ci95": ci, "pairs": {}}
    for ours, other in pairs:
        res = {}
        for m in METRICS:
            d = np.array([a[m] - b[m] for a, b in zip(boots[ours], boots[other])])
            lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
            res[m] = {"diff": round(point[ours][m] - point[other][m], 4),
                      "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict(lo, hi, m)}
        out["pairs"][f"{ours}|{other}"] = res
    return out


def main():
    report = {}
    for ds in ("hive", "alro", "mcr"):
        rows = load_dataset(ds)
        meta = A.raw_rows(ds)
        report[ds] = compare(ds, rows, ds)
        if ds == "hive":                    # domain-часть — на ней опубликованы цифры GLiNER Guard
            dom = [r for r in rows if meta[r["id"]]["meta"]["split"] == "domain"]
            report["hive_domain"] = compare(ds, dom, "hive_domain")
    for ds in ("v2", "bench", "ext"):
        report[ds] = compare(ds, load_dataset(ds), ds, PAIRS_LATEST)
    lat = {}
    for ds in ("hive", "alro", "mcr"):
        lat[ds] = {s: float(np.median([p["ms"] for p in A.load_preds(s, ds).values()]))
                   for s in ("S5-v3", "S5-r7", "S5-prev", "S6", "S7")}
    report["latency_median_ms"] = lat
    with open(os.path.join(A.RES, "competitors_paired.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for ds in ("hive", "hive_domain", "alro", "mcr", "v2", "bench", "ext"):
        print(f"== {ds}")
        for pair, res in report[ds]["pairs"].items():
            print(f"   {pair:16}", {m: (res[m]["diff"], res[m]["verdict"]) for m in ("f1", "full_masked",
                                                                                   "precision", "false_person_rate")})


if __name__ == "__main__":
    main()
