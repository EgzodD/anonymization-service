"""
Проверка критериев финальной приёмки модели этапа 3 (PROTOCOL.md, поправка 3).

Критерии записаны и закоммичены (740d399) до прогона кандидата на v2, bench, ext.
Метрики считаются функциями compare.py — те же определения, что на dev.

Запуск (после run_eval.py с метками final-prod и final-B):
    .venv/bin/python data/eval_v2/acceptance_stage3.py
Результат: results/acceptance_stage3.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare as C  # noqa: E402
from run_eval import load_dataset  # noqa: E402

RES = os.path.join(HERE, "results")
N_ITER, SEED = 5000, 2026
PRIMARY = ["ФИО: F1", "ФИО: замаскировано полностью"]


def cluster_matrix(dataset, tag):
    rows = load_dataset(dataset)
    preds = C.load_preds(os.path.join(RES, f"preds_S5-{tag}_{dataset}.jsonl"))
    clusters = {}
    for r in rows:
        v, _ = C.example_vector(r, preds[r["id"]])
        clusters[r["template_id"]] = clusters.get(r["template_id"], 0) + v
    keys = sorted(clusters)
    return keys, np.array([clusters[k] for k in keys])


def paired(dataset):
    keys, mb = cluster_matrix(dataset, "final-prod")
    keys_a, ma = cluster_matrix(dataset, "final-B")
    assert keys == keys_a
    sb, sa = C.summarize(mb.sum(axis=0)), C.summarize(ma.sum(axis=0))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), size=(N_ITER, len(keys)))
    out = {"clusters": len(keys), "prod": sb, "B": sa, "diff_ci95": {}}
    for m in C.BOOT:
        d = np.array([C.summarize(ma[i].sum(axis=0))[m] - C.summarize(mb[i].sum(axis=0))[m]
                      for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["diff_ci95"][m] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def main():
    res = {ds: paired(ds) for ds in ("v2", "bench", "ext")}
    with open(os.path.join(RES, "preds_S4-final-B_v2.jsonl"), encoding="utf-8") as f:
        ms = [json.loads(x)["ms"] for x in f]
    latency = float(np.median(ms))

    crit = [
        ("1. v2: полностью скрыто имён >= 0,93",
         res["v2"]["B"]["ФИО: замаскировано полностью"], lambda x: x >= 0.93),
        ("2. v2: ложное ФИО в текстах без ПДн <= 0,05",
         res["v2"]["B"]["Без ПДн: ложное ФИО (доля текстов)"], lambda x: x <= 0.05),
        ("3. ext: строгая точность по ФИО >= 0,60",
         res["ext"]["B"]["ФИО: точность"], lambda x: x >= 0.60),
        ("4. задержка голой модели, медиана на v2 <= 5 мс", latency, lambda x: x <= 5.0),
    ]
    worse = [f"{ds}: {m} {res[ds]['diff_ci95'][m]}" for ds in res for m in PRIMARY
             if res[ds]["diff_ci95"][m][1] < 0]
    crit.append(("5. парно не хуже боевой на v2/bench/ext (F1, полностью скрыто)",
                 "значимо хуже: " + "; ".join(worse) if worse else "нет значимых ухудшений",
                 lambda x, w=worse: not w))

    verdict = []
    print(f"{'Критерий':62}{'значение':>28}  итог")
    for name, val, ok in crit:
        passed = bool(ok(val))
        verdict.append(passed)
        sval = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"{name:62}{sval:>28}  {'ВЫПОЛНЕН' if passed else 'НЕ ВЫПОЛНЕН'}")
    decision = "ПРОДВИГАТЬ" if all(verdict) else "НЕ ПРОДВИГАТЬ"
    print(f"\nРешение по протоколу: {decision}\n")

    for ds in res:
        print(f"── {ds} (кластеров {res[ds]['clusters']})")
        print(f"   {'метрика':42}{'боевая':>9}{'B':>9}   ДИ 95 % разности")
        for m in res[ds]["prod"]:
            ci = res[ds]["diff_ci95"].get(m, "")
            fmt = "{:9.0f}" if "(шт.)" in m else "{:9.3f}"
            print(f"   {m:42}{fmt.format(res[ds]['prod'][m])}{fmt.format(res[ds]['B'][m])}   {ci}")

    out = {"protocol": "PROTOCOL.md, поправка 3 (коммит 740d399)", "n_iter": N_ITER, "seed": SEED,
           "criteria": [{"name": n, "value": v if isinstance(v, (int, float, str)) else str(v),
                         "passed": p} for (n, v, _), p in zip(crit, verdict)],
           "decision": decision, "model_latency_median_ms": latency, "datasets": res}
    with open(os.path.join(RES, "acceptance_stage3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
