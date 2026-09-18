"""
Разбор замера на трёх новых внешних наборах (PROTOCOL.md, поправка 4).

Для каждого набора (hive, alro, mcr):
  1. ФИО у всех систем: точность, полнота, строгий F1, F1 по пересечению, доля
     полностью скрытых имён, доля текстов без ФИО с ложной маской ФИО; 95 % ДИ
     (кластерный бутстрап по примерам) и парная разность «сервис v3 минус система».
  2. Сервис (v3 и prev): утечки по типам ПДн — значение целиком осталось в выводе.
     Для многословных значений (адрес) это мягкая мера: частично замаскированный
     адрес утечкой не считается; отдельно — доля адресов, закрытых хотя бы частично.
  3. Маски ФИО внутри эталонного адреса («ул. Олега Кошевого») — отдельной строкой:
     текст всё равно скрыт, это не ошибка приватности.
  4. Разбивка по категориям метаданных: где сервис теряет имена и где ставит лишние маски.
     Для hive — отдельно domain-часть: на ней в статье GLiNER Guard (arXiv 2605.05277)
     опубликован строгий F1 по NAME: 75,7 / 69,5 / 60,6 / 52,3 / 30,3 у пяти систем.

Запуск: .venv/bin/python data/eval_v2/analyze_external2.py  -> results/external2_analysis.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
from run_eval import load_dataset  # noqa: E402

RES = os.path.join(HERE, "results")
SYSTEMS = {"S5-v3": "сервис, модель v3", "S5-prev": "сервис, прежняя модель",
           "S6": "redmadrobot rubert-base-pii-ner", "S7": "конкурент alrosait spaCy"}
N_ITER, SEED = 2000, 2026


def load_preds(system, ds):
    path = os.path.join(RES, f"preds_{system}_{ds}.jsonl")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def raw_rows(ds):
    with open(os.path.join(HERE, ds + ".jsonl"), encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def counts(r, pred):
    gold = [(s["start"], s["stop"]) for s in r["spans"] if s["type"] == "PERSON"]
    has_person = bool(gold)
    c = M.example_counts(r["text"], gold, pred, is_neg=not has_person)
    return np.array([c["gold_n"], c["pred_n"], c["tp_strict"], c["tp_overlap"],
                     c["full_masked"], c["neg_texts"], c["neg_texts_with_pred"]], float)


def summ(v):
    g, p, ts, to, full, neg, negp = v
    pr = ts / p if p else 0.0
    rc = ts / g if g else 0.0
    po, ro = (to / p if p else 0.0), (to / g if g else 0.0)
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "f1_overlap": 2 * po * ro / (po + ro) if po + ro else 0.0,
            "full_masked": full / g if g else 0.0,
            "false_person_rate": negp / neg if neg else 0.0,
            "gold": int(g), "pred": int(p)}


def person_block(ds, rows):
    vec = {s: np.array([counts(r, [tuple(x) for x in load_preds(s, ds)[r["id"]]["spans"]])
                        for r in rows]) for s in SYSTEMS}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(rows), size=(N_ITER, len(rows)))
    out = {}
    for s, m in vec.items():
        est = summ(m.sum(0))
        boot = [summ(m[i].sum(0)) for i in idx]
        ci = {k: [round(float(np.percentile([b[k] for b in boot], q)), 4) for q in (2.5, 97.5)]
              for k in ("f1", "full_masked", "false_person_rate")}
        diff = {}
        if s != "S5-v3":
            base = vec["S5-v3"]
            for k in ("f1", "full_masked"):
                d = [summ(base[i].sum(0))[k] - summ(m[i].sum(0))[k] for i in idx]
                diff[k] = [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]
        out[s] = {"estimate": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in est.items()},
                  "ci95": ci, "v3_minus_system_ci95": diff}
    return out


def service_block(ds, rows):
    res = {}
    for s in ("S5-v3", "S5-prev"):
        P = load_preds(s, ds)
        tot, leak, part = Counter(), Counter(), Counter()
        in_addr = 0
        for r in rows:
            anon = P[r["id"]].get("anonymized", r["text"])
            for sp in r["spans"]:
                if sp["type"].startswith("OTHER_"):
                    continue
                val = r["text"][sp["start"]:sp["stop"]]
                tot[sp["type"]] += 1
                if val in anon:
                    leak[sp["type"]] += 1
            addr = [(sp["start"], sp["stop"]) for sp in r["spans"] if sp["type"] == "ADDRESS"]
            for a, b in P[r["id"]]["spans"]:
                if any(x <= a and b <= y for x, y in addr):
                    in_addr += 1
            if addr:
                for x, y in addr:
                    if r["text"][x:y] not in anon:
                        part["ADDRESS_touched"] += 1
        res[s] = {"leak_rate_by_type": {t: round(leak[t] / tot[t], 4) for t in sorted(tot)},
                  "values_by_type": dict(tot), "leaked_by_type": dict(leak),
                  "leak_rate_all": round(sum(leak.values()) / max(1, sum(tot.values())), 4),
                  "person_masks_inside_gold_address": in_addr,
                  "median_ms": float(np.median([p["ms"] for p in P.values()]))}
        clean = [r for r in rows if not r["spans"]]
        res[s]["clean_texts_any_mask"] = round(
            sum(P[r["id"]].get("anonymized", r["text"]) != r["text"] for r in clean) / len(clean), 4) \
            if clean else None
    return res


def category_block(ds, rows, keys):
    P = load_preds("S5-v3", ds)
    Q = load_preds("S5-prev", ds)
    out = {}
    for key in keys:
        stat = defaultdict(lambda: np.zeros(7))
        stat_prev = defaultdict(lambda: np.zeros(7))
        for r in rows:
            cat = str(r.get("meta", {}).get(key))
            stat[cat] += counts(r, [tuple(x) for x in P[r["id"]]["spans"]])
            stat_prev[cat] += counts(r, [tuple(x) for x in Q[r["id"]]["spans"]])
        out[key] = {}
        for cat, v in sorted(stat.items(), key=lambda kv: -kv[1][0] - kv[1][5]):
            a, b = summ(v), summ(stat_prev[cat])
            out[key][cat] = {"names": a["gold"], "f1_v3": round(a["f1"], 3), "f1_prev": round(b["f1"], 3),
                             "full_masked_v3": round(a["full_masked"], 3),
                             "full_masked_prev": round(b["full_masked"], 3),
                             "texts_without_name": int(v[5]),
                             "false_person_v3": round(a["false_person_rate"], 3),
                             "false_person_prev": round(b["false_person_rate"], 3)}
    return out


def main():
    report = {}
    cats = {"hive": ["split", "domain"], "alro": ["name_form", "edge_case", "neg_category", "style",
                                          "has_typo", "domain"], "mcr": []}
    for ds in ("hive", "alro", "mcr"):
        rows = load_dataset(ds)
        meta = raw_rows(ds)
        for r in rows:
            r["meta"] = meta[r["id"]].get("meta", {})
        report[ds] = {"person": person_block(ds, rows), "service": service_block(ds, rows),
                      "categories": category_block(ds, rows, cats[ds])}
        print(f"\n══ {ds} ({len(rows)} текстов)")
        print(f"  {'система':34}{'P':>7}{'R':>7}{'F1':>7}{'F1ov':>7}{'скрыто':>8}{'ложн.':>7}   F1 95% ДИ")
        for s, d in report[ds]["person"].items():
            e = d["estimate"]
            print(f"  {SYSTEMS[s]:34}{e['precision']:7.3f}{e['recall']:7.3f}{e['f1']:7.3f}"
                  f"{e['f1_overlap']:7.3f}{e['full_masked']:8.3f}{e['false_person_rate']:7.3f}   {d['ci95']['f1']}")
        for s, d in report[ds]["service"].items():
            print(f"  {s}: утечки {d['leak_rate_all']:.3f} {d['leak_rate_by_type']}  "
                  f"ФИО внутри адреса {d['person_masks_inside_gold_address']}  "
                  f"чистых с маской {d['clean_texts_any_mask']}  медиана {d['median_ms']:.1f} мс")
    with open(os.path.join(RES, "external2_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
