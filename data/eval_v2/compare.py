"""
Сравнение сервиса «до / после» правки (этап 0.2 плана улучшений).

Сравниваются два файла предсказаний сервиса (S5) на одном наборе. По умолчанию
«до» — снимок в results/baseline/, «после» — свежий прогон в results/.

Метрики:
  ФИО        — точность, полнота, F1 (строгое совпадение), доля полностью
               замаскированных имён, лишние маски, пропущенные имена;
  утечки     — доля значений ПДн каждого типа, оставшихся в выходном тексте;
  лишнее     — доля текстов без ПДн, где сервис что-либо замаскировал
               (контроль перемаскирования номеров заказов, городов и т. п.).

Разности — с 95 % доверительным интервалом (парный кластерный бутстрап по
шаблонам). Отдельно печатаются примеры, которые правка СЛОМАЛА: значение было
закрыто и стало утекать, либо в тексте без ПДн появилась лишняя маска.

Использование (набор — только dev, правило П1):
    .venv/bin/python data/eval_v2/run_eval.py --system S5 --dataset dev_v2
    .venv/bin/python data/eval_v2/compare.py --dataset dev_v2
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
from run_eval import load_dataset  # noqa: E402

FIELDS = ["gold_p", "pred_p", "tp_p", "full_p", "neg", "neg_person", "clean", "clean_anymask",
          "vals", "leaked"]


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(x) for x in f)}


def example_vector(r, p):
    text, anon = r["text"], p.get("anonymized", r["text"])
    gold = [(s["start"], s["stop"]) for s in r["spans"] if s["type"] == "PERSON"]
    c = M.example_counts(text, gold, [tuple(x) for x in p["spans"]], r["subset"] == "neg")
    vals = [s for s in r["spans"] if not s["type"].startswith("OTHER_")]
    leaked = [s for s in vals if text[s["start"]:s["stop"]] in anon]
    is_neg = r["subset"] == "neg"
    # Перемаскирование считается только по текстам ВОВСЕ без разметки. В чужих
    # наборах «neg» (нет поддерживаемых нами типов) часто содержит ПДн, для которых
    # у нас нет типа: водительское удостоверение, военный билет, ОМС. Маска на них —
    # не перемаскирование, а правильное закрытие ПДн под неточной меткой.
    is_clean = not r["spans"]
    return np.array([
        c["gold_n"], c["pred_n"], c["tp_strict"], c["full_masked"],
        int(is_neg), c["neg_texts_with_pred"], int(is_clean), int(is_clean and anon != text),
        len(vals), len(leaked),
    ], dtype=float), leaked


def summarize(t):
    t = dict(zip(FIELDS, t))
    p = t["tp_p"] / t["pred_p"] if t["pred_p"] else 0.0
    r = t["tp_p"] / t["gold_p"] if t["gold_p"] else 0.0
    return {
        "ФИО: точность": p,
        "ФИО: полнота": r,
        "ФИО: F1": 2 * p * r / (p + r) if p + r else 0.0,
        "ФИО: замаскировано полностью": t["full_p"] / t["gold_p"] if t["gold_p"] else 0.0,
        "ФИО: лишних масок (шт.)": t["pred_p"] - t["tp_p"],
        "ФИО: пропущено имён (шт.)": t["gold_p"] - t["full_p"],
        "Без ПДн: ложное ФИО (доля текстов)": t["neg_person"] / t["neg"] if t["neg"] else 0.0,
        "Без разметки: любая маска (доля текстов)": (t["clean_anymask"] / t["clean"]
                                                     if t["clean"] else 0.0),
        "Утечки ПДн, все типы (доля)": t["leaked"] / t["vals"] if t["vals"] else 0.0,
    }


BOOT = ["ФИО: замаскировано полностью", "ФИО: F1", "Без ПДн: ложное ФИО (доля текстов)",
        "Без разметки: любая маска (доля текстов)", "Утечки ПДн, все типы (доля)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--show", type=int, default=15, help="сколько сломанных примеров показать")
    a = ap.parse_args()
    if a.dataset in ("v2", "bench", "ext"):
        print(f"ВНИМАНИЕ: «{a.dataset}» — приёмочный набор. По правилу П1 настраиваться по нему "
              "нельзя; сравнение допустимо только на финальном замере (этап 4).\n")
    before = a.before or os.path.join(HERE, "results", "baseline", f"preds_S5_{a.dataset}.jsonl")
    after = a.after or os.path.join(HERE, "results", f"preds_S5_{a.dataset}.jsonl")
    rows = load_dataset(a.dataset)
    PB, PA = load_preds(before), load_preds(after)

    clusters = {}
    type_tot, type_b, type_a = Counter(), Counter(), Counter()
    broken_leak, broken_mask, fixed_leak = [], [], []
    ms_b, ms_a = [], []
    for r in rows:
        vb, lb = example_vector(r, PB[r["id"]])
        va, la = example_vector(r, PA[r["id"]])
        cb, ca = clusters.setdefault(r["template_id"], [0, 0])
        clusters[r["template_id"]] = [cb + vb, ca + va]
        ms_b.append(PB[r["id"]]["ms"])
        ms_a.append(PA[r["id"]]["ms"])
        for s in r["spans"]:
            if not s["type"].startswith("OTHER_"):
                type_tot[s["type"]] += 1
        for s in lb:
            type_b[s["type"]] += 1
        for s in la:
            type_a[s["type"]] += 1
        kb = {(s["start"], s["stop"]) for s in lb}
        ka = {(s["start"], s["stop"]) for s in la}
        for s in la:
            if (s["start"], s["stop"]) not in kb:
                broken_leak.append((r, s))
        for s in lb:
            if (s["start"], s["stop"]) not in ka:
                fixed_leak.append((r, s))
        if not r["spans"]:
            ab, aa = PB[r["id"]].get("anonymized", r["text"]), PA[r["id"]].get("anonymized", r["text"])
            if ab == r["text"] and aa != r["text"]:
                broken_mask.append((r, aa))

    keys = sorted(clusters)
    mb = np.array([clusters[k][0] for k in keys])
    ma = np.array([clusters[k][1] for k in keys])
    sb, sa = summarize(mb.sum(axis=0)), summarize(ma.sum(axis=0))

    rng = np.random.default_rng(2026)
    idx = rng.integers(0, len(keys), size=(a.n_iter, len(keys)))
    diffs = {m: np.empty(a.n_iter) for m in BOOT}
    for i in range(a.n_iter):
        xb, xa = summarize(mb[idx[i]].sum(axis=0)), summarize(ma[idx[i]].sum(axis=0))
        for m in BOOT:
            diffs[m][i] = xa[m] - xb[m]

    print(f"Набор: {a.dataset} ({len(rows)} примеров, кластеров {len(keys)})")
    print(f"  до:    {os.path.relpath(before, HERE)}")
    print(f"  после: {os.path.relpath(after, HERE)}\n")
    print(f"{'Метрика':38}{'до':>9}{'после':>9}{'разница':>10}   ДИ 95 %")
    for m in sb:
        d = sa[m] - sb[m]
        if m in BOOT:
            lo, hi = np.percentile(diffs[m], [2.5, 97.5])
            sig = "  *" if (lo > 0 or hi < 0) else ""
            ci = f"[{lo:+.3f}; {hi:+.3f}]{sig}"
        else:
            ci = ""
        fmt = "{:9.0f}" if "(шт.)" in m else "{:9.3f}"
        print(f"{m:38}{fmt.format(sb[m])}{fmt.format(sa[m])}{d:+10.3f}   {ci}")
    print(f"{'Задержка, мс (медиана)':38}{np.median(ms_b):9.1f}{np.median(ms_a):9.1f}"
          f"{np.median(ms_a) - np.median(ms_b):+10.1f}")
    print("  * — разница значима: ДИ не содержит нуля\n")

    print(f"{'Утечки по типам':38}{'до':>9}{'после':>9}{'всего':>9}")
    for t in sorted(type_tot):
        print(f"  {t:36}{type_b[t]:9}{type_a[t]:9}{type_tot[t]:9}")

    # Имя, которое было закрыто хотя бы частично, а стало открыто полностью. Метрика
    # «замаскировано полностью» этого не видит: если имя и раньше было закрыто не
    # целиком, её значение не меняется, хотя приватность стала хуже.
    uncovered = []
    for r in rows:
        text = r["text"]
        for s in r["spans"]:
            if s["type"] != "PERSON":
                continue
            g = (s["start"], s["stop"])
            def touched(p, g=g):
                return any(max(0, min(b, g[1]) - max(a, g[0])) > 0 for a, b in p["spans"])
            if touched(PB[r["id"]]) and not touched(PA[r["id"]]):
                uncovered.append((r, text[g[0]:g[1]]))
    print(f"\nИмён, ставших полностью открытыми: {len(uncovered)}")
    for r, v in uncovered[:a.show]:
        print(f"  [открыто] {r['id']}: {v!r}")

    print(f"\nИСПРАВЛЕНО утечек: {len(fixed_leak)}   СЛОМАНО (новые утечки): {len(broken_leak)}   "
          f"новых масок в текстах без разметки: {len(broken_mask)}")
    for r, s in broken_leak[:a.show]:
        print(f"  [утечка {s['type']}] {r['id']}: {r['text'][s['start']:s['stop']]!r}  "
              f"<- {r['text'][:80]!r}")
    for r, aa in broken_mask[:a.show]:
        print(f"  [лишняя маска] {r['id']}: {aa[:90]!r}")


if __name__ == "__main__":
    main()
