"""
Точка входа синтезатора: сборка набора, автопроверки, отчёт.

Запуск:
    .venv/bin/python -m synthesizer.generate --n 50
    .venv/bin/python -m synthesizer.generate --n 5000 --out synthesizer/runs/train_v4.jsonl

Что считается браком (поле checks, reject=True):
  unknown_tag     метка не из списка, либо метка человека без падежа;
  leftover        после подстановки в тексте остались скобки;
  duplicate       такой же текст (с точностью до меток) уже был;
  near_duplicate  Жаккар по тройкам слов с принятым текстом выше порога;
  no_tags         текст без меток там, где сценарий их требовал;
  no_person       сценарий просил человека, а модель его не упомянула;
  extra_pii       персональные данные вписала сама модель — в обучающих данных окажется
                  неразмеченное имя или номер.

Что браком НЕ считается. Набор синтетический, поэтому всё вне разметки персональными
данными не является по построению. Если сервис находит ФИО на трудном негативе
(«тариф «Евгений Онегин»») или на слове, которое мы испортили опечаткой («чтобыы»), —
это его ложное срабатывание, а не дефект текста. Такие случаи идут в отчёт отдельно
(person_fp_on_negatives, person_fp_on_typos) и дают бесплатную оценку лишних масок
на каждом прогоне.

Синтетика — для обучения и поиска ошибок, НЕ для приёмки: проверка на синтетике того же
происхождения завышает оценку (вывод статьи, раздел 2).
"""
import argparse
import datetime as dt
import json
import os
import random
import re

from .corrupt import corrupt
from .dedup import NearDup
from .llm import ask
from .scenarios import DOMAINS, build_prompt, scenario, wants_person
from .values import ROOT, TAG_RE, fill

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "runs", f"{dt.date.today():%Y-%m-%d}__run.jsonl")


def overlaps(res, spans):
    return any(res.start < x["stop"] and x["start"] < res.end for x in spans)


def make_row(tpl, sc, rng, opts, dup, seen, analyze_text):
    """Один текст: подстановка, порча, проверки. Возвращает (строка набора, тройки слов)."""
    text, spans, negs, errors = fill(tpl, rng)
    if sc["lower"]:                              # длины не меняются — спаны остаются верными
        text = text.lower()
    marks = []
    if rng.random() < opts.typo_share:
        text, spans, negs, marks = corrupt(text, spans, negs, rng)
        sc["typos"] = len(marks)
    key = " ".join(TAG_RE.sub("<X>", tpl).lower().split())
    sim, shingles = dup.worst(text)
    checks = {"unknown_tag": errors, "leftover": bool(re.search(r"[\[\]{}]", text)),
              "duplicate": key in seen,
              "near_duplicate": sim if sim >= opts.dup_threshold else 0,
              "no_tags": bool(sc["pii"]) and not spans,
              "no_person": wants_person(sc) and not any(x["type"] == "PERSON" for x in spans)}
    seen.add(key)

    found = [r for r in analyze_text(text) if not overlaps(r, spans)]
    on_neg = [overlaps(r, negs) for r in found]
    on_typo = [any(r.start < b and a < r.end for a, b in marks) for r in found]
    checks["extra_pii"] = [f"{r.entity_type}: {text[r.start:r.end]}"
                           for r, neg, typo in zip(found, on_neg, on_typo) if not neg and not typo]
    row = {"text": text, "spans": spans, "negative_spans": negs, "template": tpl, "scenario": sc,
           "model": opts.model, "checks": checks, "similarity": sim,
           "person_fp_on_negatives": [text[r.start:r.end] for r, neg in zip(found, on_neg)
                                      if neg and r.entity_type == "PERSON"],
           "person_fp_on_typos": [text[r.start:r.end] for r, neg, typo in zip(found, on_neg, on_typo)
                                  if typo and not neg and r.entity_type == "PERSON"],
           "reject": any(checks.values())}
    return row, shingles


def report(rows, stat, model):
    rej = sum(r["reject"] for r in rows)
    good = [r for r in rows if not r["reject"]]
    seconds = max(stat["seconds"], 1e-9)
    return {"model": model, "texts": len(rows), "rejected": rej,
            "reject_share": round(rej / len(rows), 3),
            "reasons": {k: sum(bool(r["checks"][k]) for r in rows) for k in rows[0]["checks"]},
            "requests": stat["requests"], "tokens": stat["tokens"],
            "minutes": round(stat["seconds"] / 60, 1),
            "tokens_per_sec": round(stat["tokens"] / seconds, 1),
            "sec_per_text": round(stat["seconds"] / max(len(rows), 1), 1),
            "scenarios": {"negative_only": sum(not r["scenario"]["pii"] for r in rows),
                          "with_negatives": sum(bool(r["negative_spans"]) for r in rows),
                          "with_typos": sum(bool(r["scenario"]["typos"]) for r in rows),
                          "lowercase": sum(r["scenario"]["lower"] for r in rows),
                          "domains": len({r["scenario"]["domain"] for r in rows})},
            "typos": {"words": sum(r["scenario"]["typos"] for r in rows),
                      "person_fp": sum(len(r["person_fp_on_typos"]) for r in rows),
                      "examples": sorted({x for r in rows for x in r["person_fp_on_typos"]})[:10]},
            "negatives": {"spans": sum(len(r["negative_spans"]) for r in rows),
                          "person_fp": sum(len(r["person_fp_on_negatives"]) for r in rows),
                          "examples": sorted({x for r in rows
                                              for x in r["person_fp_on_negatives"]})[:10]},
            "similarity": {"max_accepted": max((r["similarity"] for r in good), default=0.0),
                           "mean_accepted": round(sum(r["similarity"] for r in good)
                                                  / max(len(good), 1), 3)}}


def main():
    ap = argparse.ArgumentParser(description="Генерация обучающих данных с точной разметкой ПДн")
    ap.add_argument("--model", default="qwen3:8b", help="модель в Ollama")
    ap.add_argument("--n", type=int, default=50, help="сколько текстов сгенерировать")
    ap.add_argument("--per-request", type=int, default=3,
                    help="текстов за запрос; чем больше, тем хуже модель держит список требований")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--neg-share", type=float, default=0.25,
                    help="доля текстов только из трудных негативов, без ПДн")
    ap.add_argument("--typo-share", type=float, default=0.3, help="доля текстов с опечатками")
    ap.add_argument("--dup-threshold", type=float, default=0.6,
                    help="Жаккар по тройкам слов, выше которого текст считается почти-дублем")
    opts = ap.parse_args()

    model_dir = os.path.join(ROOT, "models", "person_ruBERT")
    if os.path.isdir(model_dir):          # с моделью ложные ФИО на негативах видны честно
        os.environ.setdefault("PERSON_MODEL_DIR", model_dir)
    else:
        os.environ.setdefault("ALLOW_NO_PERSON_MODEL", "true")
    from app.anonymizer import analyze_text

    rng = random.Random(opts.seed)
    dup = NearDup(opts.dup_threshold)
    rows, seen, stat = [], set(), {"requests": 0, "tokens": 0, "seconds": 0.0}
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    while len(rows) < opts.n:
        domain = rng.choice(DOMAINS)
        scens = [scenario(rng, opts.neg_share) for _ in range(opts.per_request)]
        try:
            texts, info = ask(opts.model, build_prompt(opts.per_request, domain, scens))
        except Exception as exc:  # noqa: BLE001 — генерация фоновая: фиксируем и идём дальше
            print("ошибка запроса:", exc, flush=True)
            continue
        stat["requests"] += 1
        stat["tokens"] += info["tokens"] or 0
        stat["seconds"] += info["seconds"]
        for tpl, sc in zip(texts, scens):
            row, shingles = make_row(tpl, dict(sc, domain=domain), rng, opts, dup, seen, analyze_text)
            if not row["reject"]:
                dup.add(shingles)
            rows.append(row)
        print(f"запросов {stat['requests']}, текстов {len(rows)} "
              f"(годных {sum(not r['reject'] for r in rows)}), "
              f"{stat['tokens'] / max(stat['seconds'], 1e-9):.1f} ток/с", flush=True)

    rows = rows[:opts.n]
    with open(opts.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = report(rows, stat, opts.model)
    with open(opts.out.replace(".jsonl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
