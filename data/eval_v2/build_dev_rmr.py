"""
Второй dev-набор для настройки (этап 0.1b плана) — из чужого домена.

Источник — ОБУЧАЮЩИЙ корпус redmadrobot-rnd/pii_train (MIT, 17 137 предложений).
Их тестовый набор pii_benchmark остаётся только для приёмки (правило П1),
поэтому ошибки «чужого» домена изучаются и правятся здесь.

Проверяется, что ни один текст не совпадает с pii_benchmark: иначе настройка
на этом dev незаметно подстроила бы сервис под приёмочный набор.

Преобразование — теми же функциями, что и для бенчмарка (load_bench.py):
части имени склеиваются в PERSON, позиции восстанавливаются выравниванием.

Запуск:
    .venv/bin/python data/eval_v2/build_dev_rmr.py
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import load_bench as LB  # noqa: E402

OUT = os.path.join(HERE, "dev_rmr.jsonl")
REPORT = os.path.join(HERE, "dev_rmr_report.json")
SEED = 20260917
SIZE = 1500


def norm(t):
    return " ".join(t.lower().split())


def main():
    from datasets import load_dataset
    train = load_dataset("redmadrobot-rnd/pii_train")["train"]
    bench_texts = {norm(r["text"]) for r in
                   (json.loads(x) for x in open(os.path.join(HERE, "bench.jsonl"), encoding="utf-8"))}

    idx = list(range(len(train)))
    random.Random(SEED).shuffle(idx)
    rows, dropped_align, dropped_overlap = [], 0, 0
    for i in idx:
        if len(rows) >= SIZE:
            break
        ex = train[i]
        if norm(ex["text"]) in bench_texts:
            dropped_overlap += 1
            continue
        raw = LB.spans_from_bio(ex["text"], json.loads(ex["tokens"]), json.loads(ex["ner_tags"]))
        if raw is None:
            dropped_align += 1
            continue
        spans = LB.merge_person(ex["text"], raw)
        rows.append({
            "id": f"rmr-{i:05d}", "template_id": f"rmr-{i:05d}",
            "subset": "pii" if any(not s["type"].startswith("OTHER_") for s in spans) else "neg",
            "text": ex["text"], "spans": spans,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {
        "source": "redmadrobot-rnd/pii_train (MIT), случайная выборка",
        "purpose": "настройка (не приёмка)", "seed": SEED,
        "examples": len(rows),
        "examples_neg": sum(r["subset"] == "neg" for r in rows),
        "skipped_text_also_in_benchmark": dropped_overlap,
        "skipped_alignment_failed": dropped_align,
        "entities_by_type": dict(Counter(s["type"] for r in rows for s in r["spans"]).most_common()),
        "sha256": hashlib.sha256(open(OUT, "rb").read()).hexdigest(),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
