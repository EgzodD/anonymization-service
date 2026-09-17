"""
Гейт качества на dev-наборах — храповик (этап 0.3 плана улучшений).

Зачем рядом со старым test_leakrate_regression.py. Старый гейт проверяет
held-out, порождённый тем же генератором, что и обучение: 97 % его примеров
построены по знакомым модели шаблонам, и сервис показывает там 0 утечек при
любом разумном изменении. Такой гейт не способен заметить ухудшение.

Этот гейт меряет сервис на dev-наборах, которые с обучением не пересекаются
(data/eval_v2/dev_v2.jsonl — наш домен, dev_rmr.jsonl — чужой), по трём
показателям:
  * доля значений ПДн, оставшихся в тексте (утечки);
  * доля текстов без ПДн, где сервис что-либо замаскировал (перемаскирование);
  * доля полностью замаскированных имён.

Пороги лежат в data/eval_v2/quality_thresholds.json и только ужесточаются.

БЕЗОПАСНОСТЬ: значения ПДн в сообщениях не выводятся — только счётчики,
типы и идентификаторы примеров (синтетических, но правило единое).
"""
import json
import os
import sys
from collections import Counter

import pytest

from app.anonymizer import anonymize_text

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eval_v2")
sys.path.insert(0, EVAL_DIR)
import metrics as M  # noqa: E402

with open(os.path.join(EVAL_DIR, "quality_thresholds.json"), encoding="utf-8") as _f:
    THRESHOLDS = json.load(_f)["datasets"]


def _measure(name):
    path = os.path.join(EVAL_DIR, f"{name}.jsonl")
    if not os.path.isfile(path):
        pytest.skip(f"dev-набор не найден: {path}")
    vals = leaked = neg = neg_masked = gold_p = full_p = 0
    leak_types, leak_ids = Counter(), []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    for r in rows:
        text = r["text"]
        res = anonymize_text(text)
        anon = res["anonymized"]
        pred = M.normalize(text, [(e["start"], e["end"]) for e in res["entities_found"]
                                  if e["entity_type"] == "PERSON"])
        gold = M.normalize(text, [(s["start"], s["stop"]) for s in r["spans"]
                                  if s["type"] == "PERSON"])
        gold_p += len(gold)
        full_p += M.count_coverage(text, gold, pred)
        for s in r["spans"]:
            if s["type"].startswith("OTHER_"):
                continue
            vals += 1
            if text[s["start"]:s["stop"]] in anon:
                leaked += 1
                leak_types[s["type"]] += 1
                leak_ids.append(r["id"])
        if r["subset"] == "neg":
            neg += 1
            neg_masked += int(anon != text)
    return {
        "leak_rate": leaked / vals if vals else 0.0,
        "neg_any_mask_rate": neg_masked / neg if neg else 0.0,
        "person_full_masked": full_p / gold_p if gold_p else 1.0,
        "leaked": leaked, "vals": vals, "leak_types": dict(leak_types),
        "leak_ids": leak_ids[:10],
    }


@pytest.mark.privacy
@pytest.mark.requires_model
@pytest.mark.parametrize("name", sorted(THRESHOLDS))
def test_quality_gate(name):
    m, t = _measure(name), THRESHOLDS[name]
    problems = []
    if m["leak_rate"] > t["max_leak_rate"] + 1e-9:
        problems.append(
            f"утечки {m['leak_rate']:.3f} > порога {t['max_leak_rate']:.3f} "
            f"({m['leaked']}/{m['vals']}, по типам {m['leak_types']}, примеры {m['leak_ids']})")
    if m["neg_any_mask_rate"] > t["max_neg_any_mask_rate"] + 1e-9:
        problems.append(
            f"перемаскирование текстов без ПДн {m['neg_any_mask_rate']:.3f} > порога "
            f"{t['max_neg_any_mask_rate']:.3f}")
    if m["person_full_masked"] < t["min_person_full_masked"] - 1e-9:
        problems.append(
            f"полностью замаскированных имён {m['person_full_masked']:.3f} < порога "
            f"{t['min_person_full_masked']:.3f}")
    assert not problems, f"Гейт качества «{name}» не пройден: " + "; ".join(problems) + \
        ". Значения ПДн не выводятся намеренно."
