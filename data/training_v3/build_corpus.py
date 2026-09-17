"""
Обучающий корпус модели ФИО, этап 3 плана улучшений.

Формат выхода — corpus_<вариант>.jsonl, по предложению в строке:
    {"src": "...", "tokens": [...], "tags": ["O", "B-PERSON", "I-PERSON", ...]}

Источники:
  pii    redmadrobot-rnd/pii_train (MIT). Исключены: 1500 предложений dev_rmr (по
         индексу) и любые тексты, совпадающие с dev_rmr или pii_benchmark после
         нормализации. Части имени (FIRST/LAST/MIDDLE_NAME) склеиваются в PERSON,
         если между ними нет ничего, кроме точек — то же правило, что в эталоне
         (data/eval_v2/load_bench.py). Разметка токенная, выравнивание по символам
         не нужно, поэтому используются все предложения, а не только выровненные.
  old    прежний корпус data/training/train (train.conll + train_negatives.conll),
         на котором обучена текущая модель. Лексиконы test_v2 с ним не пересекаются.
  gen    расширенный генератор (generate_v3.py), если файл gen_v3.jsonl собран.

Варианты:
  A   = pii + old            — проверка, даёт ли прирост сам pii_train
  B   = pii + old + gen      — полный корпус

Запуск:
    .venv/bin/python data/training_v3/build_corpus.py A
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVAL = os.path.join(ROOT, "data", "eval_v2")
OLD = os.path.join(ROOT, "data", "training", "train")
PARTS = {"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"}


def norm(t):
    return " ".join(t.lower().split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def person_tags(tokens, src_tags):
    """Теги источника -> B/I-PERSON. Соседние части имени (через точки) — одна сущность."""
    is_part = [t[2:] in PARTS for t in src_tags]
    out, inside = [], False
    for i, tok in enumerate(tokens):
        if is_part[i]:
            out.append("I-PERSON" if inside else "B-PERSON")
            inside = True
        elif inside and set(tok) == {"."} and i + 1 < len(tokens) and is_part[i + 1]:
            out.append("I-PERSON")          # точка между частями имени
        else:
            out.append("O")
            inside = False
    return out


def load_pii():
    from datasets import load_dataset
    ds = load_dataset("redmadrobot-rnd/pii_train")["train"]
    dev = read_jsonl(os.path.join(EVAL, "dev_rmr.jsonl"))
    dev_idx = {int(r["id"].split("-")[1]) for r in dev}
    banned = {norm(r["text"]) for r in dev}
    banned |= {norm(r["text"]) for r in read_jsonl(os.path.join(EVAL, "bench.jsonl"))}
    rows, skipped = [], Counter()
    for i, ex in enumerate(ds):
        if i in dev_idx:
            skipped["dev_rmr_index"] += 1
            continue
        if norm(ex["text"]) in banned:
            skipped["text_in_dev_or_bench"] += 1
            continue
        toks, tags = json.loads(ex["tokens"]), json.loads(ex["ner_tags"])
        rows.append({"src": "pii", "tokens": toks, "tags": person_tags(toks, tags)})
    return rows, dict(skipped)


def load_conll(path, src):
    rows, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    rows.append({"src": src, "tokens": toks, "tags": tags})
                toks, tags = [], []
                continue
            parts = line.split("\t")
            toks.append(parts[0])
            tags.append(parts[-1] if parts[-1].endswith("PERSON") else "O")
    if toks:
        rows.append({"src": src, "tokens": toks, "tags": tags})
    return rows


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "A"
    pii, skipped = load_pii()
    rows = pii + load_conll(os.path.join(OLD, "train.conll"), "old") \
        + load_conll(os.path.join(OLD, "train_negatives.conll"), "old_neg")
    if variant == "B":
        rows += read_jsonl(os.path.join(HERE, "gen_v3.jsonl"))
    out = os.path.join(HERE, f"corpus_{variant}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stat = Counter()
    for r in rows:
        stat[r["src"]] += 1
        stat[r["src"] + "_with_person"] += "B-PERSON" in r["tags"]
    print(json.dumps({"variant": variant, "sentences": len(rows), "by_source": dict(stat),
                      "pii_skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
