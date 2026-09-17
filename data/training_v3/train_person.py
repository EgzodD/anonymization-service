"""
Локальное обучение модели ФИО на процессоре (этап 3 плана улучшений).

Та же постановка, что в прежнем ноутбуке Colab: rubert-tiny2, метки O/B/I-PERSON,
метка слова ставится на его первый подтокен, продолжения слова получают I-PERSON.

Выбор эпохи — ТОЛЬКО по dev-наборам (правило П1): после каждой эпохи модель
прогоняется по dev_v2 и dev_rmr со словоуровневой склейкой "first" (как в
протоколе оценки), считаются строгий F1 и доля полностью скрытых имён теми же
функциями, что и на приёмке (data/eval_v2/metrics.py). Критерий выбран заранее:
среднее строгого F1 по двум dev-наборам. dev_v2 — другой источник, чем pii_train,
он и страхует от переобучения под домен pii_train (п. 3.3 плана).

Запуск (долго, лучше в фоне):
    .venv/bin/python data/training_v3/train_person.py --corpus corpus_A.jsonl --out models/person_v3_A
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVAL = os.path.join(ROOT, "data", "eval_v2")
sys.path.insert(0, EVAL)
import metrics as M  # noqa: E402

BASE = "cointegrated/rubert-tiny2"
LABELS = ["O", "B-PERSON", "I-PERSON"]
L2I = {x: i for i, x in enumerate(LABELS)}
DEV_SETS = ("dev_v2", "dev_rmr")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def encode(tok, row, max_len):
    enc = tok(row["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
    labels, prev = [], None
    for wid in enc.word_ids():
        if wid is None:
            labels.append(-100)
        elif wid != prev:
            labels.append(L2I[row["tags"][wid]])
        else:
            labels.append(L2I["I-PERSON"] if row["tags"][wid] != "O" else L2I["O"])
        prev = wid
    return enc["input_ids"], labels


def batches(items, batch_size, rng):
    """Батчи из предложений близкой длины: меньше паддинга — быстрее на CPU."""
    idx = sorted(range(len(items)), key=lambda i: len(items[i][0]) + rng.random())
    chunks = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(chunks)
    return chunks


def collate(items, pad_id):
    n = max(len(x[0]) for x in items)
    ids = torch.full((len(items), n), pad_id)
    lab = torch.full((len(items), n), -100)
    att = torch.zeros((len(items), n), dtype=torch.long)
    for i, (a, b) in enumerate(items):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return ids, att, lab


def evaluate_dev(model, tok):
    from transformers import pipeline
    model.eval()
    pipe = pipeline("token-classification", model=model, tokenizer=tok,
                    aggregation_strategy="first", device=-1)
    res = {}
    for name in DEV_SETS:
        tot = dict.fromkeys(M.COUNT_FIELDS, 0)
        for r in read_jsonl(os.path.join(EVAL, name + ".jsonl")):
            gold = [(s["start"], s["stop"]) for s in r["spans"] if s["type"] == "PERSON"]
            pred = [(int(e["start"]), int(e["end"])) for e in pipe(r["text"])
                    if e["entity_group"].endswith("PERSON")]
            c = M.example_counts(r["text"], gold, pred, is_neg=not r["spans"])
            for k in tot:
                tot[k] += c[k]
        s = M.summarize(tot)
        res[name] = {"f1_strict": round(s["strict"]["f1"], 4),
                     "precision": round(s["strict"]["precision"], 4),
                     "recall": round(s["strict"]["recall"], 4),
                     "full_masked": round(s["full_masked_rate"], 4),
                     "neg_fp_rate": round(s["neg_fp_rate"], 4)}
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=7)
    a = ap.parse_args()

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    rows = read_jsonl(os.path.join(HERE, a.corpus))
    items = [encode(tok, r, a.max_len) for r in rows]
    steps = a.epochs * ((len(items) + a.batch - 1) // a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best, step, t0 = -1.0, 0, time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(json.dumps({"args": vars(a), "sentences": len(items), "steps": steps}) + "\n")
        model.train()
        for epoch in range(1, a.epochs + 1):
            losses = []
            for chunk in batches(items, a.batch, rng):
                ids, att, lab = collate([items[i] for i in chunk], tok.pad_token_id)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                losses.append(loss.item())
                step += 1
                if step % 200 == 0:
                    print(f"эпоха {epoch} шаг {step}/{steps} loss {np.mean(losses[-200:]):.4f} "
                          f"{(time.time() - t0) / 60:.1f} мин", flush=True)
            dev = evaluate_dev(model, tok)
            score = float(np.mean([dev[n]["f1_strict"] for n in DEV_SETS]))
            rec = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "dev": dev,
                   "criterion": round(score, 4), "minutes": round((time.time() - t0) / 60, 1)}
            if score > best:
                best = score
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
                rec["saved"] = True
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
    print(f"готово: лучшая эпоха по критерию {best:.4f}, модель в {out_dir}")


if __name__ == "__main__":
    main()
