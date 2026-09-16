"""
Метрики оценки обнаружения ПДн + кластерный бутстрап.

Реализованы три способа сопоставления предсказанных и эталонных спанов:

  strict   — границы совпадают точно (после отбрасывания пробелов по краям);
             классическая схема CoNLL / SemEval-2013 «exact».
  overlap  — любое пересечение считается попаданием, сопоставление один-к-одному
             (жадно, по величине пересечения); схема SemEval-2013 «partial»
             в варианте, где частичное совпадение засчитывается полностью.
  coverage — эталонная сущность считается закрытой, только если КАЖДАЯ её
             словоформа полностью накрыта предсказаниями. Метрика приватности:
             частично накрытое ФИО означает, что часть имени осталась в тексте.

Самопроверка (ручной расчёт + сверка strict с seqeval):
    .venv/bin/python data/eval_v2/metrics.py
"""
import re

import numpy as np

WORD_RE = re.compile(r"\w+", re.UNICODE)


# ── подготовка спанов ─────────────────────────────────────────────────────
def normalize(text, spans):
    """Отбрасывает пробелы по краям спана, выкидывает пустые, сортирует."""
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return sorted(set(out))


# ── сопоставление ─────────────────────────────────────────────────────────
def count_strict(gold, pred):
    g, p = set(gold), set(pred)
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp        # tp, fp, fn


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def count_overlap(gold, pred):
    """Жадное сопоставление один-к-одному по величине пересечения."""
    pairs = sorted(
        ((_overlap(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    used_g, used_p, tp = set(), set(), 0
    for ov, gi, pi in pairs:
        if ov <= 0:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp, len(pred) - tp, len(gold) - tp


def count_coverage(text, gold, pred):
    """Сколько эталонных сущностей накрыты предсказаниями целиком (по словам)."""
    covered_chars = np.zeros(len(text) + 1, dtype=bool)
    for s, e in pred:
        covered_chars[s:e] = True
    full = 0
    for s, e in gold:
        words = [(s + m.start(), s + m.end()) for m in WORD_RE.finditer(text[s:e])]
        if words and all(covered_chars[ws:we].all() for ws, we in words):
            full += 1
    return full


# ── агрегация ─────────────────────────────────────────────────────────────
def prf(tp, fp, fn, beta=1.0):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = ((1 + beta ** 2) * p * r / (beta ** 2 * p + r)) if (p + r) else 0.0
    return p, r, f


COUNT_FIELDS = ["gold_n", "pred_n", "tp_strict", "tp_overlap", "full_masked",
                "neg_texts", "neg_texts_with_pred"]


def example_counts(text, gold, pred, is_neg):
    gold, pred = normalize(text, gold), normalize(text, pred)
    tp_s, _, _ = count_strict(gold, pred)
    tp_o, _, _ = count_overlap(gold, pred)
    return {
        "gold_n": len(gold), "pred_n": len(pred),
        "tp_strict": tp_s, "tp_overlap": tp_o,
        "full_masked": count_coverage(text, gold, pred),
        "neg_texts": 1 if is_neg else 0,
        "neg_texts_with_pred": 1 if (is_neg and pred) else 0,
    }


def summarize(totals):
    """totals — словарь сумм по COUNT_FIELDS."""
    g, p = totals["gold_n"], totals["pred_n"]
    out = {}
    for mode in ("strict", "overlap"):
        tp = totals[f"tp_{mode}"]
        pr, rc, f1 = prf(tp, p - tp, g - tp)
        _, _, f2 = prf(tp, p - tp, g - tp, beta=2.0)
        out[mode] = {"precision": pr, "recall": rc, "f1": f1, "f2": f2,
                     "tp": tp, "fp": p - tp, "fn": g - tp}
    out["full_masked_rate"] = totals["full_masked"] / g if g else 0.0
    out["leaked_entities"] = g - totals["full_masked"]
    out["neg_fp_rate"] = (totals["neg_texts_with_pred"] / totals["neg_texts"]
                          if totals["neg_texts"] else 0.0)
    return out


# ── кластерный бутстрап по шаблонам ───────────────────────────────────────
def bootstrap(per_template, metric_fn, n_iter=10000, seed=12345, paired_with=None):
    """
    per_template: dict template_id -> вектор сумм (np.array по COUNT_FIELDS).
    metric_fn: функция(словарь сумм) -> float.
    Возвращает (точечная оценка, нижняя граница ДИ, верхняя граница ДИ, выборка).
    paired_with: второй per_template — тогда считается разность (этот минус второй)
    на ОДНИХ И ТЕХ ЖЕ ресэмплах.
    """
    keys = sorted(per_template)
    mat = np.array([per_template[k] for k in keys], dtype=float)
    mat2 = np.array([paired_with[k] for k in keys], dtype=float) if paired_with else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_iter, len(keys)))

    def value(m, rows):
        totals = dict(zip(COUNT_FIELDS, m[rows].sum(axis=0)))
        return metric_fn(totals)

    point = value(mat, np.arange(len(keys))) - (value(mat2, np.arange(len(keys))) if mat2 is not None else 0.0)
    samples = np.empty(n_iter)
    for i in range(n_iter):
        rows = idx[i]
        samples[i] = value(mat, rows) - (value(mat2, rows) if mat2 is not None else 0.0)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, lo, hi, samples


def two_sided_p(samples):
    """Двусторонний p-value для гипотезы «разность равна нулю»."""
    n = len(samples)
    below = (samples <= 0).sum() / n
    above = (samples >= 0).sum() / n
    return float(min(1.0, 2 * min(below, above)))


# ══════════════════════════════════════════════════════════════════════════
#  САМОПРОВЕРКА
# ══════════════════════════════════════════════════════════════════════════
def _selftest():
    ok = True

    def check(name, got, exp):
        nonlocal ok
        good = got == exp
        ok &= good
        print(f"  [{'OK ' if good else 'НЕТ'}] {name}: получено {got}, ожидалось {exp}")

    # 1. точное совпадение
    text = "Иван Петров звонил Анне Ковальской вчера"
    gold = [(0, 11), (19, 34)]
    check("strict: оба совпали", count_strict(gold, [(0, 11), (19, 34)]), (2, 0, 0))
    check("strict: один пропущен", count_strict(gold, [(0, 11)]), (1, 0, 1))
    check("strict: лишний спан", count_strict(gold, [(0, 11), (19, 34), (35, 40)]), (2, 1, 0))
    # частичное попадание строгой схемой не засчитывается
    check("strict: частичное = промах", count_strict(gold, [(0, 4), (19, 34)]), (1, 1, 1))
    check("overlap: частичное засчитано", count_overlap(gold, [(0, 4), (19, 34)]), (2, 0, 0))
    # два предсказания на одну сущность: один-к-одному, второе — ложное
    check("overlap: два на одну", count_overlap([(0, 11)], [(0, 4), (5, 11)]), (1, 1, 0))

    # 2. покрытие: «Иван Петров» накрыт лишь наполовину -> имя утекло
    check("coverage: половина = не закрыт", count_coverage(text, [(0, 11)], [(5, 11)]), 0)
    check("coverage: целиком", count_coverage(text, [(0, 11)], [(0, 11)]), 1)
    check("coverage: с запасом", count_coverage(text, [(0, 11)], [(0, 18)]), 1)
    # обрезано на один символ внутри слова — слово накрыто не полностью
    check("coverage: обрезан хвост", count_coverage(text, [(0, 11)], [(0, 10)]), 0)

    # 3. нормализация краёв
    check("normalize: пробелы срезаны", normalize("  Иван  ", [(0, 8)]), [(2, 6)])

    # 4. P/R/F вручную: tp=2, fp=1, fn=1 -> P=0.667, R=0.667, F1=0.667
    p, r, f = prf(2, 1, 1)
    check("prf", (round(p, 3), round(r, 3), round(f, 3)), (0.667, 0.667, 0.667))
    # F2 весит полноту выше: tp=2, fp=2, fn=1 -> P=0.5, R=0.667, F2=0.625
    _, _, f2 = prf(2, 2, 1, beta=2.0)
    check("f2", round(f2, 3), 0.625)

    # 5. сверка strict с seqeval на токенах
    try:
        from seqeval.metrics import precision_score, recall_score
        tokens = ["Иван", "Петров", "звонил", "Анне", "Ковальской", "вчера"]
        true = ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O"]
        pred = ["B-PER", "I-PER", "O", "B-PER", "O", "O"]      # вторая сущность обрезана
        # те же данные в символьных спанах
        offs, pos = [], 0
        for t in tokens:
            offs.append((pos, pos + len(t)))
            pos += len(t) + 1
        g_spans = [(offs[0][0], offs[1][1]), (offs[3][0], offs[4][1])]
        p_spans = [(offs[0][0], offs[1][1]), (offs[3][0], offs[3][1])]
        tp, fp, fn = count_strict(g_spans, p_spans)
        our_p, our_r, _ = prf(tp, fp, fn)
        check("seqeval precision", round(our_p, 4), round(precision_score([true], [pred]), 4))
        check("seqeval recall", round(our_r, 4), round(recall_score([true], [pred]), 4))
    except ImportError:
        print("  [ — ] seqeval не установлен, сверка пропущена")

    # 6. бутстрап: на вырожденных данных ДИ обязан схлопнуться в точку
    per_t = {f"T{i}": np.array([2, 2, 2, 2, 2, 0, 0], dtype=float) for i in range(20)}
    point, lo, hi, _ = bootstrap(per_t, lambda t: summarize(t)["strict"]["recall"], n_iter=500)
    check("bootstrap на одинаковых кластерах", (round(point, 3), round(lo, 3), round(hi, 3)),
          (1.0, 1.0, 1.0))

    print("\nСАМОПРОВЕРКА:", "пройдена" if ok else "ПРОВАЛЕНА")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
