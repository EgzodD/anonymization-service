"""
Опечатки и нижний регистр — вносит код, а не языковая модель.

Стиль «с опечатками» модель в пилоте игнорировала: писала чисто. Поэтому порчу вносим
сами, и только ВНЕ подставленных значений — позиции персональных данных должны остаться
точными, а «грязные» форматы номеров и так делают генераторы build_test_v2.

Отдельно стоит перевод всего текста в нижний регистр: длины не меняются, разметка
остаётся той же, а для модели ФИО это самый трудный случай (на MultiCoNER со строчными
именами 22 % скрыто не полностью).

corrupt() возвращает ещё и список испорченных слов. Он нужен, чтобы отличать ложное
срабатывание сервиса на опечатке (его ошибка, текст годный и ценный) от персональных
данных, вписанных самой моделью (неразмеченные ПДн — текст в брак).
"""
import difflib
import re

# ── опечатки и почти-дубли ──────────────────────────────────────────────────
# соседние клавиши русской раскладки — самая частая опечатка при быстром наборе
_KEY = {"й": "цф", "ц": "йув", "у": "цка", "к": "уев", "е": "кнп", "н": "егр", "г": "ншо", "ш": "гщл",
        "щ": "шзд", "з": "щхж", "х": "зэж", "ф": "йыв", "ы": "фвс", "в": "ыам", "а": "впр", "п": "аро",
        "р": "пол", "о": "рлд", "л": "одж", "д": "лжэ", "ж": "дэх", "э": "жх", "я": "чс", "ч": "ясм",
        "с": "чми", "м": "сит", "и": "мтб", "т": "ибю", "ь": "тбю", "б": "ью", "ю": "б"}


def _apply(op, seg, rng):
    """Одна порча. None — к этому отрезку операция не применима."""
    letters = [i for i, c in enumerate(seg) if c.isalpha()]
    if op == "comma":
        pos = [i for i, c in enumerate(seg) if c == ","]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + seg[i + 1:]
    if op == "yo":
        pos = [i for i, c in enumerate(seg) if c in "ёЁ"]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + ("е" if seg[i] == "ё" else "Е") + seg[i + 1:]
    if op == "case":                       # строчная буква после точки
        pos = [m.end() - 1 for m in re.finditer(r"[.!?]\s+[А-ЯЁ]", seg)]
        if not pos:
            return None
        i = rng.choice(pos)
        return seg[:i] + seg[i].lower() + seg[i + 1:]
    if not letters:
        return None
    if op == "drop":
        i = rng.choice(letters)
        return seg[:i] + seg[i + 1:]
    if op == "double":
        i = rng.choice(letters)
        return seg[:i] + seg[i] * 2 + seg[i + 1:]
    if op == "swap":
        pairs = [i for i in letters if i + 1 < len(seg) and seg[i + 1].isalpha()]
        if not pairs:
            return None
        i = rng.choice(pairs)
        return seg[:i] + seg[i + 1] + seg[i] + seg[i + 2:]
    near = [i for i in letters if seg[i].lower() in _KEY]       # op == "key"
    if not near:
        return None
    i = rng.choice(near)
    c = rng.choice(_KEY[seg[i].lower()])
    return seg[:i] + (c.upper() if seg[i].isupper() else c) + seg[i + 1:]


def _damaged(orig, new):
    """Диапазоны изменённых слов в испорченном отрезке.

    Нужны, чтобы отличать два случая: сервис нашёл «ПДн» в слове, которое испортили мы
    (это его ложное срабатывание на опечатке — ценный обучающий пример, текст годный),
    и сервис нашёл ПДн, которые вписала сама модель (неразмеченные ПДн — текст в брак).
    """
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=orig, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        a, b = j1, max(j2, j1 + 1)
        while a > 0 and (new[a - 1].isalnum() or new[a - 1] == "-"):
            a -= 1
        while b < len(new) and (new[b].isalnum() or new[b] == "-"):
            b += 1
        out.append((a, min(b, len(new))))
    return out


def _typos(seg, rng, n):
    """n порч в одном отрезке; возвращает (отрезок, сколько внесено)."""
    done = 0
    for _ in range(n):
        ops = ["drop", "swap", "double", "key", "yo", "comma", "case"]
        rng.shuffle(ops)
        for op in ops:
            new = _apply(op, seg, rng)
            if new is not None:
                seg, done = new, done + 1
                break
    return seg, done


def corrupt(text, spans, negs, rng, rate=1.0):
    """Опечатки ВНЕ подставленных значений -> (текст, спаны, негативы, испорченные слова).

    Значения ПДн и негативов не трогаем: их позиции должны остаться точными, а «грязные»
    форматы номеров и так делают генераторы build_test_v2. Позиции пересчитываются по
    длинам отрезков, поэтому разметка остаётся верной при любой порче.
    """
    items = sorted([("p", dict(x)) for x in spans] + [("n", dict(x)) for x in negs],
                   key=lambda it: it[1]["start"])
    pieces, pos = [], 0
    for _kind, it in items:
        pieces.append(["free", text[pos:it["start"]]])
        pieces.append(["keep", text[it["start"]:it["stop"]]])
        pos = it["stop"]
    pieces.append(["free", text[pos:]])
    for p in pieces:
        if p[0] == "free" and len(p[1]) > 8:
            orig = p[1]
            p[1], _k = _typos(orig, rng, max(1, round(len(orig) / 55 * rate)))
            p.append(_damaged(orig, p[1]))
    out, k, marks = "", 0, []
    for piece in pieces:
        kind, txt = piece[0], piece[1]
        if kind == "keep":
            items[k][1]["start"], items[k][1]["stop"] = len(out), len(out) + len(txt)
            k += 1
        else:
            marks += [(len(out) + a, len(out) + b) for a, b in (piece[2] if len(piece) > 2 else ())]
        out += txt
    return (out, [it for kind, it in items if kind == "p"],
            [it for kind, it in items if kind == "n"], marks)
