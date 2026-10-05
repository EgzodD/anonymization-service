"""
Отсев почти-дублей.

Первый пилот ловил только полные совпадения, поэтому пересказ той же жалобы другими
словами проходил. Считаем Жаккар по тройкам слов; чтобы не сравнивать каждый новый текст
со всеми принятыми (на тысячах текстов это заметно), держим обратный индекс по тройкам.
"""
import re

_WORDS = re.compile(r"\w+")


class NearDup:
    """Почти-дубли: Жаккар по тройкам слов. Обратный индекс по тройкам — чтобы не
    сравнивать каждый новый текст со всеми принятыми (на тысячах текстов это заметно)."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.index, self.sets = {}, []

    @staticmethod
    def shingles(text):
        w = _WORDS.findall(text.lower())
        return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)} or set(w)

    def worst(self, text):
        """(худшее сходство с принятыми, набор тройек) — набор нужен для add()."""
        sh = self.shingles(text)
        hits = {}
        for g in sh:
            for j in self.index.get(g, ()):
                hits[j] = hits.get(j, 0) + 1
        best = 0.0
        for j, inter in hits.items():
            union = len(sh) + len(self.sets[j]) - inter
            if union:
                best = max(best, inter / union)
        return round(best, 3), sh

    def add(self, sh):
        j = len(self.sets)
        self.sets.append(sh)
        for g in sh:
            self.index.setdefault(g, []).append(j)
