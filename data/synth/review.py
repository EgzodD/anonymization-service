"""
Выгрузка результата синтезатора для ручной проверки.

Каждый текст печатается с подсветкой вставленных ПДн: ⟦значение|ТИП⟧. Сверху — сценарий,
внизу — результаты автопроверок. Проверяющему нужно ответить на три вопроса:
  1) звучит ли текст естественно (как писал бы клиент);
  2) верна ли грамматика вокруг имён (падеж, род глаголов);
  3) нет ли ПДн, которые НЕ подсвечены (это ошибка разметки).

Запуск:
    .venv/bin/python data/synth/review.py data/synth/pilot_qwen3.jsonl > pilot_review.txt
"""
import json
import sys


def mark(text, spans):
    out, pos = "", 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += text[pos:s["start"]] + f"⟦{text[s['start']:s['stop']]}|{s['type']}⟧"
        pos = s["stop"]
    return out + text[pos:]


def main(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    ok = [r for r in rows if not r["reject"]]
    print(f"Файл: {path}\nВсего {len(rows)}, прошли автопроверки {len(ok)}, брак {len(rows) - len(ok)}\n")
    for i, r in enumerate(rows, 1):
        sc = r["scenario"]
        flag = "БРАК" if r["reject"] else "ok"
        print(f"── {i:02d} [{flag}] {sc['domain']} · {sc['channel']} · {sc['style']} · ПДн: {sc['pii'] or 'нет'}")
        print("   " + mark(r["text"], r["spans"]).replace("\n", "\n   "))
        why = [f"{k}: {v}" for k, v in r["checks"].items() if v]
        if why:
            print("   проверки: " + "; ".join(why))
        print()


if __name__ == "__main__":
    main(sys.argv[1])
