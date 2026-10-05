"""
Выгрузка результата синтезатора для ручной проверки.

Каждый текст печатается с подсветкой: ⟦значение|ТИП⟧ — подставленные ПДн (их сервис
обязан скрыть), ⟪значение|НЕ ПДн⟫ — трудные негативы (их скрывать как ФИО нельзя).
Сверху — сценарий, внизу — результаты автопроверок и ложные ФИО сервиса на негативах.

Проверяющему нужно ответить на четыре вопроса:
  1) звучит ли текст естественно (как писал бы клиент);
  2) верна ли грамматика вокруг имён (падеж, род глаголов);
  3) нет ли ПДн, которые НЕ подсвечены (это ошибка разметки);
  4) не попало ли в ⟪негативы⟫ то, что на самом деле является ПДн.

Запуск:
    .venv/bin/python -m synthesizer.review synthesizer/runs/2026-10-05__run.jsonl \
        > synthesizer/runs/2026-10-05__run_review.txt
"""
import json
import sys


def mark(text, spans, negs):
    items = sorted([(s["start"], s["stop"], s["type"]) for s in spans]
                   + [(n["start"], n["stop"], "НЕ ПДн") for n in negs])
    out, pos = "", 0
    for start, stop, label in items:
        br = "⟪{}⟫" if label == "НЕ ПДн" else "⟦{}⟧"
        out += text[pos:start] + br.format(f"{text[start:stop]}|{label}")
        pos = stop
    return out + text[pos:]


def main(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    ok = [r for r in rows if not r["reject"]]
    fp = sorted({x for r in rows for x in r.get("person_fp_on_negatives", [])})
    fp_t = sorted({x for r in rows for x in r.get("person_fp_on_typos", [])})
    print(f"Файл: {path}\nВсего {len(rows)}, прошли автопроверки {len(ok)}, брак {len(rows) - len(ok)}")
    if fp:
        print(f"Ложные ФИО сервиса на негативах ({len(fp)} разных): " + ", ".join(fp))
    if fp_t:
        print(f"Ложные ФИО сервиса на словах с опечатками ({len(fp_t)} разных): " + ", ".join(fp_t))
    print()
    for i, r in enumerate(rows, 1):
        sc = r["scenario"]
        flag = "БРАК" if r["reject"] else "ok"
        how = [sc.get("domain", "?"), sc["channel"], sc["style"], sc.get("length", "?")]
        if sc.get("typos"):
            how.append(f"опечаток: {sc['typos']}")
        if sc.get("lower"):
            how.append("нижний регистр")
        print(f"── {i:02d} [{flag}] " + " · ".join(how))
        print(f"   ПДн: {sc['pii'] or 'нет'}; негативы: {sc.get('neg') or 'нет'}; "
              f"сходство с принятыми: {r.get('similarity', 0)}")
        print("   " + mark(r["text"], r["spans"], r.get("negative_spans", [])).replace("\n", "\n   "))
        why = [f"{k}: {v}" for k, v in r["checks"].items() if v]
        if why:
            print("   проверки: " + "; ".join(why))
        fps = r.get("person_fp_on_negatives", []) + r.get("person_fp_on_typos", [])
        if fps:
            print("   сервис принял за ФИО: " + ", ".join(fps))
        print()


if __name__ == "__main__":
    main(sys.argv[1])
