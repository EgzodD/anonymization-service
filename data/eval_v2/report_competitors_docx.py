"""
Отчёт Word о тестировании на общедоступных наборах: наш сервис (последний прогон)
против конкурентов.

Цветом выделены только НАШИ результаты; значения конкурентов — простым текстом.
Цвет нашей ячейки — вердикт сравнения с конкурентом по 95 % доверительному интервалу
разности (парный бутстрап):
  зелёный — выигрываем, жёлтый — на уровне, красный — проигрываем,
  серый   — сравнение некорректно, в счёт не идёт.
Выводы собираются из тех же вердиктов, что и цвета.

Источники: results/competitors_paired.json (compare_competitors.py), hive.jsonl и
results/preds_S5-r7_hive.jsonl (утечки по типам).

Запуск:
    .venv/bin/python data/eval_v2/report_competitors_docx.py --out <путь к .docx>
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OURS = "S5-r7"
COLOR = {"win": "C6EFCE", "tie": "FFEB9C", "loss": "FFC7CE", "na": "E7E6E6"}
HEAD, GROUP = "1F4E79", "DDEBF7"
WORD = {"win": "ВЫИГРЫВАЕМ", "tie": "НА УРОВНЕ", "loss": "ПРОИГРЫВАЕМ", "na": "не засчитано"}
METRICS = ["full_masked", "f1", "precision", "recall", "f1_overlap", "false_person_rate"]
METRIC_RU = {"full_masked": "Имён скрыто полностью", "f1": "Строгий F1", "precision": "Точность",
             "recall": "Полнота", "f1_overlap": "F1 по пересечению",
             "false_person_rate": "Ложное ФИО в текстах без ФИО"}
PCT = {"full_masked", "false_person_rate"}
COMP = {"S6": "redmadrobot", "S7": "PIIDetector"}
COMP_FULL = {"S6": "redmadrobot rubert-base-pii-ner", "S7": "PIIDetector (alrosait, spaCy + Presidio)"}

# набор: (название, пояснение к строке-заголовку группы, какие ячейки не засчитываются)
DATASETS = [
    ("hive", "hivetrace/pii-bench", "сообщения поддержки, разметка двух экспертов; наш замер не слепой — "
     "классы ошибок разбирались 18.09", lambda o, m: False),
    ("ext", "factRuEval-2016", "новости, ручная разметка «Диалог-2016»; независимый для всех систем",
     lambda o, m: False),
    ("v2", "test_v2", "синтетические обращения — наш домен; для конкурентов чужой", lambda o, m: False),
    ("bench", "pii_benchmark", "набор компании redmadrobot; и redmadrobot, и наша модель обучались на данных "
     "того же источника (pii_train)", lambda o, m: False),
    ("mcr", "MultiCoNER", "строчный текст; разметка неполная — точность, F1 и ложные ФИО не засчитаны",
     lambda o, m: m in ("precision", "f1", "f1_overlap", "false_person_rate")),
    ("alro", "alrosait/pii-synthetic-ru", "правила подбирались на этом наборе, PIIDetector на нём обучен — "
     "не засчитано целиком", lambda o, m: True),
]


# ── форматирование ─────────────────────────────────────────────────────────
def shade(cell, hex_color):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc.append(shd)


def num(m, v):
    return (f"{v * 100:.1f} %" if m in PCT else f"{v:.3f}").replace(".", ",")


def diff(m, d, ci):
    if m in PCT:
        s = f"{d * 100:+.1f} п.п.  [{ci[0] * 100:+.1f}; {ci[1] * 100:+.1f}]"
    else:
        s = f"{d:+.3f}  [{ci[0]:+.3f}; {ci[1]:+.3f}]"
    return s.replace(".", ",").replace("п,п,", "п.п.")


class Report:
    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(11)
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Cm(1.8)

    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, text, italic=False, bold=False, size=None):
        par = self.doc.add_paragraph()
        r = par.add_run(text)
        r.italic, r.bold = italic, bold
        if size:
            r.font.size = Pt(size)
        return par

    def bullets(self, items):
        for it in items:
            self.doc.add_paragraph(it, style="List Bullet")

    def table(self, header, rows, widths=None, note=None, font=10):
        """rows: список строк. Ячейка — str, (str, вердикт) — окрашенная, или ("GROUP", текст) —
        строка-заголовок группы на всю ширину."""
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, x in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(x)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shade(c, HEAD)
        for row in rows:
            cells = t.add_row().cells
            if isinstance(row, tuple) and row[0] == "GROUP":
                merged = cells[0].merge(cells[-1])
                merged.text = ""
                run = merged.paragraphs[0].add_run(row[1])
                run.bold = True
                if len(row) > 2:
                    r2 = merged.paragraphs[0].add_run("  —  " + row[2])
                    r2.italic = True
                shade(merged, GROUP)
                continue
            for i, x in enumerate(row):
                if isinstance(x, tuple):
                    cells[i].text = ""
                    run = cells[i].paragraphs[0].add_run(x[0])
                    run.bold = True
                    shade(cells[i], COLOR[x[1]])
                else:
                    cells[i].text = x
        for row in t.rows:
            for i, c in enumerate(row.cells):
                if widths and i < len(widths):
                    c.width = Cm(widths[i])
                for par in c.paragraphs:
                    for r in par.runs:
                        r.font.size = Pt(font)
        if note:
            self.p(note, italic=True, size=9)
        self.doc.add_paragraph()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(RES, "competitors_paired.json"), encoding="utf-8"))
    rep = Report()

    # вердикты — один раз, из них и цвета, и выводы
    verdicts = {}                         # (ds, comp, metric) -> вердикт с учётом «не засчитано»
    for ds, _, _, na in DATASETS:
        for o in COMP:
            for m in METRICS:
                v = R[ds]["pairs"][f"{OURS}|{o}"][m]["verdict"]
                verdicts[(ds, o, m)] = "na" if na(o, m) else v

    title = rep.doc.add_heading("Тестирование модуля обезличивания на общедоступных наборах", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rep.p("Сравнение с конкурентами. Последний прогон: 22.09.2026, текущая версия сервиса — модель ФИО v3 "
          "(rubert-tiny2) + правила для номеров и адресов + фильтр ложных ФИО.", italic=True)

    # ── Что это и зачем
    rep.h("1. Что это за тестирование и зачем оно нужно")
    rep.p("Модуль обезличивания убирает персональные данные (ФИО, телефоны, паспорта, адреса и т. д.) из текста "
          "перед тем, как текст уйдёт во внешний сервис или языковую модель. Ошибиться можно двумя способами:")
    rep.bullets([
        "пропустить ПДн — это утечка и нарушение 152-ФЗ; главный риск;",
        "замаскировать лишнее (например, слово «Страховой» как фамилию) — утечки нет, но текст портится.",
    ])
    rep.p("Проверять модуль только на своих тестах нельзя: их делает тот же человек, что и модуль, и они завышают "
          "результат. Мы это измерили: в прежнем внутреннем тесте 97 % примеров были построены по тем же шаблонам, "
          "что и обучение, и оценка была завышена на 0,26 по F1. Поэтому модуль проверяется на общедоступных "
          "наборах, которые размечали другие люди, и на тех же текстах сравнивается с другими системами.")
    rep.p("Что проверяется. Главное — распознавание ФИО: это единственное, что делает нейросетевая модель; "
          "остальные типы ПДн ищут правила. Утечки остальных ПДн — в разделе 6.")

    rep.h("Метрики простыми словами", 2)
    rep.table(["Метрика", "Что значит", "Лучше"], [
        ["Имён скрыто полностью", "доля имён, у которых замаскировано каждое слово. Главная метрика приватности: "
                                  "имя не осталось в тексте даже частично", "больше"],
        ["Строгий F1", "стандартная сводная оценка точности и полноты; маска засчитывается, только если её "
                       "границы в точности совпадают с именем", "больше"],
        ["Точность", "из всех масок «ФИО» — сколько действительно имена (с точными границами)", "больше"],
        ["Полнота", "из всех имён — сколько найдено с точными границами", "больше"],
        ["F1 по пересечению", "как строгий F1, но засчитывается и маска с неточными границами", "больше"],
        ["Ложное ФИО в текстах без ФИО", "доля текстов без имён, где система всё же поставила маску имени", "меньше"],
    ], widths=[4.5, 11, 1.8])

    rep.h("Наборы", 2)
    rep.table(["Набор", "Что это", "Кто размечал"], [
        ["hivetrace/pii-bench", "1 810 сообщений поддержки и мессенджеров: банк, телеком, доставка, HR…",
         "два эксперта независимо, согласие 0,87; к набору опубликована статья с результатами других систем"],
        ["factRuEval-2016", "995 абзацев новостей", "соревнование «Диалог-2016», вручную"],
        ["test_v2", "480 синтетических обращений клиентов", "мы, по протоколу, до обучения модели"],
        ["pii_benchmark", "2 841 сообщение, часть из реальных логов", "компания redmadrobot"],
        ["MultiCoNER (русская часть)", "4 000 фраз из Википедии, весь текст строчными", "сторонние авторы; разметка неполная"],
        ["alrosait/pii-synthetic-ru", "4 500 сообщений поддержки, ФИО и адреса", "сгенерировано языковой моделью"],
    ], widths=[4, 7, 6.3])

    rep.h("Конкуренты", 2)
    rep.table(["Система", "Что это", "Чем интересна"], [
        [COMP_FULL["S6"], "открытая модель компании Red Madrobot для поиска ПДн в русском тексте (MIT), rubert-base",
         "самая точная из найденных открытых моделей; крупная и медленная"],
        [COMP_FULL["S7"], "открытый проект PIIDetector: spaCy + Presidio, то же устройство, что у нас "
                          "(модель + правила)", "прямой аналог; обучен на наборе alrosait"],
        ["GLiNER Guard, GLiNER2", "модели компании HiveTrace из статьи «GLiNER Guard: Unified Encoder Family for "
                                  "Production LLM Safety and Privacy» (arXiv:2605.05277, 6 мая 2026 г.)",
         "мы их не запускали — сравниваем с опубликованными числами"],
    ], widths=[4.5, 7.5, 5.3])

    rep.h("Кто на чём проверялся", 2)
    y, pub, no = "запускали мы", "числа из статьи", "—"
    rep.table(["Система", "hivetrace", "factRuEval", "test_v2", "pii_benchmark", "MultiCoNER", "alrosait"], [
        ["Наш сервис", y, y, y, y, y, y],
        [COMP["S6"], y, y, y, y, y, y],
        [COMP["S7"], y, y, y, y, y, y],
        ["GLiNER Guard / GLiNER2", pub, no, no, no, no, no],
    ], widths=[3.8, 2.3, 2.3, 2.1, 2.5, 2.3, 2.1],
        note="Все системы с пометкой «запускали мы» прогонялись одной программой на одних и тех же текстах и "
             "считались одними формулами. Проверка программы: для PIIDetector она дала NAME F1 = 0,944 на "
             "hivetrace — ровно число, которое публикуют его авторы.")

    # ── Как читать
    rep.h("2. Как читать таблицы")
    rep.p("Цветом выделены только НАШИ результаты. Значения конкурентов — простым текстом.", bold=True)
    rep.table(["Цвет нашей ячейки", "Что значит", "Правило"], [
        [("ВЫИГРЫВАЕМ", "win"), "мы лучше конкурента", "вся 95 % зона возможной разницы — в нашу пользу"],
        [("НА УРОВНЕ", "tie"), "разница не доказана", "зона возможной разницы включает ноль"],
        [("ПРОИГРЫВАЕМ", "loss"), "конкурент лучше", "вся зона возможной разницы — не в нашу пользу"],
        [("не засчитано", "na"), "сравнивать нечестно", "конкурент обучен на этом наборе, мы на нём "
                                                        "настраивались или разметка не позволяет судить"],
    ], widths=[3.5, 4, 9.8])
    ex = R["hive"]["pairs"][f"{OURS}|S6"]["f1"]
    rep.p("Каждая строка таблицы сравнения читается слева направо: метрика → результат конкурента → НАШ результат "
          "→ насколько мы лучше (+) или хуже (−) и в скобках границы этой разницы → итог. Пример из hivetrace, строгий "
          f"F1: «{num('f1', R['hive']['systems']['S6']['f1'])} → {num('f1', R['hive']['systems'][OURS]['f1'])} → "
          f"{diff('f1', ex['diff'], ex['ci95'])} → ПРОИГРЫВАЕМ»: мы хуже redmadrobot на {num('f1', abs(ex['diff']))}, "
          f"и даже в лучшем для нас случае хуже на {num('f1', abs(ex['ci95'][1]))}. "
          "Для «ложное ФИО» знак минус — это хорошо: у нас меньше ложных масок.")

    # ── Сводка
    rep.h("3. Сводка")
    rows = []
    total = defaultdict(Counter)
    for ds, name, _, _ in DATASETS:
        row = [name]
        for o in COMP:
            c = Counter(verdicts[(ds, o, m)] for m in METRICS)
            total[o].update(c)
            if c["win"] + c["tie"] + c["loss"] == 0:
                row.append(("не засчитано", "na"))
                continue
            best = "win" if c["win"] > c["loss"] else "loss" if c["loss"] > c["win"] else "tie"
            row.append((f"выигрыш {c['win']} · вровень {c['tie']} · проигрыш {c['loss']}", best))
        rows.append(row + [num("full_masked", R[ds]["systems"][OURS]["full_masked"])
                           + f"  (redmadrobot {num('full_masked', R[ds]['systems']['S6']['full_masked'])}, "
                             f"PIIDetector {num('full_masked', R[ds]['systems']['S7']['full_masked'])})"])
    rows.append(["ИТОГО"] + [(f"выигрыш {total[o]['win']} · вровень {total[o]['tie']} · проигрыш {total[o]['loss']}",
                              "win" if total[o]["win"] > total[o]["loss"] else "loss" if total[o]["loss"] > total[o]["win"]
                              else "tie") for o in COMP] + [""])
    rep.table(["Набор", "НАШ счёт против redmadrobot", "НАШ счёт против PIIDetector", "Наших имён скрыто полностью"],
              rows, widths=[3.8, 4.6, 4.6, 4.3],
              note="Счёт — по шести метрикам раздела 1; цвет ячейки — по большинству. Подробности по каждой "
                   "метрике — в разделах 4 и 5.")

    # ── Против каждого конкурента
    def versus(o, section):
        rep.h(f"{section}. Мы против {COMP_FULL[o]}")
        rows = []
        for ds, name, why, _ in DATASETS:
            rows.append(("GROUP", name, why))
            for m in METRICS:
                res = R[ds]["pairs"][f"{OURS}|{o}"][m]
                v = verdicts[(ds, o, m)]
                rows.append([METRIC_RU[m], num(m, R[ds]["systems"][o][m]),
                             (num(m, R[ds]["systems"][OURS][m]), v), diff(m, res["diff"], res["ci95"]),
                             (WORD[v], v)])
        rep.table(["Метрика", f"{COMP[o]}", "НАШ результат", "Разница (границы)", "Итог"], rows,
                  widths=[4.6, 2.6, 2.8, 4.6, 2.7])
    versus("S6", 4)
    versus("S7", 5)

    # ── Опубликованные
    rep.h("6. Сравнение с системами из статьи (hivetrace, domain-часть)")
    rep.p("Статья: " + 'Minko B., Sadiekh S., Kokuykin E. «GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy». arXiv:2605.05277, опубликована 6 мая 2026 г. (https://arxiv.org/abs/2605.05277). Авторы — компания HiveTrace; в той же работе выпущен набор hivetrace/pii-bench.', italic=True)
    dom = R["hive_domain"]
    ours, ci = dom["systems"][OURS]["f1"], dom["systems_ci95"][OURS]["f1"]
    rep.p(f"У опубликованных чисел нет границ разницы, поэтому их сравниваем с границами нашего результата: "
          f"наш строгий F1 по ФИО = {num('f1', ours)}, границы {num('f1', ci[0])} … {num('f1', ci[1])}. "
          f"Число ниже наших границ — выигрываем, внутри — на уровне, выше — проигрываем. Правило подсчёта — как "
          f"в статье: строгое совпадение границ, domain-часть набора.")
    rows = []
    for name, v in [("GLiNER Guard uni-encoder", 0.757), ("GLiNER Guard bi-encoder", 0.695),
                    ("GLiNER2 Multi", 0.606), ("GLiNER Guard Omni", 0.523), ("GLiNER2 Large", 0.303)]:
        vd = "win" if v < ci[0] else "loss" if v > ci[1] else "tie"
        rows.append([name + " (из статьи)", num("f1", v), (num("f1", ours), vd), f"{ours - v:+.3f}".replace(".", ","),
                     (WORD[vd], vd)])
    for o in COMP:
        res = dom["pairs"][f"{OURS}|{o}"]["f1"]
        rows.append([COMP[o] + " (наш замер)", num("f1", dom["systems"][o]["f1"]), (num("f1", ours), res["verdict"]),
                     diff("f1", res["diff"], res["ci95"]), (WORD[res["verdict"]], res["verdict"])])
    rep.table(["Система", "Её F1", "НАШ F1", "Разница", "Итог"], rows, widths=[5.5, 2.2, 2.4, 4.6, 2.6])

    # ── Скорость
    rep.h("7. Скорость")
    L = R["latency_median_ms"]
    rep.p("Сравнивается модель с моделью: медиана миллисекунд на текст на процессоре без видеокарты. «На уровне» — "
          "разница не больше 25 %. Наша модель — 3,47 мс (замер на test_v2); весь наш сервис с правилами — "
          f"{L['hive'][OURS]:.0f}–{L['alro'][OURS]:.0f} мс, с одной моделью конкурента его не сравниваем.")
    rows = []
    for o in COMP:
        vals = [L[ds][o] for ds in ("hive", "alro", "mcr")]
        lo, hi = min(vals), max(vals)
        r_ = 3.47 / ((lo + hi) / 2)
        vd = "win" if r_ < 0.8 else "loss" if r_ > 1.25 else "tie"
        how = f"в {1 / r_:.0f} раз быстрее" if vd == "win" else "примерно столько же" if vd == "tie" else "медленнее"
        rows.append([COMP_FULL[o], f"{lo:.1f}–{hi:.1f} мс".replace(".", ","), ("3,5 мс", vd), how, (WORD[vd], vd)])
    rep.table(["Конкурент", "Его модель", "НАША модель", "Разница", "Итог"], rows, widths=[5.5, 2.8, 2.6, 3.8, 2.6])

    # ── Утечки по типам
    rep.h("8. Утечки остальных ПДн (hivetrace, последний прогон)")
    rep.p("Конкуренты в этой проверке не участвуют: redmadrobot и PIIDetector мы запускали только на ФИО. Поэтому "
          "здесь без цвета — просто наш результат: сколько значений каждого типа осталось в тексте.")
    raw = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "hive.jsonl"), encoding="utf-8"))}
    P = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, f"preds_{OURS}_hive.jsonl"), encoding="utf-8"))}
    tot, leak = Counter(), Counter()
    for rid, r in raw.items():
        for s in r["spans"]:
            if not s["type"].startswith("OTHER_"):
                tot[s["type"]] += 1
                leak[s["type"]] += r["text"][s["start"]:s["stop"]] in P[rid]["anonymized"]
    ru = {"PERSON": "ФИО", "PHONE_NUMBER": "телефон", "ADDRESS": "адрес", "EMAIL_ADDRESS": "почта",
          "PASSPORT": "паспорт", "INN": "ИНН", "SNILS": "СНИЛС", "CREDIT_CARD": "номер карты"}
    rows = [[ru.get(t, t), str(tot[t]), str(leak[t]), f"{leak[t] / tot[t] * 100:.1f} %".replace(".", ",")]
            for t in sorted(tot, key=lambda t: -leak[t])]
    rows.append(["всего", str(sum(tot.values())), str(sum(leak.values())),
                 f"{sum(leak.values()) / sum(tot.values()) * 100:.1f} %".replace(".", ",")])
    rep.table(["Тип ПДн", "Значений в наборе", "Осталось в тексте", "Доля утечек"], rows, widths=[4, 4, 4, 4],
              note="Адреса: часть «утечек» — город без улицы («Проживаю в Москве»), который по нашей политике не "
                   "скрывается, а в наборе размечен как адрес.")

    # ── Выводы — из тех же вердиктов
    rep.h("9. Выводы")
    names = {ds: name for ds, name, _, _ in DATASETS}
    for v, title_ in (("win", "Где выигрываем"), ("tie", "Где на уровне"), ("loss", "Где проигрываем")):
        rep.h(title_, 2)
        for o in COMP:
            by_metric = defaultdict(list)
            for (ds, oo, m), vv in verdicts.items():
                if vv == v and oo == o:
                    by_metric[m].append(names[ds])
            par = rep.doc.add_paragraph(style="List Bullet")
            par.add_run(f"Против {COMP[o]}: ").bold = True
            par.add_run("; ".join(f"{METRIC_RU[m]} — {', '.join(d_)}" for m, d_ in
                                  sorted(by_metric.items(), key=lambda kv: METRICS.index(kv[0]))) or "нет")
    rep.h("Что это значит", 2)
    rep.p("Наш сервис сильнее там, где для обезличивания важнее всего, — имена реже остаются открытыми, особенно в "
          "неаккуратном тексте и на нашем домене. Слабее — в точности: чаще маскирует лишнее и не всегда попадает "
          "в границы имени. Для задачи «не выпустить ПДн» это правильная сторона ошибки, но лишние маски портят "
          "текст для дальнейшей обработки. Что делать дальше — в БУДУЩИЕ_ПЛАНЫ.md, пункты 3 и 3а.")
    rep.h("Ограничения", 2)
    rep.bullets([
        "Реальных обращений среди наборов нет; наборы домена поддержки синтетические.",
        "hivetrace — замер не слепой: классы его ошибок разбирались при доработке правил 18.09.",
        "test_v2 — наш домен, поэтому преимущество на нём ожидаемо; pii_benchmark — «свой» и для redmadrobot, и для нас.",
        "Все внешние наборы теперь использованы; следующая версия требует нового слепого набора.",
    ])
    rep.p("Файлы: data/eval_v2/results/competitors_paired.json (цифры), compare_competitors.py (расчёт), "
          "report_competitors_docx.py (этот отчёт), PROTOCOL.md (протокол).", italic=True, size=9)
    rep.doc.save(a.out)
    print(a.out)
    print({o: dict(total[o]) for o in COMP})


if __name__ == "__main__":
    main()
