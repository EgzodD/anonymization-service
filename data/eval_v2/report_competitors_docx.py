"""
Отчёт Word о тестировании на общедоступных наборах с цветовой разметкой сравнений.

Цвет ячейки — вердикт сравнения нашего сервиса с другой системой:
  зелёный — выигрываем значимо (95 % ДИ разности целиком в нашу пользу);
  жёлтый  — на уровне (ДИ разности содержит ноль; для величин без ДИ — см. правило в отчёте);
  красный — проигрываем значимо;
  серый   — сравнение некорректно (система обучена на этом наборе, мы настраивались на нём,
            или разметка набора не позволяет судить о метрике) — показано, но не засчитано.

Источники: results/competitors_paired.json (compare_competitors.py),
results/acceptance_stage3.json, results/preds_S5-v3_hive.jsonl и preds_S5-r7_hive.jsonl.

Запуск:
    .venv/bin/python data/eval_v2/report_competitors_docx.py --out <путь к .docx>
"""
import argparse
import json
import os
from collections import Counter

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COLOR = {"win": "C6EFCE", "tie": "FFEB9C", "loss": "FFC7CE", "na": "E7E6E6"}
WORD = {"win": "выигрываем", "tie": "на уровне", "loss": "проигрываем", "na": "не засчитано"}
METRIC_RU = {"precision": "Точность", "recall": "Полнота", "f1": "Строгий F1",
             "f1_overlap": "F1 по пересечению", "full_masked": "Имён скрыто полностью",
             "false_person_rate": "Ложное ФИО в текстах без ФИО"}
PCT = {"full_masked", "false_person_rate"}
SYS_RU = {"S6": "redmadrobot rubert-base-pii-ner", "S7": "PIIDetector (alrosait, spaCy)",
          "S5-prev": "наш сервис, прежняя модель", "S5-v3": "наш сервис, модель v3",
          "S5-r7": "наш сервис сейчас (v3 + правила 18.09)"}


def shade(cell, hex_color):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc.append(shd)


def fmt(m, v):
    return f"{v * 100:.1f} %".replace(".", ",") if m in PCT else f"{v:.3f}".replace(".", ",")


def fmt_diff(m, d):
    s = f"{d * 100:+.1f} п.п." if m in PCT else f"{d:+.3f}"
    return s.replace(".", ",")


def fmt_ci(m, ci):
    k = 100 if m in PCT else 1
    return f"[{ci[0] * k:+.{1 if m in PCT else 3}f}; {ci[1] * k:+.{1 if m in PCT else 3}f}]".replace(".", ",")


class Report:
    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(11)
        self.score = Counter()

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

    def table(self, header, rows, note=None):
        """rows: список строк; ячейка — str или (str, вердикт)."""
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        for i, x in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = x
            c.paragraphs[0].runs[0].bold = True
            shade(c, "D9E1F2")
        for row in rows:
            cells = t.add_row().cells
            for i, x in enumerate(row):
                if isinstance(x, tuple):
                    cells[i].text = x[0]
                    shade(cells[i], COLOR[x[1]])
                else:
                    cells[i].text = x
        for row in t.rows:
            for c in row.cells:
                for par in c.paragraphs:
                    for r in par.runs:
                        r.font.size = Pt(9)
        if note:
            self.p(note, italic=True, size=9)
        self.doc.add_paragraph()


def competitor_table(rep, R, ds, ours, others, na_rules, count=True, note=None):
    """Значения конкурентов — простым текстом; окрашены только наши ячейки.

    Против каждого конкурента — своя наша ячейка: наше значение, разница с ним и
    её интервал; цвет — вердикт этого сравнения.
    """
    d = R[ds]
    short = {"S6": "redmadrobot", "S7": "PIIDetector"}
    header = ["Метрика"] + [SYS_RU[o] for o in others] + [f"МЫ против {short[o]}" for o in others]
    rows = []
    for m in ["f1", "precision", "recall", "f1_overlap", "full_masked", "false_person_rate"]:
        row = [METRIC_RU[m]] + [fmt(m, d["systems"][o][m]) for o in others]
        for o in others:
            res = d["pairs"][f"{ours}|{o}"][m]
            verdict = "na" if na_rules(o, m) else res["verdict"]
            text = (f"{fmt(m, d['systems'][ours][m])}\nразница {fmt_diff(m, res['diff'])}\n"
                    f"ДИ {fmt_ci(m, res['ci95'])}\n{WORD[verdict]}")
            row.append((text, verdict))
            if count and verdict != "na":
                rep.score[(o, verdict)] += 1
        rows.append(row)
    rep.table(header, rows, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(os.path.join(RES, "competitors_paired.json"), encoding="utf-8"))
    ACC = json.load(open(os.path.join(RES, "acceptance_stage3.json"), encoding="utf-8"))
    rep = Report()

    t = rep.doc.add_heading("Тестирование модуля обезличивания на общедоступных наборах: "
                            "сравнение с конкурентами", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rep.p("Дата: 22.09.2026. Предмет: распознавание ФИО и утечки ПДн. Наш сервис — модель ФИО v3 "
          "(rubert-tiny2) + правила + фильтр ложных ФИО. Все цифры получены по протоколу "
          "data/eval_v2/PROTOCOL.md (поправки 2–5), скрипты и предсказания — в data/eval_v2/.", italic=True)

    # ── 0. Что это за тестирование
    rep.h("Что это за тестирование и зачем оно нужно")
    rep.h("Зачем", 2)
    rep.p("Модуль обезличивания убирает персональные данные (ФИО, телефоны, паспорта, адреса и т. д.) из текста "
          "перед тем, как текст уйдёт во внешний сервис или языковую модель. Ошибиться можно двумя способами:")
    rep.bullets([
        "пропустить ПДн — это утечка и нарушение 152-ФЗ; главный риск;",
        "замаскировать лишнее (например, слово «Страховой» как фамилию) — утечки нет, но текст портится "
        "и хуже обрабатывается дальше.",
    ])
    rep.p("Проверять модуль только на своих тестах нельзя: свои тесты делает тот же человек, что и модуль, и они "
          "завышают результат. Мы это измерили сами: в прежнем внутреннем тесте 97 % примеров были построены по "
          "тем же шаблонам, что и обучение, и оценка была завышена на 0,26 по F1. Поэтому модуль проверяется на "
          "общедоступных наборах, которые размечали другие люди, и на тех же текстах сравнивается с другими системами.")
    rep.h("Что проверяется", 2)
    rep.bullets([
        "Главное — распознавание ФИО: это единственное, что делает нейросетевая модель; остальные типы ищут правила.",
        "Дополнительно — утечки остальных ПДн (телефон, почта, ИНН, СНИЛС, паспорт, карта, адрес) там, где они размечены.",
    ])
    rep.h("Метрики простыми словами", 2)
    rep.table(["Метрика", "Что значит", "Что лучше"], [
        ["Имён скрыто полностью", "доля имён, у которых замаскировано каждое слово — имя не осталось в тексте даже частично. "
                                  "Главная метрика приватности", "больше"],
        ["Точность", "из всех масок «ФИО» — сколько действительно имена, причём с точными границами", "больше"],
        ["Полнота", "из всех имён — сколько найдено с точными границами", "больше"],
        ["Строгий F1", "среднее точности и полноты; стандартная метрика для сравнения систем", "больше"],
        ["F1 по пересечению", "то же, но засчитывается и маска с неточными границами", "больше"],
        ["Ложное ФИО в текстах без ФИО", "доля текстов без имён, где система всё же поставила маску имени", "меньше"],
        ["Утечки ПДн", "доля значений ПДн, оставшихся в тексте после обезличивания", "меньше"],
    ])
    rep.h("На каких наборах", 2)
    rep.table(["Набор", "Что это", "Кто размечал", "Зачем он в проверке"], [
        ["hivetrace/pii-bench", "1 810 сообщений поддержки и мессенджеров (банк, телеком, доставка, HR…)",
         "два эксперта независимо, согласие 0,87", "самый близкий к нашей задаче; к нему опубликована статья "
                                                    "с результатами других систем"],
        ["alrosait/pii-synthetic-ru", "4 500 сообщений поддержки, ФИО и адреса", "сгенерировано языковой моделью",
         "много имён в разных формах и падежах, трудные негативы"],
        ["MultiCoNER (русская часть)", "4 000 фраз из Википедии, весь текст строчными",
         "сторонние авторы, разметка неполная", "проверка на тексте без заглавных букв"],
        ["test_v2", "480 синтетических обращений", "мы, по протоколу, до обучения модели",
         "приёмка модели на нашем домене"],
        ["pii_benchmark (redmadrobot)", "2 841 сообщение, часть из реальных логов", "redmadrobot",
         "приёмка на чужом домене"],
        ["factRuEval-2016", "995 абзацев новостей", "соревнование «Диалог-2016»", "приёмка на новостях"],
    ])
    rep.h("Кто конкуренты", 2)
    rep.table(["Система", "Что это", "Чем интересна"], [
        [SYS_RU["S6"], "открытая модель Red Madrobot (MIT) для поиска ПДн в русском тексте, rubert-base",
         "самая точная из найденных открытых моделей, но крупная и медленная"],
        [SYS_RU["S7"], "открытый проект PIIDetector: spaCy + Presidio, та же архитектура, что у нас (модель + правила)",
         "прямой аналог по устройству; обучен на наборе alrosait"],
        ["GLiNER Guard, GLiNER2", "модели компании HiveTrace из статьи к набору hivetrace (arXiv 2605.05277)",
         "мы их не запускали — берём опубликованные числа"],
        [SYS_RU["S5-prev"], "наш сервис до переобучения модели", "показывает, что дала новая модель"],
    ])
    rep.h("Кто на чём проверялся", 2)
    yes, pub, no = "запускали мы", "опубликовано авторами", "—"
    rep.table(["Система", "hivetrace", "alrosait", "MultiCoNER", "test_v2", "pii_benchmark", "factRuEval"], [
        ["Наш сервис", yes, yes, yes, yes, yes, yes],
        [SYS_RU["S6"], yes, yes, yes, yes, yes, no],
        [SYS_RU["S7"], yes, yes, yes, no, no, no],
        ["GLiNER Guard / GLiNER2", pub + " (domain-часть)", no, no, no, no, no],
        [SYS_RU["S5-prev"], yes, yes, yes, yes, yes, yes],
    ], "Все системы, помеченные «запускали мы», прогонялись одной и той же программой на одних и тех же текстах "
       "и считались одними и теми же формулами. Проверка программы: для PIIDetector она дала NAME F1 = 0,944 "
       "на hivetrace — ровно число, опубликованное его авторами.")
    rep.h("Как проводилось", 2)
    rep.bullets([
        "Критерии и перечень систем записывались в протокол и фиксировались в git до первого прогона.",
        "Каждый набор для слепого замера использовался один раз; ни один не использовался для обучения.",
        "Разница с конкурентом считается с 95 % доверительным интервалом (парный бутстрап): случайное различие "
        "не выдаётся за выигрыш или проигрыш.",
    ])

    # ── 1. Правило цвета
    rep.h("1. Как читать цвета")
    rep.p("Цветом выделены только НАШИ результаты. Значения конкурентов даны простым текстом. Против каждого "
          "конкурента — своя наша ячейка: цвет показывает, выиграли мы у него, на уровне или проиграли.", bold=True)
    rep.table(["Цвет", "Значение", "Правило"], [
        [("зелёный", "win"), "выигрываем", "95 % доверительный интервал разности «мы минус конкурент» "
                                           "целиком в нашу пользу (парный бутстрап, 2000 итераций)"],
        [("жёлтый", "tie"), "на уровне", "интервал разности содержит ноль — различие не доказано; "
                                         "для величин без интервала — правило в разделе таблицы"],
        [("красный", "loss"), "проигрываем", "интервал разности целиком не в нашу пользу"],
        [("серый", "na"), "не засчитано", "сравнение некорректно: конкурент обучен на этом наборе, "
                                          "мы настраивались на нём или разметка не позволяет судить"],
    ], "Для «ложное ФИО» лучше меньшее значение, для остальных метрик — большее. В нашей ячейке: наше значение, "
       "разница «мы минус конкурент» и её интервал.")

    # ── 2. Наборы и системы
    rep.h("2. Наборы и системы")
    rep.table(["Набор", "Что внутри", "Разметка", "Лицензия", "Статус для нас"], [
        ["hivetrace/pii-bench", "1 810 текстов поддержки и мессенджеров, 228 имён", "два эксперта, согласие 0,87",
         "Apache-2.0", "слепой замер после приёмки; после правил 18.09 — не слепой"],
        ["alrosait/pii-synthetic-ru", "4 500 текстов, 2 941 имя, адреса", "сгенерировано LLM",
         "MIT", "слепой замер после приёмки; затем — dev для правил"],
        ["MultiCoNER ru", "4 000 фраз из Википедии, строчные, 829 имён", "сторонняя, неполная",
         "CC BY 4.0", "слепой замер после приёмки"],
        ["test_v2 / pii_benchmark / factRuEval", "обращения / логи / новости", "наша / redmadrobot / «Диалог»",
         "— / MIT / MIT", "приёмка модели v3 по предрегистрации"],
    ])
    rep.table(["Система", "Что это", "Скорость"], [
        [SYS_RU["S5-v3"], "rubert-tiny2 + правила + морфологический фильтр", "сервис 9–16 мс, модель 3,5 мс"],
        [SYS_RU["S6"], "rubert-base, обучена на русских ПДн (MIT)", "36–56 мс (одна модель)"],
        [SYS_RU["S7"], "spaCy ru_core_news_lg, дообучена на alrosait (MIT)", "3–4 мс (одна модель)"],
        ["GLiNER Guard, GLiNER2", "системы из статьи авторов hivetrace (arXiv 2605.05277)", "—"],
    ])

    # ── 3. Сравнение с конкурентами (слепой замер)
    rep.h("3. Наш сервис против конкурентов — слепой замер (модель v3)")
    rep.p("Замер выполнен один раз, до каких-либо правок по этим наборам.")
    rep.h("3.1. hivetrace/pii-bench — эксперты, домен поддержки", 2)
    competitor_table(rep, R, "hive", "S5-v3", ["S6", "S7"], lambda o, m: False)
    rep.h("3.2. alrosait/pii-synthetic-ru", 2)
    competitor_table(rep, R, "alro", "S5-v3", ["S6", "S7"], lambda o, m: o == "S7",
                     note="PIIDetector обучен на данных alrosait — его столбец серый, в итог не засчитан.")
    rep.h("3.3. MultiCoNER — строчный текст без регистра", 2)
    competitor_table(rep, R, "mcr", "S5-v3", ["S6", "S7"],
                     lambda o, m: m in ("precision", "f1", "f1_overlap", "false_person_rate"),
                     note="Разметка MultiCoNER неполная: больше половины «ложных» ФИО — неразмеченные настоящие имена. "
                          "Точность, F1 и ложные ФИО на нём не показательны (серые).")

    # ── 4. Итоговый счёт
    rep.h("4. Итоговый счёт (раздел 3, серые ячейки не считаются)")
    rows = []
    for o in ("S6", "S7"):
        w, ti, lo = rep.score[(o, "win")], rep.score[(o, "tie")], rep.score[(o, "loss")]
        rows.append([SYS_RU[o], (str(w), "win"), (str(ti), "tie"), (str(lo), "loss")])
    rep.table(["Против", "Выигрываем", "На уровне", "Проигрываем"], rows)

    # ── 5. Опубликованные результаты
    rep.h("5. Сравнение с опубликованными результатами (hivetrace, domain-часть)")
    dom = R["hive_domain"]
    ours, ci = dom["systems"]["S5-v3"]["f1"], dom["systems_ci95"]["S5-v3"]["f1"]
    ci_txt = f"[{fmt('f1', ci[0])}; {fmt('f1', ci[1])}]"
    rep.p(f"Правило: у опубликованных чисел нет интервалов, поэтому сравнение идёт с интервалом нашей метрики. "
          f"Наш строгий F1 по ФИО = {fmt('f1', ours)}, 95 % ДИ {ci_txt}. Число ниже интервала — "
          f"выигрываем, внутри — на уровне, выше — проигрываем. Правило подсчёта — как в статье: строгое совпадение "
          f"границ, domain-часть набора.")
    published = [("GLiNER Guard uni-encoder", 0.757), ("GLiNER Guard bi-encoder", 0.695), ("GLiNER2 Multi", 0.606),
                 ("GLiNER Guard Omni", 0.523), ("GLiNER2 Large", 0.303)]
    rows = []
    for name, v in published:
        verdict = "win" if v < ci[0] else "loss" if v > ci[1] else "tie"
        rows.append([name + " (статья)", fmt("f1", v),
                     (f"{fmt('f1', ours)}\nразница {fmt_diff('f1', ours - v)}\n{WORD[verdict]}", verdict)])
    for s in ("S6", "S7"):
        res = dom["pairs"][f"S5-v3|{s}"]["f1"]
        rows.append([SYS_RU[s] + " (наш замер)", fmt("f1", dom["systems"][s]["f1"]),
                     (f"{fmt('f1', ours)}\nразница {fmt_diff('f1', res['diff'])}\nДИ {fmt_ci('f1', res['ci95'])}\n"
                      f"{WORD[res['verdict']]}", res["verdict"])])
    rep.table(["Система", "Её F1 по ФИО", "НАШ F1 по ФИО"], rows,
              "Проверка методики: для PIIDetector наш замер на всём hivetrace дал NAME F1 = 0,944 — ровно число, "
              "опубликованное его авторами.")

    # ── 6. Текущая версия
    rep.h("6. Текущая версия (v3 + правила 18.09) против конкурентов")
    rep.p("Правила настраивались на alrosait (dev по поправке 5), а классы ошибок hivetrace были известны по разбору. "
          "Поэтому hivetrace здесь — не слепой замер, alrosait — серый целиком. MultiCoNER для правил не использовался.")
    competitor_table(rep, R, "hive", "S5-r7", ["S6", "S7"], lambda o, m: False, count=False,
                     note="Не слепой замер: цвета информативны, но в итоговый счёт раздела 4 не входят.")
    competitor_table(rep, R, "alro", "S5-r7", ["S6", "S7"], lambda o, m: True, count=False,
                     note="Серое целиком: правила настраивались на этом наборе.")
    competitor_table(rep, R, "mcr", "S5-r7", ["S6", "S7"],
                     lambda o, m: m in ("precision", "f1", "f1_overlap", "false_person_rate"), count=False)

    # ── 7. Скорость
    rep.h("7. Скорость (медиана, мс на текст, процессор без GPU)")
    rep.p("Правило: сравниваются одинаковые вещи (модель с моделью). «На уровне» — разница не больше 25 %. "
          "Время всего нашего сервиса (правила + модель + фильтр) с временем одной модели конкурента не сравнивается — серое.")
    L = R["latency_median_ms"]

    def speed(v_ours, v_other):
        r = v_ours / v_other
        return "win" if r < 0.8 else "loss" if r > 1.25 else "tie"
    rows = []
    for ds, name in (("hive", "hivetrace"), ("alro", "alrosait"), ("mcr", "MultiCoNER")):
        v6, v7 = speed(3.47, L[ds]["S6"]), speed(3.47, L[ds]["S7"])
        rows.append([name, f"{L[ds]['S6']:.1f}".replace(".", ","), f"{L[ds]['S7']:.1f}".replace(".", ","),
                     (f"3,5 — {WORD[v6]}", v6), (f"3,5 — {WORD[v7]}", v7),
                     (f"{L[ds]['S5-v3']:.1f} (весь сервис, не сравнивается)".replace(".", ","), "na")])
    rep.table(["Набор", SYS_RU["S6"] + ", мс", SYS_RU["S7"] + ", мс", "НАША модель против redmadrobot",
               "НАША модель против PIIDetector", "Наш сервис целиком, мс"], rows,
              "Наша модель — медиана 3,47 мс на test_v2 (замер приёмки).")

    # ── 8. Приёмка: мы против себя прежних
    rep.h("8. Приёмка модели v3 по протоколу: новая модель против прежней")
    rep.p("Код сервиса одинаковый, меняется только модель. Интервалы — из предрегистрированного замера (поправка 3).")
    keys = [("ФИО: F1", False, "f1"), ("ФИО: замаскировано полностью", False, "full_masked"),
            ("Без ПДн: ложное ФИО (доля текстов)", True, "false_person_rate"),
            ("Утечки ПДн, все типы (доля)", True, "false_person_rate")]
    rows = []
    for ds, name in (("v2", "test_v2"), ("bench", "pii_benchmark"), ("ext", "factRuEval")):
        dd = ACC["datasets"][ds]
        row = [name]
        for k, lower, m in keys:
            lo, hi = dd["diff_ci95"][k]
            if lower:
                lo, hi = -hi, -lo
            verdict = "win" if lo > 0 else "loss" if hi < 0 else "tie"
            pct = m in PCT or "Утечки" in k
            before, after = dd["prod"][k], dd["B"][k]
            b_txt = (f"{before * 100:.1f} %" if pct else f"{before:.3f}").replace(".", ",")
            a_txt = (f"{after * 100:.1f} %" if pct else f"{after:.3f}").replace(".", ",")
            row.append(b_txt)
            row.append((a_txt + f"\n{WORD[verdict]}", verdict))
        rows.append(row)
    rep.table(["Набор", "F1: прежняя", "F1: НАША v3", "Скрыто: прежняя", "Скрыто: НАША v3",
               "Ложное ФИО: прежняя", "Ложное ФИО: НАША v3", "Утечки: прежняя", "Утечки: НАША v3"], rows,
              "Красная ячейка factRuEval: одиночные фамилии без имени в новостях стали утекать чаще (3,1 → 7,9 %).")

    # ── 9. Утечки по типам на hivetrace: до и после правил
    rep.h("9. Утечки по типам ПДн на hivetrace: до и после правил 18.09 (не слепой замер)")
    raw = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "hive.jsonl"), encoding="utf-8"))}
    cnt = {}
    for tag in ("v3", "r7"):
        P = {r["id"]: r for r in map(json.loads, open(os.path.join(RES, f"preds_S5-{tag}_hive.jsonl"), encoding="utf-8"))}
        tot, leak = Counter(), Counter()
        for rid, r in raw.items():
            for s in r["spans"]:
                if s["type"].startswith("OTHER_"):
                    continue
                tot[s["type"]] += 1
                leak[s["type"]] += r["text"][s["start"]:s["stop"]] in P[rid]["anonymized"]
        cnt[tag] = (tot, leak)
    rows = []
    for typ in sorted(cnt["v3"][0], key=lambda t: -cnt["v3"][1][t]):
        b, a_, n = cnt["v3"][1][typ], cnt["r7"][1][typ], cnt["v3"][0][typ]
        verdict = "win" if a_ < b else "loss" if a_ > b else "tie"
        rows.append([typ, str(n), str(b), (f"{a_}  ({WORD[verdict] if verdict != 'tie' else 'без изменений'})", verdict)])
    rep.table(["Тип", "Значений", "Утекло до", "Утекло после"], rows,
              "Цвет здесь — направление изменения без интервала по типу. По всем типам вместе разница значима: "
              "8,0 → 2,4 %, ДИ [−7,0; −4,4] п.п. Жёлтое — нулевые утечки до и после.")

    # ── 10. Выводы
    rep.h("10. Выводы")
    rep.h("Где выигрываем", 2)
    rep.bullets([
        "Доля полностью скрытых имён: значимо выше PIIDetector на hivetrace и MultiCoNER и выше redmadrobot на alrosait "
        "и MultiCoNER; на hivetrace — на уровне redmadrobot (100 % против 99,6 %).",
        "Строчный текст без регистра (MultiCoNER): скрыто 78 % имён против 50 % у redmadrobot и 1 % у PIIDetector.",
        "Системы из статьи авторов hivetrace: все пять ниже нашего интервала F1.",
        "Скорость модели против redmadrobot: в 10–16 раз быстрее.",
        "Против себя прежних: F1 по ФИО вырос значимо на всех трёх наборах приёмки.",
    ])
    rep.h("Где на уровне", 2)
    rep.bullets([
        "Скорость модели против PIIDetector (3,5 мс против 3–4 мс).",
        "Доля скрытых имён против redmadrobot на hivetrace (100 % против 99,6 %, различие не доказано).",
    ])
    rep.h("Где проигрываем", 2)
    rep.bullets([
        "Точность и строгий F1 по ФИО на hivetrace — и redmadrobot, и PIIDetector: у нас больше лишних масок "
        "(захват соседнего слова, опечатки как имена).",
        "Ложные ФИО в текстах без ФИО — на hivetrace против обоих конкурентов, на alrosait против redmadrobot "
        "(на MultiCoNER метрика не показательна).",
        "Строгая полнота и F1 по пересечению на hivetrace против redmadrobot: имена мы скрываем не хуже "
        "(доля скрытых — на уровне), но границы маски чаще не совпадают с эталоном.",
        "Точность и F1 против redmadrobot на alrosait (слепой замер модели v3).",
        "Прежняя версия против новой на новостях (factRuEval): утечки одиночных фамилий.",
    ])
    rep.h("Что это значит", 2)
    rep.p("Наш сервис сильнее там, где важнее всего для обезличивания, — в полноте: имя реже остаётся открытым, "
          "особенно в неаккуратном тексте. Слабее — в точности: чаще маскирует лишнее. Для задачи «не выпустить ПДн» "
          "это правильная сторона ошибки, но лишние маски портят текст для дальнейшей обработки. Следующий шаг — "
          "переобучение на трудных негативах и дистилляция из redmadrobot (БУДУЩИЕ_ПЛАНЫ.md, пп. 3 и 3а).")
    rep.h("Ограничения", 2)
    rep.bullets([
        "Реальных обращений среди наборов нет; наборы домена поддержки синтетические.",
        "Раздел 6 и 9 — не слепые замеры (классы ошибок были известны).",
        "Все внешние наборы теперь использованы; следующая модель требует нового слепого набора.",
    ])
    rep.doc.save(a.out)
    print(a.out)
    print({f"{o}:{v}": rep.score[(o, v)] for o in ("S6", "S7") for v in ("win", "tie", "loss")})


if __name__ == "__main__":
    main()
