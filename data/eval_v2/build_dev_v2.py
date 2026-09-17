"""
Dev-набор для НАСТРОЙКИ сервиса (этап 0.1a плана улучшений).

Правило П1 плана: набор test_v2.jsonl — только финальная приёмка, настраивать
по нему нельзя. Поэтому правила, пороги и фильтры подбираются на этом наборе.

Механика та же, что у build_test_v2.py (склонение по правилам, генераторы
значений, проверки), но всё содержательное своё:
  * другие шаблоны фраз, другие словари имён, отчеств, фамилий, топонимов;
  * непересечение проверяется ДВАЖДЫ — с обучающими данными И с набором v2.
    Если dev пересекается с v2, то настройка на dev косвенно подгоняет
    сервис под финальный тест.

Запуск (из корня репозитория):
    .venv/bin/python data/eval_v2/build_dev_v2.py
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "dev_v2.jsonl")
REPORT_PATH = os.path.join(HERE, "dev_v2_report.json")
SEED = 20260917

spec = importlib.util.spec_from_file_location("b", os.path.join(HERE, "build_test_v2.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

# ── словари v2 запоминаем ДО подмены — с ними проверяется непересечение ──
V2_LEX = {
    "first": b.MALE_NAMES + b.FEMALE_NAMES,
    "patronymic": [p for pair in b.PATRONYMICS for p in pair],
    "surname": [s for s, _ in b.SURNAMES],
    "city": list(b.CITIES), "street": list(b.STREETS),
    "org": list(b.ORGS), "noun": list(b.NOUNS),
}
V2_TEMPLATES = list(b.PII_TEMPLATES) + list(b.NEG_TEMPLATES)
V2_FIXED = list(b.NEG_FIXED)

# ── словари dev: не пересекаются ни с обучением, ни с v2 ─────────────────
# Имена подобраны под правила склонения сборщика: мужские на согласный,
# -ий, -ей; женские на -а, -я, -ия. Избегались беглые гласные (Пётр → Петра,
# Лев → Льва) и мужские имена на -а/-я (Никита, Илья) — их правила не покрывают.
b.MALE_NAMES = [
    "Антон", "Евгений", "Борис", "Василий", "Фёдор", "Анатолий", "Константин",
    "Вячеслав", "Герман", "Ефим", "Даниил", "Арсений", "Родион", "Савелий",
    "Климент", "Эрик", "Альберт", "Артур", "Нестор", "Лаврентий",
]
b.FEMALE_NAMES = [
    "Лидия", "Майя", "Раиса", "Тамара", "Эмилия", "Юлиана", "Кристина", "Диана",
    "Евгения", "Маргарита", "Изабелла", "Стефания", "Агата", "Аделина",
    "Анастасия", "Виолетта", "Серафима", "Мирослава", "Снежана", "Ева",
]
b.PATRONYMICS = [
    ("Антонович", "Антоновна"), ("Евгеньевич", "Евгеньевна"),
    ("Фёдорович", "Фёдоровна"), ("Даниилович", "Данииловна"),
    ("Арсеньевич", "Арсеньевна"), ("Родионович", "Родионовна"),
    ("Савельевич", "Савельевна"), ("Климентович", "Климентовна"),
    ("Альбертович", "Альбертовна"), ("Артурович", "Артуровна"),
    ("Семёнович", "Семёновна"), ("Тихонович", "Тихоновна"),
    ("Демьянович", "Демьяновна"), ("Несторович", "Несторовна"),
    ("Эрикович", "Эриковна"),
]
b.SURNAMES = [
    ("Тихомиров", "ov"), ("Воронцов", "ov"), ("Галкин", "ov"), ("Журавлёв", "ov"),
    ("Кудрявцев", "ov"), ("Маслов", "ov"), ("Ершов", "ov"), ("Щукин", "ov"),
    ("Троицкий", "sky"), ("Покровский", "sky"), ("Раевский", "sky"),
    ("Добровольский", "sky"), ("Ольшанский", "sky"),
    ("Шевченко", "fixed"), ("Тимошенко", "fixed"), ("Кравченко", "fixed"),
    ("Белых", "fixed"), ("Хитрово", "fixed"), ("Живаго", "fixed"),
    ("Ковальчук", "cons"), ("Саркисян", "cons"), ("Бойчук", "cons"),
    ("Стасюк", "cons"), ("Оганесян", "cons"),
]
b.CITIES = [
    "Калуга", "Тула", "Рязань", "Брянск", "Орёл", "Белгород", "Псков", "Сургут",
    "Нальчик", "Абакан", "Ухта", "Сыктывкар", "Выборг", "Коломна", "Обнинск",
    "Дубна", "Муром", "Камышин",
]
b.STREETS = [
    "Гоголя", "Пирогова", "Маяковского", "Горького", "Суворова", "Кутузова",
    "Мичурина", "Вокзальная", "Октябрьская", "Набережная",
]
b.ORGS = ["Орбита", "Полюс", "Сигма", "Вертикаль", "Импульс", "Гранит", "Сапфир",
          "Фрегат", "Каскад", "Атлант"]
b.NOUNS = ["Инструкция", "Анкета", "Выгрузка", "Отчёт", "Шаблон", "Тикет", "Регламент",
           "Акт", "Черновик", "Архив", "Резюме", "Прайс", "Лицензия", "Ордер", "Бланк"]

PII_TEMPLATES = [
    "Бронь на имя {P:gent:full} подтверждена, заезд в пятницу.",
    "Техподдержка, это {P:nomn:any}, у меня не работает личный кабинет.",
    "{P:nomn:full} {g:оплатил|оплатила} полис, копия ушла на {email}.",
    "Для связи с {P:ablt:full} используйте номер {phone}.",
    "Страховой случай заявлен {P:ablt:full}, СНИЛС {snils}.",
    "Абонентский договор переоформлен на {P:accs:full}.",
    "Мастер {P:nomn:full} приедет после 14:00.",
    "{P:nomn:address}, напоминаем о записи на техосмотр.",
    "Рецепт выписан на {P:accs:full}, дата рождения {dob}.",
    "{g:Ученик|Ученица} {P:nomn:full} не {g:сдал|сдала} домашнее задание.",
    "Заявление на Госуслугах подано от имени {P:gent:full}.",
    "Проверьте, пожалуйста, ИНН {inn}: владелец {P:nomn:full}?",
    "Отпуск {P:gent:full} согласован с 3 по 17 августа.",
    "Номер {phone} закреплён за {P:ablt:full}.",
    "Посылку заберёт {P:nomn:full}, паспорт {passport}.",
    "Ваш персональный менеджер — {P:nomn:full}, e-mail {email}.",
    "{P:nomn:short}, добрый день! Документы готовы.",
    "Возврат по карте {card} оформлен на {P:accs:full}.",
    "Арендатор {P:nomn:full} просит продлить договор.",
    "Со слов соседки, {P:nomn:full} давно не {g:появлялся|появлялась}.",
    "В табеле отсутствуют часы {P:gent:full} за вторую неделю.",
    "Пропуск на территорию выписан {P:datv:full}.",
    "Водитель {P:nomn:full} {g:доставил|доставила} груз с опозданием.",
    "Анкета кандидата: {P:nomn:full}, дата рождения {dob}, тел. {phone}.",
    "Прошу аннулировать доверенность на {P:accs:full}.",
    "Участник конкурса {P:nomn:full} {g:прошёл|прошла} во второй тур.",
    "Вчера звонила мама {P:gent:short}, просила перезвонить.",
    "Бухгалтерия: зарплату {P:datv:full} перечислить на карту {card}.",
    "Экзамен у {P:gent:full} перенесён на следующую сессию.",
    "Реквизиты получателя: {P:nomn:full} / ИНН {inn} / паспорт {passport} / СНИЛС {snils}",
    "Тикет #5521 закреплён за инженером {P:ablt:full}.",
    "Спасибо {P:datv:address} за помощь с переездом!",
    "Нотариус {P:nomn:full} {g:удостоверил|удостоверила} подпись.",
    "Номер брони 77-4410, гость — {P:nomn:full}.",
    "Сообщение для {P:gent:full}: ваш заказ задерживается.",
    "На осмотр {g:пришёл|пришла} {P:nomn:full} без направления.",
    "Тренер {P:nomn:full} заменит занятие в среду.",
    "Учётную запись {P:gent:full} заблокировали после трёх попыток входа.",
    "Подписано электронной подписью: {P:nomn:ini}",
    "Ответственные: {P:nomn:ini} (снабжение), {P2:nomn:ini} (логистика).",
    "Кредит одобрен, заёмщик {P:nomn:full}, телефон {phone}.",
    "По поручению {P:gent:full} направляем акт сверки.",
    "{P:nomn:full} {g:переехал|переехала}, новый номер для связи {phone}.",
    "Замечание сделано {P:datv:full} устно.",
    "Родителям {P:gent:full}: собрание в четверг в 18:00.",
    "Карта {card} принадлежит {P:datv:full}, выпущена в прошлом году.",
    "Автомобиль записан на {P:accs:full}, осмотр завтра.",
    "Письмо с адреса {email} подписано {P:ablt:full}.",
    "На линии {P:nomn:full}, уточняет статус возврата.",
    "Поздравляем {P:accs:full} с юбилеем!",
]
NEG_TEMPLATES = [
    "Отделение в г. {city} работает без перерыва.",
    "Поезд до станции {city} отправляется с третьего пути.",
    "Доставка по г. {city} — бесплатно от 3000 рублей.",
    "Мероприятие пройдёт в городе {city} в конце месяца.",
    "Магазин на ул. {street} закрыт на ремонт.",
    "Парковка у дома на улице {street} платная.",
    "Остановка «ул. {street}» перенесена на 200 метров.",
    "Компания «{org}» выиграла тендер.",
    "Счёт выставлен от имени ООО «{org}».",
    "{noun} во вложении, проверьте до обеда.",
    "Раздел «{noun}» временно недоступен.",
    "{noun}: изменения сохранены.",
]
NEG_FIXED = [
    "Любовь клиентов к бренду растёт.",
    "Сервер «Орион» перезагружен в 03:00.",
    "Статус заявки: в работе.",
    "Пункт выдачи «Север» переехал.",
    "Код подтверждения отправлен повторно.",
    "Скидка 15 % действует до конца месяца.",
    "Номер договора 2024-117 — укажите его в назначении платежа.",
    "ТЕЛ. горячей линии указан на обороте.",
    "Кв. 12, эт. 3, подъезд 2 — вход со двора.",
    "Г. Тула и г. Калуга временно не обслуживаются.",
]


def main():
    rng = random.Random(SEED)
    failures = []

    def low(xs):
        return {x.lower().replace("ё", "е") for x in xs}

    # ── обучающая сторона (как в сборщике v2) ────────────────────────────
    gen = b.load_module("generate_dataset", os.path.join(b.TRAIN_DIR, "generate_dataset.py"))
    aug = b.load_module("augment_negatives", os.path.join(b.TRAIN_DIR, "augment_negatives.py"))
    train_rows = []
    for rel in ["train/train.jsonl", "dev/dev.jsonl", "test/test.jsonl",
                "train/train_negatives.jsonl", "dev/dev_negatives.jsonl"]:
        train_rows += b.read_jsonl(os.path.join(b.TRAIN_DIR, rel))
    train_frames = {b.frame_of_example(r) for r in train_rows}
    for t in list(gen.TEMPLATES) + [x for n in ("NEG_TEMPLATES_TOPO", "NEG_TEMPLATES_NOUN",
                                                "NEG_TEMPLATES_STREET", "NEG_TEMPLATES_ORG",
                                                "PERSON_TOPO_TEMPLATES")
                                     for x in getattr(aug, n)]:
        train_frames |= b.frames_of_template(t)
    train_lex = {
        "first": low(gen.MALE_NAMES + gen.FEMALE_NAMES),
        "patronymic": low(gen.MALE_PATR + gen.FEMALE_PATR),
        "surname": low(gen.SURNAMES_M + [s + "а" for s in gen.SURNAMES_M]),
        "city": low(aug.TOPONYMS), "street": low(aug.STREETS),
        "org": low(aug.ORGS), "noun": low(aug.NOUNS),
    }

    # ── сторона v2 ───────────────────────────────────────────────────────
    v2_rows = b.read_jsonl(os.path.join(HERE, "test_v2.jsonl"))
    v2_frames = set()
    for t in V2_TEMPLATES:
        v2_frames |= b.frames_of_template(t)
    v2_frames |= {b.norm_frame(t) for t in V2_FIXED}
    v2_lex = {k: low(v) for k, v in V2_LEX.items()}

    dev_lex = {
        "first": low(b.MALE_NAMES + b.FEMALE_NAMES),
        "patronymic": low([p for pair in b.PATRONYMICS for p in pair]),
        "surname": low([s for s, _ in b.SURNAMES]),
        "city": low(b.CITIES), "street": low(b.STREETS), "org": low(b.ORGS), "noun": low(b.NOUNS),
    }

    # 1. словари
    overlap = {}
    for k in dev_lex:
        t_int, v_int = sorted(dev_lex[k] & train_lex[k]), sorted(dev_lex[k] & v2_lex[k])
        overlap[k] = {"training": t_int, "v2": v_int}
        if t_int:
            failures.append(f"словарь «{k}» пересекается с обучением: {t_int}")
        if v_int:
            failures.append(f"словарь «{k}» пересекается с v2: {v_int}")
    person_keys = ("first", "patronymic", "surname")
    dev_person = set().union(*(dev_lex[k] for k in person_keys))
    other_person = set().union(*(train_lex[k] for k in person_keys), *(v2_lex[k] for k in person_keys))
    cross = sorted(dev_person & other_person)
    if cross:
        failures.append(f"элементы ФИО dev встречаются в словарях обучения или v2: {cross}")

    # 2. склонение сверено с pymorphy
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()

    def pm(word, gender, case, need):
        for p in morph.parse(word):
            if need in p.tag.grammemes:
                inf = p.inflect({case, gender})
                if inf:
                    return inf.word.capitalize()
        return None

    declension_mismatch = []
    for g, names in (("masc", b.MALE_NAMES), ("femn", b.FEMALE_NAMES)):
        for w in names:
            for c in b.CASES:
                ours, ref = b.decline_first_name(w, g, c), pm(w, g, c, "Name")
                if ref and ref.replace("ё", "е") != ours.replace("ё", "е"):
                    declension_mismatch.append(f"{w} {c}: {ours} / pymorphy {ref}")
    for pm_, pf_ in b.PATRONYMICS:
        for g, w in (("masc", pm_), ("femn", pf_)):
            for c in b.CASES:
                ours, ref = b.decline_patronymic(w, g, c), pm(w, g, c, "Patr")
                if ref and ref.replace("ё", "е") != ours.replace("ё", "е"):
                    declension_mismatch.append(f"{w} {c}: {ours} / pymorphy {ref}")
    for base, cls in b.SURNAMES:
        for g in ("masc", "femn"):
            forms = [b.decline_surname(base, cls, g, c) for c in b.CASES]
            for c, f in zip(b.CASES, forms):
                ref = pm(base if g == "masc" else forms[0], g, c, "Surn")
                if ref and ref.replace("ё", "е") != f.replace("ё", "е"):
                    declension_mismatch.append(f"{base} {g} {c}: {f} / pymorphy {ref}")
    if declension_mismatch:
        failures.append(f"склонение расходится с pymorphy ({len(declension_mismatch)}): "
                        f"{declension_mismatch[:8]}")

    # 3. сборка
    examples, n = [], 0
    for ti, tpl in enumerate(PII_TEMPLATES, 1):
        for _ in range(b.INSTANCES_PII):
            text, spans = b.build_pii(tpl, rng)
            n += 1
            examples.append({"id": f"dev-{n:04d}", "template_id": f"DP{ti:03d}",
                             "subset": "pii", "text": text, "spans": spans})
    for ti, tpl in enumerate(NEG_TEMPLATES, 1):
        seen = set()
        for _ in range(b.INSTANCES_NEG_SLOT):
            text = b.build_neg(tpl, rng)
            if text in seen:
                continue
            seen.add(text)
            n += 1
            examples.append({"id": f"dev-{n:04d}", "template_id": f"DN{ti:03d}",
                             "subset": "neg", "text": text, "spans": []})
    for fi, text in enumerate(NEG_FIXED, 1):
        n += 1
        examples.append({"id": f"dev-{n:04d}", "template_id": f"DF{fi:03d}",
                         "subset": "neg", "text": text, "spans": []})

    # 4. целостность и лексика внутри спанов
    def person_words(rows):
        out = set()
        for r in rows:
            for s in r["spans"]:
                if s["type"] == "PERSON":
                    out |= {w.replace("ё", "е") for w in
                            b.WORD.findall(r["text"][s["start"]:s["stop"]].lower()) if len(w) > 1}
        return out

    forbidden_words = person_words(train_rows) | person_words(v2_rows)
    all_names = dev_person | other_person
    for ex in examples:
        prev = -1
        for s in sorted(ex["spans"], key=lambda s: s["start"]):
            val = ex["text"][s["start"]:s["stop"]]
            if not val or val != val.strip():
                failures.append(f"{ex['id']}: пустой спан или пробел на краю: {val!r}")
            if s["start"] < prev:
                failures.append(f"{ex['id']}: пересекающиеся спаны")
            prev = s["stop"]
        if "{" in ex["text"] or "}" in ex["text"]:
            failures.append(f"{ex['id']}: незаполненный слот")
        hit = person_words([ex]) & forbidden_words
        if hit:
            failures.append(f"{ex['id']}: слова ФИО встречались в обучении или v2: {sorted(hit)}")
        if ex["subset"] == "neg":
            bad = [w for w in b.WORD.findall(ex["text"].lower().replace("ё", "е")) if w in all_names]
            if bad:
                failures.append(f"{ex['id']}: в негативе слово из словарей ФИО: {bad}")

    # 5. шаблоны не совпадают ни с обучением, ни с v2
    exact, sims = [], []
    for kind, tpls in (("DP", PII_TEMPLATES), ("DN", NEG_TEMPLATES)):
        for ti, tpl in enumerate(tpls, 1):
            fr = b.frames_of_template(tpl)
            tid = f"{kind}{ti:03d}"
            if fr & train_frames:
                exact.append(f"{tid}~обучение")
            if fr & v2_frames:
                exact.append(f"{tid}~v2")
            bt = max(b.jaccard(a, x) for a in fr for x in train_frames)
            bv = max(b.jaccard(a, x) for a in fr for x in v2_frames)
            sims.append({"template_id": tid, "max_jaccard_training": round(bt, 3),
                         "max_jaccard_v2": round(bv, 3)})
    for fi, text in enumerate(NEG_FIXED, 1):
        fr = b.norm_frame(text)
        if fr in train_frames or fr in v2_frames:
            exact.append(f"DF{fi:03d}")
    if exact:
        failures.append(f"шаблоны совпадают с обучением или v2: {exact}")
    close = [s for s in sims if max(s["max_jaccard_training"], s["max_jaccard_v2"]) >= 0.5]
    if close:
        failures.append(f"шаблоны слишком похожи (Жаккар >= 0,5): {close}")

    pii = [e for e in examples if e["subset"] == "pii"]
    ents = [s for e in pii for s in e["spans"]]
    persons = [s for s in ents if s["type"] == "PERSON"]
    report = {
        "seed": SEED, "purpose": "настройка (не приёмка)",
        "examples_total": len(examples), "examples_pii": len(pii),
        "examples_neg": len(examples) - len(pii),
        "entities_by_type": dict(Counter(s["type"] for s in ents)),
        "person_by_format": dict(Counter(s["meta"]["format"] for s in persons)),
        "person_lowercase": sum(s["meta"]["lower"] for s in persons),
        "max_jaccard_training": max(s["max_jaccard_training"] for s in sims),
        "max_jaccard_v2": max(s["max_jaccard_v2"] for s in sims),
        "lexicon_overlap": overlap,
        "declension_mismatches_with_pymorphy": declension_mismatch,
        "failures": failures,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if failures:
        print("ПРОВЕРКИ НЕ ПРОЙДЕНЫ — набор не записан:")
        for x in failures[:30]:
            print("  -", x)
        sys.exit(1)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    report["sha256"] = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("lexicon_overlap", "declension_mismatches_with_pymorphy")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
