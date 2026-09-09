"""Генерирует тестовые документы с синтетическими ПДн в tests/web-тест/тестовые_файлы/.

Запуск (из корня репозитория):
    .venv/bin/python tests/web-тест/сгенерировать_тестовые_файлы.py

Три набора, по нарастанию сложности:

1. `тестовый_документ.*`   — базовый: немного текста и таблица «метка/значение».
2. `много_текста.*`        — объём: ~12 страниц сплошного текста, ПДн разных
   людей разбросаны по всему документу. Проверяет recall на длинном входе
   и скорость (для PDF растеризация всех страниц — самая тяжёлая операция).
3. `с_картинками.*`        — ПДн ВНУТРИ изображения, то есть пикселями.

Про третий набор отдельно. Оба обработчика ищут ПДн только в ТЕКСТЕ: docx —
по runs абзацев и таблиц, pdf — по словам с координатами из pdfplumber.
У текста на картинке текстового слоя нет, поэтому он не будет найден и
останется в выдаче. Это не баг генератора, а известный предел модуля:
закрыть его может только OCR. Файлы нужны, чтобы этот предел был виден
глазами, а не жил в предположениях.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "тестовые_файлы")
os.makedirs(OUT, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ── синтетические (вымышленные) ПДн ───────────────────────────────────────
PERSON = "Сидоров Пётр Иванович"
FIELDS = [
    ("ФИО", PERSON),
    ("ИНН", "7707083893"),
    ("Паспорт", "45 08 731902"),
    ("СНИЛС", "112-233-445 95"),
    ("Телефон", "+7 (921) 555-18-43"),
    ("E-mail", "petr.sidorov@example.com"),
    ("Дата рождения", "15.03.1985"),
    ("Карта", "5536 9102 7741 3086"),
    ("Адрес", "г. Москва, ул. Ленина, дом 5, квартира 12"),
]
BODY = (
    f"Заявление. Я, {PERSON}, прошу оформить услугу. "
    "Мои данные: ИНН 7707083893, СНИЛС 112-233-445 95, паспорт серия 45 08 номер 731902. "
    "Телефон +7 (921) 555-18-43, почта petr.sidorov@example.com, дата рождения 15.03.1985. "
    "Карта для оплаты 5536 9102 7741 3086. "
    "Проживаю по адресу: Санкт-Петербург, Невский проспект, дом 18, квартира 47."
)

# Разные люди — чтобы на длинном документе было видно, кого модуль пропустил.
PEOPLE = [
    {"fio": "Ковалёва Анна Сергеевна", "inn": "500100732259",
     "snils": "224-958-123 11", "pass": "40 12 583214",
     "tel": "+7 (495) 221-09-87", "mail": "a.kovaleva@example.com",
     "dr": "02.07.1990", "card": "4276 1842 9031 5570",
     "addr": "г. Казань, ул. Баумана, дом 34, квартира 7"},
    {"fio": "Морозов Дмитрий Олегович", "inn": "645311458702",
     "snils": "318-402-771 63", "pass": "60 09 114725",
     "tel": "8-903-447-15-62", "mail": "d.morozov@example.com",
     "dr": "19.11.1978", "card": "5469 3800 1234 5678",
     "addr": "г. Новосибирск, Красный проспект, дом 12, квартира 105"},
    {"fio": "Тихонова Елена Владимировна", "inn": "771902334815",
     "snils": "905-114-238 47", "pass": "45 17 902345",
     "tel": "+7 921 330-77-01", "mail": "e.tihonova@example.com",
     "dr": "27.04.1983", "card": "4058 1290 5566 7788",
     "addr": "г. Екатеринбург, ул. Малышева, дом 51, квартира 22"},
    {"fio": "Абрамов Сергей Петрович", "inn": "236104558931",
     "snils": "441-207-663 09", "pass": "03 14 667201",
     "tel": "+7(812)445-92-13", "mail": "s.abramov@example.com",
     "dr": "08.01.1995", "card": "2200 7001 4455 9012",
     "addr": "г. Самара, ул. Полевая, дом 8, квартира 3"},
]

# Наполнитель: обычный канцелярит без ПДн — имитирует реальный документ,
# где полезных данных мало, а текста много.
FILLER = [
    "Настоящий документ составлен в двух экземплярах, имеющих равную "
    "юридическую силу, по одному для каждой из сторон.",
    "Стороны подтверждают, что обладают необходимой правоспособностью для "
    "заключения настоящего соглашения и исполнения принятых обязательств.",
    "Срок рассмотрения обращения составляет не более тридцати календарных дней "
    "с момента его регистрации в системе документооборота.",
    "В случае возникновения разногласий стороны принимают меры к их "
    "урегулированию путём переговоров до обращения в судебные инстанции.",
    "Исполнитель обязуется обеспечить конфиденциальность сведений, ставших ему "
    "известными в ходе исполнения настоящего договора.",
    "Изменения и дополнения к настоящему документу оформляются в письменной "
    "форме и подписываются уполномоченными представителями сторон.",
    "Заявитель уведомлён о порядке обработки представленных сведений и о праве "
    "отозвать согласие в любой момент.",
    "Ответственность сторон определяется в соответствии с действующим "
    "законодательством Российской Федерации.",
]


def person_block(p, n):
    """Абзацы про одного человека — ПДн вперемешку с обычным текстом."""
    return [
        f"Раздел {n}. Сведения о заявителе.",
        f"Заявитель: {p['fio']}, дата рождения {p['dr']}. "
        f"Документ, удостоверяющий личность: паспорт {p['pass']}.",
        f"Идентификаторы: ИНН {p['inn']}, СНИЛС {p['snils']}.",
        f"Контакты для связи: телефон {p['tel']}, электронная почта {p['mail']}.",
        f"Адрес регистрации: {p['addr']}.",
        f"Реквизиты для перечисления: карта {p['card']}.",
    ]


# ══════════════════════════════════════════════════════════════════════════
#  1. Базовый документ
# ══════════════════════════════════════════════════════════════════════════
def make_docx():
    from docx import Document
    doc = Document()
    doc.add_heading("Тестовый документ с ПДн", level=1)
    doc.add_paragraph(BODY)
    doc.add_paragraph("Таблица данных (метка и значение в разных ячейках):")
    t = doc.add_table(rows=1, cols=2)
    t.style = "Table Grid"
    t.rows[0].cells[0].text = "Поле"
    t.rows[0].cells[1].text = "Значение"
    for label, val in FIELDS:
        c = t.add_row().cells
        c[0].text = label
        c[1].text = val
    path = os.path.join(OUT, "тестовый_документ.docx")
    doc.save(path)
    return path


def _pdf_canvas(path):
    """Канвас A4 с зарегистрированным кириллическим шрифтом."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError("нет DejaVuSans.ttf — PDF не собрать")
    pdfmetrics.registerFont(TTFont("DV", FONT_PATH))
    return canvas.Canvas(path, pagesize=A4), A4


def make_pdf():
    path = os.path.join(OUT, "тестовый_документ.pdf")
    c, _ = _pdf_canvas(path)
    c.setFont("DV", 12)
    y = 800
    c.drawString(50, y, "Тестовый документ с ПДн")
    y -= 30
    c.setFont("DV", 11)
    for label, val in FIELDS:
        c.drawString(50, y, f"{label}: {val}")
        y -= 22
    c.save()
    return path


# ══════════════════════════════════════════════════════════════════════════
#  2. Много текста, без картинок
# ══════════════════════════════════════════════════════════════════════════
def big_text_paragraphs(repeats=9):
    """~12 страниц: ПДн четырёх человек, разбавленные канцеляритом."""
    out = ["Сводный реестр обращений граждан"]
    n = 1
    for r in range(repeats):
        for p in PEOPLE:
            out += person_block(p, n)
            out += FILLER[(n + r) % len(FILLER):] + FILLER[:(n + r) % len(FILLER)]
            n += 1
    return out


def make_docx_bigtext():
    from docx import Document
    doc = Document()
    doc.add_heading("Много текста без изображений", level=1)
    for para in big_text_paragraphs():
        doc.add_paragraph(para)
    path = os.path.join(OUT, "много_текста.docx")
    doc.save(path)
    return path


def _wrap(c, text, font, size, width):
    """Разбивает строку по ширине — reportlab сам переносы не делает."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words, lines, cur = text.split(), [], ""
    for w in words:
        probe = f"{cur} {w}".strip()
        if stringWidth(probe, font, size) <= width:
            cur = probe
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_pdf_bigtext():
    path = os.path.join(OUT, "много_текста.pdf")
    c, (pw, ph) = _pdf_canvas(path)
    left, right, top, bottom, size = 50, 50, 800, 60, 10.5
    y = top
    c.setFont("DV", size)
    for para in big_text_paragraphs():
        for line in _wrap(c, para, "DV", size, pw - left - right):
            if y < bottom:
                c.showPage()
                c.setFont("DV", size)
                y = top
            c.drawString(left, y, line)
            y -= size * 1.5
        y -= size * 0.6
    c.save()
    return path


# ══════════════════════════════════════════════════════════════════════════
#  3. С картинками — ПДн пикселями, текстового слоя нет
# ══════════════════════════════════════════════════════════════════════════
def make_pii_image(path, title, lines):
    """Имитация скана: белый лист, рамка, текст с ПДн — всё растром."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = 1000, 90 + 46 * len(lines)
    img = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.truetype(FONT_BOLD if os.path.exists(FONT_BOLD) else FONT_PATH, 30)
        f_body = ImageFont.truetype(FONT_PATH, 24)
    except OSError:
        f_title = f_body = ImageFont.load_default()
    d.rectangle([4, 4, w - 5, h - 5], outline="#8a8a8a", width=3)
    d.text((30, 26), title, fill="#101010", font=f_title)
    d.line([30, 70, w - 30, 70], fill="#c0c0c0", width=2)
    y = 92
    for ln in lines:
        d.text((30, y), ln, fill="#101010", font=f_body)
        y += 46
    img.save(path)
    return path


def _scan_images():
    """Две «сканированные справки» с ПДн внутри пикселей."""
    p = PEOPLE[0]
    a = make_pii_image(
        os.path.join(OUT, "_скан_справка_1.png"),
        "СПРАВКА (скан) — данные внутри изображения",
        [f"ФИО: {p['fio']}", f"Паспорт: {p['pass']}", f"ИНН: {p['inn']}",
         f"Телефон: {p['tel']}", f"E-mail: {p['mail']}"],
    )
    q = PEOPLE[1]
    b = make_pii_image(
        os.path.join(OUT, "_скан_справка_2.png"),
        "ВЫПИСКА (скан) — данные внутри изображения",
        [f"ФИО: {q['fio']}", f"СНИЛС: {q['snils']}", f"Дата рождения: {q['dr']}",
         f"Адрес: {q['addr']}", f"Карта: {q['card']}"],
    )
    return a, b


def make_docx_images():
    from docx import Document
    from docx.shared import Inches
    img1, img2 = _scan_images()
    doc = Document()
    doc.add_heading("Документ с изображениями", level=1)
    doc.add_paragraph(
        "Ниже два «скана». ПДн на них нарисованы пикселями и текстового слоя "
        "не имеют — обработчик docx их не увидит. Абзац ниже, наоборот, "
        "обычный текст и должен быть обезличен."
    )
    doc.add_paragraph(
        f"Текстовая часть: обращение подал {PEOPLE[2]['fio']}, "
        f"ИНН {PEOPLE[2]['inn']}, телефон {PEOPLE[2]['tel']}, "
        f"почта {PEOPLE[2]['mail']}, адрес {PEOPLE[2]['addr']}."
    )
    doc.add_picture(img1, width=Inches(6.0))
    doc.add_paragraph("Между изображениями — ещё текст, он обезличиванию подлежит:")
    doc.add_paragraph(
        f"Второй заявитель: {PEOPLE[3]['fio']}, паспорт {PEOPLE[3]['pass']}, "
        f"СНИЛС {PEOPLE[3]['snils']}, карта {PEOPLE[3]['card']}."
    )
    doc.add_picture(img2, width=Inches(6.0))
    path = os.path.join(OUT, "с_картинками.docx")
    doc.save(path)
    return path


def make_pdf_images():
    from reportlab.lib.utils import ImageReader
    img1, img2 = _scan_images()
    path = os.path.join(OUT, "с_картинками.pdf")
    c, (pw, ph) = _pdf_canvas(path)
    c.setFont("DV", 14)
    c.drawString(50, 800, "Документ с изображениями")
    c.setFont("DV", 10.5)
    y = 775
    for line in [
        "ПДн на «сканах» ниже нарисованы пикселями — текстового слоя у них нет,",
        "поэтому обработчик их не найдёт. Текст в этом абзаце обезличен быть должен:",
        f"Заявитель {PEOPLE[2]['fio']}, ИНН {PEOPLE[2]['inn']}, "
        f"телефон {PEOPLE[2]['tel']}.",
        f"Почта {PEOPLE[2]['mail']}, адрес {PEOPLE[2]['addr']}.",
    ]:
        c.drawString(50, y, line)
        y -= 18
    for img in (img1, img2):
        ir = ImageReader(img)
        iw, ih = ir.getSize()
        draw_w = pw - 100
        draw_h = draw_w * ih / iw
        y -= draw_h + 16
        if y < 60:
            c.showPage()
            c.setFont("DV", 10.5)
            y = ph - 60 - draw_h
        c.drawImage(ir, 50, y, width=draw_w, height=draw_h)
    c.save()
    return path


if __name__ == "__main__":
    targets = [
        ("базовый docx", make_docx),
        ("базовый pdf", make_pdf),
        ("много текста docx", make_docx_bigtext),
        ("много текста pdf", make_pdf_bigtext),
        ("с картинками docx", make_docx_images),
        ("с картинками pdf", make_pdf_images),
    ]
    for name, fn in targets:
        try:
            p = fn()
            kb = os.path.getsize(p) / 1024
            print(f"{name:22}: {os.path.basename(p)}  ({kb:.0f} КБ)")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22}: пропущен — {e}")
