"""
Отдельный веб-стенд для ручной проверки обезличивания (НЕ часть прод-кода app/).

Переиспользует ядро модуля (app.anonymizer и обработчики документов), но живёт
в отдельной папке и запускается сам по себе. Auth нет — это локальный тест.

Запуск (из корня репозитория, python ИЗ venv — в системном нет зависимостей):
    .venv/bin/python tests/web-тест/server.py
затем открыть http://localhost:8080 в браузере.
"""
import json
import os
import sys

# Корень репозитория — ищем вверх папку, в которой лежит пакет app/.
# К глубине вложенности не привязываемся: стенд уже переезжал
# (web-тест/ -> tests/web-тест/), и жёсткий путь тогда сломался.
ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, "app")) and ROOT != os.path.dirname(ROOT):
    ROOT = os.path.dirname(ROOT)
if not os.path.isdir(os.path.join(ROOT, "app")):
    sys.exit("Не найден корень репозитория (папка app/). Запускайте стенд изнутри репозитория.")
sys.path.insert(0, ROOT)
# модель PERSON грузится из .env (PERSON_MODEL_DIR); без неё стенд не должен падать
os.environ.setdefault("ALLOW_NO_PERSON_MODEL", "true")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.anonymizer import (  # noqa: E402
    OPERATORS,
    POLICY_ENTITIES,
    anonymize_text,
    deanonymize_text,
    resolve_disabled_entities,
)

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Обезличивание — веб-тест", docs_url=None, redoc_url=None)


class TextIn(BaseModel):
    text: str
    # кастомный параметр: типы, которые НЕ надо маскировать.
    # Может только СУЖАТЬ политику — расширить её через него нельзя.
    disable_entities: list[str] | None = None


class DeIn(BaseModel):
    text: str
    mapping: dict


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/policy")
def policy():
    """
    Типы ПДн, которые разрешено отключать кастомным параметром disable_entities.

    `core` — типы с собственным плейсхолдером, то есть те, что наши распознаватели
    реально выдают. Остальное в `entities` приходит из встроенного набора presidio
    (MEDICAL_LICENSE, IN_VOTER и т.п.): параметр их примет, но на практике они
    не срабатывают. Стенд показывает их отдельно, чтобы не путать при проверке.
    """
    core = sorted((set(OPERATORS) - {"DEFAULT"}) & POLICY_ENTITIES)
    return {"entities": sorted(POLICY_ENTITIES), "core": core}


@app.post("/anonymize")
def anonymize(body: TextIn):
    # в тесте показываем и значения, и mapping (для подсветки и восстановления)
    res = anonymize_text(body.text, disable_entities=body.disable_entities)
    if body.disable_entities:
        # Диагностика кастомного параметра: что реально применилось, а что политика
        # отбросила. Недопустимые типы игнорируются молча — данные остаются скрытыми,
        # и стенд должен это показывать явно, иначе правило не проверить глазами.
        requested = {str(x).strip().upper() for x in body.disable_entities if str(x).strip()}
        applied = resolve_disabled_entities(body.disable_entities)
        res = {**res,
               "disable_applied": sorted(applied),
               "disable_ignored": sorted(requested - applied)}
    return res


@app.post("/deanonymize")
def deanonymize(body: DeIn):
    return deanonymize_text(body.text, body.mapping)


@app.post("/document")
async def document(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(422, "Пустой файл")
    name = (file.filename or "").lower()
    if name.endswith(".docx"):
        from app.integrations.documents.docx_handler import anonymize_docx
        out, summary = anonymize_docx(data)
        media, ext = ("application/vnd.openxmlformats-officedocument."
                      "wordprocessingml.document"), ".docx"
    elif name.endswith(".pdf"):
        from app.integrations.documents.pdf_handler import anonymize_pdf
        out, summary = anonymize_pdf(data, max_pages=100, dpi=150)
        media, ext = "application/pdf", ".pdf"
    else:
        raise HTTPException(415, "Поддерживаются только .docx и .pdf")
    return Response(
        content=out, media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="anon{ext}"',
            "X-Anonymization-Summary": json.dumps(summary, ensure_ascii=False),
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("Веб-тест: http://localhost:8080  (Ctrl+C — стоп)")
    uvicorn.run(app, host="127.0.0.1", port=8080)
