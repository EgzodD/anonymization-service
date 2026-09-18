"""
Отдельный веб-стенд для ручной проверки обезличивания (НЕ часть прод-кода app/).

Переиспользует ядро модуля (app.anonymizer и обработчики документов), но живёт
в отдельной папке и запускается сам по себе. Auth нет — это локальный тест.

Запуск (из корня репозитория, python ИЗ venv — в системном нет зависимостей):
    .venv/bin/python tests/web-тест/server.py
затем открыть http://localhost:8080 в браузере.

Сравнение «было / стало» (блок 3 страницы): рядом с текущим кодом стенд поднимает
СТАРУЮ версию сервиса отдельным процессом — код до плана улучшений (коммит
WEBTEST_OLD_REF, по умолчанию ca69325 от 16.09.2026) с прежней моделью ФИО
(models/person_ruBERT_prev_20260918). Старый код выкладывается через
`git worktree` в .cache/web-test-old (в git не попадает), процесс слушает
127.0.0.1:8081 и останавливается вместе со стендом. Первый запуск — 20–40 секунд.
"""
import atexit
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

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


# ── старая версия для сравнения «было / стало» ─────────────────────────────
OLD_REF = os.environ.get("WEBTEST_OLD_REF", "ca69325")
OLD_MODEL = os.environ.get("WEBTEST_OLD_MODEL",
                           os.path.join(ROOT, "models", "person_ruBERT_prev_20260918"))
OLD_PORT = int(os.environ.get("WEBTEST_OLD_PORT", "8081"))
OLD_DIR = os.path.join(ROOT, ".cache", "web-test-old")
OLD_URL = f"http://127.0.0.1:{OLD_PORT}"
_old = {"proc": None, "state": "не запущена", "error": ""}


def _start_old():
    """Выкладывает старый код и запускает его процессом uvicorn (в фоне)."""
    try:
        if not os.path.isdir(OLD_DIR):
            _old["state"] = "выкладываю старый код…"
            subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", OLD_DIR, OLD_REF],
                           check=True, capture_output=True, text=True)
        if not os.path.isfile(os.path.join(OLD_MODEL, "config.json")):
            raise RuntimeError(f"нет прежней модели: {OLD_MODEL}")
        env = {**os.environ, "PERSON_MODEL_DIR": OLD_MODEL, "API_KEY": "",
               "ALLOW_NO_PERSON_MODEL": "false", "CHATWOOT_ENABLED": "false"}
        _old["state"] = "запускаю…"
        _old["proc"] = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(OLD_PORT)], cwd=OLD_DIR, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if _old["proc"].poll() is not None:
                raise RuntimeError("процесс старой версии завершился при запуске")
            try:
                urllib.request.urlopen(OLD_URL + "/health", timeout=2)
                _old["state"] = "готова"
                return
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        raise RuntimeError("старая версия не ответила за 2 минуты")
    except Exception as exc:  # noqa: BLE001 — стенд работает и без сравнения
        _old["state"], _old["error"] = "ошибка", str(exc)


@atexit.register
def _stop_old():
    if _old["proc"] is not None and _old["proc"].poll() is None:
        _old["proc"].terminate()


def _entities(text, found):
    return [{"type": e["entity_type"], "start": e["start"], "end": e["end"],
             "value": text[e["start"]:e["end"]]} for e in found]


def _diff(old, new):
    """Сопоставляет маски двух версий по позициям.

    Оценку «хорошо/плохо» стенд не ставит: снятая маска может быть исправленным
    ложным срабатыванием («гражданине»), а может — новой утечкой. Это видно по значению.
    """
    def ov(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]
    out = []
    for o in old:
        same = [n for n in new if n["start"] == o["start"] and n["end"] == o["end"]]
        if same:
            if same[0]["type"] != o["type"]:
                out.append({"kind": "тип изменён", "old": o, "new": same[0]})
            continue
        near = [n for n in new if ov(n, o)]
        if near:
            out.append({"kind": "граница изменена", "old": o, "new": near[0]})
        else:
            out.append({"kind": "маска снята", "old": o, "new": None})
    for n in new:
        if not any(ov(n, o) for o in old):
            out.append({"kind": "добавлена маска", "old": None, "new": n})
    return out


@app.get("/compare/status")
def compare_status():
    return {"state": _old["state"], "error": _old["error"], "old_ref": OLD_REF,
            "old_model": os.path.basename(OLD_MODEL)}


class CompareIn(BaseModel):
    text: str


@app.post("/compare")
def compare(body: CompareIn):
    if _old["state"] != "готова":
        raise HTTPException(503, f"Старая версия: {_old['state']} {_old['error']}".strip())
    t0 = time.perf_counter()
    new = anonymize_text(body.text)
    ms_new = (time.perf_counter() - t0) * 1000
    req = urllib.request.Request(OLD_URL + "/anonymize/text", method="POST",
                                 data=json.dumps({"text": body.text}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        old = json.loads(r.read().decode("utf-8"))
    ms_old = (time.perf_counter() - t0) * 1000
    eo, en = _entities(body.text, old["entities_found"]), _entities(body.text, new["entities_found"])
    return {"old": {"anonymized": old["anonymized"], "entities": eo, "ms": round(ms_old, 1)},
            "new": {"anonymized": new["anonymized"], "entities": en, "ms": round(ms_new, 1)},
            "diff": _diff(eo, en)}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/policy")
def policy():
    """
    Типы ПДн, которые разрешено отключать кастомным параметром disable_entities.

    `core` — типы с собственным плейсхолдером (<PERSON>, <INN>, ...). Остальное в
    `entities` приходит из встроенного набора presidio: часть из них срабатывает
    (DATE_TIME, IP_ADDRESS, URL, IBAN_CODE, CRYPTO) и маскируется общим <PII>,
    часть рассчитана на иностранные форматы (IN_VOTER, MEDICAL_LICENSE). Стенд
    показывает их отдельно, чтобы основные типы не терялись в списке.
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

    threading.Thread(target=_start_old, daemon=True).start()
    print("Веб-тест: http://localhost:8080  (Ctrl+C — стоп)")
    print(f"Сравнение: старая версия {OLD_REF} поднимается на {OLD_URL} (20–40 с)")
    uvicorn.run(app, host="127.0.0.1", port=8080)
