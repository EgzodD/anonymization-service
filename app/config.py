import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Сервис по умолчанию работает standalone, без Chatwoot. Интеграция
    # включается явно через CHATWOOT_ENABLED=true (см. app/integrations/chatwoot).
    chatwoot_enabled: bool = False

    # Пусто = БД не сконфигурирована. Обязательна только при chatwoot_enabled=true
    # (проверяется fail-fast при старте — см. app/integrations/chatwoot/database.py).
    database_url: str = ""
    api_key: str = ""                    # пустая строка = аутентификация отключена
    chatwoot_webhook_secret: str = ""    # пустая строка = проверка подписи отключена; используется только при chatwoot_enabled=true

    # Путь к дообученной ruBERT-модели PERSON. Обязателен: без модели ФИО не
    # распознаются вообще, поэтому сервис откажется стартовать (см. app/main.py).
    person_model_dir: str = ""

    # Человеческая метка модели ФИО для /health и снимков замеров (benchmarks/).
    # Пусто = берётся имя папки из PERSON_MODEL_DIR. Задавать вручную стоит, когда
    # папка называется невыразительно: цифры качества зависят от модели не меньше,
    # чем от кода, и по номеру версии её не опознать.
    person_model_name: str = ""

    # Аварийный флаг: разрешить запуск БЕЗ модели PERSON. Только для тестов и CI,
    # где модели нет в чекауте. В проде включать нельзя — ФИО будут утекать.
    allow_no_person_model: bool = False

    # Опциональный адаптер обезличивания документов (PDF/Word). По умолчанию
    # выключен — ядро остаётся лёгким и не тянет python-docx/pdfplumber/pypdfium2.
    # Включается флагом DOCUMENT_ENABLED=true (см. app/integrations/documents).
    document_enabled: bool = False
    # Лимиты защиты от DoS (гигантский файл, zip-bomb в .docx, PDF на тысячи страниц).
    document_max_bytes: int = 20 * 1024 * 1024   # 20 МБ на файл
    document_max_pdf_pages: int = 100            # предел страниц PDF
    document_pdf_dpi: int = 150                  # DPI растеризации PDF

    class Config:
        # Абсолютный путь к .env в корне проекта: относительный (".env")
        # резолвится от папки ЗАПУСКА процесса, из-за чего настройки (в т.ч.
        # PERSON_MODEL_DIR) молча терялись при запуске скриптов не из корня.
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")


settings = Settings()
