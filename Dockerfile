FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Дообученная модель PERSON (ruBERT) — transformers + CPU-torch.
# Ставим CPU-сборку torch (лёгкая ~190МБ) вместо дефолтной CUDA-сборки (~2ГБ).
# Версии закреплены под проверенное локальное окружение: без пина pip берёт
# свежий мажор transformers, и образ перестаёт совпадать с тем, что протестировано.
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    "transformers==5.12.1" "torch==2.12.1" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN python -m spacy download ru_core_news_lg --timeout=300

RUN adduser --disabled-password --gecos "" appuser

# В рантайме нужны только код и модель PERSON — копируем их точечно, а не `COPY . .`:
# тесты, обучающие данные и служебные папки в поставляемый образ не попадают.
# Если приложению понадобится новый файл вне app/, его нужно добавить сюда явно.
COPY app/ ./app/

# Модель — с владельцем appuser. На хосте model.safetensors бывает с правами 600,
# COPY их сохраняет, и сервис, запущенный от appuser, не смог бы прочитать веса.
COPY --chown=appuser:appuser models/person_ruBERT/ ./models/person_ruBERT/

EXPOSE 8000

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
