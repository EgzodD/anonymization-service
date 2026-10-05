#!/usr/bin/env bash
# Локальный прогон проверок — то же, что делает CI, но без CI-сервера.
#
# Зачем нужен: тесты — это обычный Python (pytest), они лежат в tests/ и не
# требуют никакого CI. CI — лишь «будильник», который запускает их сам на каждый
# push. На рабочем gitea раннера Actions нет, поэтому там будильника не будет —
# а проверки нужны. Этот скрипт и есть замена: один вход для человека и для
# git-хука.
#
# Использование:
#   ./scripts/check.sh
#
# Чтобы гонялось само перед каждым push (CI на своей машине, без сервера):
#   ln -s ../../scripts/check.sh .git/hooks/pre-push
set -euo pipefail

cd "$(dirname "$0")/.."

# Предпочитаем venv проекта, иначе системный python
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi

echo "▶ Линтер (статический анализ, ruff)"
if [ -x .venv/bin/ruff ]; then
    .venv/bin/ruff check app/ tests/ synthesizer/
else
    "$PY" -m ruff check app/ tests/ synthesizer/
fi

echo
echo "▶ Тесты (модель-независимый контур)"
if "$PY" -c "import pytest_cov" 2>/dev/null; then
    "$PY" -m pytest -m "not requires_model" --cov=app --cov-report=term-missing
else
    echo "  (покрытие пропущено: нет pytest-cov — поставить: pip install pytest-cov)"
    "$PY" -m pytest -m "not requires_model"
fi

echo
# Приватностный гейт гоняется только если дообученная PERSON-модель есть локально.
# Без неё активна стоковая Natasha, которая даёт утечки — гейт бы падал не по делу.
if "$PY" -c "from app.person_transformer_recognizer import person_model_available
raise SystemExit(0 if person_model_available() else 1)" 2>/dev/null; then
    echo "▶ Приватностный гейт LeakRate — 0 утечек ПДн (PERSON-модель найдена)"
    "$PY" -m pytest -m requires_model -q
else
    echo "⊘ Приватностный гейт LeakRate пропущен: нет дообученной PERSON-модели."
    echo "  Задать PERSON_MODEL_DIR или загрузить:"
    echo "    PERSON_MODEL_URL=<url> ./scripts/fetch_person_model.sh"
fi

echo
echo "✔ Все проверки пройдены"
