"""Версия модуля: один номер во всех местах, где он объявлен.

Категория: модульные. Запуск: pytest tests/test_version.py

Зачем тест. Номер версии физически лежит в трёх местах: app/__init__.py
(источник), pyproject.toml (pyproject не умеет читать его из кода без системы
сборки) и CHANGELOG.md (журнал). Разойтись они могут молча — и тогда по
работающему контейнеру нельзя понять, что в нём запущено. Этот тест ловит
расхождение на проверках, а не на поставке.
"""

import re
import tomllib
from pathlib import Path

import pytest

from app import __version__

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), (
        f"версия {__version__!r} не по SemVer MAJOR.MINOR.PATCH"
    )


def test_pyproject_version_matches_code():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__, (
        "pyproject.toml и app/__init__.py разошлись — поправить оба"
    )


def test_changelog_has_section_for_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog, (
        f"в CHANGELOG.md нет раздела ## [{__version__}] — описать выпуск перед тегом"
    )


def test_health_reports_version(client):
    data = client.get("/health").json()
    assert data["version"] == __version__
    # person_model присутствует всегда: null = модель ФИО не подключена
    # (так работает контур CI с ALLOW_NO_PERSON_MODEL=true).
    assert "person_model" in data
