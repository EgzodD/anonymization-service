#!/usr/bin/env python3
"""Снимок замеров качества для текущей версии модуля: benchmarks/<версия>/.

Зачем. Качество зависит от трёх вещей — кода, модели ФИО и набора данных.
Если хранить одни графики, потом не отличить «стало лучше от правки правил» от
«стало лучше, потому что поменялся набор». Снимок кладёт рядом цифры и то,
чем они получены: коммит, имя модели, хеши наборов, версии библиотек.

Что делает:
  * читает готовые результаты прогона из data/eval_v2/results/;
  * сводит их в компактный metrics.json (сравнение двух версий = diff двух файлов);
  * пишет meta.json с происхождением цифр;
  * копирует подробные результаты рядом, как есть.

Прогон сам НЕ запускает: сначала data/eval_v2/run_eval.py и compare_*.py,
потом этот скрипт. Так и задумано — прогон на шести наборах идёт часами,
а снимок должен быть мгновенным и повторяемым.

Использование:
    python scripts/snapshot_benchmark.py              # версия из app/__init__.py
    python scripts/snapshot_benchmark.py --version 1.1.0
    python scripts/snapshot_benchmark.py --our-system S5-r8   # id нашей системы в прогоне
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402

RESULTS = ROOT / "data" / "eval_v2" / "results"
EVAL = ROOT / "data" / "eval_v2"
BENCHMARKS = ROOT / "benchmarks"

# Человеческие названия наборов — в сыром прогоне они сокращены до ключей.
DATASET_TITLES = {
    "v2": "наш test_v2 (приёмочный, 480)",
    "dev_v2": "наш dev_v2 (настройка правил)",
    "dev_rmr": "dev_rmr (настройка правил)",
    "hive": "hivetrace/pii-bench",
    "hive_domain": "hivetrace/pii-bench, срез hive_domain",
    "alro": "alrosait/pii-synthetic-ru",
    "mcr": "MultiCoNER ru",
    "bench": "redmadrobot/pii_benchmark",
    "ext": "factRuEval-2016",
}

# Подробные результаты прогона — кладём в снимок как есть, в подпапку raw/.
# Отдельная папка не для красоты: в прогоне тоже есть metrics.json, и при копировании
# в корень снимка он затирал сводку (поймано на первом же запуске).
DETAIL_FILES = ["competitors_paired.json", "all_types_paired.json", "metrics.json"]

METRIC_KEYS = ("precision", "recall", "f1", "f1_overlap", "full_masked", "false_person_rate")


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def sha256_short(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def lib_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "n/a"


def person_model_label() -> str | None:
    """Имя модели ФИО. Берём через модуль сервиса, чтобы метка совпадала с /health."""
    try:
        from app.person_transformer_recognizer import person_model_name

        return person_model_name()
    except Exception:
        return None


def build_metrics(paired: dict, our_system: str) -> dict:
    """Сводка по наборам: наша система плюс те, с кем сравнивались."""
    out: dict[str, dict] = {}
    latency = paired.get("latency_median_ms", {})
    for name, block in paired.items():
        if not isinstance(block, dict) or "systems" not in block:
            continue
        systems = block["systems"]
        if our_system not in systems:
            # Набор, на котором наша система не прогонялась, в сводку не попадает:
            # пустая строка в таблице читается как «ноль», а это неправда.
            continue
        entry = {
            "title": DATASET_TITLES.get(name, name),
            "texts": block.get("texts"),
            "ours": {k: systems[our_system][k] for k in METRIC_KEYS if k in systems[our_system]},
            "others": {
                sid: {k: m[k] for k in METRIC_KEYS if k in m}
                for sid, m in systems.items()
                if sid != our_system
            },
        }
        ci = block.get("systems_ci95", {}).get(our_system)
        if ci:
            entry["ours_ci95"] = ci
        ms = latency.get(name, {}).get(our_system)
        if ms is not None:
            entry["ours_latency_ms_median"] = ms
        out[name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=__version__, help="номер версии (по умолчанию из кода)")
    ap.add_argument(
        "--our-system",
        default="S5-r7",
        help="id нашей системы в результатах прогона (по умолчанию S5-r7)",
    )
    ap.add_argument("--force", action="store_true", help="перезаписать существующий снимок")
    args = ap.parse_args()

    paired_path = RESULTS / "competitors_paired.json"
    if not paired_path.is_file():
        print(f"нет {paired_path.relative_to(ROOT)} — сначала прогон (data/eval_v2/run_eval.py)")
        return 1

    target = BENCHMARKS / f"v{args.version}"
    if target.exists() and not args.force:
        print(f"{target.relative_to(ROOT)} уже есть — перезапись только с --force")
        return 1
    target.mkdir(parents=True, exist_ok=True)

    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    metrics = build_metrics(paired, args.our_system)
    if not metrics:
        print(f"система {args.our_system!r} не найдена в результатах; есть: "
              f"{sorted({s for b in paired.values() if isinstance(b, dict) for s in b.get('systems', {})})}")
        return 1

    (target / "metrics.json").write_text(
        json.dumps(
            {"version": args.version, "our_system": args.our_system, "datasets": metrics},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "version": args.version,
        "snapshot_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "commit_date": git("log", "-1", "--date=short", "--pretty=%ad"),
        "dirty": bool(git("status", "--porcelain")),
        "our_system": args.our_system,
        "person_model": person_model_label(),
        "datasets_sha256": {
            p.stem: sha256_short(p) for p in sorted(EVAL.glob("*.jsonl"))
        },
        "libs": {
            lib: lib_version(lib)
            for lib in ("presidio-analyzer", "spacy", "transformers", "torch", "pymorphy3")
        },
        "python": sys.version.split()[0],
        "source": "data/eval_v2/results/",
    }
    (target / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw_dir = target / "raw"
    raw_dir.mkdir(exist_ok=True)
    copied = []
    for name in DETAIL_FILES:
        src = RESULTS / name
        if src.is_file():
            shutil.copy2(src, raw_dir / name)
            copied.append(f"raw/{name}")

    print(f"снимок: {target.relative_to(ROOT)}")
    print(f"  наборов в сводке: {len(metrics)} ({', '.join(metrics)})")
    print(f"  коммит: {meta['commit']}{' (есть незакоммиченные правки)' if meta['dirty'] else ''}")
    print(f"  модель ФИО: {meta['person_model'] or 'не подключена'}")
    print(f"  подробности скопированы: {', '.join(copied) or 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
