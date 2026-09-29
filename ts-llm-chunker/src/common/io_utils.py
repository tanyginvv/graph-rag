"""IO-хелперы: чтение/запись MD и JSON, загрузка YAML-конфига."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_hash(cfg: dict[str, Any]) -> str:
    """Стабильный SHA1-хэш конфига для versioning чанков."""
    canonical = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def iter_md_files(directory: str | Path) -> list[Path]:
    """Возвращает отсортированный список .md файлов в директории (без рекурсии)."""
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix == ".md")
