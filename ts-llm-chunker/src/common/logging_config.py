"""Единая настройка логирования для всех CLI и модулей."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Настраивает root logger. Идемпотентно — повторные вызовы просто меняют уровень."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        for h in root.handlers:
            h.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
