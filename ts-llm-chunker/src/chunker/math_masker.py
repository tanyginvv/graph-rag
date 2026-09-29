"""Маскирование и восстановление LaTeX-формул для корректного токенизирования и сплиттинга."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger

log = get_logger(__name__)

# Display math: $$...$$ (greedy внутри одной строки, и multiline)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# Inline math: $...$ (не жадный, в пределах одной строки)
# Исключаем ложные срабатывания: $100 (число после $) и одиночные $
_INLINE_MATH = re.compile(
    r"(?<!\$)"      # не перед другим $
    r"\$"
    r"(?!\$)"       # не $$
    r"(?!\d)"       # не $100
    r"(.+?)"
    r"(?<!\$)"
    r"\$"
    r"(?!\$)"
)


def mask_formulas(text: str) -> tuple[str, dict[str, str]]:
    """Заменяет LaTeX-формулы на плейсхолдеры.

    Display math: $$...$$ -> [DISPLAY_MATH_NNN]
    Inline math:  $...$  -> [MATH_NNN]

    Returns:
        (masked_text, mapping) — mapping позволяет восстановить формулы через unmask.
    """
    mapping: dict[str, str] = {}
    counter = {"display": 0, "inline": 0}

    def _replace_display(m: re.Match[str]) -> str:
        idx = counter["display"]
        counter["display"] += 1
        placeholder = f"[DISPLAY_MATH_{idx:03d}]"
        mapping[placeholder] = m.group(0)
        return placeholder

    def _replace_inline(m: re.Match[str]) -> str:
        idx = counter["inline"]
        counter["inline"] += 1
        placeholder = f"[MATH_{idx:03d}]"
        mapping[placeholder] = m.group(0)
        return placeholder

    # Сначала display ($$...$$), потом inline ($...$) — порядок важен
    text = _DISPLAY_MATH.sub(_replace_display, text)
    text = _INLINE_MATH.sub(_replace_inline, text)

    log.info(
        "Masked %d display + %d inline formula(s)",
        counter["display"],
        counter["inline"],
    )
    return text, mapping


def unmask_formulas(text: str, mapping: dict[str, str]) -> str:
    """Восстанавливает формулы из плейсхолдеров."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text
