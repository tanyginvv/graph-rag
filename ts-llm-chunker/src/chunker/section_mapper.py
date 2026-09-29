"""Маппинг заголовков секций на онтологические labels."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger
from src.common.schemas import SectionLabel

log = get_logger(__name__)

# Общий префикс для номера секции: арабские "3." "3.1." или римские "I." "III." "VIII."
_NUM = r"(?:\d+(?:\.\d+)*\.?|[IVX]+\.)"

# Паттерны (regex, label) — проверяются по порядку, первое совпадение побеждает.
# Матчим по тексту заголовка БЕЗ `#`-префикса.
SECTION_PATTERNS: list[tuple[re.Pattern[str], SectionLabel]] = [
    (re.compile(r"^Abstract", re.IGNORECASE), "Abstract"),
    (re.compile(rf"^{_NUM}\s*Introduction", re.IGNORECASE), "Introduction"),
    (re.compile(rf"^{_NUM}\s*Related\s+Work", re.IGNORECASE), "RelatedWork"),
    (re.compile(rf"^{_NUM}\s*Preliminar", re.IGNORECASE), "Preliminaries"),
    (re.compile(rf"^{_NUM}\s*Methodolog", re.IGNORECASE), "Methodology"),
    (re.compile(rf"^{_NUM}\s*(?:Proposed\s+)?(?:Method|Approach|Framework|Model)", re.IGNORECASE), "Methodology"),
    (re.compile(rf"^{_NUM}\s*Experiment", re.IGNORECASE), "Experiments"),
    (re.compile(rf"^{_NUM}\s*Results?\b", re.IGNORECASE), "Results"),
    (re.compile(rf"^{_NUM}\s*Discussion", re.IGNORECASE), "Discussion"),
    (re.compile(rf"^{_NUM}\s*Conclusion", re.IGNORECASE), "Conclusion"),
    (re.compile(rf"^{_NUM}\s*(?:Visual\s+Analy|Ablation)", re.IGNORECASE), "Experiments"),
    # Appendix
    (re.compile(r"^[A-Z]\.?\s*(?:Implementation|Experimental\s+(?:Detail|Setup))", re.IGNORECASE), "Appendix_Implementation"),
    (re.compile(r"^[A-Z]\.\d+\.?\s*Implementation", re.IGNORECASE), "Appendix_Implementation"),
    (re.compile(r"^[A-Z]\.\d+\.?\s*Baseline", re.IGNORECASE), "Appendix_Baselines"),
    (re.compile(r"^[A-Z]\.\d+\.?\s*.*Dataset", re.IGNORECASE), "Appendix_Datasets"),
    (re.compile(r"^[A-Z]\.\d+\.?\s*.*Metric", re.IGNORECASE), "Appendix_Metrics"),
    (re.compile(r"^[B-F]\.?\s+", re.IGNORECASE), "Appendix_Results"),
    # Catch-all для Appendix
    (re.compile(r"^[A-F]\.\d+", re.IGNORECASE), "Appendix_Other"),
]


def _strip_heading_prefix(heading_content: str) -> str:
    """Убирает MD-маркеры заголовка: '## 3. Methodology' -> '3. Methodology'."""
    return re.sub(r"^#{1,6}\s+", "", heading_content).strip()


def classify_heading(heading_content: str) -> SectionLabel:
    """Определяет онтологический label для заголовка.

    Args:
        heading_content: полный текст строки заголовка (может содержать `#`-префикс).

    Returns:
        SectionLabel. Если ни один паттерн не подошёл — "Unknown".
    """
    text = _strip_heading_prefix(heading_content)

    for pattern, label in SECTION_PATTERNS:
        if pattern.match(text):
            return label

    log.warning("Unknown section heading: '%s'", text)
    return "Unknown"
