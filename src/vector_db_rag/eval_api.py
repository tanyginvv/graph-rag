from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EVAL_API_BASE_URL = "http://89.169.33.54:8000"
DEFAULT_EVAL_QUESTIONS_PATH = Path("eval-api/docs/questions_raw.json")


class EvalApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    article_id: str
    article_title: str


@dataclass(frozen=True)
class EvalQuestionSet:
    version: str
    items: list[EvalQuestion]


def load_eval_question_set(path: str | Path = DEFAULT_EVAL_QUESTIONS_PATH) -> EvalQuestionSet:
    questions_path = Path(path)
    with questions_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    version = str(data.get("version") or "").strip()
    if not version:
        raise EvalApiError(f"Eval questions file does not contain version: {questions_path}")

    raw_items = data.get("items") or []
    items: list[EvalQuestion] = []
    for item in raw_items:
        question_id = str(item.get("id") or "").strip()
        question_text = str(item.get("question") or "").strip()
        article_id = str(item.get("article_id") or "").strip()
        article_title = str(item.get("article_title") or "").strip()
        if not question_id or not question_text or not article_id or not article_title:
            raise EvalApiError(f"Malformed eval question entry in {questions_path}: {item!r}")
        items.append(
            EvalQuestion(
                id=question_id,
                question=question_text,
                article_id=article_id,
                article_title=article_title,
            )
        )
    return EvalQuestionSet(version=version, items=items)


def detect_eval_question_type(question: str) -> str:
    normalized = question.strip().lower()
    if "завершается формирование прогноза" in normalized:
        return "М"
    if (
        ("преобразования временных рядов" in normalized or "преобразование временных рядов" in normalized)
        and ("нормировк" in normalized or "токенизаци" in normalized or "подготовки" in normalized)
        and "представления" not in normalized
    ):
        return "Л"
    if "цели преследуются" in normalized or (
        "цели" in normalized and ("предобучения" in normalized or "дообучения" in normalized)
    ):
        return "К"
    if "тип задач" in normalized and "горизонт" in normalized:
        return "Ж"
    if "шаги подготовки данных" in normalized or "подготовки данных и обучения" in normalized:
        return "И"
    if (
        "шагов состоит пайплайн" in normalized
        or "шагов состоят пайплайны" in normalized
        or "из каких шагов состоит пайплайн" in normalized
    ):
        return "Е"
    if "нейросетевую архитектуру" in normalized and "backbone" in normalized:
        return "Д"
    if "способы улучшения" in normalized or "улучшения прогноза" in normalized:
        return "Б"
    if (
        "представления временных рядов" in normalized
        or "использовать llm для представления" in normalized
        or "используется llm для представления" in normalized
        or "использовать llm-подобную" in normalized
        or "использовать предобученные" in normalized
    ):
        return "Г"
    if "метрики используются" in normalized or "метрики оценки" in normalized:
        return "З"
    if (
        ("бенчмарки" in normalized or "бенчмарках" in normalized)
        and ("используются" in normalized or "датасет" in normalized or "проводится оценка" in normalized)
    ):
        return "В"
    if (
        "считаются наилучшими" in normalized
        or "наилучшими по результатам" in normalized
        or "наилучшими согласно" in normalized
    ):
        return "А+"
    if "существуют способы использования llm" in normalized or "способ использования llm" in normalized:
        if "наилучш" in normalized or "считаются" in normalized:
            return "А+"
        return "А"
    if "демонстрирует ли подход улучшение точности" in normalized or "обобщающую способность" in normalized:
        return "Спец"
    if "достигает ли предложенный метод улучшения" in normalized:
        return "Спец"
    raise EvalApiError(f"Unsupported Eval API question type: {question}")


def select_eval_questions(
    question_set: EvalQuestionSet,
    *,
    question_ids: tuple[str, ...] = (),
    article_ids: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[EvalQuestion]:
    selected = question_set.items
    if question_ids:
        allowed_question_ids = set(question_ids)
        selected = [question for question in selected if question.id in allowed_question_ids]
    if article_ids:
        allowed_article_ids = set(article_ids)
        selected = [question for question in selected if question.article_id in allowed_article_ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_submission_payload(*, run_id: str | None, items: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"items": items}
    if run_id:
        payload["run_id"] = run_id
    return payload


def submit_eval_payload(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise EvalApiError(
            "httpx is not installed. Install project dependencies first: pip install -e ."
        ) from exc

    url = f"{base_url.rstrip('/')}/v1/eval/submit"
    response = httpx.post(
        url,
        headers={"X-API-Key": api_key},
        json=payload,
        timeout=timeout_seconds,
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        detail = response.text.strip()
        raise EvalApiError(f"Eval API submit failed: {response.status_code} {detail}") from exc

    data = response.json()
    if not isinstance(data, dict):
        raise EvalApiError("Eval API response is not a JSON object.")
    return data