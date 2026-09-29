from __future__ import annotations

from typing import Any


class LLMError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        base_url: str,
        model: str,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        timeout_seconds: int = 120,
    ) -> None:
        if not api_key:
            raise LLMError(
                "Не задан API key для LLM. Установите DEEPSEEK_API_KEY "
                "(или переменную из llm.api_key_env), либо запустите команду с --dry-run, "
                "чтобы проверить grounded context без вызова модели."
            )
        self.base_url = base_url
        self.model = model

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai is not installed. Install project dependencies first: pip install -e ."
            ) from exc

        self.provider = provider
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                **build_chat_request_kwargs(
                    provider=self.provider,
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_enabled=self.thinking_enabled,
                    reasoning_effort=self.reasoning_effort,
                )
            )
        except Exception as exc:
            raise LLMError(f"Failed to generate answer: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("LLM response does not contain choices.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("LLM response does not contain a message.")
        content = getattr(message, "content", None)
        text = _coerce_message_content(content)
        if not text:
            finish_reason = str(getattr(choices[0], "finish_reason", "") or "")
            reasoning_content = str(getattr(message, "reasoning_content", "") or "").strip()
            if reasoning_content and finish_reason == "length":
                raise LLMError(
                    "Модель исчерпала max_tokens на reasoning до выдачи финального ответа. "
                    "Увеличьте --max-tokens, уменьшите контекст или отключите thinking mode."
                )
            raise LLMError("LLM response is empty.")
        return text


def build_chat_request_kwargs(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_enabled: bool,
    reasoning_effort: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "deepseek":
        request["extra_body"] = {
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled",
            }
        }
        if thinking_enabled:
            request["reasoning_effort"] = reasoning_effort
    return request


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            else:
                text = str(getattr(item, "text", "") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""
