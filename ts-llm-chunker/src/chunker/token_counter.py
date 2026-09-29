"""Подсчёт токенов через tiktoken."""

from __future__ import annotations

import tiktoken


def get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = get_encoder(encoding_name)
    return len(enc.encode(text))
