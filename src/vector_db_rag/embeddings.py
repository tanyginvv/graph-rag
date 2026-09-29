from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .constants import DENSE_DIM, DENSE_MODEL


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = DENSE_MODEL,
        expected_dim: int = DENSE_DIM,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.expected_dim = expected_dim
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "openai is not installed. Install project dependencies first: pip install -e ."
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise EmbeddingError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            text = text.strip()
            if not text:
                raise EmbeddingError("Cannot embed an empty text.")
            cached = self._read_cache(text)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(text)
            else:
                vectors[index] = cached

        if missing_texts:
            new_vectors = self._request_embeddings(missing_texts)
            for index, text, vector in zip(missing_indices, missing_texts, new_vectors, strict=True):
                self._validate_dim(vector)
                self._write_cache(text, vector)
                vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise EmbeddingError("Internal embedding error: missing vector in batch result.")
        return [vector for vector in vectors if vector is not None]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts)
                vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
                for vector in vectors:
                    self._validate_dim(vector)
                return vectors
            except Exception as exc:  # OpenAI SDK exception hierarchy varies across versions.
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(2 ** (attempt - 1))
        raise EmbeddingError(f"Failed to create embeddings: {last_error}") from last_error

    def _validate_dim(self, vector: list[float]) -> None:
        if len(vector) != self.expected_dim:
            raise EmbeddingError(
                f"Unexpected embedding dimension: got {len(vector)}, expected {self.expected_dim}"
            )

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(f"{self.model}\n{self.expected_dim}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        vector = data.get("embedding")
        if not isinstance(vector, list):
            return None
        self._validate_dim(vector)
        return [float(value) for value in vector]

    def _write_cache(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "model": self.model,
                    "dim": self.expected_dim,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "embedding": vector,
                },
                file,
            )
