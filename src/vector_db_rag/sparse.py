from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[^\W_]+(?:[-_.][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def token_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % 2_147_483_647


@dataclass(frozen=True)
class SparseVectorData:
    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        return {"indices": self.indices, "values": self.values}


@dataclass
class BM25SparseEncoder:
    doc_count: int
    avg_doc_len: float
    document_frequencies: dict[str, int]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def fit(cls, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> "BM25SparseEncoder":
        if not documents:
            raise ValueError("Cannot fit BM25SparseEncoder on an empty document list.")

        document_frequencies: Counter[str] = Counter()
        doc_lengths: list[int] = []
        for document in documents:
            tokens = tokenize(document)
            doc_lengths.append(len(tokens))
            document_frequencies.update(set(tokens))

        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        return cls(
            doc_count=len(documents),
            avg_doc_len=avg_doc_len,
            document_frequencies=dict(document_frequencies),
            k1=k1,
            b=b,
        )

    def encode_document(self, text: str) -> SparseVectorData:
        tokens = tokenize(text)
        return self._encode(tokens, document_length=len(tokens), apply_bm25_tf=True)

    def encode_query(self, text: str) -> SparseVectorData:
        tokens = tokenize(text)
        return self._encode(tokens, document_length=len(tokens), apply_bm25_tf=False)

    def _encode(
        self,
        tokens: list[str],
        *,
        document_length: int,
        apply_bm25_tf: bool,
    ) -> SparseVectorData:
        if not tokens:
            return SparseVectorData(indices=[], values=[])

        term_counts = Counter(tokens)
        sparse_values: dict[int, float] = {}
        for token, tf in term_counts.items():
            if apply_bm25_tf:
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * document_length / max(self.avg_doc_len, 1.0)
                )
                value = (tf * (self.k1 + 1)) / denominator
            else:
                value = tf
            index = token_to_index(token)
            sparse_values[index] = sparse_values.get(index, 0.0) + float(value)

        sorted_items = sorted(sparse_values.items())
        return SparseVectorData(
            indices=[index for index, _ in sorted_items],
            values=[value for _, value in sorted_items],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "local_bm25",
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "document_frequencies": self.document_frequencies,
            "k1": self.k1,
            "b": self.b,
            "tokenizer": TOKEN_RE.pattern,
            "token_index": "blake2b-8-mod-2147483647",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25SparseEncoder":
        return cls(
            doc_count=int(data["doc_count"]),
            avg_doc_len=float(data["avg_doc_len"]),
            document_frequencies={str(k): int(v) for k, v in data["document_frequencies"].items()},
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )


def save_sparse_encoder(path: str | Path, encoder: BM25SparseEncoder) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(encoder.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_sparse_encoder(path: str | Path) -> BM25SparseEncoder:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return BM25SparseEncoder.from_dict(data)
