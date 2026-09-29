from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


def point_id_from_chunk_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"vector-db-rag/chunk/{chunk_id}"))

