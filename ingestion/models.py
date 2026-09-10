from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    source: str
    file_type: str
    content_hash: str
    text: str
    ingestion_timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    page: int | None = None


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    source_chunk_id: str


@dataclass(frozen=True)
class RelationshipRecord:
    source_entity_id: str
    relationship: str
    target_entity_id: str
    source_chunk_id: str
    confidence: float = 1.0


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    status: str = "failed"
    chunks_created: int = 0
    embeddings_created: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    duplicate_detected: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    validation: dict[str, Any] = field(default_factory=dict)
