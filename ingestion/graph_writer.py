from ingestion.models import ChunkRecord, DocumentRecord, EntityRecord, RelationshipRecord
from neo4j.connection import Neo4jClient
from neo4j.queries import UPSERT_CHUNK, UPSERT_DOCUMENT, UPSERT_ENTITY, UPSERT_RELATIONSHIP


class GraphWriter:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def write_document(self, document: DocumentRecord, chunks: list[ChunkRecord], entities: list[EntityRecord], relationships: list[RelationshipRecord]) -> None:
        self.client.execute(UPSERT_DOCUMENT, {"document_id": document.document_id, "filename": document.filename, "source": document.source, "file_type": document.file_type, "content_hash": document.content_hash, "ingestion_timestamp": document.ingestion_timestamp.isoformat()})
        for chunk in chunks:
            self.client.execute(UPSERT_CHUNK, {"chunk_id": chunk.chunk_id, "document_id": chunk.document_id, "filename": chunk.filename, "source": chunk.source, "chunk_index": chunk.chunk_index, "text": chunk.text, "embedding": chunk.embedding, "page": chunk.page})
        for entity in entities:
            self.client.execute(UPSERT_ENTITY, entity.__dict__)
        for relationship in relationships:
            self.client.execute(UPSERT_RELATIONSHIP, relationship.__dict__)
