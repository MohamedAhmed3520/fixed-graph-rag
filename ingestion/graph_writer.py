from ingestion.models import (
    ChunkRecord,
    DocumentRecord,
    EntityRecord,
    RelationshipRecord,
)

from neo4j.connection import Neo4jClient

from neo4j.queries import (
    UPSERT_CHUNK,
    UPSERT_DOCUMENT,
    UPSERT_ENTITY,
    UPSERT_RELATIONSHIP,
)


class GraphWriter:
    """Write one complete document graph in a single Neo4j transaction."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    def write_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        entities: list[EntityRecord],
        relationships: list[RelationshipRecord],
    ) -> None:
        """Persist the complete document graph atomically.

        If any statement fails, Neo4j rolls back the entire transaction.
        """

        def write_transaction(tx) -> None:

            # -------------------------------------------------
            # Document
            # -------------------------------------------------

            tx.run(
                UPSERT_DOCUMENT,
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "source": document.source,
                    "file_type": document.file_type,
                    "content_hash": document.content_hash,
                    "ingestion_timestamp": (
                        document.ingestion_timestamp.isoformat()
                    ),
                },
            ).consume()

            # -------------------------------------------------
            # Chunks
            # -------------------------------------------------

            for chunk in chunks:
                tx.run(
                    UPSERT_CHUNK,
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "embedding": chunk.embedding,
                        "page": chunk.page,
                    },
                ).consume()

            # -------------------------------------------------
            # Entities
            # -------------------------------------------------

            for entity in entities:
                tx.run(
                    UPSERT_ENTITY,
                    entity.__dict__,
                ).consume()

            # -------------------------------------------------
            # Relationships
            # -------------------------------------------------

            for relationship in relationships:
                tx.run(
                    UPSERT_RELATIONSHIP,
                    relationship.__dict__,
                ).consume()

        self.client.execute_write_transaction(
            write_transaction
        )
