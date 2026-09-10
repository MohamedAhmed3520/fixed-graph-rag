import hashlib
import logging
import time
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from config.settings import get_settings
from embeddings.embedding_model import embed_documents
from ingestion.chunker import split_documents
from ingestion.extractor import extract_entities, extract_relationships
from ingestion.loaders import load_document
from ingestion.models import ChunkRecord, DocumentRecord, IngestionResult
from ingestion.normalizer import normalize_text
from ingestion.graph_writer import GraphWriter
from ingestion.validator import validate_document
from neo4j.connection import Neo4jClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, graph_writer: GraphWriter | None = None, client: Neo4jClient | None = None, llm=None, embedder=embed_documents):
        self.graph_writer = graph_writer
        self.client = client
        self.llm = llm
        self.embedder = embedder

    def ingest_path(self, file_path: str | Path, allow_duplicate: bool = False) -> IngestionResult:
        path = Path(file_path)
        started = time.perf_counter()
        try:
            return self.ingest_documents(load_document(path), allow_duplicate=allow_duplicate)
        except Exception as exc:
            result = IngestionResult(
                document_id=f"doc-{hashlib.sha256(path.read_bytes()).hexdigest()[:24]}",
                filename=path.name,
                errors=[str(exc)],
                duration_seconds=time.perf_counter() - started,
            )
            if path.suffix.lower() == ".pdf" and "No text could be extracted" in str(exc):
                result.warnings.append("This PDF has no extractable text layer. OCR is required for scanned/image-only PDFs.")
            logger.warning("Document loading failed for %s: %s", path.name, exc)
            return result

    def ingest_bytes(self, filename: str, content: bytes, allow_duplicate: bool = False) -> IngestionResult:
        safe_filename = Path(filename).name
        with tempfile.NamedTemporaryFile(prefix="graprag-", suffix=Path(safe_filename).suffix, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            result = self.ingest_path(temporary_path, allow_duplicate)
            result.filename = safe_filename
            result.errors = [error.replace(temporary_path.name, safe_filename) for error in result.errors]
            result.warnings = [warning.replace(temporary_path.name, safe_filename) for warning in result.warnings]
            return result
        finally:
            temporary_path.unlink(missing_ok=True)

    def ingest_documents(self, loaded_documents, allow_duplicate: bool = False) -> IngestionResult:
        started = time.perf_counter()
        if not loaded_documents:
            raise ValueError("No documents were loaded")
        first_metadata = loaded_documents[0].metadata
        document_id = first_metadata["document_id"]
        digest = first_metadata["content_hash"]
        result = IngestionResult(document_id, first_metadata["filename"])
        try:
            normalized_documents = [Document(page_content=normalize_text(document.page_content), metadata=dict(document.metadata)) for document in loaded_documents]
            text = "\n\n".join(document.page_content for document in normalized_documents)
            if self.client and not allow_duplicate:
                duplicate_rows = self.client.execute(
                    "MATCH (d:Document {content_hash: $content_hash}) RETURN d.document_id AS document_id LIMIT 1",
                    {"content_hash": digest},
                )
                if duplicate_rows:
                    result.status = "duplicate"
                    result.duplicate_detected = True
                    result.warnings.append(f"Document already ingested as {duplicate_rows[0]['document_id']}")
                    return result
            document = DocumentRecord(document_id, first_metadata["filename"], first_metadata["source"], first_metadata["file_type"], digest, text)
            settings = get_settings()
            chunk_documents = split_documents(normalized_documents, settings.chunk_size, settings.chunk_overlap)
            vectors = self.embedder([chunk.page_content for chunk in chunk_documents]) if chunk_documents else []
            chunks = [
                self._chunk_record(chunk, vector)
                for chunk, vector in zip(chunk_documents, vectors)
            ]
            entities = []
            relationships = []
            for chunk in chunks:
                extracted = extract_entities(chunk.text, chunk.chunk_id, self.llm)
                entities.extend(extracted)
                relationships.extend(extract_relationships(chunk.text, extracted, self.llm))
            if self.graph_writer:
                self.graph_writer.write_document(document, chunks, entities, relationships)
            result.status = "success"
            result.chunks_created = len(chunks)
            result.embeddings_created = len(vectors)
            result.entities_created = len({entity.entity_id for entity in entities})
            result.relationships_created = len(relationships)
            if self.client:
                result.validation = validate_document(self.client, document_id, len(chunks))
                if not result.validation["valid"]:
                    result.status = "failed"
                    result.errors.append("Post-ingestion validation failed")
        except Exception as exc:
            logger.exception("Ingestion failed for %s", result.filename)
            result.errors.append(str(exc))
        result.duration_seconds = time.perf_counter() - started
        return result

    @staticmethod
    def _chunk_record(chunk, embedding):
        metadata = chunk.metadata
        return ChunkRecord(
            metadata["chunk_id"], metadata["document_id"], metadata["filename"], metadata["source"], metadata["chunk_index"], chunk.page_content, embedding, metadata.get("page")
        )
