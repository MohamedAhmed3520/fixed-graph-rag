from langchain_core.documents import Document
from ingestion.pipeline import IngestionPipeline

from ingestion.chunker import split_documents
from ingestion.normalizer import normalize_text


def test_normalization_preserves_paragraphs():
    assert normalize_text(" A\r\n\r\n\r\nB  \n") == "A\n\nB"


def test_chunking_is_deterministic_and_traceable():
    document = Document(page_content="one two three four five", metadata={"document_id": "doc-1", "filename": "notes.txt", "source": "notes.txt"})
    first = split_documents([document], chunk_size=100, chunk_overlap=10)
    second = split_documents([document], chunk_size=100, chunk_overlap=10)
    assert first == second
    assert all(chunk.metadata["document_id"] == "doc-1" for chunk in first)
    assert [chunk.metadata["chunk_index"] for chunk in first] == [0]


def test_loader_failure_returns_structured_result(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"pdf bytes")
    monkeypatch.setattr("ingestion.pipeline.load_document", lambda _: (_ for _ in ()).throw(ValueError("No text could be extracted from: scan.pdf")))

    result = IngestionPipeline().ingest_path(pdf_path)

    assert result.status == "failed"
    assert result.errors == ["No text could be extracted from: scan.pdf"]
    assert "OCR is required" in result.warnings[0]
