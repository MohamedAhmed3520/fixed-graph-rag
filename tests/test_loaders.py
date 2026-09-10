from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.loaders import UnsupportedFileError, get_loader, load_document


@pytest.mark.parametrize(
    ("filename", "loader_name"),
    [("notes.txt", "TextLoader"), ("notes.md", "TextLoader"), ("notes.docx", "Docx2txtLoader"), ("notes.pdf", "PyPDFLoader")],
)
def test_loader_factory_selects_langchain_loader(tmp_path: Path, filename: str, loader_name: str):
    path = tmp_path / filename
    path.write_text("content", encoding="utf-8")
    assert type(get_loader(path)).__name__ == loader_name


def test_text_loader_returns_documents_with_enriched_metadata(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    documents = load_document(path)
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "hello"
    assert documents[0].metadata["filename"] == "notes.txt"
    assert documents[0].metadata["file_type"] == "txt"
    assert documents[0].metadata["document_id"].startswith("doc-")


def test_loader_preserves_page_metadata(monkeypatch, tmp_path: Path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"pdf")

    class FakeLoader:
        def load(self):
            return [Document(page_content="page text", metadata={"source": "source.pdf", "page": 3})]

    monkeypatch.setattr("ingestion.loaders.get_loader", lambda _: FakeLoader())
    documents = load_document(path)
    assert documents[0].metadata["page"] == 3
    assert documents[0].metadata["source"] == "source.pdf"


def test_empty_and_unsupported_files_fail(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.touch()
    with pytest.raises(ValueError):
        load_document(empty)
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedFileError):
        load_document(unsupported)
