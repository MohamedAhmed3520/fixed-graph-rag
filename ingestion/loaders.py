import hashlib
from pathlib import Path

from langchain_core.documents import Document


class UnsupportedFileError(ValueError):
    pass


SUPPORTED_TYPES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def _ocr_pdf(file_path: Path) -> list[Document]:
    import fitz
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    documents = []
    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            result, _ = ocr(image)
            text = "\n".join(item[1] for item in (result or []) if len(item) > 1 and item[1].strip())
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": str(file_path), "page": page_number}))
    return documents


def get_loader(file_path: str | Path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileError(f"Unsupported file type: {suffix or '<none>'}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path.name}")

    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix == ".txt":
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    if suffix in {".md", ".markdown"}:
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return Docx2txtLoader(str(path))


def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    documents = get_loader(path).load()
    text = "\n\n".join(document.page_content for document in documents).strip()
    if not text and path.suffix.lower() == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        documents = PyMuPDFLoader(str(path)).load()
        text = "\n\n".join(document.page_content for document in documents).strip()
    if not text and path.suffix.lower() == ".pdf":
        try:
            documents = _ocr_pdf(path)
        except ImportError as exc:
            raise ValueError(
                "OCR dependencies are unavailable. Install rapidocr-onnxruntime and its "
                f"dependencies to process scanned PDFs: {exc}"
            ) from exc
        text = "\n\n".join(document.page_content for document in documents).strip()
    if not text:
        raise ValueError(f"No text could be extracted from: {path.name}")

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    base_metadata = {
        "source": str(path),
        "filename": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "document_id": f"doc-{content_hash[:24]}",
        "content_hash": content_hash,
    }
    enriched = []
    for document in documents:
        metadata = {**base_metadata, **document.metadata}
        enriched.append(Document(page_content=document.page_content, metadata=metadata))
    return enriched
