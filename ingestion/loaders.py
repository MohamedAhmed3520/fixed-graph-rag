import hashlib
from pathlib import Path

from langchain_core.documents import Document


class UnsupportedFileError(ValueError):
    """Raised when an unsupported file type is provided."""


SUPPORTED_TYPES = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
}


def _combine_text(documents: list[Document]) -> str:
    """Combine document contents into one string."""

    return "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content
        and document.page_content.strip()
    ).strip()


def _has_text(documents: list[Document]) -> bool:
    """Return True if at least one document contains text."""

    return bool(_combine_text(documents))


def _ocr_pdf(file_path: Path) -> list[Document]:
    """
    OCR a scanned/image-only PDF using RapidOCR.

    Each PDF page is rendered to an image with PyMuPDF and
    processed independently by RapidOCR.

    Returns:
        A list containing one LangChain Document per page
        where OCR text was successfully extracted.
    """

    # Import these here so normal TXT/MD/DOCX files don't require
    # OCR initialization.
    try:
        import fitz
        import numpy as np
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are not installed. "
            "Make sure requirements.txt contains: "
            "rapidocr, onnxruntime, pymupdf, numpy, Pillow"
        ) from exc

    documents: list[Document] = []

    try:
        ocr = RapidOCR()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize RapidOCR: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        pdf = fitz.open(str(file_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open PDF '{file_path.name}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        for page_number, page in enumerate(pdf):
            try:
                # Render page at 2x resolution for better OCR accuracy.
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                # Convert PyMuPDF image to NumPy array.
                image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )

                # Run RapidOCR.
                result = ocr(image)

                # Current RapidOCR exposes recognized text
                # through the `txts` property.
                texts = getattr(result, "txts", None)

                if not texts:
                    continue

                page_text_parts: list[str] = []

                for item in texts:
                    if item is None:
                        continue

                    text = str(item).strip()

                    if text:
                        page_text_parts.append(text)

                page_text = "\n".join(page_text_parts).strip()

                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": str(file_path),
                                "page": page_number,
                                "page_number": page_number + 1,
                                "ocr": True,
                            },
                        )
                    )

            except Exception as exc:
                raise RuntimeError(
                    f"OCR failed on page {page_number + 1} "
                    f"of '{file_path.name}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    finally:
        pdf.close()

    return documents


def get_loader(file_path: str | Path):
    """
    Return the appropriate LangChain document loader.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileError(
            f"Unsupported file type: {suffix or '<none>'}"
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        raise ValueError(
            f"File is empty: {path.name}"
        )

    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    if suffix == ".pdf":
        return PyPDFLoader(str(path))

    if suffix == ".txt":
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    if suffix in {".md", ".markdown"}:
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    if suffix == ".docx":
        return Docx2txtLoader(str(path))

    raise UnsupportedFileError(
        f"Unsupported file type: {suffix}"
    )


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a document and return LangChain Documents.

    PDF extraction strategy:

        1. PyPDFLoader
        2. PyMuPDFLoader
        3. RapidOCR

    This means normal PDFs use their existing text layer,
    while scanned/image-only PDFs automatically use OCR.

    Returns:
        list[Document]

    Raises:
        ValueError:
            If no text can be extracted.

        RuntimeError:
            If PDF OCR or another extraction step fails.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    # ---------------------------------------------------------
    # 1. Standard loader
    # ---------------------------------------------------------

    documents: list[Document] = []

    try:
        loader = get_loader(path)
        documents = loader.load()

    except Exception as exc:
        # For PDFs, continue to the fallback extraction methods.
        if suffix != ".pdf":
            raise RuntimeError(
                f"Failed to load '{path.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    text = _combine_text(documents)

    # ---------------------------------------------------------
    # 2. PyMuPDF fallback
    # ---------------------------------------------------------

    if not text and suffix == ".pdf":
        try:
            from langchain_community.document_loaders import (
                PyMuPDFLoader,
            )

            pymupdf_documents = PyMuPDFLoader(
                str(path)
            ).load()

            if _has_text(pymupdf_documents):
                documents = pymupdf_documents
                text = _combine_text(documents)

        except Exception:
            # If PyMuPDF doesn't extract anything, continue to OCR.
            documents = []
            text = ""

    # ---------------------------------------------------------
    # 3. OCR fallback
    # ---------------------------------------------------------

    if not text and suffix == ".pdf":

        documents = _ocr_pdf(path)

        text = _combine_text(documents)

    # ---------------------------------------------------------
    # 4. Final validation
    # ---------------------------------------------------------

    if not text:
        raise ValueError(
            f"No text could be extracted from '{path.name}'. "
            "The PDF appears to contain scanned/image-only "
            "content, but OCR did not recognize any text."
        )

    # ---------------------------------------------------------
    # 5. Create deterministic document ID
    # ---------------------------------------------------------

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    base_metadata = {
        "source": str(path),
        "filename": path.name,
        "file_type": suffix.lstrip("."),
        "document_id": f"doc-{content_hash[:24]}",
        "content_hash": content_hash,
    }

    # ---------------------------------------------------------
    # 6. Enrich every Document with common metadata
    # ---------------------------------------------------------

    enriched: list[Document] = []

    for document in documents:
        metadata = {
            **base_metadata,
            **document.metadata,
        }

        enriched.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    return enriched
