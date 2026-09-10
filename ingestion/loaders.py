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


def _ocr_pdf(file_path: Path) -> list[Document]:
    """
    Extract text from a scanned/image-only PDF using RapidOCR.

    Each PDF page is rendered to an image using PyMuPDF and then
    processed independently by RapidOCR.

    Returns:
        list[Document]: One LangChain Document per page containing
        OCR-extracted text.
    """

    import fitz
    import numpy as np
    from rapidocr import RapidOCR

    documents: list[Document] = []

    # Initialize OCR engine once instead of once per page.
    ocr = RapidOCR()

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf):
            try:
                # Render the page at 2x resolution.
                # This generally gives OCR better input quality
                # than rendering at the default resolution.
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                # Convert PyMuPDF image buffer into a NumPy array.
                image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )

                # Run OCR.
                result = ocr(image)

                # Modern RapidOCR exposes recognized text through
                # the `txts` attribute.
                texts = getattr(result, "txts", None)

                if texts is None:
                    texts = []

                # Clean OCR output.
                page_text_parts = []

                for text in texts:
                    if text is None:
                        continue

                    text = str(text).strip()

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

    return documents


def get_loader(file_path: str | Path):
    """
    Return the appropriate LangChain loader for a file.

    PDFs are initially processed with PyPDFLoader. If no text is
    extracted, load_document() will try PyMuPDF and finally OCR.
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


def _has_text(documents: list[Document]) -> bool:
    """
    Check whether loaded documents contain meaningful text.
    """

    return any(
        document.page_content
        and document.page_content.strip()
        for document in documents
    )


def _combine_text(documents: list[Document]) -> str:
    """
    Combine document text into a single string for validation.
    """

    return "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content
    ).strip()


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a document using a multi-stage extraction strategy.

    PDF extraction order:

        1. PyPDFLoader
        2. PyMuPDFLoader
        3. RapidOCR

    This allows normal text PDFs to use normal text extraction while
    scanned/image-only PDFs automatically fall back to OCR.

    Returns:
        list[Document]: Extracted LangChain documents with metadata.

    Raises:
        ValueError: If no text can be extracted.
        RuntimeError: If OCR fails.
    """

    path = Path(file_path)

    suffix = path.suffix.lower()

    # ---------------------------------------------------------
    # Stage 1: Standard LangChain loader
    # ---------------------------------------------------------

    loader = get_loader(path)

    try:
        documents = loader.load()
    except Exception as exc:
        # For PDFs, continue to PyMuPDF/OCR even if the first loader
        # itself encounters an extraction problem.
        if suffix == ".pdf":
            documents = []
        else:
            raise RuntimeError(
                f"Failed to load '{path.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    text = _combine_text(documents)

    # ---------------------------------------------------------
    # Stage 2: PyMuPDF fallback for PDFs
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
            # Do not stop here. The PDF may be scanned and require OCR.
            documents = []
            text = ""

    # ---------------------------------------------------------
    # Stage 3: OCR fallback for scanned PDFs
    # ---------------------------------------------------------

    if not text and suffix == ".pdf":
        try:
            documents = _ocr_pdf(path)
        except ImportError as exc:
            raise ValueError(
                "This PDF appears to be scanned/image-only and "
                "requires OCR, but the OCR dependencies are not "
                "installed. Install `rapidocr` and `onnxruntime`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not OCR scanned PDF '{path.name}'. "
                f"OCR error: {type(exc).__name__}: {exc}"
            ) from exc

        text = _combine_text(documents)

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if not text:
        raise ValueError(
            f"No text could be extracted from '{path.name}'. "
            f"The PDF may contain unsupported image content, "
            f"or the OCR engine could not recognize its text."
        )

    # ---------------------------------------------------------
    # Content hash / document ID
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
    # Add common metadata to every page/document
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
