import hashlib
from pathlib import Path

from langchain_core.documents import Document


class UnsupportedFileError(ValueError):
    pass


SUPPORTED_TYPES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def _ocr_pdf(file_path: Path) -> list[Document]:
    """
    OCR fallback for scanned/image-only PDFs.

    Raises a detailed RuntimeError if OCR dependencies,
    initialization, or page processing fails.
    """

    # Import OCR dependencies
    try:
        import pymupdf
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError(
            "OCR dependencies failed to load: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Initialize OCR engine
    try:
        ocr = RapidOCR()
    except Exception as exc:
        raise RuntimeError(
            "RapidOCR initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    documents = []

    try:
        pdf = pymupdf.open(file_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open PDF for OCR: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        for page_number, page in enumerate(pdf, start=1):

            # Render PDF page to image
            try:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )

                image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to render PDF page {page_number}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # Run OCR
            try:
                result, _ = ocr(image)
            except Exception as exc:
                raise RuntimeError(
                    f"OCR failed on page {page_number}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # Extract recognized text
            page_text = []

            for item in result or []:
                if len(item) > 1:
                    detected_text = str(item[1]).strip()

                    if detected_text:
                        page_text.append(detected_text)

            text = "\n".join(page_text).strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_number,
                            "ocr": True,
                        },
                    )
                )

    finally:
        pdf.close()

    return documents


def get_loader(file_path: str | Path):
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

    return Docx2txtLoader(str(path))


def load_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    # ---------------------------------------------------------
    # Normal document loading
    # ---------------------------------------------------------
    documents = get_loader(path).load()

    text = "\n\n".join(
        document.page_content
        for document in documents
    ).strip()

    # ---------------------------------------------------------
    # PDF fallback 1: PyMuPDF
    # ---------------------------------------------------------
    if not text and suffix == ".pdf":
        try:
            from langchain_community.document_loaders import (
                PyMuPDFLoader,
            )

            documents = PyMuPDFLoader(str(path)).load()

            text = "\n\n".join(
                document.page_content
                for document in documents
            ).strip()

        except Exception:
            # If PyMuPDF extraction itself fails,
            # continue to OCR fallback.
            documents = []
            text = ""

    # ---------------------------------------------------------
    # PDF fallback 2: OCR
    # ---------------------------------------------------------
    if not text and suffix == ".pdf":

        documents = _ocr_pdf(path)

        text = "\n\n".join(
            document.page_content
            for document in documents
        ).strip()

        if not text:
            raise ValueError(
                f"No text could be extracted from PDF: {path.name}. "
                "The PDF may contain scanned images that OCR "
                "could not recognize."
            )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------
    if not text:
        raise ValueError(
            f"No text could be extracted from: {path.name}"
        )

    # ---------------------------------------------------------
    # Content hash
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
    # Enrich metadata
    # ---------------------------------------------------------
    enriched = []

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
