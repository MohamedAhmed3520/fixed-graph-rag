import hashlib
import io
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
    """
    Combine all non-empty document contents into one string.
    """

    return "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content
        and document.page_content.strip()
    ).strip()


def _has_text(documents: list[Document]) -> bool:
    """
    Return True if at least one document contains text.
    """

    return bool(_combine_text(documents))


def _ocr_pdf(file_path: Path) -> list[Document]:
    """
    OCR a scanned/image-only PDF using Tesseract.

    Each PDF page is rendered to an image using PyMuPDF,
    then processed independently by Tesseract.

    Supports English and Arabic OCR.

    Required Python packages:
        pymupdf
        pytesseract
        Pillow

    Required Linux packages on Streamlit Cloud:
        tesseract-ocr
        tesseract-ocr-eng
        tesseract-ocr-ara

    Returns:
        A list containing one LangChain Document per page
        where OCR text was successfully extracted.
    """

    # ---------------------------------------------------------
    # Import OCR dependencies lazily.
    #
    # This means normal TXT/MD/DOCX files do not initialize
    # OCR dependencies unnecessarily.
    # ---------------------------------------------------------

    try:
        import fitz
        import pytesseract
        from PIL import Image

    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are not installed. "
            "Make sure requirements.txt contains: "
            "pymupdf, pytesseract, Pillow."
        ) from exc

    # ---------------------------------------------------------
    # Check that Tesseract executable is available.
    # ---------------------------------------------------------

    try:
        tesseract_version = pytesseract.get_tesseract_version()

    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed on the system. "
            "For Streamlit Cloud, add the following to "
            "packages.txt:\n\n"
            "tesseract-ocr\n"
            "tesseract-ocr-eng\n"
            "tesseract-ocr-ara\n\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    documents: list[Document] = []

    # ---------------------------------------------------------
    # Open PDF
    # ---------------------------------------------------------

    try:
        pdf = fitz.open(str(file_path))

    except Exception as exc:
        raise RuntimeError(
            f"Failed to open PDF '{file_path.name}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:

        # -----------------------------------------------------
        # Process every page independently
        # -----------------------------------------------------

        for page_number, page in enumerate(pdf):

            try:

                # -------------------------------------------------
                # Render PDF page at 2x resolution.
                #
                # 2x gives Tesseract significantly better input
                # quality than using the original PDF resolution.
                # -------------------------------------------------

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                # -------------------------------------------------
                # Convert PyMuPDF image to PIL Image.
                # -------------------------------------------------

                image_bytes = pixmap.tobytes("png")

                image = Image.open(
                    io.BytesIO(image_bytes)
                )

                # -------------------------------------------------
                # OCR
                #
                # eng+ara means:
                # English + Arabic
                #
                # If your application only processes English,
                # you can change this to "eng".
                # -------------------------------------------------

                text = pytesseract.image_to_string(
                    image,
                    lang="eng+ara",
                    config="--psm 3",
                )

                text = text.strip()

                # -------------------------------------------------
                # Skip pages where OCR found nothing.
                # -------------------------------------------------

                if not text:
                    continue

                # -------------------------------------------------
                # Store page as LangChain Document.
                # -------------------------------------------------

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "page": page_number,
                            "page_number": page_number + 1,
                            "ocr": True,
                            "ocr_engine": "tesseract",
                            "ocr_languages": "eng+ara",
                            "tesseract_version": str(
                                tesseract_version
                            ),
                        },
                    )
                )

            except Exception as exc:

                raise RuntimeError(
                    f"OCR failed on page "
                    f"{page_number + 1} "
                    f"of '{file_path.name}': "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    finally:
        pdf.close()

    return documents


def get_loader(file_path: str | Path):
    """
    Return the appropriate LangChain document loader.

    Supported:
        .txt
        .md
        .markdown
        .pdf
        .docx
    """

    path = Path(file_path)

    suffix = path.suffix.lower()

    # ---------------------------------------------------------
    # Validate extension
    # ---------------------------------------------------------

    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileError(
            f"Unsupported file type: "
            f"{suffix or '<none>'}"
        )

    # ---------------------------------------------------------
    # Validate file exists
    # ---------------------------------------------------------

    if not path.is_file():
        raise FileNotFoundError(path)

    # ---------------------------------------------------------
    # Validate file is not empty
    # ---------------------------------------------------------

    if path.stat().st_size == 0:
        raise ValueError(
            f"File is empty: {path.name}"
        )

    # ---------------------------------------------------------
    # Import LangChain loaders
    # ---------------------------------------------------------

    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    if suffix == ".pdf":
        return PyPDFLoader(str(path))

    # ---------------------------------------------------------
    # TXT
    # ---------------------------------------------------------

    if suffix == ".txt":
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    # ---------------------------------------------------------
    # Markdown
    # ---------------------------------------------------------

    if suffix in {".md", ".markdown"}:
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------

    if suffix == ".docx":
        return Docx2txtLoader(str(path))

    raise UnsupportedFileError(
        f"Unsupported file type: {suffix}"
    )


def load_document(
    file_path: str | Path,
) -> list[Document]:
    """
    Load a document and return LangChain Documents.

    PDF extraction strategy:

        1. PyPDFLoader
        2. PyMuPDFLoader
        3. Tesseract OCR

    Normal PDFs use their existing text layer.

    Scanned/image-only PDFs automatically fall back
    to OCR.

    Supported:
        TXT
        Markdown
        DOCX
        PDF
        Scanned PDF

    Returns:
        list[Document]

    Raises:
        UnsupportedFileError:
            Unsupported file type.

        FileNotFoundError:
            File does not exist.

        ValueError:
            File is empty or no text could be extracted.

        RuntimeError:
            Document loading or OCR failed.
    """

    # ---------------------------------------------------------
    # Normalize path
    # ---------------------------------------------------------

    path = Path(file_path)

    suffix = path.suffix.lower()

    # ---------------------------------------------------------
    # Validate basic file information
    # ---------------------------------------------------------

    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedFileError(
            f"Unsupported file type: "
            f"{suffix or '<none>'}"
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        raise ValueError(
            f"File is empty: {path.name}"
        )

    # ---------------------------------------------------------
    # Documents returned by the standard loader
    # ---------------------------------------------------------

    documents: list[Document] = []

    # =========================================================
    # 1. STANDARD LOADER
    # =========================================================

    try:

        loader = get_loader(path)

        documents = loader.load()

    except Exception as exc:

        # -----------------------------------------------------
        # Non-PDF files don't have additional fallback
        # strategies.
        # -----------------------------------------------------

        if suffix != ".pdf":

            raise RuntimeError(
                f"Failed to load '{path.name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # -----------------------------------------------------
        # PDF errors are intentionally ignored here.
        #
        # We continue to PyMuPDF and then OCR.
        # -----------------------------------------------------

        documents = []

    # ---------------------------------------------------------
    # Check extracted text.
    # ---------------------------------------------------------

    text = _combine_text(documents)

    # =========================================================
    # 2. PYMUPDF FALLBACK
    # =========================================================

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

                text = _combine_text(
                    pymupdf_documents
                )

        except Exception:
            # -------------------------------------------------
            # PyMuPDF did not extract usable text.
            # Continue to OCR.
            # -------------------------------------------------

            documents = []

            text = ""

    # =========================================================
    # 3. TESSERACT OCR FALLBACK
    # =========================================================

    if not text and suffix == ".pdf":

        documents = _ocr_pdf(path)

        text = _combine_text(documents)

    # =========================================================
    # 4. FINAL VALIDATION
    # =========================================================

    if not text:

        raise ValueError(
            f"No text could be extracted from "
            f"'{path.name}'. "
            "The PDF may contain scanned/image-only "
            "content, and OCR did not recognize any text."
        )

    # =========================================================
    # 5. CREATE DETERMINISTIC DOCUMENT ID
    # =========================================================

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    document_id = (
        f"doc-{content_hash[:24]}"
    )

    # =========================================================
    # 6. COMMON METADATA
    # =========================================================

    base_metadata = {
        "source": str(path),
        "filename": path.name,
        "file_type": suffix.lstrip("."),
        "document_id": document_id,
        "content_hash": content_hash,
    }

    # =========================================================
    # 7. ENRICH EVERY DOCUMENT
    # =========================================================

    enriched: list[Document] = []

    for document in documents:

        # -----------------------------------------------------
        # Common metadata + loader-specific metadata
        #
        # Loader metadata comes after base metadata so that
        # page/source information generated by LangChain is
        # preserved.
        # -----------------------------------------------------

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
