import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{document_id}:{digest}"


def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 120, separators: list[str] | None = None) -> list[Document]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        document_id = metadata["document_id"]
        metadata.update({"chunk_id": _chunk_id(document_id, index, chunk.page_content), "chunk_index": index})
        chunks[index] = Document(page_content=chunk.page_content, metadata=metadata)
    return chunks
