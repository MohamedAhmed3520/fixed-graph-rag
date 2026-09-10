from functools import lru_cache

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for embeddings")
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)
