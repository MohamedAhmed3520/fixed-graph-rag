from functools import lru_cache

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_llm():
    from langchain_openai import ChatOpenAI
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for answer generation")
    return ChatOpenAI(model=settings.openrouter_model, api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url, temperature=0, max_tokens=settings.llm_max_tokens)
