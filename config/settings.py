from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL")
    openrouter_model: str = Field(default="openai/gpt-4o-mini-2024-07-18", validation_alias="OPENROUTER_MODEL")
    neo4j_uri: str = Field(default="neo4j://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", validation_alias="NEO4J_DATABASE")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    app_name: str = Field(default="Local GraphRAG", validation_alias="APP_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=5, validation_alias="TOP_K", gt=0)
    graph_depth: int = Field(default=2, validation_alias="GRAPH_DEPTH", ge=1)
    llm_max_tokens: int = Field(default=500, validation_alias="LLM_MAX_TOKENS", gt=0)

    def missing_services(self) -> list[str]:
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
