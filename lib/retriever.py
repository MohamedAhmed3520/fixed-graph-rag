from dataclasses import dataclass, field

from config.settings import get_settings
from tools.entity_search import entity_search
from tools.graph_search import graph_search
from tools.vector_search import vector_search


@dataclass
class RetrievalResult:
    vector_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def combined_context(self) -> str:
        seen = set()
        sections = []
        for item in self.vector_results + self.graph_results:
            key = item.get("chunk_id") or (item.get("entity"), item.get("related_entity"))
            if key in seen:
                continue
            seen.add(key)
            sections.append(item.get("text") or f"{item.get('entity')} related to {item.get('related_entity')}")
        return "\n\n".join(sections)


def hybrid_retrieve(question: str, client, entities: list[str] | None = None, top_k: int | None = None, depth: int | None = None) -> RetrievalResult:
    settings = get_settings()
    result = RetrievalResult()
    try:
        result.vector_results = vector_search(client, question, top_k or settings.top_k)
    except Exception as exc:
        result.errors.append(f"Vector retrieval failed: {exc}")
    try:
        entity_rows = entity_search(client, entities or [])
        result.graph_results = graph_search(client, [row["entity_id"] for row in entity_rows], depth or settings.graph_depth)
    except Exception as exc:
        result.errors.append(f"Graph retrieval failed: {exc}")
    unique_sources = {}
    for item in result.vector_results + result.graph_results:
        if item.get("chunk_id"):
            unique_sources[item["chunk_id"]] = {key: item.get(key) for key in ("filename", "document_id", "chunk_id", "source", "page", "score")}
    result.sources = list(unique_sources.values())
    return result
