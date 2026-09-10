from typing import Any, TypedDict


class GraphRAGState(TypedDict, total=False):
    user_question: str
    rewritten_question: str
    extracted_entities: list[str]
    vector_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    combined_context: str
    sources: list[dict[str, Any]]
    final_answer: str
    debug_info: dict[str, Any]
    errors: list[str]
