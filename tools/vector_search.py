from embeddings.embedding_model import embed_query
from neo4j.connection import Neo4jClient

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.document_id AS document_id, node.text AS text,
    node.source AS source, node.filename AS filename, node.page AS page, score
"""


def vector_search(client: Neo4jClient, question: str, top_k: int = 5, embedder=embed_query) -> list[dict]:
    return client.execute(VECTOR_SEARCH, {"top_k": top_k, "embedding": embedder(question)})
