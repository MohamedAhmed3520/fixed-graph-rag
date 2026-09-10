from neo4j.connection import Neo4jClient


VALIDATE_DOCUMENT = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN count(d) AS documents, count(c) AS chunks
"""


def validate_document(client: Neo4jClient, document_id: str, expected_chunks: int) -> dict[str, object]:
    rows = client.execute(VALIDATE_DOCUMENT, {"document_id": document_id})
    row = rows[0] if rows else {"documents": 0, "chunks": 0}
    valid = row["documents"] == 1 and row["chunks"] == expected_chunks
    return {"valid": valid, "documents": row["documents"], "chunks": row["chunks"], "expected_chunks": expected_chunks}
