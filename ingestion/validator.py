from neo4j.connection import Neo4jClient


VALIDATE_DOCUMENT = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN count(DISTINCT d) AS documents,
       count(DISTINCT c) AS chunks
"""


def validate_document(
    client: Neo4jClient,
    document_id: str,
    expected_chunks: int,
) -> dict[str, object]:
    """Validate that one document and its expected chunks exist in Neo4j.

    DISTINCT is important here because the OPTIONAL MATCH creates one result
    row per document/chunk pair. Without DISTINCT, a document with 14 chunks
    would incorrectly report ``documents == 14`` instead of ``documents == 1``.
    """
    rows = client.execute(
        VALIDATE_DOCUMENT,
        {"document_id": document_id},
    )

    row = rows[0] if rows else {"documents": 0, "chunks": 0}

    documents = int(row["documents"] or 0)
    chunks = int(row["chunks"] or 0)
    expected_chunks = int(expected_chunks)

    valid = documents == 1 and chunks == expected_chunks

    return {
        "valid": valid,
        "documents": documents,
        "chunks": chunks,
        "expected_chunks": expected_chunks,
    }
