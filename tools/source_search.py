from neo4j.connection import Neo4jClient

SOURCE_SEARCH = """
MATCH (d:Document)
WHERE toLower(d.filename) CONTAINS toLower($filename)
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.document_id AS document_id, d.filename AS filename, c.chunk_id AS chunk_id, c.text AS text
ORDER BY c.chunk_index
"""


def source_search(client: Neo4jClient, filename: str) -> list[dict]:
    return client.execute(SOURCE_SEARCH, {"filename": filename})
