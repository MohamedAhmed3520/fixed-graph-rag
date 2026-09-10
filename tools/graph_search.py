from neo4j.connection import Neo4jClient

GRAPH_SEARCH = """
MATCH (e:Entity)
WHERE e.entity_id IN $entity_ids
MATCH (e)-[r:RELATED_TO*1..$depth]-(neighbor:Entity)
OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(neighbor)
RETURN DISTINCT e.name AS entity, neighbor.name AS related_entity,
    [path_rel IN r | path_rel.relationship] AS relationships,
       chunk.chunk_id AS chunk_id, chunk.document_id AS document_id,
    chunk.text AS text, chunk.source AS source, chunk.filename AS filename, chunk.page AS page
"""


def graph_search(client: Neo4jClient, entity_ids: list[str], depth: int = 2) -> list[dict]:
    if not entity_ids:
        return []
    # Neo4j does not parameterize relationship bounds in all supported versions.
    # Keep the bound application-controlled and select a bounded query.
    bounded = GRAPH_SEARCH.replace("$depth", str(max(1, min(depth, 4))))
    return client.execute(bounded, {"entity_ids": entity_ids})
