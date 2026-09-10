from neo4j.connection import Neo4jClient

ENTITY_SEARCH = """
UNWIND $names AS requested
MATCH (e:Entity)
WHERE e.normalized_name CONTAINS toLower(requested)
RETURN e.entity_id AS entity_id, e.name AS name, e.entity_type AS entity_type
"""


def entity_search(client: Neo4jClient, names: list[str]) -> list[dict]:
    return client.execute(ENTITY_SEARCH, {"names": names}) if names else []
