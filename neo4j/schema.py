from neo4j.connection import Neo4jClient


SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
    "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
]


def initialize_schema(client: Neo4jClient) -> None:
    for statement in SCHEMA_STATEMENTS:
        client.execute(statement)
