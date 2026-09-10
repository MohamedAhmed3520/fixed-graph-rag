UPSERT_DOCUMENT = """
MERGE (d:Document {document_id: $document_id})
SET d.filename = $filename, d.source = $source, d.file_type = $file_type,
    d.content_hash = $content_hash, d.ingestion_timestamp = $ingestion_timestamp
RETURN d.document_id AS document_id
"""

UPSERT_CHUNK = """
MATCH (d:Document {document_id: $document_id})
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.document_id = $document_id, c.filename = $filename, c.source = $source,
    c.chunk_index = $chunk_index, c.text = $text, c.embedding = $embedding, c.page = $page
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN c.chunk_id AS chunk_id
"""

UPSERT_ENTITY = """
MATCH (c:Chunk {chunk_id: $source_chunk_id})
MERGE (e:Entity {entity_id: $entity_id})
SET e.name = $name, e.normalized_name = $normalized_name, e.entity_type = $entity_type
MERGE (c)-[:MENTIONS]->(e)
RETURN e.entity_id AS entity_id
"""

UPSERT_RELATIONSHIP = """
MATCH (source:Entity {entity_id: $source_entity_id})
MATCH (target:Entity {entity_id: $target_entity_id})
MERGE (source)-[r:RELATED_TO {source_chunk_id: $source_chunk_id}]->(target)
SET r.relationship = $relationship, r.confidence = $confidence
RETURN source.entity_id AS source_entity_id
"""
