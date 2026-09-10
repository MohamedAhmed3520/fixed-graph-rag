import hashlib
import re
from pydantic import BaseModel, Field

from ingestion.models import EntityRecord, RelationshipRecord
from llm import get_llm
from prompts.entity_extraction import get_entity_extraction_prompt
from prompts.relationship_extraction import get_relationship_extraction_prompt


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = "concept"


class ExtractedEntities(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    confidence: float = Field(default=1.0, ge=0, le=1)


class ExtractedRelationships(BaseModel):
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _entity_id(normalized_name: str, entity_type: str) -> str:
    return hashlib.sha256(f"{entity_type}:{normalized_name}".encode()).hexdigest()[:24]


def extract_entities(text: str, chunk_id: str, llm=None) -> list[EntityRecord]:
    structured = (llm or get_llm()).with_structured_output(ExtractedEntities)
    result = structured.invoke(get_entity_extraction_prompt().invoke({"text": text}))
    records = []
    for entity in result.entities:
        normalized = normalize_name(entity.name)
        if normalized:
            records.append(EntityRecord(_entity_id(normalized, entity.entity_type), entity.name.strip(), normalized, entity.entity_type.lower(), chunk_id))
    return records


def extract_relationships(text: str, entities: list[EntityRecord], llm=None) -> list[RelationshipRecord]:
    if not entities:
        return []
    structured = (llm or get_llm()).with_structured_output(ExtractedRelationships)
    names = ", ".join(entity.name for entity in entities)
    result = structured.invoke(get_relationship_extraction_prompt().invoke({"text": text, "entities": names}))
    by_name = {entity.normalized_name: entity for entity in entities}
    relationships = []
    for item in result.relationships:
        source = by_name.get(normalize_name(item.source))
        target = by_name.get(normalize_name(item.target))
        relation = normalize_name(item.relationship).replace(" ", "_")
        if source and target and source.entity_id != target.entity_id and relation:
            relationships.append(RelationshipRecord(source.entity_id, relation, target.entity_id, source.source_chunk_id, item.confidence))
    return relationships
