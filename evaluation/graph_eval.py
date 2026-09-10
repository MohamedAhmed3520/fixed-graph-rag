def evaluate_graph(expected_entities: set[str], actual_entities: set[str]) -> dict[str, float]:
    if not expected_entities:
        return {"entity_precision": 0.0, "entity_recall": 0.0}
    intersection = expected_entities & actual_entities
    precision = len(intersection) / len(actual_entities) if actual_entities else 0.0
    return {"entity_precision": precision, "entity_recall": len(intersection) / len(expected_entities)}
