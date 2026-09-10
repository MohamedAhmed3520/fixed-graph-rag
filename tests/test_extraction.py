from ingestion.extractor import normalize_name


def test_entity_names_are_normalized_for_matching():
    assert normalize_name("  Acme   Corp ") == "acme corp"
