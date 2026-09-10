def source_hit(expected_sources: list[str], retrieved_sources: list[dict]) -> float:
    if not expected_sources:
        return 0.0
    actual = {item.get("filename") or item.get("document_id") for item in retrieved_sources}
    return float(any(source in actual for source in expected_sources))


def token_recall(expected: str, actual: str) -> float:
    expected_tokens = {token.lower() for token in expected.split() if token.strip()}
    if not expected_tokens:
        return 0.0
    actual_tokens = {token.lower() for token in actual.split()}
    return len(expected_tokens & actual_tokens) / len(expected_tokens)
