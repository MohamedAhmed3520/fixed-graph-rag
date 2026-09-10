from evaluation.metrics import source_hit


def evaluate_retrieval(cases: list[dict], retrieve) -> list[dict]:
    return [{"question": case["question"], "source_hit": source_hit(case.get("expected_sources", []), retrieve(case["question"]))} for case in cases]
