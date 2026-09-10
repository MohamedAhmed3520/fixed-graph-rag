from evaluation.metrics import source_hit, token_recall


def evaluate_case(case: dict, output: dict) -> dict[str, float | str]:
    answer = output.get("final_answer", "")
    sources = output.get("sources", [])
    return {"question": case["question"], "answer_recall": token_recall(case.get("expected_answer", ""), answer), "source_hit": source_hit(case.get("expected_sources", []), sources)}


def summarize(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    return {key: sum(float(row.get(key, 0)) for row in results) / len(results) for key in ("answer_recall", "source_hit")}
