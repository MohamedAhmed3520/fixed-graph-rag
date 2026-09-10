from evaluation.metrics import token_recall


def evaluate_generation(cases: list[dict], answers: list[str]) -> list[dict]:
    return [{"question": case["question"], "answer_recall": token_recall(case.get("expected_answer", ""), answer)} for case, answer in zip(cases, answers)]
