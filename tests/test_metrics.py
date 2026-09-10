from evaluation.metrics import source_hit, token_recall


def test_metrics_are_deterministic():
    assert source_hit(["a.pdf"], [{"filename": "a.pdf"}]) == 1.0
    assert token_recall("alpha beta", "alpha only") == 0.5
