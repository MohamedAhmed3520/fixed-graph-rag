from ingestion.validator import validate_document


class FakeClient:

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(
        self,
        query,
        parameters=None,
    ):
        self.calls.append(
            (query, parameters)
        )

        return self.rows


def test_validation_counts_document_once_for_multiple_chunks():

    client = FakeClient(
        [
            {
                "documents": 1,
                "chunks": 14,
            }
        ]
    )

    result = validate_document(
        client,
        "doc-123",
        14,
    )

    assert result == {
        "valid": True,
        "documents": 1,
        "chunks": 14,
        "expected_chunks": 14,
    }

    assert (
        "count(DISTINCT d)"
        in client.calls[0][0]
    )

    assert (
        "count(DISTINCT c)"
        in client.calls[0][0]
    )


def test_validation_fails_when_chunk_count_is_wrong():

    client = FakeClient(
        [
            {
                "documents": 1,
                "chunks": 13,
            }
        ]
    )

    result = validate_document(
        client,
        "doc-123",
        14,
    )

    assert result["valid"] is False
    assert result["documents"] == 1
    assert result["chunks"] == 13
    assert result["expected_chunks"] == 14


def test_validation_fails_when_document_is_missing():

    client = FakeClient([])

    result = validate_document(
        client,
        "missing",
        0,
    )

    assert result == {
        "valid": False,
        "documents": 0,
        "chunks": 0,
        "expected_chunks": 0,
    }
