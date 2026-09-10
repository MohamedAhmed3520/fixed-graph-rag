import json
from pathlib import Path


def load_dataset(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list")
    return data
