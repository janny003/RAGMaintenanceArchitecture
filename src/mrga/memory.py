import json
from pathlib import Path
from typing import Dict, Any


class PersistentMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"records": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append(self, record: Dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("records", []).append(record)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
