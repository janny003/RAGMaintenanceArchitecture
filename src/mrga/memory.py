import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class PersistentMemory:
    SCHEMA_VERSION = "1.1"

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(self._empty_schema())

    def _empty_schema(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "static_memory": {
                "equipment_profiles": [],
                "limits": {},
            },
            "dynamic_memory": {
                "latest_observations": [],
            },
            "episode_memory": {
                "history": [],
                "preferences": {},
                "resolved_priority": {},
                "last_interview": {},
                "interview_history": [],
            },
            "verification_memory": {
                "history": [],
            },
        }

    def _read_raw(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _migrate_if_needed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "records" in data:
            migrated = self._empty_schema()
            migrated["episode_memory"]["history"] = data.get("records", [])
            return migrated

        # normalize partially-populated schema
        normalized = self._empty_schema()
        for key in normalized:
            if key in data:
                if isinstance(normalized[key], dict) and isinstance(data[key], dict):
                    normalized[key].update(data[key])
                else:
                    normalized[key] = data[key]
        normalized["schema_version"] = self.SCHEMA_VERSION
        return normalized

    def load(self) -> Dict[str, Any]:
        data = self._read_raw()
        migrated = self._migrate_if_needed(data)
        if migrated != data:
            self._write(migrated)
        return migrated

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def append_episode(self, record: Dict[str, Any]) -> None:
        data = self.load()
        rec = {"timestamp_utc": self._utc_now(), **record}
        data["episode_memory"].setdefault("history", []).append(rec)
        self._write(data)

    def append_verification(self, record: Dict[str, Any]) -> None:
        data = self.load()
        rec = {"timestamp_utc": self._utc_now(), **record}
        data["verification_memory"].setdefault("history", []).append(rec)
        self._write(data)
