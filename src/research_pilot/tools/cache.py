"""Tiny disk cache so benchmark re-runs don't spend Tavily credits or refetch pages."""

import hashlib
import json
from pathlib import Path


class DiskCache:
    def __init__(self, directory: Path, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled

    def _path(self, namespace: str, key: dict) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        return self.directory / namespace / f"{digest}.json"

    def get(self, namespace: str, key: dict) -> str | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["value"]

    def set(self, namespace: str, key: dict, value: str) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "value": value}, ensure_ascii=False), encoding="utf-8")
