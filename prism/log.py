"""Logger interface: one JSONL file per run (logs/<run_id>.jsonl), one event per line."""

from pathlib import Path
from typing import Protocol

from prism.events import to_json_line


class Logger(Protocol):
    def log(self, event) -> None:
        """Write one event as a JSONL line (see prism.events.to_json_line)."""
        ...


class JsonlLogger:
    """Writes each event as one line to logs/<run_id>.jsonl, flushing after every line
    so a crashed run still leaves a usable log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, event) -> None:
        self._f.write(to_json_line(event) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()