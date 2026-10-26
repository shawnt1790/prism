"""Logger interface: one JSONL file per run (logs/<run_id>.jsonl), one event per line."""

from typing import Protocol


class Logger(Protocol):
    def log(self, event) -> None:
        """Write one event as a JSONL line (see prism.events.to_json_line)."""
        ...
