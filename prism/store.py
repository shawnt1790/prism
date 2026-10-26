"""Storage for offloaded outputs (Policy B writes files inside the task container)."""

from typing import Protocol


class Store(Protocol):
    path_prefix: str        # e.g. "/tmp/prism"; follow-up reads are commands that reference it

    def put(self, run_id: str, step: int, call_idx: int, text: str) -> str:
        """Save the full output and return the path the model is told about."""
        ...
