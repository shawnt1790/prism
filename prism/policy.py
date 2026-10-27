"""CompressionPolicy interface, registry, and the compress() wrapper that logs tokens before/after and time spent.

Policies only return text. compress() is the one place that measures, so every policy is
measured the same way. Adding a policy = one file in prism/policies/ with @register("name"),
one YAML config, one unit test.
"""

import time
from collections.abc import Callable
from typing import Protocol

from prism import capabilities, tokens
from prism.events import CompressionRecord, ToolResultEvent


class CompressionPolicy(Protocol):
    name: str
    requires: set[str]      # e.g. {"shell_fs"}

    def apply(self, ev: ToolResultEvent, store) -> str: ...


# ---------- registry ----------

REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: makes a policy available to configs as `policy: <name>`."""
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def build_policy(policy_cfg: dict):
    """{"policy": name, "params": {...}} -> policy instance."""
    import prism.policies  # noqa: F401  (registers the built-in policies)

    name = policy_cfg.get("policy", "none")
    if name not in REGISTRY:
        raise ValueError(f"Unknown policy {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**(policy_cfg.get("params") or {}))


# ---------- stores ----------

class NullStore:
    """Default store for harnesses without offloading: any put() is a bug."""
    path_prefix = ""

    def put(self, run_id: str, step: int, call_idx: int, text: str) -> str:
        raise RuntimeError("This policy tried to offload output, but no store was configured")


class _RecordingStore:
    """Wraps a store so compress() learns the offload path without policies reporting it."""

    def __init__(self, store):
        self._store = store
        self.path_prefix = store.path_prefix
        self.last_path: str | None = None

    def put(self, run_id: str, step: int, call_idx: int, text: str) -> str:
        self.last_path = self._store.put(run_id, step, call_idx, text)
        return self.last_path


# ---------- the wrapper ----------

def compress(policy, ev: ToolResultEvent, store, logger) -> str:
    """Run the policy on one tool result and log a CompressionRecord. Returns the text the model sees."""
    recording = _RecordingStore(store)
    start = time.perf_counter()
    shown = policy.apply(ev, recording)
    ms = (time.perf_counter() - start) * 1000

    logger.log(CompressionRecord(
        run_id=ev.run_id,
        step=ev.step,
        call_idx=ev.call_idx,
        policy=policy.name,
        chars_before=len(ev.output),
        chars_after=len(shown),
        tokens_before=tokens.count(ev.output),
        tokens_after=tokens.count(shown),
        ms=round(ms, 3),
        offload_path=recording.last_path,
    ))
    return shown


def make_compress(policy_cfg: dict, *, run_id: str, logger, provides: set[str],
                  harness: str = "mini", store=None) -> Callable[[ToolResultEvent], str]:
    """Build the policy from config, check capabilities, and return the function an adapter
    calls on each ToolResultEvent. `run_id` is accepted for the runner's call signature;
    each record takes its run_id from the event."""
    policy = build_policy(policy_cfg)
    capabilities.check(policy, provides, harness)
    store = store if store is not None else NullStore()
    return lambda ev: compress(policy, ev, store, logger)
