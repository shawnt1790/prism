"""Standardized events. This file is the contract between team members (v1, frozen Week 10).

Changes need both members' sign-off. See docs/design.md for field meanings.
"""

import json
import time
from dataclasses import asdict, dataclass, fields

SCHEMA_VERSION = 1


@dataclass
class RunStart:
    run_id: str
    task_id: str
    harness: str            # "mini" | "openhands" | ...
    config: str             # config name, e.g. "filter"
    model: str
    repeat: int
    harness_version: str
    prism_commit: str
    schema_version: int = SCHEMA_VERSION


@dataclass
class ToolResultEvent:
    run_id: str
    step: int
    call_idx: int           # index of this call within the step (parallel tool calls)
    tool: str               # "bash" in mini-SWE-agent; harness tool name elsewhere
    command: str            # the command or a readable form of the tool call
    output: str             # raw output before any truncation, stdout+stderr merged
    exit_code: int | None   # None when the harness raised instead of finishing
    error: str | None       # harness exception text (timeout, crash), else None
    meta: dict              # harness-specific extras; policies ignore it


@dataclass
class CompressionRecord:
    run_id: str
    step: int
    call_idx: int
    policy: str
    chars_before: int
    chars_after: int
    tokens_before: int      # prism.tokens.count(), approximate
    tokens_after: int
    ms: float
    offload_path: str | None  # set only when the policy offloaded the full output


@dataclass
class ModelCallEvent:
    run_id: str
    step: int
    model: str
    input_tokens: int       # uncached input only (Anthropic usage semantics)
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: float         # computed by prism.pricing, not the harness
    latency_s: float


@dataclass
class RunResult:
    run_id: str
    task_id: str
    harness: str
    config: str
    model: str
    repeat: int
    solved: bool            # from SWE-bench evaluation, so this line is written last
    exit_status: str        # "submitted" | "step_limit" | "cost_limit" | "error"
    total_steps: int
    n_model_calls: int
    total_cost_usd: float
    wall_clock_s: float


EVENT_TYPES = {
    "run_start": RunStart,
    "tool_result": ToolResultEvent,
    "compression": CompressionRecord,
    "model_call": ModelCallEvent,
    "result": RunResult,
}
_TYPE_NAMES = {cls: name for name, cls in EVENT_TYPES.items()}


def to_json_line(event, ts: float | None = None) -> str:
    """Serialize an event as one JSONL line: {"type", "ts", ...fields}."""
    record = {"type": _TYPE_NAMES[type(event)], "ts": time.time() if ts is None else ts}
    record.update(asdict(event))
    return json.dumps(record, ensure_ascii=False)


def from_json_line(line: str):
    """Parse one JSONL line back into its event dataclass (drops "type" and "ts")."""
    record = json.loads(line)
    cls = EVENT_TYPES[record["type"]]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in record.items() if k in names})
