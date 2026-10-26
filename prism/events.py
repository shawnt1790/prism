"""Standardized events. This file is the contract between team members (frozen Week 10)."""

from dataclasses import dataclass


@dataclass
class ToolResultEvent:
    run_id: str
    step: int
    tool: str               # "bash" in mini-SWE-agent; harness tool name elsewhere
    command: str            # the command or a readable form of the tool call
    output: str
    exit_code: int | None
    meta: dict              # harness-specific extras; policies ignore it


@dataclass
class ModelCallEvent:
    run_id: str
    step: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class RunResult:
    run_id: str
    task_id: str
    harness: str
    config: str
    repeat: int
    solved: bool
    total_steps: int
    wall_clock_s: float
