# Prism

A libCacheSim-style framework for **tool-output compression** in coding agents: one event format, one narrow plugin interface, thin harness adapters, and shared evaluation.

**Core claim:** a compression policy written once against Prism's interface runs unchanged in two harnesses (mini-SWE-agent and a second one, likely OpenHands), and we can measure whether it saves money once follow-up reads and task success are counted.

Team: Shawn Thomas, Grace Sung · CS 6501 AI Systems

## Layout

```
prism/          the library: events, policy interface + registry, capability check, store, policies
adapters/       harness adapters (mini-SWE-agent, second harness)
eval/           run.py (harness × config × task × repeat), replay.py (offline), metrics.py
configs/        one YAML per policy config (none, filter, offload)
tasks/          SWE-bench task IDs
logs/           one JSONL file per run (git-ignored)
docs/           survey, design, adding_a_policy
tests/          unit tests
```

## Policies

| | Policy A: Filtering | Policy B: Offload and retrieve |
|---|---|---|
| What it does | Keep head/tail lines, error/fail/traceback/assert lines, collapse repeats | Save full output to `/tmp/prism/<run>/<step>.txt`; model sees a filtered preview + path |
| Information lost? | Yes | No (agent can read the file) |
| Requires | nothing | `shell_fs` |

## The contract

`prism/events.py` and the JSONL log format are the contract between team members. They are frozen in Week 10; changes need both members' sign-off.

Example log lines:

```jsonl
{"type": "tool_result", "step": 4, "tool": "bash", "command": "python -m pytest tests/", "output": "...", "exit_code": 1}
{"type": "compression", "step": 4, "policy": "filter", "tokens_before": 9120, "tokens_after": 610, "ms": 3}
{"type": "model_call", "step": 5, "input_tokens": 41200, "cache_read_tokens": 30100, "cache_write_tokens": 900, "output_tokens": 310, "cost_usd": 0.021, "latency_s": 6.2}
{"type": "result", "task_id": "...", "harness": "mini", "config": "filter", "repeat": 1, "solved": true}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

See [PLAN.md](PLAN.md) for milestones and division of work.
