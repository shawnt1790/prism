# Design

## The contract (v1)

`prism/events.py` and the JSONL log format below are the contract between team members. Changes need both members' sign-off (a PR approved by the other person). Bump `SCHEMA_VERSION` on any change.

### Events

**RunStart**: first line of every log.

| Field | Meaning |
|---|---|
| `run_id` | unique per run; also the log file name |
| `task_id` | SWE-bench instance ID |
| `harness` | `mini`, `openhands`, ... |
| `config` | config name (`none`, `filter`, `offload`) |
| `model` | pinned model ID |
| `repeat` | 1, 2, 3 |
| `harness_version`, `prism_commit` | for reproducibility |
| `schema_version` | contract version (1) |

**ToolResultEvent**: one per tool call, built by the adapter.

| Field | Meaning |
|---|---|
| `step` | model step that issued the call |
| `call_idx` | index of the call within the step. A step can run several calls in parallel, so `(step, call_idx)` is the key |
| `tool` | `bash` in mini-SWE-agent; the harness's tool name elsewhere |
| `command` | the command, or a readable form of the tool call |
| `output` | **raw** output before any truncation, stdout and stderr merged |
| `exit_code` | the process exit code; `None` when the harness raised instead of finishing |
| `error` | the harness's exception text (timeout, crash), else `None` |
| `meta` | harness-specific extras; policies ignore it |

**CompressionRecord**: one per `prism.compress()` call. Written by Prism, never by policies.

| Field | Meaning |
|---|---|
| `step`, `call_idx` | which ToolResultEvent this compressed |
| `policy` | policy name |
| `chars_before`, `chars_after` | exact sizes |
| `tokens_before`, `tokens_after` | `prism.tokens.count()` (approximate, see below) |
| `ms` | time spent in the policy |
| `offload_path` | where the full output was saved, if offloaded |

**ModelCallEvent**: one per model call.

| Field | Meaning |
|---|---|
| `step`, `model` | |
| `input_tokens` | **uncached** input only (Anthropic `usage` semantics). Do not use litellm's `prompt_tokens`, which includes cached tokens |
| `cache_read_tokens`, `cache_write_tokens`, `output_tokens` | from `usage` |
| `cost_usd` | from `prism.pricing.cost()`, not from the harness |
| `latency_s` | wall-clock time of the call |

Total prompt size = `input_tokens + cache_read_tokens + cache_write_tokens`.

**RunResult**: last line, written after SWE-bench evaluation.

| Field | Meaning |
|---|---|
| `solved` | from the SWE-bench evaluation harness |
| `exit_status` | `submitted`, `step_limit`, `cost_limit`, or `error`. Runs with `error` are re-run; failed tasks are not |
| `total_steps`, `n_model_calls`, `total_cost_usd`, `wall_clock_s` | totals |

### JSONL log

One file per run: `logs/<run_id>.jsonl`. Each line has `type` and `ts` (Unix time) plus the event's fields. Use `prism.events.to_json_line()` / `from_json_line()`; don't hand-build lines.

```jsonl
{"type": "run_start", "ts": 1793050000.0, "run_id": "mini-filter-sympy__sympy-20590-r1", "task_id": "sympy__sympy-20590", "harness": "mini", "config": "filter", "model": "claude-haiku-5-5", "repeat": 1, "harness_version": "2.4.6", "prism_commit": "abc1234", "schema_version": 1}
{"type": "tool_result", "ts": 1793050012.3, "run_id": "...", "step": 4, "call_idx": 0, "tool": "bash", "command": "python -m pytest tests/", "output": "...", "exit_code": 1, "error": null, "meta": {}}
{"type": "compression", "ts": 1793050012.3, "run_id": "...", "step": 4, "call_idx": 0, "policy": "filter", "chars_before": 36480, "chars_after": 2440, "tokens_before": 9120, "tokens_after": 610, "ms": 3.1, "offload_path": null}
{"type": "model_call", "ts": 1793050018.5, "run_id": "...", "step": 5, "model": "claude-haiku-5-5", "input_tokens": 1200, "cache_read_tokens": 30100, "cache_write_tokens": 900, "output_tokens": 310, "cost_usd": 0.000738, "latency_s": 6.2}
{"type": "result", "ts": 1793050400.0, "run_id": "...", "task_id": "sympy__sympy-20590", "harness": "mini", "config": "filter", "model": "claude-haiku-5-5", "repeat": 1, "solved": true, "exit_status": "submitted", "total_steps": 31, "n_model_calls": 31, "total_cost_usd": 0.021, "wall_clock_s": 400.0}
```

Logs keep the full raw output and are git-ignored.

### Interfaces

- `Store.put(run_id, step, call_idx, text) -> str` saves the full output and returns the path the model is told about. `Store.path_prefix` (e.g. `/tmp/prism`): any later command that references it counts as a follow-up read.
- `Logger.log(event) -> None` writes one JSONL line.
- Capabilities in v1: `shell_fs` (the agent can read files in its environment).

### Token counting

`prism.tokens.count(text) = ceil(len(text) / 4)`. It is deterministic and free, so it adds no latency to the agent loop. Report it as approximate. Offline replay calibrates the ratio once against Anthropic's `count_tokens` on ~50 saved outputs. Money is always measured from `ModelCallEvent`.

### Cost

`prism.pricing.cost()` prices every call from `usage` with one table for all harnesses. The tier is chosen by total prompt size: Claude Haiku 5.5 costs 5× more above 100K prompt tokens. So compression that keeps prompts under 100K saves more than the token count alone suggests. Set `MSWEA_COST_TRACKING=ignore_errors` so mini-SWE-agent doesn't fail when litellm lacks a price.

## Baseline fairness (mini-SWE-agent)

mini-SWE-agent's SWE-bench config already truncates: output under 10,000 chars is shown in full; longer output becomes the first 5,000 + last 5,000 chars plus a warning. It stays on in **every** config ("harness as shipped").

The adapter wraps the environment's `execute()`, so Prism sees raw output and runs **before** that template. A policy that gets output under 10,000 chars therefore skips the harness truncation entirely.

Prompt caching is on by default for Claude in mini-SWE-agent. Policies must be **deterministic** (same input, same output), and nothing may rewrite earlier messages, or the cache prefix breaks.
