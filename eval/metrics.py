"""Logs -> results table.

Usage:
    python -m eval.metrics                 # all logs/*.jsonl
    python -m eval.metrics --logs-dir logs --out logs/results.csv

Writes one row per run to a CSV and prints a summary per config.
Token sizes of tool outputs use prism.tokens.count() (approximate); money comes from model calls.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from prism import tokens
from prism.events import CompressionRecord, ModelCallEvent, RunResult, RunStart, ToolResultEvent, from_json_line

FOLLOW_UP_PREFIX = "/tmp/prism"   # Store.path_prefix: commands mentioning it are follow-up reads

COLUMNS = ["run_id", "task_id", "config", "model", "repeat", "exit_status", "solved",
           "steps", "model_calls", "input_tokens", "cache_read_tokens", "cache_write_tokens",
           "output_tokens", "prompt_tokens_total", "cost_usd", "wall_clock_s",
           "tool_calls", "tool_output_tokens_raw", "compressions", "tokens_before", "tokens_after",
           "follow_up_reads", "complete"]


def summarize_run(path: Path) -> dict:
    events = [from_json_line(line) for line in path.read_text().splitlines() if line.strip()]
    start = next((e for e in events if isinstance(e, RunStart)), None)
    result = next((e for e in events if isinstance(e, RunResult)), None)
    calls = [e for e in events if isinstance(e, ModelCallEvent)]
    tools = [e for e in events if isinstance(e, ToolResultEvent)]
    comps = [e for e in events if isinstance(e, CompressionRecord)]

    costs = [c.cost_usd for c in calls]
    row = {
        "run_id": (start or result).run_id if (start or result) else path.stem,
        "task_id": start.task_id if start else "",
        "config": start.config if start else "",
        "model": start.model if start else "",
        "repeat": start.repeat if start else "",
        "exit_status": result.exit_status if result else "incomplete",
        "solved": "" if result is None or result.solved is None else result.solved,
        "steps": result.total_steps if result else len(calls),
        "model_calls": len(calls),
        "input_tokens": sum(c.input_tokens for c in calls),
        "cache_read_tokens": sum(c.cache_read_tokens for c in calls),
        "cache_write_tokens": sum(c.cache_write_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "cost_usd": round(sum(costs), 6) if costs and None not in costs else "",
        "wall_clock_s": result.wall_clock_s if result else "",
        "tool_calls": len(tools),
        "tool_output_tokens_raw": sum(tokens.count(t.output) for t in tools),
        "compressions": len(comps),
        "tokens_before": sum(c.tokens_before for c in comps),
        "tokens_after": sum(c.tokens_after for c in comps),
        "follow_up_reads": sum(FOLLOW_UP_PREFIX in t.command for t in tools),
        "complete": result is not None,
    }
    row["prompt_tokens_total"] = row["input_tokens"] + row["cache_read_tokens"] + row["cache_write_tokens"]
    return row


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(values) / len(values) if values else None


def summarize_by_config(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["config"]].append(r)
    out = []
    for config, rs in sorted(groups.items()):
        solved_known = [r for r in rs if r["solved"] != ""]
        out.append({
            "config": config,
            "runs": len(rs),
            "submitted": sum(r["exit_status"] == "submitted" for r in rs),
            "solved": f"{sum(r['solved'] is True for r in solved_known)}/{len(solved_known)}" if solved_known else "n/a",
            "mean_steps": _mean([r["steps"] for r in rs]),
            "mean_prompt_tokens": _mean([r["prompt_tokens_total"] for r in rs]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rs]),
            "mean_cost_usd": _mean([r["cost_usd"] for r in rs]),
            "mean_tool_output_tokens": _mean([r["tool_output_tokens_raw"] for r in rs]),
            "follow_up_reads": sum(r["follow_up_reads"] for r in rs),
        })
    return out


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if v < 1 else f"{v:,.0f}"
    return str(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--out", default=None, help="CSV path (default: <logs-dir>/results.csv)")
    args = p.parse_args(argv)

    logs_dir = Path(args.logs_dir)
    paths = sorted(logs_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No .jsonl logs in {logs_dir}/")
    rows = [summarize_run(p) for p in paths]

    out = Path(args.out) if args.out else logs_dir / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_by_config(rows)
    headers = list(summary[0].keys())
    print("  ".join(f"{h:>22}" for h in headers))
    for s in summary:
        print("  ".join(f"{_fmt(s[h]):>22}" for h in headers))
    incomplete = [r["run_id"] for r in rows if not r["complete"]]
    if incomplete:
        print(f"\nIncomplete runs (no result line): {incomplete}")
    print(f"\nPer-run table: {out}")


if __name__ == "__main__":
    main()