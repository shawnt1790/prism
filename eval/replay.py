"""Replay recorded tool outputs through a policy offline (libCacheSim-style).

Usage:
    python -m eval.replay                                   # configs/filter.yaml over logs/*.jsonl
    python -m eval.replay --configs configs/filter.yaml configs/none.yaml --out logs/replay.csv

Reads every ToolResultEvent from baseline (`none`) runs, runs each policy on it through
prism.policy.compress(), and writes one row per (event, policy). Cheap: no model calls.
Gives compression ratio by command type and policy overhead; it can't show what the agent
would have done next, which is what the live runs are for.
"""

import argparse
import csv
import re
import shlex
from collections import defaultdict
from pathlib import Path

import yaml

from prism.events import RunStart, ToolResultEvent, from_json_line
from prism.policy import build_policy, compress

HARNESS_TRUNCATION_CHARS = 10_000   # mini-SWE-agent shows longer outputs as head + tail

COLUMNS = ["run_id", "task_id", "step", "call_idx", "command_type", "policy",
           "chars_before", "chars_after", "tokens_before", "tokens_after", "ms",
           "harness_would_truncate"]

TEST_RE = re.compile(r"\b(pytest|py\.test|runtests\.py|bin/test|tox|unittest|nosetests)\b")
EDIT_RE = re.compile(r"(\bsed\s+-i\b|\bcat\s*>|<<\s*['\"]?\w+|\btee\b|\bpatch\b|\bapply_patch\b)")
SEARCH_CMDS = {"grep", "egrep", "rg", "ag", "find", "ls", "tree", "locate"}
READ_CMDS = {"cat", "head", "tail", "nl", "less", "more", "sed", "awk", "wc"}


class _ListLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


def classify_command(command: str) -> str:
    """test | edit | git | search | read | other. Looks at every part of a compound command."""
    if TEST_RE.search(command):
        return "test"
    if EDIT_RE.search(command):
        return "edit"
    firsts = []
    for part in re.split(r"&&|\|\||;|\|", command):
        try:
            words = shlex.split(part)
        except ValueError:
            words = part.split()
        while words and ("=" in words[0] or words[0] in {"timeout", "sudo", "env"}):
            words = words[2:] if words[0] == "timeout" else words[1:]
        if words:
            firsts.append(words[0].rsplit("/", 1)[-1])
    if "git" in firsts:
        return "git"
    if SEARCH_CMDS.intersection(firsts):
        return "search"
    if READ_CMDS.intersection(firsts):
        return "read"
    return "other"


def load_baseline_events(logs_dir: Path) -> list[tuple[str, ToolResultEvent]]:
    """(task_id, event) for every tool result in `none` runs."""
    out = []
    for path in sorted(logs_dir.glob("*.jsonl")):
        events = [from_json_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        start = next((e for e in events if isinstance(e, RunStart)), None)
        if start is None or start.config != "none":
            continue
        out.extend((start.task_id, e) for e in events if isinstance(e, ToolResultEvent))
    return out


def replay(baseline: list[tuple[str, ToolResultEvent]], policy_cfgs: list[dict]) -> list[dict]:
    rows = []
    for cfg in policy_cfgs:
        policy = build_policy(cfg)
        logger = _ListLogger()
        for task_id, ev in baseline:
            compress(policy, ev, store=_NoOffloadStore(), logger=logger)
            rec = logger.events[-1]
            rows.append({
                "run_id": ev.run_id, "task_id": task_id, "step": ev.step, "call_idx": ev.call_idx,
                "command_type": classify_command(ev.command), "policy": rec.policy,
                "chars_before": rec.chars_before, "chars_after": rec.chars_after,
                "tokens_before": rec.tokens_before, "tokens_after": rec.tokens_after, "ms": rec.ms,
                "harness_would_truncate": rec.chars_before > HARNESS_TRUNCATION_CHARS,
            })
    return rows


class _NoOffloadStore:
    path_prefix = "/tmp/prism"

    def put(self, run_id, step, call_idx, text):
        # Offline there is no container; report where the file would have gone.
        return f"{self.path_prefix}/{run_id}/{step}_{call_idx}.txt"


def summarize(rows: list[dict]) -> list[dict]:
    """Per (policy, command_type), plus an "all" row per policy."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["command_type"])].append(r)
        groups[(r["policy"], "all")].append(r)
    out = []
    for (policy, ctype), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] == "all", kv[0][1])):
        before = sum(r["tokens_before"] for r in rs)
        after = sum(r["tokens_after"] for r in rs)
        out.append({
            "policy": policy, "command_type": ctype, "events": len(rs),
            "tokens_before": before, "tokens_after": after,
            "ratio": round(after / before, 3) if before else None,
            "harness_truncated": sum(r["harness_would_truncate"] for r in rs),
            "mean_ms": round(sum(r["ms"] for r in rs) / len(rs), 3),
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--configs", nargs="+", default=["configs/filter.yaml"])
    p.add_argument("--out", default=None, help="CSV path (default: <logs-dir>/replay.csv)")
    args = p.parse_args(argv)

    logs_dir = Path(args.logs_dir)
    baseline = load_baseline_events(logs_dir)
    if not baseline:
        raise SystemExit(f"No tool results from `none` runs in {logs_dir}/ (run the baseline first)")
    policy_cfgs = [yaml.safe_load(Path(c).read_text()) for c in args.configs]
    rows = replay(baseline, policy_cfgs)

    out = Path(args.out) if args.out else logs_dir / "replay.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    headers = list(summary[0].keys())
    print("  ".join(f"{h:>17}" for h in headers))
    for s in summary:
        print("  ".join(f"{'-' if s[h] is None else s[h]!s:>17}" for h in headers))
    print(f"\n{len(baseline)} tool results from `none` runs; per-event table: {out}")


if __name__ == "__main__":
    main()
