"""eval.replay over a synthetic `none` log built from the saved fixture outputs."""

from pathlib import Path

import pytest

from eval import replay
from prism.events import RunResult, RunStart, ToolResultEvent, to_json_line

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
COMMANDS = [("cd /testbed && python -m pytest -v tests/", "pytest_log.txt"),
            ("cat lib/argparse.py", "cat_big.txt"),
            ("grep -rn 'def ' json email", "grep.txt"),
            ("python reproduce.py", None)]


def write_log(path: Path, config: str):
    lines = [to_json_line(RunStart("r-" + config, "demo__demo-1", "mini", config,
                                   "claude-haiku-5-5", 1, "2.4.6", "abc"))]
    for step, (cmd, fixture) in enumerate(COMMANDS, start=1):
        output = (FIXTURES / fixture).read_text(encoding="utf-8") if fixture else "README.md\nsetup.py\n"
        lines.append(to_json_line(ToolResultEvent("r-" + config, step, 0, "bash", cmd, output, 0, None, {})))
    lines.append(to_json_line(RunResult("r-" + config, "demo__demo-1", "mini", config, "claude-haiku-5-5",
                                        1, None, "submitted", 4, 4, 0.01, 10.0)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("command, expected", [
    ("cd /testbed && python -m pytest -x tests/test_a.py", "test"),
    ("./tests/runtests.py migrations", "test"),
    ("bin/test sympy/core", "test"),
    ("sed -i 's/a/b/' x.py", "edit"),
    ("cat > fix.py <<'EOF'\nprint(1)\nEOF", "edit"),
    ("git diff", "git"),
    ("grep -rn foo . | head -20", "search"),
    ("find . -name '*.py'", "search"),
    ("cat setup.py", "read"),
    ("sed -n '10,40p' a.py", "read"),
    ("python reproduce.py", "other"),
])
def test_classify_command(command, expected):
    assert replay.classify_command(command) == expected


def test_replay_only_uses_baseline_runs(tmp_path):
    write_log(tmp_path / "base.jsonl", "none")
    write_log(tmp_path / "filtered.jsonl", "filter")
    baseline = replay.load_baseline_events(tmp_path)
    assert len(baseline) == len(COMMANDS) and all(t == "demo__demo-1" for t, _ in baseline)


def test_replay_end_to_end(tmp_path, capsys):
    write_log(tmp_path / "base.jsonl", "none")
    replay.main(["--logs-dir", str(tmp_path),
                 "--configs", str(ROOT / "configs" / "filter.yaml"), str(ROOT / "configs" / "none.yaml")])
    assert (tmp_path / "replay.csv").exists()

    rows = replay.replay(replay.load_baseline_events(tmp_path),
                         [{"policy": "filter"}, {"policy": "none"}])
    by = {(r["policy"], r["command_type"]): r for r in rows}
    for ctype in ["test", "read", "search"]:
        assert by[("filter", ctype)]["tokens_after"] < by[("filter", ctype)]["tokens_before"] / 2
        assert by[("filter", ctype)]["harness_would_truncate"]
        assert by[("none", ctype)]["tokens_after"] == by[("none", ctype)]["tokens_before"]
    assert by[("filter", "other")]["chars_after"] == by[("filter", "other")]["chars_before"]  # short `ls`

    summary = {(s["policy"], s["command_type"]): s for s in replay.summarize(rows)}
    assert summary[("filter", "all")]["events"] == len(COMMANDS)
    assert summary[("filter", "all")]["ratio"] < 0.5 and summary[("none", "all")]["ratio"] == 1.0
    assert "per-event table" in capsys.readouterr().out
