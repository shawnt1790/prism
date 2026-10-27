"""eval.run and eval.metrics end to end with a scripted model and a local shell (no Docker, no API)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("minisweagent")

from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.test_models import DeterministicModel  # noqa: E402

from eval import metrics, run  # noqa: E402
from prism.events import RunResult, RunStart, from_json_line  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
USAGE = {"prompt_tokens": 1500, "completion_tokens": 20,
         "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200}
MINI_CONFIG = {
    "agent": {"system_template": "test", "instance_template": "Task: {{task}}", "step_limit": 10},
    "model": {"model_name": "anthropic/claude-haiku-5-5"},
}
INSTANCE = {"instance_id": "demo__demo-1", "problem_statement": "count to 500"}


def reply(command):
    return {"role": "assistant", "content": command,
            "extra": {"actions": [{"command": command}], "response": {"usage": USAGE}}}


def do_run(tmp_path, commands, cost_limit=0.0):
    model = DeterministicModel(outputs=[reply(c) for c in commands])
    return run.run_one(instance=INSTANCE, config_name="none", policy_cfg={"policy": "none"},
                       mini_config=MINI_CONFIG, repeat=1, cost_limit=cost_limit,
                       logs_dir=tmp_path, env=LocalEnvironment(), model=model)


def test_read_tasks_ignores_comments(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("# header\n\ndjango__django-1   # note\nsympy__sympy-2\n")
    assert run.read_tasks(f) == ["django__django-1", "sympy__sympy-2"]


def test_repo_configs_load():
    base, policy_cfg, name = run.load_configs(ROOT / "configs" / "none.yaml")
    assert name == "none" and policy_cfg["policy"] == "none" and "model" in base


def test_run_writes_contract_log_and_preds(tmp_path):
    s = do_run(tmp_path, ["seq 1 500", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho my-patch"])
    assert s["exit_status"] == "submitted"

    events = [from_json_line(l) for l in (tmp_path / f"{s['run_id']}.jsonl").read_text().splitlines()]
    assert isinstance(events[0], RunStart) and isinstance(events[-1], RunResult)
    assert events[-1].solved is None and events[-1].n_model_calls == 2
    assert events[-1].total_cost_usd and events[-1].total_cost_usd > 0
    assert (tmp_path / f"{s['run_id']}.traj.json").exists()
    preds = json.loads((tmp_path / "preds" / "none-r1.json").read_text())
    assert "my-patch" in preds["demo__demo-1"]["model_patch"]

    row = metrics.summarize_run(tmp_path / f"{s['run_id']}.jsonl")
    assert row["complete"] and row["tool_calls"] == 1 and row["model_calls"] == 2
    assert row["input_tokens"] == 600 and row["prompt_tokens_total"] == 2 * 1500
    assert row["tool_output_tokens_raw"] > 0 and row["follow_up_reads"] == 0
    metrics.main(["--logs-dir", str(tmp_path)])
    assert (tmp_path / "results.csv").exists()


def test_cost_cap_stops_the_run(tmp_path):
    s = do_run(tmp_path, ["echo a", "echo b", "echo c"], cost_limit=1e-9)
    assert s["exit_status"] == "cost_limit" and s["steps"] == 1