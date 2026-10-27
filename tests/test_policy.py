"""Registry, capability check, the compress() wrapper, and Policy A inside PrismAgent."""

from pathlib import Path

import pytest
import yaml

from prism import capabilities, tokens
from prism.events import CompressionRecord, ToolResultEvent
from prism.policies.filter import FilterPolicy
from prism.policy import REGISTRY, build_policy, compress, make_compress, register

ROOT = Path(__file__).resolve().parent.parent


class ListLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


def tool_event(output: str, step: int = 3, call_idx: int = 1) -> ToolResultEvent:
    return ToolResultEvent(run_id="r1", step=step, call_idx=call_idx, tool="bash",
                           command="python -m pytest", output=output, exit_code=1,
                           error=None, meta={})


def test_build_policy_from_repo_config():
    cfg = yaml.safe_load((ROOT / "configs" / "filter.yaml").read_text())
    policy = build_policy(cfg)
    assert isinstance(policy, FilterPolicy) and policy.name == "filter"
    assert policy.max_kept_lines == cfg["params"]["max_kept_lines"]


def test_build_policy_none_and_unknown():
    assert build_policy({"policy": "none"}).name == "none"
    with pytest.raises(ValueError, match="registered"):
        build_policy({"policy": "does-not-exist"})


def test_capability_mismatch_is_a_clear_error():
    @register("needs_fs_test")
    class NeedsFs:
        requires = {"shell_fs"}

        def apply(self, ev, store):
            return ev.output

    try:
        with pytest.raises(capabilities.CapabilityError, match="missing \\['shell_fs'\\]"):
            make_compress({"policy": "needs_fs_test"}, run_id="r1", logger=ListLogger(),
                          provides=set(), harness="toy")
        make_compress({"policy": "needs_fs_test"}, run_id="r1", logger=ListLogger(),
                      provides={"shell_fs"})
    finally:
        REGISTRY.pop("needs_fs_test")


def test_make_compress_logs_one_record_per_call():
    logger = ListLogger()
    fn = make_compress({"policy": "filter", "params": {"threshold_tokens": 10}},
                       run_id="r1", logger=logger, provides={"shell_fs"})
    raw = "\n".join(f"line {i}" for i in range(1_000))
    shown = fn(tool_event(raw))
    fn(tool_event("short"))

    assert len(logger.events) == 2 and all(isinstance(e, CompressionRecord) for e in logger.events)
    rec = logger.events[0]
    assert (rec.run_id, rec.step, rec.call_idx, rec.policy) == ("r1", 3, 1, "filter")
    assert (rec.chars_before, rec.chars_after) == (len(raw), len(shown))
    assert (rec.tokens_before, rec.tokens_after) == (tokens.count(raw), tokens.count(shown))
    assert rec.tokens_after < rec.tokens_before and rec.ms >= 0 and rec.offload_path is None
    assert logger.events[1].chars_before == logger.events[1].chars_after == 5


def test_compress_records_offload_path():
    class MemStore:
        path_prefix = "/tmp/prism"

        def put(self, run_id, step, call_idx, text):
            return f"{self.path_prefix}/{run_id}/{step}_{call_idx}.txt"

    class Offload:
        name, requires = "offload_test", {"shell_fs"}

        def apply(self, ev, store):
            return f"saved to {store.put(ev.run_id, ev.step, ev.call_idx, ev.output)}"

    logger = ListLogger()
    shown = compress(Offload(), tool_event("x" * 100), MemStore(), logger)
    assert logger.events[0].offload_path == "/tmp/prism/r1/3_1.txt"
    assert shown.endswith("/tmp/prism/r1/3_1.txt")


def test_filter_policy_inside_prism_agent(tmp_path):
    """Policy A end to end in mini-SWE-agent: scripted model, local shell, no Docker or API."""
    pytest.importorskip("minisweagent")
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel

    from adapters.mini_swe_agent import PROVIDES, PrismAgent
    from eval import metrics, run
    from prism.events import from_json_line
    from prism.log import JsonlLogger

    usage = {"prompt_tokens": 1500, "completion_tokens": 20,
             "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200}

    def reply(command):
        return {"role": "assistant", "content": command,
                "extra": {"actions": [{"command": command}], "response": {"usage": usage}}}

    cfg = yaml.safe_load((ROOT / "configs" / "filter.yaml").read_text())
    log_path = tmp_path / "run.jsonl"
    with JsonlLogger(log_path) as logger:
        compress_fn = run.get_compress(cfg, "test-run", logger)   # the runner's real path
        agent = PrismAgent(DeterministicModel(outputs=[
                               reply("seq 1 5000"),
                               reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]),
                           LocalEnvironment(), run_id="test-run", prism_logger=logger,
                           compress=compress_fn, model_name="claude-haiku-5-5",
                           system_template="test", instance_template="Task: {{task}}",
                           step_limit=10, cost_limit=0)
        agent.run("count to 5000")

    events = [from_json_line(line) for line in log_path.read_text().splitlines()]
    tool = next(e for e in events if isinstance(e, ToolResultEvent))
    rec = next(e for e in events if isinstance(e, CompressionRecord))
    assert tool.output.splitlines()[-1] == "5000"                 # log keeps the raw output
    assert (rec.step, rec.call_idx, rec.policy) == (tool.step, tool.call_idx, "filter")
    assert rec.tokens_after < rec.tokens_before

    observations = [str(m.get("content")) for m in agent.messages if m.get("role") == "user"][1:]
    assert any("[prism filter: showing" in o for o in observations)   # model saw the filtered text

    row = metrics.summarize_run(log_path)
    assert row["compressions"] == 1 and row["tokens_after"] < row["tokens_before"]
    assert "shell_fs" in PROVIDES
