"""PrismAgent runs a scripted model against a local shell (no Docker, no API key needed)."""

import pytest

pytest.importorskip("minisweagent")

from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.test_models import DeterministicModel  # noqa: E402

from adapters.mini_swe_agent import PrismAgent  # noqa: E402
from prism.events import ModelCallEvent, ToolResultEvent, from_json_line  # noqa: E402
from prism.log import JsonlLogger  # noqa: E402

AGENT_CONFIG = {
    "system_template": "You are a test agent.",
    "instance_template": "Task: {{task}}",
    "step_limit": 10,
    "cost_limit": 0,
}

USAGE = {"prompt_tokens": 1500, "completion_tokens": 20,
         "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 200}


def scripted(command: str) -> dict:
    """One model reply that runs `command`, with Anthropic-style usage attached."""
    return {"role": "assistant", "content": f"running {command}",
            "extra": {"actions": [{"command": command}], "response": {"usage": USAGE}}}


def run_agent(tmp_path, compress=None):
    model = DeterministicModel(outputs=[
        scripted("seq 1 500"),
        scripted("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ])
    log_path = tmp_path / "run.jsonl"
    kwargs = {} if compress is None else {"compress": compress}
    with JsonlLogger(log_path) as logger:
        agent = PrismAgent(model, LocalEnvironment(), run_id="test-run", prism_logger=logger,
                           model_name="claude-haiku-5-5", **kwargs, **AGENT_CONFIG)
        result = agent.run("count to 500")
    events = [from_json_line(line) for line in log_path.read_text().splitlines()]
    return agent, result, events


def test_passthrough_logs_events_and_changes_nothing(tmp_path):
    agent, result, events = run_agent(tmp_path)

    assert result["exit_status"] == "Submitted"
    tools = [e for e in events if isinstance(e, ToolResultEvent)]
    calls = [e for e in events if isinstance(e, ModelCallEvent)]
    assert len(calls) == 2                       # one per model reply
    assert len(tools) == 1                       # submit command is not a tool result
    assert tools[0].output.splitlines()[-1] == "500"
    assert (tools[0].step, tools[0].call_idx, tools[0].exit_code) == (1, 0, 0)
    # uncached input = prompt_tokens - cache_read - cache_write
    assert (calls[0].input_tokens, calls[0].cache_read_tokens, calls[0].cache_write_tokens) == (300, 1000, 200)
    assert calls[0].cost_usd is not None and calls[0].cost_usd > 0
    # with pass-through, the model saw the full raw output
    assert any("500" in str(m.get("content")) for m in agent.messages if m.get("role") == "user")


def test_compress_changes_what_the_model_sees_but_not_the_log(tmp_path):
    agent, _, events = run_agent(tmp_path, compress=lambda ev: "[compressed]")

    tool = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tool.output.splitlines()[-1] == "500"          # log keeps the raw output
    observations = [str(m.get("content")) for m in agent.messages if m.get("role") == "user"][1:]
    assert any("[compressed]" in o for o in observations)  # model saw the compressed text
    assert not any("\n500\n" in o for o in observations)