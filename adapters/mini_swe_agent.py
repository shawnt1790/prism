"""Adapter for mini-SWE-agent (v2.4.6).

Hook point (see docs/design.md, "mini-SWE-agent hook"):
  minisweagent/agents/default.py, DefaultAgent.execute_actions
    line 156: outputs = [self.env.execute(action) ...]        <- raw tool output
    line 157: self.model.format_observation_messages(...)       <- harness formats it
                                                                   (10,000-char truncation)
  Prism runs between those two lines.

PrismAgent subclasses DefaultAgent, so mini-SWE-agent itself is not modified.
It overrides two methods:
  - query():           logs one ModelCallEvent per model call (tokens, cost, latency)
  - execute_actions(): logs one ToolResultEvent per command, calls compress(),
                       and puts the compressed text back before the harness formats it

Usage:
    agent = PrismAgent(model, env, run_id="...", prism_logger=logger,
                       compress=my_compress_fn, **agent_config)
    agent.run(task)

`compress` takes a ToolResultEvent and returns the text the model should see.
Until prism.compress() exists, the default is a pass-through (returns the raw output),
so the agent behaves exactly like the unmodified harness.
"""

import time
import warnings
from collections.abc import Callable

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError

from prism import pricing
from prism.events import ModelCallEvent, ToolResultEvent

HARNESS_NAME = "mini"
PROVIDES = {"shell_fs"}  # the agent can read files in its container with cat/grep


def passthrough(event: ToolResultEvent) -> str:
    """The `none` behaviour: show the model the raw output, unchanged."""
    return event.output


class PrismAgent(DefaultAgent):
    def __init__(self, model, env, *, run_id: str, prism_logger,
                 compress: Callable[[ToolResultEvent], str] = passthrough,
                 model_name: str | None = None, **kwargs):
        super().__init__(model, env, **kwargs)
        # Prefixed names so nothing clashes with DefaultAgent's own attributes
        # (DefaultAgent already uses self.logger for Python logging).
        self.prism_run_id = run_id
        self.prism_logger = prism_logger
        self.prism_compress = compress
        self.prism_model_name = model_name or getattr(model.config, "model_name", "unknown")
        self._warned_unpriced = False

    # ---------- model calls ----------

    def query(self) -> dict:
        start = time.time()
        try:
            message = super().query()
        except FormatError as e:
            # The call was billed even though the reply couldn't be parsed; log it anyway.
            self._log_model_call(e.messages[0], time.time() - start)
            raise
        self._log_model_call(message, time.time() - start)
        return message

    def _log_model_call(self, message: dict, latency_s: float) -> None:
        response = message.get("extra", {}).get("response")
        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}

        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        # litellm's prompt_tokens includes cached tokens; the contract wants uncached input only.
        prompt_total = int(usage.get("prompt_tokens") or 0)
        input_tokens = max(prompt_total - cache_read - cache_write, 0)

        try:
            cost_usd = pricing.cost(self.prism_model_name, input_tokens, cache_read,
                                    cache_write, output_tokens)
        except KeyError:
            # Model missing from prism/pricing.py. Tokens are still logged, so cost
            # can be recomputed later once the price table is updated.
            cost_usd = None
            if not self._warned_unpriced:
                warnings.warn(f"No price for {self.prism_model_name!r} in prism/pricing.py; "
                              "logging cost_usd=None.")
                self._warned_unpriced = True

        self.prism_logger.log(ModelCallEvent(
            run_id=self.prism_run_id,
            step=self.n_calls,
            model=self.prism_model_name,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_s=round(latency_s, 3),
        ))

    # ---------- tool calls (the Prism hook) ----------

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for call_idx, action in enumerate(actions):
            # Raises Submitted for the final submit command; nothing to compress then.
            out = self.env.execute(action)

            raw = out.get("output", "") or ""
            error = out.get("exception_info") or None
            event = ToolResultEvent(
                run_id=self.prism_run_id,
                step=self.n_calls,
                call_idx=call_idx,
                tool="bash",
                command=action.get("command", ""),
                output=raw,
                exit_code=None if error else out.get("returncode"),
                error=error,
                meta={},
            )
            self.prism_logger.log(event)

            shown = self.prism_compress(event)
            if shown != raw:
                out = {**out, "output": shown}
            outputs.append(out)

        # Same as DefaultAgent.execute_actions line 157: the harness formats (and truncates) the result.
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )