import json

import pytest

from prism import pricing, tokens
from prism.events import (
    CompressionRecord,
    ModelCallEvent,
    RunResult,
    RunStart,
    ToolResultEvent,
    from_json_line,
    to_json_line,
)

EVENTS = [
    RunStart("r1", "sympy__sympy-20590", "mini", "filter", "claude-haiku-5-5", 1, "2.4.6", "abc1234"),
    ToolResultEvent("r1", 4, 0, "bash", "pytest", "line\n" * 3, 1, None, {"cwd": "/testbed"}),
    ToolResultEvent("r1", 5, 1, "bash", "sleep 99", "", None, "timeout", {}),
    CompressionRecord("r1", 4, 0, "filter", 15, 10, 4, 3, 0.5, None),
    ModelCallEvent("r1", 5, "claude-haiku-5-5", 1200, 30100, 900, 310, 0.0007, 6.2),
    RunResult("r1", "sympy__sympy-20590", "mini", "filter", "claude-haiku-5-5", 1, True,
              "submitted", 31, 31, 0.021, 400.0),
]


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: type(e).__name__)
def test_round_trip(event):
    line = to_json_line(event, ts=123.0)
    assert "\n" not in line
    record = json.loads(line)
    assert record["ts"] == 123.0 and "type" in record
    assert from_json_line(line) == event


def test_token_count_is_deterministic():
    assert tokens.count("") == 0
    assert tokens.count("abcd") == 1
    assert tokens.count("abcde") == 2


def test_pricing_tier_switches_above_100k():
    under = pricing.cost("claude-haiku-5-5", 100_000, 0, 0, 0)
    over = pricing.cost("anthropic/claude-haiku-5-5", 100_001, 0, 0, 0)
    assert under == pytest.approx(0.01)
    assert over == pytest.approx(100_001 * 0.50 / 1_000_000)


def test_pricing_tier_counts_cached_tokens():
    # 50K uncached + 60K cache read = 110K prompt -> long-context tier
    c = pricing.cost("claude-haiku-5-5", 50_000, 60_000, 0, 0)
    assert c == pytest.approx((50_000 * 0.50 + 60_000 * 0.05) / 1_000_000)
