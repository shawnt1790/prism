"""Policy A (filter) on real saved outputs: a long pytest log, a big cat, a grep."""

import re
from pathlib import Path

import pytest

from prism import tokens
from prism.policies.filter import FilterPolicy

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAMES = ["pytest_log.txt", "cat_big.txt", "grep.txt"]
HARNESS_TRUNCATION_CHARS = 10_000   # mini-SWE-agent's own head/tail cut


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def counts(filtered: str) -> tuple[int, int, int]:
    """(shown, total, dropped) from the header, plus a check that gap markers add up."""
    shown, total, dropped = map(int, re.search(
        r"showing (\d+) of (\d+) lines.*; (\d+) lines dropped", filtered).groups())
    gaps = sum(int(n) for n in re.findall(r"^\[\.\.\. (\d+) lines dropped \.\.\.\]$", filtered, re.M))
    assert gaps == dropped
    return shown, total, dropped


def test_short_output_passes_through_unchanged():
    text = "collected 3 items\n\ntest_a.py ...  [100%]\n"
    assert FilterPolicy().filter(text) == text


@pytest.mark.parametrize("name", NAMES)
def test_fixtures_shrink_below_threshold(name):
    policy = FilterPolicy()
    raw = fixture(name)
    out = policy.filter(raw)
    assert tokens.count(raw) > policy.threshold_tokens
    assert tokens.count(out) < tokens.count(raw) / 2
    assert len(out) < HARNESS_TRUNCATION_CHARS   # small enough that the harness won't cut it again


@pytest.mark.parametrize("name", NAMES)
def test_head_and_tail_are_kept(name):
    policy = FilterPolicy(head_lines=10, tail_lines=10)
    raw_lines = fixture(name).rstrip("\n").split("\n")
    out = policy.filter(fixture(name))
    out_lines = out.rstrip("\n").split("\n")
    assert out_lines[1:6] == [policy._cut(l) for l in raw_lines[:5]]   # line 0 is the header
    assert out_lines[-5:] == [policy._cut(l) for l in raw_lines[-5:]]


@pytest.mark.parametrize("name", NAMES)
def test_marker_counts_add_up(name):
    raw = fixture(name)
    shown, total, dropped = counts(FilterPolicy().filter(raw))
    assert total == len(raw.rstrip("\n").split("\n"))
    assert shown + dropped == total and dropped > 0


def test_every_failure_survives_in_pytest_log():
    out = FilterPolicy().filter(fixture("pytest_log.txt"))
    for needle in ["AssertionError: square of 37", "AssertionError: square of 151",
                   "AssertionError: square of 288", "ZeroDivisionError: division by zero",
                   "AssertionError: legacy_mode still enabled",
                   "5 failed, 399 passed"]:
        assert needle in out, needle
    for failed in re.findall(r"^FAILED .*$", fixture("pytest_log.txt"), re.M):
        assert failed in out


def test_repeated_lines_collapse():
    out = FilterPolicy().filter(fixture("pytest_log.txt"))
    warning = "DeprecationWarning: legacy_mode is deprecated and will be removed"
    assert out.split("\n").count(warning) == 1     # the printed lines (the source line also names it)
    assert "[line above repeated 299 more times]" in out


def test_collapse_counts_lines_correctly():
    text = "\n".join(["start"] + ["same"] * 1000 + ["end"] * 2) + "\n"
    out = FilterPolicy(threshold_tokens=10).filter(text)
    assert "[line above repeated 999 more times]" in out
    shown, total, dropped = counts(out)
    assert (shown, total, dropped) == (1003, 1003, 0)


def test_long_lines_are_cut():
    text = "x" * 50_000
    out = FilterPolicy(max_line_chars=300).filter(text)
    assert "[+49700 chars]" in out and len(out) < 1_000


def test_kept_match_cap():
    text = "\n".join(f"line {i} ERROR" for i in range(5_000))
    out = FilterPolicy(head_lines=5, tail_lines=5, max_kept_lines=20, context_lines=0).filter(text)
    shown, total, dropped = counts(out)
    assert shown == 30 and total == 5_000


def test_deterministic():
    raw = fixture("pytest_log.txt")
    assert FilterPolicy().filter(raw) == FilterPolicy().filter(raw)


def test_trailing_newline_preserved():
    raw = fixture("grep.txt")
    assert raw.endswith("\n") and FilterPolicy().filter(raw).endswith("\n")
