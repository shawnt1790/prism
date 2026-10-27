"""Capability check: compares policy.requires with adapter.provides before a run.

A mismatch stops the run with a clear message instead of failing silently mid-run.
Capabilities in v1: "shell_fs" (the agent can read files in its environment).
"""


class CapabilityError(Exception):
    pass


def check(policy, provides: set[str], harness: str) -> None:
    missing = set(policy.requires) - set(provides)
    if missing:
        raise CapabilityError(
            f"policy {policy.name!r} requires {sorted(policy.requires)}; "
            f"harness {harness!r} provides {sorted(provides)}; missing {sorted(missing)}"
        )
