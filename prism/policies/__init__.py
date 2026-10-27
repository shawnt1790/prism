"""Compression policies. Each policy is one file here plus one YAML config and one unit test.

Importing this package registers every built-in policy (see prism.policy.register).
"""

from prism.policies import filter, passthrough  # noqa: F401
