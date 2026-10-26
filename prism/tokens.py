"""Token counting used for every tokens_before/tokens_after in CompressionRecord.

v1 is a deterministic chars/4 approximation: free and adds no latency to the agent loop.
Real spend is always measured from ModelCallEvent usage. Offline replay calibrates this
ratio against Anthropic's count_tokens on saved outputs.
"""

import math

CHARS_PER_TOKEN = 4


def count(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)
