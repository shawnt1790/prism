"""Price table and cost function. One cost definition for every harness.

Prices are USD per million tokens. Cache read/write rates for Haiku 5.5 assume the usual
0.1x / 1.25x of input; confirm against the pricing page before the Week 11 budget.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    input: float
    output: float
    cache_read: float
    cache_write: float


# model -> list of (max prompt tokens for this tier, rates), checked in order
PRICES: dict[str, list[tuple[float, Rates]]] = {
    "claude-haiku-5-5": [
        (100_000, Rates(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125)),
        (float("inf"), Rates(input=0.50, output=2.50, cache_read=0.05, cache_write=0.625)),
    ],
    "claude-sonnet-5-5": [
        (float("inf"), Rates(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50)),
    ],
}


def _base_model(model: str) -> str:
    # accept litellm-style names such as "anthropic/claude-haiku-5-5"
    return model.split("/")[-1]


def prompt_tokens(input_tokens: int, cache_read_tokens: int, cache_write_tokens: int) -> int:
    """Total prompt size; Anthropic's input_tokens excludes cached tokens."""
    return input_tokens + cache_read_tokens + cache_write_tokens


def cost(model: str, input_tokens: int, cache_read_tokens: int,
         cache_write_tokens: int, output_tokens: int) -> float:
    """Cost of one model call in USD. The tier is chosen by total prompt size."""
    prompt = prompt_tokens(input_tokens, cache_read_tokens, cache_write_tokens)
    for max_prompt, rates in PRICES[_base_model(model)]:
        if prompt <= max_prompt:
            break
    return (input_tokens * rates.input
            + cache_read_tokens * rates.cache_read
            + cache_write_tokens * rates.cache_write
            + output_tokens * rates.output) / 1_000_000
