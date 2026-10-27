# Test fixtures: real tool outputs

Saved command outputs for Policy A's unit tests and the replay test. They are real outputs, generated locally (Python 3.10). Replace or extend them with outputs from baseline logs once those exist.

| File | Command | Size |
|---|---|---|
| `pytest_log.txt` | `python -m pytest -v test_numbers.py` over the suite below: 404 tests, 5 failures (3 assertion errors, a `ZeroDivisionError`, and a failure whose captured stdout repeats one warning 300 times) | 785 lines, 55 KB |
| `cat_big.txt` | `cat argparse.py` from the Python 3.10 stdlib | 2,577 lines, 98 KB |
| `grep.txt` | `grep -rn "def " json email`, run inside the Python 3.10 stdlib folder | 543 lines, 33 KB |

The `rootdir` and `platform` lines in `pytest_log.txt` were rewritten to `/testbed` and Linux, so no local paths are stored.

The suite behind `pytest_log.txt`:

```python
import pytest


def parse_ratio(text):
    num, den = text.split("/")
    return int(num) / int(den)


@pytest.mark.parametrize("n", range(400))
def test_square_is_nonnegative(n):
    if n in (37, 151, 288):
        assert n * n < 0, f"square of {n} should be negative?"
    assert n * n >= 0


@pytest.mark.parametrize("text", ["1/2", "3/4", "5/0"])
def test_parse_ratio(text):
    assert parse_ratio(text) <= 1


def test_deprecated_api():
    for _ in range(300):
        print("DeprecationWarning: legacy_mode is deprecated and will be removed")
    assert False, "legacy_mode still enabled"
```
