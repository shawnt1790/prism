"""Policy A: filtering (head/tail lines, keep error/fail/traceback/assert lines, collapse repeats).

Lossy: dropped lines are gone for good, but the model reads nothing extra. Deterministic, so the
same output always compresses to the same text and the prompt cache stays valid.

Steps, for outputs over threshold_tokens (shorter ones pass through unchanged):
  1. Collapse runs of >= 3 identical consecutive lines into the line plus a repeat marker.
  2. Keep the first head_lines and last tail_lines lines.
  3. In between, keep lines matching keep_patterns plus context_lines after each,
     up to max_kept_lines in total.
  4. Cut any kept line longer than max_line_chars.
  5. Replace each gap with "[... N lines dropped ...]" and put a summary header on top.
"""

import re

from prism import tokens
from prism.policy import register

DEFAULT_PATTERNS = ["error", "fail", "traceback", "assert"]
MIN_REPEAT = 3


@register("filter")
class FilterPolicy:
    requires: set[str] = set()

    def __init__(self, threshold_tokens: int = 2000, head_lines: int = 50, tail_lines: int = 50,
                 keep_patterns: list[str] | None = None, context_lines: int = 2,
                 max_kept_lines: int = 100, max_line_chars: int = 300):
        self.threshold_tokens = threshold_tokens
        self.head_lines = head_lines
        self.tail_lines = tail_lines
        self.keep_patterns = list(keep_patterns or DEFAULT_PATTERNS)
        self.context_lines = context_lines
        self.max_kept_lines = max_kept_lines
        self.max_line_chars = max_line_chars
        self._pattern = re.compile("|".join(self.keep_patterns), re.IGNORECASE)

    def apply(self, ev, store) -> str:
        return self.filter(ev.output)

    def filter(self, text: str) -> str:
        if tokens.count(text) <= self.threshold_tokens:
            return text

        trailing_newline = text.endswith("\n")
        lines = text.split("\n")
        if trailing_newline:
            lines.pop()
        total = len(lines)

        entries = self._collapse(lines)          # [(text, original line count, is_marker)]
        keep = self._select(entries)

        out, dropped, gap = [], 0, 0
        for i, (line, count, is_marker) in enumerate(entries):
            if keep[i]:
                if gap:
                    out.append(f"[... {gap} lines dropped ...]")
                    gap = 0
                out.append(line if is_marker else self._cut(line))
            else:
                gap += count
                dropped += count
        if gap:
            out.append(f"[... {gap} lines dropped ...]")

        header = (f"[prism filter: showing {total - dropped} of {total} lines: first {self.head_lines}, "
                  f"last {self.tail_lines}, and lines matching {'/'.join(self.keep_patterns)}; "
                  f"{dropped} lines dropped]")
        return "\n".join([header, *out]) + ("\n" if trailing_newline else "")

    def _collapse(self, lines: list[str]) -> list[tuple[str, int, bool]]:
        entries, i = [], 0
        while i < len(lines):
            j = i
            while j + 1 < len(lines) and lines[j + 1] == lines[i]:
                j += 1
            run = j - i + 1
            if run >= MIN_REPEAT:
                entries.append((lines[i], 1, False))
                entries.append((f"[line above repeated {run - 1} more times]", run - 1, True))
            else:
                entries.extend((lines[k], 1, False) for k in range(i, j + 1))
            i = j + 1
        return entries

    def _select(self, entries: list[tuple[str, int, bool]]) -> list[bool]:
        n = len(entries)
        keep = [False] * n
        head_end = min(self.head_lines, n)
        tail_start = max(n - self.tail_lines, head_end)
        for i in range(head_end):
            keep[i] = True
        for i in range(tail_start, n):
            keep[i] = True

        kept_middle = 0
        for i in range(head_end, tail_start):
            if kept_middle >= self.max_kept_lines:
                break
            line, _, is_marker = entries[i]
            if is_marker or not self._pattern.search(line):
                continue
            for k in range(i, min(i + 1 + self.context_lines, tail_start)):
                if not keep[k] and kept_middle < self.max_kept_lines:
                    keep[k] = True
                    kept_middle += 1

        # A repeat marker is shown only together with the line it repeats.
        for i in range(1, n):
            if entries[i][2]:
                keep[i] = keep[i - 1]
        return keep

    def _cut(self, line: str) -> str:
        if len(line) <= self.max_line_chars:
            return line
        return f"{line[:self.max_line_chars]} [+{len(line) - self.max_line_chars} chars]"
