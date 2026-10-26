# Plan

Full project plan: https://claude.ai/artifact/8XRszpZYooytCrgxfBLPDK

## Milestones

| Week | Goal |
|---|---|
| 10 (Oct 26) | Core library + Policy A in mini-SWE-agent; freeze contract; midterm checkpoint |
| 11 (Nov 2) | Policy B, eval runner/replay/metrics, pick 25 tasks, baselines, OpenHands trial, budget estimate |
| 12 (Nov 9) | Second adapter; both policies in both harnesses on 1–2 tasks; record adapter effort |
| 13 (Nov 16) | Pilots; freeze tasks/configs/model; start full runs |
| 14 (Nov 23) | Finish 450 runs (25 tasks × 3 configs × 3 repeats × 2 harnesses) |
| 15 (Nov 30) | Analysis (RQ1, RQ2); docs/adding_a_policy.md |
| 16 (Dec 7) | Final report, presentation, live demo |

## Ownership

| Area | Owner |
|---|---|
| Event types, policy interface, registry, capability check | Person A |
| Policy A, Policy B, offload store, follow-up read detection | Person A |
| Harness adapters | Person B |
| Docker + SWE-bench setup, eval runner, replay, metrics | Person B |
| Survey, design doc, analysis, report, slides | Both |

Each member reviews the other's pull requests.
