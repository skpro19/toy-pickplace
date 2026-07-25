# Flywheel ablation: `dagger_intervention_ratio` (seed-0 replication)

Partial 1D sweep on Vast.ai RTX 4090 (2026-07-19). Each run is a full flywheel: expert collection → 10 DAgger rounds → train/eval.

All metrics below come from the **held-out 100-episode re-evaluation** in `final_scores.json` (via `scripts/final_score.py`), not the 25-episode flywheel eval recorded in `metrics.json` during training.

**Placement success rate** is the primary comparison metric (matching mode-b checkpoint selection). **Score** is reported as auxiliary context.

Per-run curve PNGs compare **both eval passes side by side**: gray = flywheel (25 ep, seed 42), blue = held-out (100 ep, 5 seeds × 20 ep).

Raw metrics live under `results/flywheel/ablation-dagger_intervention_ratio-seed0-20260719-152042/`; this folder holds exported plots only.

## What changed vs Study 2A (2026-07-18)

| Setting | Jul 18 sweep | This sweep |
|---|---|---|
| `intervention_threshold` | 0.1 | 0.1 |
| `global_seed` | default (non-zero) | **0** |
| Ratios completed | 0.2, 0.5, 0.8, 1.0 | **0.2, 0.5, 0.8 only** (DIR=1.0 not run) |

This is a seed-controlled replication of [Study 2A](../README.md#sweep-a--fixed-intervention_threshold--01-default-threshold) with `global_seed=0`, intended to test whether the Jul 18 peak-vs-final regression pattern holds under a fixed master seed.

## Shared evaluation setup

| Setting | Value |
|---|---|
| Flywheel in-loop eval | 25 episodes, seed 42 |
| **Reported re-eval** | **100 episodes** (5 held-out seeds × 20 episodes) |
| Eval max steps | 1400 |
| Selection mode | mode-b (placement success rate first) |
| Eval metric version | 5 |
| DAgger rounds | 10 (rounds 0–10) |
| `global_seed` | 0 |

## Summary

![dagger_intervention_ratio seed-0 summary](summary.png)

## Results (held-out eval)

| Ratio | Best score | Placement success | Best checkpoint | Final-round score | Final placement |
|---:|---:|---:|---|---:|---:|
| **0.2** | **0.788** | **70%** | round-002 | 0.740 | **64%** |
| **0.5** | 0.686 | 54% | round-002 | 0.560 | 24% |
| **0.8** *(default)* | 0.758 | 64% | round-007 | 0.741 | **64%** |

Per-run held-out eval curves:

- [`…0p2…`](ablation-dagger_intervention_ratio-seed0-20260719-152042_abl-dagger_intervention_ratio-0p2-final-score-curve.png)
- [`…0p5…`](ablation-dagger_intervention_ratio-seed0-20260719-152042_abl-dagger_intervention_ratio-0p5-final-score-curve.png)
- [`…0p8…`](ablation-dagger_intervention_ratio-seed0-20260719-152042_abl-dagger_intervention_ratio-0p8-final-score-curve.png)

## Comparison to Jul 18 Study 2A (IT=0.1, default seed)

| Ratio | Jul 18 best placement | Jul 18 final placement | Seed-0 best placement | Seed-0 final placement |
|---:|---:|---:|---:|---:|
| 0.2 | 86% | 42% | 70% | **64%** |
| 0.5 | 72% | 40% | 54% | 24% |
| 0.8 | 66% | **64%** | 64% | **64%** |

## Findings

- **DIR=0.8 remains the most stable** under seed 0 — 64% placement at both best checkpoint (round 7) and final round (round 10), matching the Jul 18 final-round behavior.
- **DIR=0.2 peaks earlier** (round 2, 70% held-out) but **does not collapse** at round 10 the way the Jul 18 run did (64% final vs 42%). Peak magnitude is lower (70% vs 86%) but end-of-sweep retention is much better.
- **DIR=0.5 underperforms** on this seed — best held-out placement is only 54% (round 2) and final round drops to 24%, worse than both Jul 18 and the other two ratios here.
- **DIR=1.0 was not run** in this sweep; conclusions about expert-only data under seed 0 remain open.

## Recommendations

1. **For end-of-sweep stability with seed 0:** keep **DIR=0.8** (64% final placement, 0.741 score at round 10).
2. **For early stopping with seed 0:** **DIR=0.2** at round 2 (70%, 0.788) or **DIR=0.8** at round 7 (64%, 0.758) — both beat DIR=0.5; round-10 retention favors 0.8.
3. **Treat DIR=0.5 as seed-sensitive** — 72% peak on Jul 18 but 40% final there and 24% final here; do not pick it without held-out re-eval on the target seed.
4. **Complete DIR=1.0** under seed 0 before revising the Jul 18 recommendation that 1.0 peaks highest at IT=0.1.

## Folder contents

```
dagger-intervention-ratio-seed0/
├── README.md                          ← this file
├── summary.png
└── …-final-score-curve.png            (3 files: DIR 0.2, 0.5, 0.8)
```

## Source data

| Sweep | Fixed parameter | Results directory | Run date |
|---|---|---|---|
| `dagger_intervention_ratio` | `intervention_threshold=0.1`, `global_seed=0` | `results/flywheel/ablation-dagger_intervention_ratio-seed0-20260719-152042/` | 2026-07-19 |

Held-out scoring: [`scripts/final_score.py`](../../scripts/final_score.py) (default 100 episodes).

Related sweeps: [`../README.md`](../README.md) (Studies 1–2), [`../it-vs-dir/`](../it-vs-dir/) (2D grid).
