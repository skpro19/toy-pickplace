# Flywheel ablation: `dagger_intervention_ratio` (seed-10 replication)

Partial 1D sweep on Vast.ai RTX 4090 (2026-07-19). Each run is a full flywheel: expert collection → 10 DAgger rounds → train/eval.

All metrics below come from the **held-out 100-episode re-evaluation** in `final_scores.json` (via `scripts/final_score.py`), not the 25-episode flywheel eval recorded in `metrics.json` during training.

**Placement success rate** is the primary comparison metric (matching mode-b checkpoint selection). **Score** is reported as auxiliary context.

Per-run curve PNGs compare **both eval passes side by side**: gray = flywheel (25 ep, seed 42), blue = held-out (100 ep, 5 seeds × 20 ep).

Raw metrics live under `results/flywheel/ablation-dagger-intervention-ratio-seed10-20260719-152313/`; this folder holds exported plots only.

## What changed vs Study 2A (2026-07-18)

| Setting | Jul 18 sweep | This sweep |
|---|---|---|
| `intervention_threshold` | 0.1 | 0.1 |
| `global_seed` | default (non-zero) | **10** |
| Ratios completed | 0.2, 0.5, 0.8, 1.0 | **0.2, 0.5, 0.8 only** (DIR=1.0 not run) |

This is a seed-controlled replication of [Study 2A](../README.md#sweep-a--fixed-intervention_threshold--01-default-threshold) with `global_seed=10`, paired with the [seed-0 replication](../dagger-intervention-ratio-seed0/) from the same day.

## Shared evaluation setup

| Setting | Value |
|---|---|
| Flywheel in-loop eval | 25 episodes, seed 42 |
| **Reported re-eval** | **100 episodes** (5 held-out seeds × 20 episodes) |
| Eval max steps | 1400 |
| Selection mode | mode-b (placement success rate first) |
| Eval metric version | 5 |
| DAgger rounds | 10 (rounds 0–10) |
| `global_seed` | 10 |

## Summary

![dagger_intervention_ratio seed-10 summary](summary.png)

## Results (held-out eval)

| Ratio | Best score | Placement success | Best checkpoint | Final-round score | Final placement |
|---:|---:|---:|---|---:|---:|
| **0.2** | 0.859 | 79% | round-010 | **0.859** | **79%** |
| **0.5** | 0.823 | 74% | round-009 | 0.499 | 32% |
| **0.8** *(default)* | **0.958** | **93%** | round-010 | **0.958** | **93%** |

Per-run held-out eval curves:

- [`…0.2…`](ablation-dagger-intervention-ratio-seed10-20260719-152313_abl-dagger_intervention_ratio-0.2-final-score-curve.png)
- [`…0.5…`](ablation-dagger-intervention-ratio-seed10-20260719-152313_abl-dagger_intervention_ratio-0.5-final-score-curve.png)
- [`…0.8…`](ablation-dagger-intervention-ratio-seed10-20260719-152313_abl-dagger_intervention_ratio-0.8-final-score-curve.png)

## Cross-seed comparison (IT=0.1, held-out eval)

| Ratio | Jul 18 final | Seed-0 final | **Seed-10 final** |
|---:|---:|---:|---:|
| 0.2 | 42% | 64% | **79%** |
| 0.5 | 40% | 24% | 32% |
| 0.8 | 64% | 64% | **93%** |

Peak placement follows the same ranking: seed 10 reaches 93% (DIR=0.8), 79% (DIR=0.2), and 74% (DIR=0.5) vs Jul 18 peaks of 88%, 86%, and 72% respectively — but only DIR=0.8 and DIR=0.2 **retain** their peaks through round 10 on this seed.

## Findings

- **DIR=0.8 is the standout** under seed 10 — 93% placement at both best checkpoint and final round (0.958 score), with no peak-to-final regression. This exceeds Jul 18's best single ratio (88% at DIR=1.0, round 9) while training through round 10.
- **DIR=0.2 also holds through round 10** — 79% final placement (0.859), vs 42% final on Jul 18 and 64% on seed 0. The early-peak-then-collapse pattern does not appear on this seed.
- **DIR=0.5 still regresses** — 74% peak at round 9 drops to 32% at round 10, confirming ratio 0.5 is seed-sensitive even when its peak is strong.
- **DIR=1.0 was not run**; seed 10's DIR=0.8 result (93%) already surpasses Jul 18's DIR=1.0 peak (88%) on held-out eval.

## Recommendations

1. **For this task under seed 10:** use **DIR=0.8** — 93% final placement with no regression (0.958 at round 10).
2. **DIR=0.2 is a viable alternative** on seed 10 if you want more policy-mixed training data — 79% final (0.859), also stable through round 10.
3. **Avoid DIR=0.5 without held-out re-eval** — strong mid-sweep peaks (74%) collapse to 32% by round 10 on both seed 0 and seed 10.
4. **Seed choice dominates variance** — compare [`seed-0`](../dagger-intervention-ratio-seed0/) (DIR=0.8 flat at 64%) vs this sweep before locking flywheel defaults.

## Folder contents

```
dagger-intervention-ratio-seed10/
├── README.md                          ← this file
├── summary.png
└── …-final-score-curve.png            (3 files: DIR 0.2, 0.5, 0.8)
```

## Source data

| Sweep | Fixed parameter | Results directory | Run date |
|---|---|---|---|
| `dagger_intervention_ratio` | `intervention_threshold=0.1`, `global_seed=10` | `results/flywheel/ablation-dagger-intervention-ratio-seed10-20260719-152313/` | 2026-07-19 |

Held-out scoring: [`scripts/final_score.py`](../../scripts/final_score.py) (default 100 episodes).

Related sweeps: [`../README.md`](../README.md) (Studies 1–2), [`../dagger-intervention-ratio-seed0/`](../dagger-intervention-ratio-seed0/), [`../it-vs-dir/`](../it-vs-dir/) (2D grid).
