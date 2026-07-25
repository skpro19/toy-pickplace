# Flywheel ablation results

Summary of flywheel parameter sweeps on Vast.ai RTX 4090 instances (2026-07-18). Each sweep holds all other flywheel settings at baseline and varies one parameter across four values. Every run is a full flywheel: expert collection → 10 DAgger rounds → train/eval.

All scores below come from the **held-out 100-episode re-evaluation** in `final_scores.json` (via `scripts/final_score.py`), not the 25-episode flywheel eval recorded in `metrics.json` during training.

Summary plots compare **both eval passes side by side**: light blue = flywheel (25 ep, seed 42), dark blue = held-out (100 ep, 5 seeds × 20 ep). Curve panels show per-round **score** and **placement success** for each eval type.

Raw metrics live under `results/flywheel/`; this folder holds the exported score-curve plots.

## Shared evaluation setup

| Setting | Value |
|---|---|
| Flywheel in-loop eval | 25 episodes, seed 42 |
| **Reported re-eval** | **100 episodes** (5 held-out seeds × 20 episodes) |
| Eval max steps | 1400 |
| Selection mode | mode-b (placement success rate first) |
| Eval metric version | 5 |
| DAgger rounds | 10 (rounds 0–10) |

During the flywheel, checkpoints were scored with 25 episodes. After each sweep, `final_score.py` re-scored every round's `best.pt` on 100 held-out episodes. The tables and summary plots use those larger-eval numbers.

---

## Study 1: `intervention_threshold`

**Fixed:** `dagger_intervention_ratio = 0.8` (config default)

L2 arm-action threshold for triggering expert takeover during DAgger rollouts. Lower values intervene more often.

| Threshold | Best score | Placement success | Best checkpoint | Final-round score |
|---:|---:|---:|---|---:|
| **0.05** | 0.765 | **69%** | round-004 | 0.569 |
| **0.10** *(default)* | **0.783** | 66% | round-009 | **0.752** |
| **0.20** | 0.720 | 54% | round-003 | 0.493 |
| **0.30** | 0.465 | 15% | round-009 | 0.415 |

![intervention_threshold summary](intervention-threshold/summary.png)

Per-run held-out eval curves: [`intervention-threshold/`](intervention-threshold/)

### Findings

- **0.05 edges 0.10 on placement success** under mode-b (69% vs 66%), but **0.10 reaches the highest score** in the sweep (0.783) and retains 64% success at the final round.
- **0.05 peaked early** at round 4 (0.765, 69%) then collapsed to 29% placement by round 10 — more expert corrections did not sustain gains.
- **0.2 peaked at round 3** (0.720, 54%) then declined — similar mid-sweep regression.
- **0.3 remains clearly too lenient** — placement success never exceeded 15% on the held-out eval.

---

## Study 2: `dagger_intervention_ratio`

Share of expert-executed frames within DAgger training data (retrain rounds only). Lower values keep more policy-executed frames in the training mix.

Two sweeps were run at different fixed `intervention_threshold` values.

### Sweep A — fixed `intervention_threshold = 0.1` *(default threshold)*

Follow-up sweep combining the Study 1 winner (`intervention_threshold = 0.1`) with varying `dagger_intervention_ratio`.

| Ratio | Best score | Placement success | Best checkpoint | Final-round score |
|---:|---:|---:|---|---:|
| **0.2** | 0.880 | 86% | round-009 | 0.544 |
| **0.5** | 0.805 | 72% | round-007 | 0.573 |
| **0.8** *(default)* | 0.783 | 66% | round-009 | **0.752** |
| **1.0** | **0.911** | **88%** | round-009 | 0.676 |

![dagger_intervention_ratio summary — threshold 0.1 sweep](dagger-intervention-ratio/summary-threshold-0.1.png)

Per-run held-out eval curves: [`dagger-intervention-ratio/`](dagger-intervention-ratio/) (`…-20260718-045251_…`)

#### Findings

- **Ratio 1.0 peaks highest** on held-out eval (0.911 score, 88% placement) — a reversal from Sweep B at threshold 0.2, where 1.0 was worst.
- **Ratio 0.2** also peaks strongly (0.880, 86%) but **regresses to 42% placement** by round 10.
- **Default ratio 0.8 is the most stable** — best checkpoint matches the Study 1 threshold-0.1 run (0.783, 66%) and **final round retains 64% placement** (0.752).
- Peak vs final-round tradeoff is sharp: the highest peak ratios (1.0, 0.2) both collapse by the final round.

### Sweep B — fixed `intervention_threshold = 0.2`

Original ratio sweep (run before Study 1 confirmed 0.1 as the better threshold).

| Ratio | Best score | Placement success | Best checkpoint | Final-round score |
|---:|---:|---:|---|---:|
| **0.2** | **0.874** | **79%** | round-009 | 0.813 |
| **0.5** | 0.708 | 56% | round-005 | 0.585 |
| **0.8** *(default)* | 0.720 | 54% | round-003 | 0.493 |
| **1.0** | 0.509 | 27% | round-005 | 0.455 |

![dagger_intervention_ratio summary — threshold 0.2 sweep](dagger-intervention-ratio/summary-threshold-0.2.png)

Per-run held-out eval curves: [`dagger-intervention-ratio/`](dagger-intervention-ratio/) (`…-20260718-033707_…`)

#### Findings

- **Lower ratio wins** at threshold 0.2 — ratio 0.2 best (0.874, 79%), +0.154 score over default 0.8.
- **Expert-only data (1.0) is worst** at threshold 0.2 (0.509, 27%).
- Default 0.8 matches the Study 1 threshold-0.2 run (identical curves), confirming sweep controls.

---

## Study 2A replication: `dagger_intervention_ratio` · `global_seed=0`

Partial rerun of Study 2A Sweep A (fixed `intervention_threshold = 0.1`) with **`global_seed = 0`**. Completed ratios: 0.2, 0.5, 0.8 — **DIR=1.0 was not run**.

| Ratio | Best score | Placement success | Best checkpoint | Final-round score | Final placement |
|---:|---:|---:|---|---:|---:|
| **0.2** | **0.788** | **70%** | round-002 | 0.740 | **64%** |
| **0.5** | 0.686 | 54% | round-002 | 0.560 | 24% |
| **0.8** *(default)* | 0.758 | 64% | round-007 | 0.741 | **64%** |

![dagger_intervention_ratio seed-0 summary](dagger-intervention-ratio-seed0/summary.png)

Per-run held-out eval curves and full analysis: [`dagger-intervention-ratio-seed0/`](dagger-intervention-ratio-seed0/)

### Findings

- **DIR=0.8 is again the most stable** — 64% final placement matches the Jul 18 Study 2A result for the default ratio.
- **DIR=0.2 retains 64% at round 10** (vs 42% on Jul 18) — lower peak (70% vs 86%) but much less final-round regression under seed 0.
- **DIR=0.5 is seed-sensitive** — 40% final on Jul 18, 24% final here (both well below its 72% peak).

---

## Study 2A replication: `dagger_intervention_ratio` · `global_seed=10`

Partial rerun of Study 2A Sweep A (fixed `intervention_threshold = 0.1`) with **`global_seed = 10`**. Completed ratios: 0.2, 0.5, 0.8 — **DIR=1.0 was not run**.

| Ratio | Best score | Placement success | Best checkpoint | Final-round score | Final placement |
|---:|---:|---:|---|---:|---:|
| **0.2** | 0.859 | 79% | round-010 | **0.859** | **79%** |
| **0.5** | 0.823 | 74% | round-009 | 0.499 | 32% |
| **0.8** *(default)* | **0.958** | **93%** | round-010 | **0.958** | **93%** |

![dagger_intervention_ratio seed-10 summary](dagger-intervention-ratio-seed10/summary.png)

Per-run held-out eval curves and full analysis: [`dagger-intervention-ratio-seed10/`](dagger-intervention-ratio-seed10/)

### Findings

- **DIR=0.8 reaches 93% final placement** on seed 10 — no peak-to-final regression; exceeds Jul 18's best held-out peak (88% at DIR=1.0).
- **DIR=0.2 retains 79% at round 10** — stable end-of-sweep performance unlike Jul 18 (42% final).
- **DIR=0.5 still collapses** (74% peak → 32% final), matching the seed-sensitivity seen on seed 0 and Jul 18.
- **Seed choice dominates** — same IT/DIR grid yields 64% (seed 0) vs 93% (seed 10) final placement at DIR=0.8.

---

## Recommendations

1. **Keep `intervention_threshold = 0.1`** — best balance of peak score and final-round stability at the default `dagger_intervention_ratio = 0.8` (66% peak / 64% final placement).
2. **At IT=0.1, for peak checkpoint with early stopping:** try **DIR=1.0** (88% placement, 0.911 score at round 9) or **DIR=0.2** (86%, 0.880) — both regress sharply by round 10 without re-eval.
3. **At IT=0.1, for end-of-sweep stability:** keep **DIR=0.8** (64% final placement, 0.752 score at round 10).
4. **At IT=0.2:** prefer **DIR=0.2** (79% peak, 69% final placement) over the default ratio.
5. **Peak vs final-round regression** remains the main flywheel risk — use held-out eval for checkpoint selection.

---

## Folder contents

```
ablations/
├── README.md                          ← this file (Studies 1–2 + seed replications)
├── intervention-threshold/            ← Study 1 (1D threshold sweep)
│   ├── summary.png
│   └── …-final-score-curve.png        (4 files)
├── dagger-intervention-ratio/         ← Study 2 (1D ratio sweeps)
│   ├── summary-threshold-0.1.png
│   ├── summary-threshold-0.2.png
│   └── …-final-score-curve.png        (8 files)
├── dagger-intervention-ratio-seed0/   ← Study 2A replication (global_seed=0, partial)
│   ├── summary.png
│   └── …-final-score-curve.png        (3 files)
├── dagger-intervention-ratio-seed10/  ← Study 2A replication (global_seed=10, partial)
│   ├── summary.png
│   └── …-final-score-curve.png        (3 files)
└── it-vs-dir/                         ← 2D IT × DIR grid
```

## Source data

| Sweep | Fixed parameter | Results directory | Run date |
|---|---|---|---|
| intervention_threshold | `dagger_intervention_ratio = 0.8` | `results/flywheel/ablation-intervention-threshold-20260718-020039/` | 2026-07-18 |
| dagger_intervention_ratio | `intervention_threshold = 0.1` | `results/flywheel/ablation-dagger-intervention-ratio-20260718-045251/` | 2026-07-18 |
| dagger_intervention_ratio | `intervention_threshold = 0.2` | `results/flywheel/ablation-dagger-intervention-ratio-20260718-033707/` | 2026-07-18 |
| dagger_intervention_ratio | `intervention_threshold = 0.1`, `global_seed = 0` | `results/flywheel/ablation-dagger_intervention_ratio-seed0-20260719-152042/` | 2026-07-19 |
| dagger_intervention_ratio | `intervention_threshold = 0.1`, `global_seed = 10` | `results/flywheel/ablation-dagger-intervention-ratio-seed10-20260719-152313/` | 2026-07-19 |

Run protocols: [`docs/ablation/intervention-threshold.md`](../docs/ablation/intervention-threshold.md), [`docs/ablation/dagger-intervention-ratio.md`](../docs/ablation/dagger-intervention-ratio.md).

Held-out scoring: [`scripts/final_score.py`](../scripts/final_score.py) (default 100 episodes).
