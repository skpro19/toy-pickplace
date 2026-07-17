# Flywheel ablation results

Summary of two parameter sweeps run on a single Vast.ai RTX 4090 instance (2026-07-18). Each sweep holds all other flywheel settings at baseline and varies one parameter across four values. Every run is a full flywheel: expert collection → 10 DAgger rounds → train/eval.

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

**Fixed:** `intervention_threshold = 0.2`

Share of expert-executed frames within DAgger training data (retrain rounds only). Lower values keep more policy-executed frames in the training mix.

| Ratio | Best score | Placement success | Best checkpoint | Final-round score |
|---:|---:|---:|---|---:|
| **0.2** | **0.874** | **79%** | round-009 | 0.813 |
| **0.5** | 0.708 | 56% | round-005 | 0.585 |
| **0.8** *(default)* | 0.720 | 54% | round-003 | 0.493 |
| **1.0** | 0.509 | 27% | round-005 | 0.455 |

![dagger_intervention_ratio summary](dagger-intervention-ratio/summary.png)

Per-run held-out eval curves: [`dagger-intervention-ratio/`](dagger-intervention-ratio/)

### Findings

- **Lower ratio wins decisively.** Ratio 0.2 achieved the best result across both studies (0.874 score, 79% success) — +0.154 score over the default 0.8 on the held-out eval.
- **Expert-only DAgger data (1.0) is worst** — best checkpoint at round 5 (0.509, 27%) with no sustained recovery.
- **Default 0.8 matches the intervention-threshold-0.2 run** (identical score curves), confirming the sweeps were controlled correctly.
- Most runs peaked before the final round, suggesting **continued DAgger rounds may cause regression** when the training mix is suboptimal.

---

## Recommendations

1. **Keep `intervention_threshold = 0.1` as default** — highest held-out score (0.783) and stable late-round performance (64% at round 10). Consider 0.05 only if peak placement matters more than end-of-sweep reliability.
2. **Lower `dagger_intervention_ratio` to ~0.2** — the largest single improvement found (+0.154 score vs default 0.8 at the same threshold). Worth a follow-up sweep around 0.1–0.3.
3. **Investigate flywheel regression** — several runs peak mid-sweep then decline on the held-out eval. Consider early stopping on eval, checkpoint selection, or fewer DAgger rounds.
4. **Combine the winners** — a run with `intervention_threshold=0.1` and `dagger_intervention_ratio=0.2` is the logical next experiment.

---

## Folder contents

```
ablations/
├── README.md                          ← this file
├── intervention-threshold/
│   ├── summary.png                    ← bar + curve summary for threshold sweep
│   ├── …-0.05-final-score-curve.png
│   ├── …-0.1-final-score-curve.png
│   ├── …-0.2-final-score-curve.png
│   └── …-0.3-final-score-curve.png
└── dagger-intervention-ratio/
    ├── summary.png                    ← bar + curve summary for ratio sweep
    ├── …-0.2-final-score-curve.png
    ├── …-0.5-final-score-curve.png
    ├── …-0.8-final-score-curve.png
    └── …-1.0-final-score-curve.png
```

## Source data

| Sweep | Results directory | Run date |
|---|---|---|
| intervention_threshold | `results/flywheel/ablation-intervention-threshold-20260718-020039/` | 2026-07-18 |
| dagger_intervention_ratio | `results/flywheel/ablation-dagger-intervention-ratio-20260718-033707/` | 2026-07-18 |

Run protocols: [`docs/ablation/intervention-threshold.md`](../docs/ablation/intervention-threshold.md), [`docs/ablation/dagger-intervention-ratio.md`](../docs/ablation/dagger-intervention-ratio.md).

Held-out scoring: [`scripts/final_score.py`](../scripts/final_score.py) (default 100 episodes).
