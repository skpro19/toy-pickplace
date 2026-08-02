"""Compute baseline-results canvas metrics for a seed-variant flywheel strategy.

Given run names that share the same config slug (e.g. BASEv3_num-expert-episodes-200)
and differ only by global seed suffix (_seed0, _seed420, _seed5693), load
final_scores.json and metrics.json and emit every numeric series used by
baseline-results canvases. Launch timestamps in run names may differ per seed.

- held-out placement curves (100-ep re-eval from final_scores.json)
- in-loop train placement curves (metrics.json eval_metrics)
- cross-seed means for held-out and train
- task funnel at each run's mode-b best round
- aggregate bar-chart stats (peak, r10, mean windows, worst, max drop)
- per-seed run-registry rows (best round, best placement, train at best)

By default the script runs in strict mode: every seed must have final_scores.json
and metrics.json with 11 held-out rounds (r0-r10), and all seeds must match.
Pass --allow-partial to restore warning-based fallback (in-loop eval when
final_scores.json is missing, unequal round counts, etc.).

Usage:
    uv run python scripts/get_canvas_metrics.py \\
        --runs \\
        2026-08-02_14-04-30_1785659670318738614_BASEv3_num-expert-episodes-200_seed0 \\
        2026-08-02_14-04-30_1785659670658765093_BASEv3_num-expert-episodes-200_seed420 \\
        2026-08-02_14-06-17_1785659777609462603_BASEv3_num-expert-episodes-200_seed5693 \\
        --from-s3

    uv run python scripts/get_canvas_metrics.py --runs ... --format typescript \\
        --constant-prefix BASEV3_EXPERT_200
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

FUNNEL_METRIC_KEYS = (
    "grasp_rate",
    "lift_rate",
    "tray_reach_rate",
    "lowered_to_tray_rate",
    "released_over_tray_rate",
    "placement_success_rate",
)

AGGREGATE_LABELS = (
    "Peak",
    "Round-10",
    "Mean r1–10",
    "Mean r3–10",
    "Mean r6–10",
    "Worst round",
    "Max one-round drop",
)

EXPECTED_HELD_OUT_ROUNDS = 11


class CanvasMetricsError(Exception):
    """Run artifacts are missing or inconsistent for canvas held-out metrics."""

RUN_NAME_PATTERN = re.compile(
    r"^(?P<launch>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+_)(?P<config>.+)_seed(?P<seed>\d+)$",
)


def normalize_run_name(*, run_name: str) -> str:
    return run_name.strip().rstrip("/")


def parse_run_name(*, run_name: str) -> tuple[str, str, str]:
    normalized = normalize_run_name(run_name=run_name)
    match = RUN_NAME_PATTERN.match(normalized)
    if match is None:
        raise ValueError(
            f"Run name must match "
            f"YYYY-MM-DD_HH-MM-SS_<id>_<config>_seed<number>: {run_name!r}",
        )
    seed_number = match.group("seed")
    seed_key = f"seed{seed_number}"
    seed_label = f"seed {seed_number}"
    return match.group("config"), seed_key, seed_label


def validate_run_family(*, run_names: list[str]) -> str:
    config_slugs: list[str] = []
    for run_name in run_names:
        config_slug, _, _ = parse_run_name(run_name=run_name)
        config_slugs.append(config_slug)
    unique_config_slugs = sorted(set(config_slugs))
    if len(unique_config_slugs) != 1:
        raise ValueError(
            "All runs must share the same config slug before _seed<number>; "
            f"found: {unique_config_slugs}",
        )
    return unique_config_slugs[0]


def pct(*, value: float) -> float:
    if value <= 1.0:
        return round(value * 100.0, 1)
    return round(value, 1)


def mean_sd(*, values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return round(values[0], 1), 0.0
    return round(statistics.mean(values), 1), round(statistics.stdev(values), 1)


def window_mean(*, series: list[float], start: int, end: int) -> float:
    return statistics.mean(series[start : end + 1])


def max_one_round_drop(*, series: list[float]) -> float:
    peak = max(series)
    peak_index = series.index(peak)
    return peak - min(series[peak_index:])


def cross_seed_mean_per_round(*, seed_series: dict[str, list[float]]) -> list[float]:
    max_rounds = max(len(values) for values in seed_series.values())
    means: list[float] = []
    for round_index in range(max_rounds):
        round_values = [
            values[round_index]
            for values in seed_series.values()
            if round_index < len(values)
        ]
        means.append(round(statistics.mean(round_values), 1))
    return means


def cross_seed_stdev_per_stage(*, seed_funnels: dict[str, list[float]]) -> list[float]:
    stage_count = len(next(iter(seed_funnels.values())))
    return [
        round(statistics.stdev([funnel[stage_index] for funnel in seed_funnels.values()]), 1)
        if len(seed_funnels) > 1
        else 0.0
        for stage_index in range(stage_count)
    ]


def fail_or_warn(
    *,
    strict: bool,
    message: str,
    warnings: list[str],
) -> None:
    if strict:
        raise CanvasMetricsError(message)
    warnings.append(message)


def format_checkpoint_rounds(*, round_count: int, has_final_scores: bool) -> str:
    last_round = round_count - 1
    if not has_final_scores:
        return f"0–{last_round} (in progress)"
    if last_round == 10:
        return "0–10"
    return f"0–{last_round}"


def load_json_from_path(*, path: Path) -> dict:
    return json.loads(path.read_text())


def fetch_s3_json(
    *,
    s3_uri: str,
    aws_profile: str | None,
) -> dict:
    command = ["aws", "s3", "cp", s3_uri, "-"]
    if aws_profile:
        command[2:2] = ["--profile", aws_profile]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise FileNotFoundError(s3_uri) from error
    return json.loads(output)


def resolve_results_file(
    *,
    run_name: str,
    filename: str,
    results_root: Path,
    arch: str,
    from_s3: bool,
    s3_bucket: str,
    aws_profile: str | None,
) -> tuple[dict | None, str | None]:
    local_path = results_root / arch / run_name / filename
    if local_path.is_file():
        return load_json_from_path(path=local_path), None

    s3_uri = f"{s3_bucket.rstrip('/')}/results/flywheel/{arch}/{run_name}/{filename}"
    if not from_s3:
        return None, f"missing local file and --from-s3 not set: {local_path}"

    try:
        return fetch_s3_json(s3_uri=s3_uri, aws_profile=aws_profile), None
    except FileNotFoundError:
        return None, f"missing on disk and S3: {s3_uri}"


def held_out_curve_from_final_scores(*, final_scores: dict) -> list[float]:
    return [
        pct(value=round_data["final_placement_success_rate"])
        for round_data in final_scores["rounds"]
    ]


def train_curve_from_metrics(*, metrics: dict) -> list[float] | None:
    rounds = metrics.get("rounds", [])
    if not rounds:
        return None
    train_values: list[float] = []
    for round_data in rounds:
        eval_metrics = round_data.get("eval_metrics")
        if not isinstance(eval_metrics, dict):
            return None
        if "placement_success_rate" not in eval_metrics:
            return None
        train_values.append(pct(value=eval_metrics["placement_success_rate"]))
    return train_values


def funnel_at_best_round(
    *,
    final_scores: dict,
) -> tuple[int, list[float], float]:
    best_round = int(final_scores["overall_best_round"])
    best_place = pct(value=final_scores["overall_best_placement_success_rate"])
    final_metrics = final_scores["rounds"][best_round]["final_metrics"]
    funnel = [pct(value=final_metrics[key]) for key in FUNNEL_METRIC_KEYS]
    return best_round, funnel, best_place


def train_at_best_round(
    *,
    metrics: dict | None,
    best_round: int,
) -> float | None:
    if metrics is None:
        return None
    rounds = metrics.get("rounds", [])
    if best_round >= len(rounds):
        return None
    eval_metrics = rounds[best_round].get("eval_metrics")
    if not isinstance(eval_metrics, dict):
        return None
    if "placement_success_rate" not in eval_metrics:
        return None
    return pct(value=eval_metrics["placement_success_rate"])


def compute_aggregate_metrics(
    *,
    held_out_by_seed: dict[str, list[float]],
) -> dict[str, object]:
    peaks = [max(series) for series in held_out_by_seed.values()]
    round_10_values = [series[10] if len(series) > 10 else series[-1] for series in held_out_by_seed.values()]
    mean_r1_10 = [
        window_mean(series=series, start=1, end=min(10, len(series) - 1))
        for series in held_out_by_seed.values()
    ]
    mean_r3_10 = [
        window_mean(series=series, start=3, end=min(10, len(series) - 1))
        for series in held_out_by_seed.values()
    ]
    mean_r6_10 = [
        window_mean(series=series, start=6, end=min(10, len(series) - 1))
        for series in held_out_by_seed.values()
    ]
    worst_values = [min(series) for series in held_out_by_seed.values()]
    max_drops = [max_one_round_drop(series=series) for series in held_out_by_seed.values()]

    placement_means: list[float] = []
    placement_sds: list[float] = []
    for values in (peaks, round_10_values, mean_r1_10, mean_r3_10, mean_r6_10, worst_values):
        mean, sd = mean_sd(values=values)
        placement_means.append(mean)
        placement_sds.append(sd)

    max_drop_mean, max_drop_sd = mean_sd(values=max_drops)

    aggregate_table: list[str] = []
    for index, (mean, sd) in enumerate(zip(placement_means, placement_sds, strict=True)):
        if index < 6:
            aggregate_table.append(f"{mean:.1f}% ± {sd:.1f}")
        else:
            aggregate_table.append(f"{mean:.1f} pp ± {sd:.1f}")
    aggregate_table.append(f"{max_drop_mean:.1f} pp ± {max_drop_sd:.1f}")

    return {
        "labels": list(AGGREGATE_LABELS),
        "placement_means": placement_means,
        "placement_sds": placement_sds,
        "max_drop_mean": max_drop_mean,
        "max_drop_sd": max_drop_sd,
        "aggregate_table": aggregate_table,
    }


def compute_strategy_metrics(
    *,
    run_names: list[str],
    results_root: Path,
    arch: str,
    from_s3: bool,
    s3_bucket: str,
    aws_profile: str | None,
    strict: bool = True,
) -> dict[str, object]:
    normalized_runs = [normalize_run_name(run_name=name) for name in run_names]
    validate_run_family(run_names=normalized_runs)

    parsed_runs = [
        (run_name, *parse_run_name(run_name=run_name))
        for run_name in normalized_runs
    ]
    parsed_runs.sort(key=lambda item: int(item[2].replace("seed", "")))

    warnings: list[str] = []
    held_out_by_seed: dict[str, list[float]] = {}
    train_by_seed: dict[str, list[float]] = {}
    funnel_by_seed: dict[str, list[float]] = {}
    funnel_round_by_seed: dict[str, int] = {}
    run_registry: list[dict[str, object]] = []

    round_counts: list[int] = []

    for run_name, _prefix, seed_key, seed_label in parsed_runs:
        final_scores, final_warning = resolve_results_file(
            run_name=run_name,
            filename="final_scores.json",
            results_root=results_root,
            arch=arch,
            from_s3=from_s3,
            s3_bucket=s3_bucket,
            aws_profile=aws_profile,
        )
        metrics, metrics_warning = resolve_results_file(
            run_name=run_name,
            filename="metrics.json",
            results_root=results_root,
            arch=arch,
            from_s3=from_s3,
            s3_bucket=s3_bucket,
            aws_profile=aws_profile,
        )
        if final_warning:
            fail_or_warn(strict=strict, message=final_warning, warnings=warnings)
        if metrics_warning:
            fail_or_warn(strict=strict, message=metrics_warning, warnings=warnings)

        if final_scores is not None:
            held_out = held_out_curve_from_final_scores(final_scores=final_scores)
            best_round, funnel, best_place = funnel_at_best_round(final_scores=final_scores)
            train_at_best = train_at_best_round(metrics=metrics, best_round=best_round)
            has_final_scores = True
        elif metrics is not None:
            fallback_message = (
                f"{seed_key}: final_scores.json missing for run {run_name}; "
                "held-out metrics require 100-ep re-eval from final_scores.json "
                "(pass --allow-partial to use in-loop eval fallback)"
            )
            fail_or_warn(strict=strict, message=fallback_message, warnings=warnings)
            held_out = train_curve_from_metrics(metrics=metrics)
            if held_out is None:
                raise CanvasMetricsError(
                    f"{seed_key}: cannot derive held-out curve from metrics.json",
                )
            best_round = max(
                range(len(metrics["rounds"])),
                key=lambda index: metrics["rounds"][index]["eval_metrics"]["placement_success_rate"],
            )
            eval_metrics = metrics["rounds"][best_round]["eval_metrics"]
            funnel = [pct(value=eval_metrics[key]) for key in FUNNEL_METRIC_KEYS]
            best_place = pct(value=eval_metrics["placement_success_rate"])
            train_at_best = best_place
            has_final_scores = False
        else:
            raise CanvasMetricsError(f"No results found for run {run_name}")

        held_out_by_seed[seed_key] = held_out
        round_counts.append(len(held_out))
        funnel_by_seed[seed_key] = funnel
        funnel_round_by_seed[seed_key] = best_round

        train_curve = train_curve_from_metrics(metrics=metrics) if metrics is not None else None
        if train_curve is not None:
            train_by_seed[seed_key] = train_curve

        run_registry.append(
            {
                "run_id": run_name,
                "seed_key": seed_key,
                "label": seed_label,
                "best_round": best_round,
                "best_place": best_place,
                "train_at_best": train_at_best,
                "checkpoint_rounds": format_checkpoint_rounds(
                    round_count=len(held_out),
                    has_final_scores=has_final_scores,
                ),
            },
        )

    round_count_by_seed = dict(
        zip([entry["seed_key"] for entry in run_registry], round_counts, strict=True),
    )
    if len(set(round_counts)) != 1:
        fail_or_warn(
            strict=strict,
            message=(
                "Seeds have different round counts "
                f"{round_count_by_seed}; cross-seed means require the same r0-r10 coverage "
                "(pass --allow-partial to average only seeds with data at each round)"
            ),
            warnings=warnings,
        )

    if strict:
        for seed_key, round_count in round_count_by_seed.items():
            if round_count != EXPECTED_HELD_OUT_ROUNDS:
                raise CanvasMetricsError(
                    f"{seed_key}: expected {EXPECTED_HELD_OUT_ROUNDS} held-out rounds "
                    f"(r0-r10), got {round_count}",
                )

    mean_held = cross_seed_mean_per_round(seed_series=held_out_by_seed)
    train_excluded_seeds = [
        seed_key
        for _, _, seed_key, _ in parsed_runs
        if seed_key not in train_by_seed
    ]
    if train_excluded_seeds:
        fail_or_warn(
            strict=strict,
            message=(
                "train cross-seed mean computed over available seeds only; "
                f"excluded seeds without a train curve: {', '.join(train_excluded_seeds)}"
            ),
            warnings=warnings,
        )
    mean_train = (
        cross_seed_mean_per_round(seed_series=train_by_seed)
        if train_by_seed
        else None
    )

    funnel_mean = cross_seed_mean_per_round(seed_series=funnel_by_seed)
    funnel_sd = cross_seed_stdev_per_stage(seed_funnels=funnel_by_seed)

    aggregate = compute_aggregate_metrics(held_out_by_seed=held_out_by_seed)

    config_slug = validate_run_family(run_names=normalized_runs)

    return {
        "config_slug": config_slug,
        "run_names": [run_name for run_name, _, _, _ in parsed_runs],
        "seed_keys": [seed_key for _, _, seed_key, _ in parsed_runs],
        "warnings": warnings,
        "held_out": held_out_by_seed,
        "mean_held": mean_held,
        "train": train_by_seed or None,
        "mean_train": mean_train,
        "funnel": funnel_by_seed,
        "funnel_round_by_seed": funnel_round_by_seed,
        "funnel_mean": funnel_mean,
        "funnel_sd": funnel_sd,
        "aggregate": aggregate,
        "run_registry": run_registry,
    }


def format_typescript_block(
    *,
    metrics: dict[str, object],
    constant_prefix: str,
    comment: str | None,
) -> str:
    held_out = metrics["held_out"]
    train = metrics["train"]
    funnel = metrics["funnel"]
    aggregate = metrics["aggregate"]

    lines: list[str] = []
    if comment:
        lines.append(f"/** {comment} */")

    lines.append(f"const {constant_prefix}_HELD_OUT = {json.dumps(held_out, indent=2)};")
    lines.append(f"const {constant_prefix}_MEAN_HELD = {json.dumps(metrics['mean_held'])};")

    if train is not None and metrics["mean_train"] is not None:
        lines.append(f"const {constant_prefix}_TRAIN = {json.dumps(train, indent=2)};")
        lines.append(f"const {constant_prefix}_MEAN_TRAIN = {json.dumps(metrics['mean_train'])};")

    lines.append(f"const {constant_prefix}_FUNNEL = {json.dumps(funnel, indent=2)};")
    lines.append(f"const {constant_prefix}_FUNNEL_MEAN = {json.dumps(metrics['funnel_mean'])};")
    lines.append(f"const {constant_prefix}_FUNNEL_SD = {json.dumps(metrics['funnel_sd'])};")
    lines.append("")
    lines.append("// AGGREGATE_STRATEGIES row")
    lines.append(
        "placementMeans: "
        f"{json.dumps(aggregate['placement_means'])},",
    )
    lines.append(
        "placementSds: "
        f"{json.dumps(aggregate['placement_sds'])},",
    )
    lines.append(
        f"maxDropMean: {aggregate['max_drop_mean']},",
    )
    lines.append(
        f"maxDropSd: {aggregate['max_drop_sd']},",
    )
    lines.append("")
    lines.append("// AGGREGATE_TABLE metrics")
    lines.append(f"metrics: {json.dumps(aggregate['aggregate_table'], indent=2)},")
    lines.append("")
    lines.append("// RUNS entries")
    for entry in metrics["run_registry"]:
        train_at_best = entry["train_at_best"]
        train_literal = "null" if train_at_best is None else f"{train_at_best:.1f}"
        lines.append(
            "{ "
            f"runId: {json.dumps(entry['run_id'])}, "
            f"label: {json.dumps(entry['label'])}, "
            f"bestRound: {entry['best_round']}, "
            f"bestPlace: {entry['best_place']:.1f}, "
            f"trainAtBest: {train_literal}, "
            f"checkpointRounds: {json.dumps(entry['checkpoint_rounds'])} "
            "},",
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute baseline-results canvas metrics for a seed-variant strategy.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directory names sharing the same config prefix (_seed0, _seed420, ...).",
    )
    parser.add_argument(
        "--arch",
        default="vision_mlp",
        help="Flywheel architecture subdirectory (default: vision_mlp).",
    )
    parser.add_argument(
        "--results-root",
        default="results/flywheel",
        help="Local results root (default: results/flywheel).",
    )
    parser.add_argument(
        "--from-s3",
        action="store_true",
        help="Fetch missing artifacts from S3 when local files are absent.",
    )
    parser.add_argument(
        "--s3-bucket",
        default="s3://toy-pickplace",
        help="S3 bucket for --from-s3 (default: s3://toy-pickplace).",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        help="AWS profile for S3 reads (default: AWS_PROFILE env or toy-pickplace-backup).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "typescript", "both"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--constant-prefix",
        default=None,
        help="TypeScript constant prefix, e.g. BASEV3_EXPERT_200 (required for typescript output).",
    )
    parser.add_argument(
        "--comment",
        default=None,
        help="Optional comment for the TypeScript block.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow missing final_scores.json, unequal round counts, and in-loop held-out "
            "fallback; emit warnings instead of raising CanvasMetricsError (default: strict)."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    aws_profile = args.aws_profile
    if aws_profile is None:
        aws_profile = __import__("os").environ.get("AWS_PROFILE", "toy-pickplace-backup")

    try:
        metrics = compute_strategy_metrics(
            run_names=args.runs,
            results_root=Path(args.results_root),
            arch=args.arch,
            from_s3=args.from_s3,
            s3_bucket=args.s3_bucket,
            aws_profile=aws_profile,
            strict=not args.allow_partial,
        )
    except CanvasMetricsError as error:
        print(f"canvas metrics error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if args.format in ("json", "both"):
        json.dump(metrics, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if args.format in ("typescript", "both"):
        if not args.constant_prefix:
            raise SystemExit("--constant-prefix is required when --format is typescript or both")
        typescript = format_typescript_block(
            metrics=metrics,
            constant_prefix=args.constant_prefix,
            comment=args.comment,
        )
        if args.format == "both":
            sys.stdout.write("\n")
        sys.stdout.write(typescript)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
