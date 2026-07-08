#!/usr/bin/env python3
"""Create an LLM-failure-focused Test14-hard scenario filter.

The LLM-failure set uses available Test14-hard quick-test parquet metrics from:
zero-shot, loss-based, RandomBucket, and LLM-guided PLUTO runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLUTO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PLUTO_ROOT.parent

DEFAULT_EXP_ROOT = WORKSPACE_ROOT / "nuplan-devkit" / "nuplan" / "exp" / "exp"
DEFAULT_FILTER = PLUTO_ROOT / "config" / "scenario_filter" / "test14-hard-llm-failure.yaml"
DEFAULT_ARTIFACT_DIR = PLUTO_ROOT / "artifacts" / "records" / "test14_hard_llm_failure"

METHODS = {
    "zeroshot": "quick_test_zeroshot_test14_hard",
    "lossbased": "quick_test_lossbased_test14_hard",
    "randombucket": "quick_test_curriculum_randombucket_test14_hard",
    "llm_guided": "quick_test_curriculum_llm_guided_v2_test14_hard",
}

EVAL_METHODS = list(METHODS)
NON_LLM_METHODS = ["zeroshot", "lossbased", "randombucket"]

METRICS = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
]

MULTIPLIER_METRICS = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
]

WEIGHTED_METRICS = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}

BUCKET_ORDER = [
    "collision_failure",
    "drivable_offroute_failure",
    "traffic_rule_failure",
    "low_progress",
    "common_success",
]

# Select scarce failure categories first, then emit rows in the requested order.
SELECTION_ORDER = [
    "drivable_offroute_failure",
    "collision_failure",
    "traffic_rule_failure",
    "low_progress",
    "common_success",
]

BUCKET_LABELS = {
    "collision_failure": "collision failure",
    "drivable_offroute_failure": "drivable/off-route failure",
    "traffic_rule_failure": "red-light/traffic-rule failure",
    "low_progress": "low progress",
    "common_success": "common success / nearest fallback",
}


SceneMetrics = dict[str, dict[str, Any]]
AllMetrics = dict[str, SceneMetrics]


def read_method_metrics(exp_root: Path, method: str, experiment: str) -> SceneMetrics:
    metrics_dir = exp_root / experiment / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics directory for {method}: {metrics_dir}")

    scenes: SceneMetrics = {}
    for metric in METRICS:
        metric_file = metrics_dir / f"{metric}.parquet"
        if not metric_file.exists():
            raise FileNotFoundError(f"Missing metric parquet for {method}: {metric_file}")
        frame = pd.read_parquet(metric_file)
        required_columns = {"scenario_name", "scenario_type", "log_name", "metric_score"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{metric_file} missing columns: {sorted(missing_columns)}")

        for _, row in frame.iterrows():
            token = str(row["scenario_name"])
            scene = scenes.setdefault(
                token,
                {
                    "scenario_name": token,
                    "scenario_type": str(row["scenario_type"]),
                    "log_name": str(row["log_name"]),
                },
            )
            score = row["metric_score"]
            scene[metric] = float(score) if pd.notna(score) else math.nan

    total_weight = sum(WEIGHTED_METRICS.values())
    complete = {}
    for token, scene in scenes.items():
        if not all(metric in scene for metric in METRICS):
            continue
        multiplier = 1.0
        for metric in MULTIPLIER_METRICS:
            multiplier *= float(scene[metric])
        weighted = sum(float(scene[metric]) * weight for metric, weight in WEIGHTED_METRICS.items()) / total_weight
        scene["nr_cls"] = multiplier * weighted
        scene["success"] = all(float(scene[metric]) == 1.0 for metric in METRICS)
        complete[token] = scene
    return complete


def load_metrics(exp_root: Path) -> AllMetrics:
    return {
        method: read_method_metrics(exp_root, method, experiment)
        for method, experiment in METHODS.items()
    }


def fail_category(data: AllMetrics, token: str, method: str, category: str) -> bool:
    scene = data[method][token]
    if category == "collision_failure":
        return float(scene["no_ego_at_fault_collisions"]) < 1.0
    if category == "drivable_offroute_failure":
        return (
            float(scene["drivable_area_compliance"]) < 1.0
            or float(scene["driving_direction_compliance"]) < 1.0
        )
    if category == "traffic_rule_failure":
        return float(scene["speed_limit_compliance"]) < 1.0
    if category == "low_progress":
        return (
            float(scene["ego_is_making_progress"]) < 1.0
            or float(scene["ego_progress_along_expert_route"]) < 0.5
        )
    raise KeyError(category)


def method_count(data: AllMetrics, token: str, predicate: Callable[[str], bool]) -> int:
    return sum(1 for method in METHODS if token in data[method] and predicate(method))


def avg_nr_cls(data: AllMetrics, token: str) -> float:
    scores = [float(data[method][token]["nr_cls"]) for method in METHODS if token in data[method]]
    return sum(scores) / len(scores)


def method_success_count(data: AllMetrics, token: str) -> int:
    return method_count(data, token, lambda method: bool(data[method][token]["success"]))


def method_failure_count(data: AllMetrics, token: str) -> int:
    return method_count(data, token, lambda method: not bool(data[method][token]["success"]))


def category_failure_count(data: AllMetrics, token: str, category: str) -> int:
    return sum(1 for method in EVAL_METHODS if fail_category(data, token, method, category))


def category_priority_tier(data: AllMetrics, token: str, category: str) -> int | None:
    if not fail_category(data, token, "llm_guided", category):
        return None
    failure_count = category_failure_count(data, token, category)
    if failure_count == len(EVAL_METHODS):
        return 0
    if any(fail_category(data, token, method, category) for method in NON_LLM_METHODS):
        return 1
    return 2


def category_candidates(
    data: AllMetrics,
    tokens: set[str],
    category: str,
    preserve_tokens: set[str],
) -> list[str]:
    return sorted(
        [
            token
            for token in tokens
            if category_priority_tier(data, token, category) is not None
        ],
        key=lambda token: (
            category_priority_tier(data, token, category),
            token in preserve_tokens,
            -category_failure_count(data, token, category),
            avg_nr_cls(data, token),
            token,
        ),
    )


def common_success_candidates(data: AllMetrics, tokens: set[str]) -> list[str]:
    return sorted(
        tokens,
        key=lambda token: (
            -method_success_count(data, token),
            not bool(data["llm_guided"][token]["success"]),
            -avg_nr_cls(data, token),
            token,
        ),
    )


def select_bucket(
    candidates: list[str],
    selected: set[str],
    bucket_size: int,
) -> list[str]:
    bucket = []
    for token in candidates:
        if token in selected:
            continue
        bucket.append(token)
        selected.add(token)
        if len(bucket) >= bucket_size:
            break
    return bucket


def scene_summary(data: AllMetrics, token: str, bucket: str, rank: int) -> dict[str, Any]:
    ref_method = "llm_guided" if token in data["llm_guided"] else next(method for method in METHODS if token in data[method])
    ref = data[ref_method][token]
    row: dict[str, Any] = {
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS[bucket],
        "bucket_rank": rank,
        "scenario_name": token,
        "scenario_type": ref["scenario_type"],
        "log_name": ref["log_name"],
        "avg_nr_cls": round(avg_nr_cls(data, token), 6),
        "method_success_count": method_success_count(data, token),
        "method_failure_count": method_failure_count(data, token),
    }
    if bucket in {
        "collision_failure",
        "drivable_offroute_failure",
        "traffic_rule_failure",
        "low_progress",
    }:
        row["failure_bucket_priority_tier"] = category_priority_tier(data, token, bucket)
        row["category_failure_count"] = category_failure_count(data, token, bucket)
    for method in METHODS:
        if token not in data[method]:
            row[f"{method}_available"] = False
            continue
        scene = data[method][token]
        row[f"{method}_available"] = True
        row[f"{method}_success"] = bool(scene["success"])
        row[f"{method}_nr_cls"] = round(float(scene["nr_cls"]), 6)
        for metric in METRICS:
            row[f"{method}_{metric}"] = round(float(scene[metric]), 6)
    return row


def write_filter(path: Path, rows: list[dict[str, Any]]) -> None:
    scenario_types = sorted({str(row["scenario_type"]) for row in rows})
    lines = [
        "_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_: all",
        "scenario_types:",
    ]
    lines.extend(f"  - {scenario_type}" for scenario_type in scenario_types)
    lines.extend(["scenario_tokens:"])
    lines.extend(f"  - \"{row['scenario_name']}\"" for row in rows)
    lines.extend(
        [
            "log_names: null",
            "map_names: null",
            "num_scenarios_per_type: null",
            "limit_total_scenarios: null",
            "timestamp_threshold_s: 15",
            "ego_displacement_minimum_m: null",
            "ego_start_speed_threshold: null",
            "ego_stop_speed_threshold: null",
            "speed_noise_tolerance: null",
            "expand_scenarios: null",
            "remove_invalid_goals: true",
            "shuffle: false",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    front = [
        "bucket",
        "bucket_label",
        "bucket_rank",
        "scenario_name",
        "scenario_type",
        "log_name",
        "avg_nr_cls",
        "method_success_count",
        "method_failure_count",
    ]
    fields = front + [field for field in fields if field not in front]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_llm_failure_filter(
    data: AllMetrics,
    failure_bucket_size: int,
    common_success_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all4_tokens = set.intersection(*(set(data[method]) for method in METHODS))

    selected: set[str] = set()
    bucket_tokens: dict[str, list[str]] = {}
    bucket_size_targets = {
        "collision_failure": failure_bucket_size,
        "drivable_offroute_failure": failure_bucket_size,
        "traffic_rule_failure": failure_bucket_size,
        "low_progress": failure_bucket_size,
        "common_success": common_success_size,
    }

    for bucket in SELECTION_ORDER:
        if bucket in {
            "collision_failure",
            "drivable_offroute_failure",
            "traffic_rule_failure",
            "low_progress",
        }:
            candidates = category_candidates(data, all4_tokens, bucket, all4_tokens)
        elif bucket == "common_success":
            candidates = common_success_candidates(data, all4_tokens)
        else:
            raise KeyError(bucket)
        bucket_tokens[bucket] = select_bucket(candidates, selected, bucket_size_targets[bucket])

    rows = [
        scene_summary(data, token, bucket, rank)
        for bucket in BUCKET_ORDER
        for rank, token in enumerate(bucket_tokens[bucket], start=1)
    ]

    summary = {
        "bucket_size_targets": bucket_size_targets,
        "total_selected": len(rows),
        "unique_selected": len({row["scenario_name"] for row in rows}),
        "method_scene_counts": {method: len(scenes) for method, scenes in data.items()},
        "all4_intersection_count": len(all4_tokens),
        "bucket_counts": {bucket: len(tokens) for bucket, tokens in bucket_tokens.items()},
        "underfilled_buckets": {
            bucket: len(tokens)
            for bucket, tokens in bucket_tokens.items()
            if len(tokens) < bucket_size_targets[bucket]
        },
        "bucket_tokens": bucket_tokens,
        "selection_notes": [
            "Failure-type buckets use the 272-scene intersection of zeroshot, lossbased, randombucket, and llm_guided.",
            "Failure-type buckets require llm_guided to fail the bucket category.",
            "Failure-type priority tiers are: 0=zeroshot+lossbased+randombucket+llm_guided all fail the category, 1=llm_guided plus at least one of zeroshot/lossbased/randombucket fails the category, 2=llm_guided alone fails the category.",
            "Buckets remain underfilled if the llm_guided category-failure pool has fewer scenes than the target size.",
            "Common success uses the strict four-method 272-scene intersection and targets 40 scenes by default.",
            "Red-light/traffic-rule is represented by speed_limit_compliance because these quick-test metrics do not include a red-light-specific parquet metric.",
            "Common success fallback ranks by more methods succeeding, then llm_guided success, then higher average NR-CLS.",
        ],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--output-filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--failure-bucket-size", type=int, default=15)
    parser.add_argument("--common-success-size", type=int, default=40)
    args = parser.parse_args()

    data = load_metrics(args.exp_root)
    rows, summary = create_llm_failure_filter(data, args.failure_bucket_size, args.common_success_size)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_filter(args.output_filter, rows)
    write_csv(args.artifact_dir / "selection.csv", rows)
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote filter: {args.output_filter}")
    print(f"Wrote selection CSV: {args.artifact_dir / 'selection.csv'}")
    print(f"Wrote summary JSON: {args.artifact_dir / 'summary.json'}")
    print(json.dumps(summary["bucket_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
