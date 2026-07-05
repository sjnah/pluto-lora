#!/usr/bin/env python3
"""Create a fast Val14 benchmark filter with five representatives per cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLUTO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PLUTO_ROOT.parent

DEFAULT_EXP_ROOT = WORKSPACE_ROOT / "nuplan-devkit" / "nuplan" / "exp" / "exp"
DEFAULT_FILTER = PLUTO_ROOT / "config" / "scenario_filter" / "val14-fast.yaml"
DEFAULT_ARTIFACT_DIR = PLUTO_ROOT / "artifacts" / "records" / "val14_fast"
DEFAULT_SOURCE_EXPERIMENT = "quick_test_curriculum_randombucket_val14_benchmark"

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

DIFFICULTY_ORDER = ["hard", "medium", "easy"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def metric_dirs(exp_root: Path, experiment: str) -> list[Path]:
    direct = exp_root / experiment / "metrics"
    dirs = [direct] if direct.exists() else []
    dirs.extend(sorted(exp_root.glob(f"{experiment}_batch*/metrics")))
    if not dirs:
        raise FileNotFoundError(f"No metric directories found for {experiment} under {exp_root}")
    return dirs


def load_source_rows(exp_root: Path, experiment: str) -> list[dict[str, Any]]:
    dirs = metric_dirs(exp_root, experiment)
    scenes: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        for metrics_dir in dirs:
            metric_file = metrics_dir / f"{metric}.parquet"
            if not metric_file.exists():
                continue
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

    rows = []
    total_weight = sum(WEIGHTED_METRICS.values())
    for token, scene in scenes.items():
        if not all(metric in scene for metric in METRICS):
            continue
        multiplier = 1.0
        for metric in MULTIPLIER_METRICS:
            multiplier *= float(scene[metric])
        weighted = sum(float(scene[metric]) * weight for metric, weight in WEIGHTED_METRICS.items()) / total_weight
        nr_cls = multiplier * weighted
        row: dict[str, Any] = {
            "scenario_name": token,
            "scenario_type": scene["scenario_type"],
            "log_name": scene["log_name"],
            "source_method": "curriculum_randombucket",
            "source_nr_cls": round(nr_cls, 6),
            "source_success": all(float(scene[metric]) == 1.0 for metric in METRICS),
        }
        for metric in METRICS:
            row[metric] = round(float(scene[metric]), 6)
        rows.append(row)
    return rows


def assign_difficulty(rows: list[dict[str, Any]]) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["scenario_type"])].append(row)

    for scenario_type, group in by_type.items():
        ordered = sorted(group, key=lambda row: (float(row["source_nr_cls"]), str(row["scenario_name"])))
        n = len(ordered)
        hard_cutoff = n // 3
        medium_cutoff = (2 * n) // 3
        for idx, row in enumerate(ordered):
            if idx < hard_cutoff:
                bucket = "hard"
            elif idx < medium_cutoff:
                bucket = "medium"
            else:
                bucket = "easy"
            row["difficulty_bucket"] = bucket
            row["type_difficulty_rank"] = idx + 1
            row["type_candidate_count"] = n
            row["type_difficulty_percentile"] = round((idx + 1) / n, 6)


def select_quantile_representatives(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: (float(row["source_nr_cls"]), str(row["scenario_name"])))
    if len(ordered) < count:
        return []

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx in range(count):
        target_quantile = (idx + 1) / (count + 1)
        target_position = target_quantile * (len(ordered) - 1)
        position, row = sorted(
            enumerate(ordered),
            key=lambda item: (
                str(item[1]["scenario_name"]) in used,
                abs(item[0] - target_position),
                str(item[1]["scenario_name"]),
            ),
        )[0]
        selected.append(row)
        used.add(str(row["scenario_name"]))
        row["selection_strategy"] = "source_nr_cls_quantile_representative"
        row["selection_cell_rank"] = position + 1
        row["selection_cell_size"] = len(ordered)
        row["selection_cell_target_count"] = count
        row["selection_quantile_target"] = round(target_quantile, 6)
    return selected


def select_fast_rows(rows: list[dict[str, Any]], per_cell: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assign_difficulty(rows)
    by_type_difficulty: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type_difficulty[(str(row["scenario_type"]), str(row["difficulty_bucket"]))].append(row)

    type_counts = Counter(str(row["scenario_type"]) for row in rows)
    eligible_types = [
        scenario_type
        for scenario_type, count in sorted(type_counts.items())
        if all(len(by_type_difficulty[(scenario_type, difficulty)]) >= per_cell for difficulty in DIFFICULTY_ORDER)
    ]

    selected: list[dict[str, Any]] = []
    for scenario_type in eligible_types:
        for difficulty in DIFFICULTY_ORDER:
            selected.extend(select_quantile_representatives(by_type_difficulty[(scenario_type, difficulty)], per_cell))

    selected = sorted(
        selected,
        key=lambda row: (
            str(row["scenario_type"]),
            DIFFICULTY_ORDER.index(str(row["difficulty_bucket"])),
            int(row["selection_cell_rank"]),
            str(row["scenario_name"]),
        ),
    )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    excluded_types = {
        scenario_type: {
            "total": type_counts[scenario_type],
            **{
                difficulty: len(by_type_difficulty[(scenario_type, difficulty)])
                for difficulty in DIFFICULTY_ORDER
            },
        }
        for scenario_type in sorted(type_counts)
        if scenario_type not in eligible_types
    }

    summary = {
        "candidate_count": len(rows),
        "selected_size": len(selected),
        "per_cell": per_cell,
        "selection_strategy": "source_nr_cls_quantile_representative",
        "source_method": "curriculum_randombucket",
        "difficulty_order": DIFFICULTY_ORDER,
        "eligible_scenario_types": eligible_types,
        "excluded_scenario_types": excluded_types,
        "selection_notes": [
            "Candidate pool is the available full Val14 RandomBucket metric output.",
            "Difficulty is assigned within each actual scenario_type by RandomBucket NR-CLS tertiles.",
            "The fast filter selects five quantile representatives per scenario-type-by-difficulty cell.",
            "This is a balanced Val14 fast subset, not a four-method calibrated proxy.",
        ],
    }
    return selected, summary


def method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = [float(row["source_nr_cls"]) for row in rows]
    entry: dict[str, Any] = {
        "method": "curriculum_randombucket",
        "method_label": "RandomBucket",
        "n": len(scores),
        "score": round(mean(scores), 6),
        "score_std": round(std(scores), 6),
        "perfect_count": sum(1 for row in rows if bool(row["source_success"])),
        "zero_score_count": sum(1 for score in scores if score == 0.0),
    }
    for metric in METRICS:
        entry[metric] = round(mean([float(row[metric]) for row in rows]), 6)
    return [entry]


def grouped_summary(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_field])].append(row)
    out = []
    for group, group_rows in sorted(groups.items()):
        scores = [float(row["source_nr_cls"]) for row in group_rows]
        out.append(
            {
                group_field: group,
                "method": "curriculum_randombucket",
                "method_label": "RandomBucket",
                "n": len(scores),
                "score": round(mean(scores), 6),
                "perfect_count": sum(1 for row in group_rows if bool(row["source_success"])),
                "zero_score_count": sum(1 for score in scores if score == 0.0),
            }
        )
    return out


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(str(row["scenario_type"]) for row in rows)
    difficulty_counts = Counter(str(row["difficulty_bucket"]) for row in rows)
    type_difficulty_counts = Counter(
        f"{row['scenario_type']}::{row['difficulty_bucket']}" for row in rows
    )
    return {
        "scenario_type_counts": dict(sorted(type_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "type_difficulty_counts": dict(sorted(type_difficulty_counts.items())),
        "selected_source_nr_cls_mean": round(mean([float(row["source_nr_cls"]) for row in rows]), 6),
        "selected_source_nr_cls_std": round(std([float(row["source_nr_cls"]) for row in rows]), 6),
    }


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
        "selection_rank",
        "scenario_name",
        "scenario_type",
        "difficulty_bucket",
        "type_difficulty_rank",
        "type_candidate_count",
        "source_nr_cls",
        "selection_strategy",
        "selection_quantile_target",
        "selection_cell_rank",
        "selection_cell_size",
        "selection_cell_target_count",
        "source_success",
        "log_name",
    ]
    fields = front + [field for field in fields if field not in front]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_named_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--output-filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--per-cell", type=int, default=5)
    args = parser.parse_args()

    candidates = load_source_rows(args.exp_root, args.source_experiment)
    rows, summary = select_fast_rows(candidates, args.per_cell)
    summary.update(selection_summary(rows))
    summary["method_summary"] = method_summary(rows)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_filter(args.output_filter, rows)
    write_csv(args.artifact_dir / "selection.csv", rows)
    write_named_csv(args.artifact_dir / "method_summary.csv", summary["method_summary"])
    write_named_csv(args.artifact_dir / "by_difficulty.csv", grouped_summary(rows, "difficulty_bucket"))
    write_named_csv(args.artifact_dir / "by_scenario_type.csv", grouped_summary(rows, "scenario_type"))
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote filter: {args.output_filter}")
    print(f"Wrote selection CSV: {args.artifact_dir / 'selection.csv'}")
    print(f"Wrote summary JSON: {args.artifact_dir / 'summary.json'}")
    print(json.dumps(selection_summary(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
