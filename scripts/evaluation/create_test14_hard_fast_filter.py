#!/usr/bin/env python3
"""Create a method-balanced fast Test14-hard benchmark filter.

The fast filter uses the four-method Test14-hard intersection only as a
candidate pool and as a historical difficulty signal. It does not prioritize
failures from any single method. Selection is stratified by scenario type and
within-type average NR-CLS tertile.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLUTO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PLUTO_ROOT.parent

DEFAULT_EXP_ROOT = WORKSPACE_ROOT / "nuplan-devkit" / "nuplan" / "exp" / "exp"
DEFAULT_FILTER = PLUTO_ROOT / "config" / "scenario_filter" / "test14-hard-fast.yaml"
DEFAULT_ARTIFACT_DIR = PLUTO_ROOT / "artifacts" / "records" / "test14_hard_fast"

METHODS = {
    "zeroshot": "quick_test_zeroshot_test14_hard",
    "lossbased": "quick_test_lossbased_test14_hard",
    "randombucket": "quick_test_curriculum_randombucket_test14_hard",
    "llm_guided": "quick_test_curriculum_llmbased_test14_hard",
}

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
METHOD_LABELS = {
    "zeroshot": "Zero-shot",
    "lossbased": "Loss-based",
    "randombucket": "RandomBucket",
    "llm_guided": "LLM-guided",
}
METHOD_KEYS = list(METHODS)
CALIBRATION_SCORE_WEIGHT = 1000.0
CALIBRATION_CELL_ERROR_WEIGHT = 0.03
CALIBRATION_RESTARTS = 64
CALIBRATION_MAX_PASSES = 80

SceneMetrics = dict[str, dict[str, Any]]
AllMetrics = dict[str, SceneMetrics]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


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


def avg_nr_cls(data: AllMetrics, token: str) -> float:
    return mean([float(data[method][token]["nr_cls"]) for method in METHODS])


def method_success_count(data: AllMetrics, token: str) -> int:
    return sum(1 for method in METHODS if bool(data[method][token]["success"]))


def assign_within_type_difficulty(rows: list[dict[str, Any]]) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["scenario_type"])].append(row)

    for scenario_type, group in by_type.items():
        ordered = sorted(group, key=lambda row: (float(row["avg_nr_cls"]), str(row["scenario_name"])))
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


def base_rows(data: AllMetrics) -> list[dict[str, Any]]:
    all4_tokens = sorted(set.intersection(*(set(data[method]) for method in METHODS)))
    rows = []
    for token in all4_tokens:
        ref = data["zeroshot"][token]
        scores = {method: float(data[method][token]["nr_cls"]) for method in METHODS}
        row: dict[str, Any] = {
            "scenario_name": token,
            "scenario_type": ref["scenario_type"],
            "log_name": ref["log_name"],
            "avg_nr_cls": round(mean(list(scores.values())), 6),
            "method_success_count": method_success_count(data, token),
            "method_failure_count": len(METHODS) - method_success_count(data, token),
        }
        for method, score in scores.items():
            row[f"{method}_nr_cls"] = round(score, 6)
            row[f"{method}_success"] = bool(data[method][token]["success"])
            for metric in METRICS:
                row[f"{method}_{metric}"] = round(float(data[method][token][metric]), 6)
        rows.append(row)
    assign_within_type_difficulty(rows)
    return rows


def distribute(total: int, buckets: list[str]) -> dict[str, int]:
    base = total // len(buckets)
    remainder = total % len(buckets)
    return {bucket: base + (1 if idx < remainder else 0) for idx, bucket in enumerate(buckets)}


def score_vector(row: dict[str, Any]) -> list[float]:
    return [float(row[f"{method}_nr_cls"]) for method in METHOD_KEYS]


def score_sums(rows: Sequence[dict[str, Any]]) -> list[float]:
    sums = [0.0 for _ in METHOD_KEYS]
    for row in rows:
        for idx, score in enumerate(score_vector(row)):
            sums[idx] += score
    return sums


def build_cell_options(
    scenario_type: str,
    difficulty: str,
    candidates: list[dict[str, Any]],
    count: int,
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda row: (float(row["avg_nr_cls"]), str(row["scenario_name"])))
    if len(ordered) < count:
        raise ValueError(
            f"Not enough candidates for {scenario_type}::{difficulty}: "
            f"need {count}, found {len(ordered)}"
        )

    cell_method_means = [
        mean([float(row[f"{method}_nr_cls"]) for row in ordered])
        for method in METHOD_KEYS
    ]
    cell_avg_nr_cls_mean = mean([float(row["avg_nr_cls"]) for row in ordered])
    rank_by_token = {str(row["scenario_name"]): idx + 1 for idx, row in enumerate(ordered)}

    if len(ordered) == count:
        raw_combos = [tuple(ordered)]
    else:
        raw_combos = list(itertools.combinations(ordered, count))

    options: list[dict[str, Any]] = []
    for combo in raw_combos:
        sums = score_sums(combo)
        combo_means = [value / count for value in sums]
        combo_avg_nr_cls_mean = mean([float(row["avg_nr_cls"]) for row in combo])
        method_error = sum(
            (combo_means[idx] - cell_method_means[idx]) ** 2
            for idx in range(len(METHOD_KEYS))
        )
        avg_error = (combo_avg_nr_cls_mean - cell_avg_nr_cls_mean) ** 2
        options.append(
            {
                "rows": combo,
                "score_sums": sums,
                "cell_error": method_error + 0.25 * avg_error,
                "token_key": tuple(str(row["scenario_name"]) for row in combo),
            }
        )

    options = sorted(options, key=lambda option: (float(option["cell_error"]), option["token_key"]))
    for idx, option in enumerate(options, start=1):
        option["cell_error_rank"] = idx

    return {
        "key": f"{scenario_type}::{difficulty}",
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "target_count": count,
        "candidate_count": len(ordered),
        "rank_by_token": rank_by_token,
        "options": options,
    }


def calibration_eval(
    selected_sums: list[float],
    selected_count: int,
    target_means: list[float],
    cell_error_total: float,
) -> dict[str, Any]:
    selected_means = [score_sum / selected_count for score_sum in selected_sums]
    score_sse = sum(
        (selected_means[idx] - target_means[idx]) ** 2
        for idx in range(len(METHOD_KEYS))
    )
    objective = (
        CALIBRATION_SCORE_WEIGHT * score_sse
        + CALIBRATION_CELL_ERROR_WEIGHT * cell_error_total
    )
    return {
        "objective": objective,
        "score_sse": score_sse,
        "cell_error_total": cell_error_total,
        "selected_means": selected_means,
    }


def run_calibration_search(
    cells: list[dict[str, Any]],
    selected_count: int,
    target_means: list[float],
    seed: int,
) -> dict[str, Any]:
    if not cells:
        return {
            "indices": [],
            "eval": calibration_eval([0.0 for _ in METHOD_KEYS], 1, target_means, 0.0),
            "passes": 0,
            "improvements": 0,
            "starts": 0,
        }

    starts: list[list[int]] = [[0 for _ in cells]]
    rng = random.Random(seed)
    for _ in range(CALIBRATION_RESTARTS):
        starts.append([rng.randrange(len(cell["options"])) for cell in cells])

    unique_starts: list[list[int]] = []
    seen_starts: set[tuple[int, ...]] = set()
    for start in starts:
        key = tuple(start)
        if key not in seen_starts:
            seen_starts.add(key)
            unique_starts.append(start)

    best_result: Optional[dict[str, Any]] = None
    for start in unique_starts:
        indices = list(start)
        current_sums = [0.0 for _ in METHOD_KEYS]
        current_cell_error = 0.0
        for cell, option_idx in zip(cells, indices):
            option = cell["options"][option_idx]
            current_cell_error += float(option["cell_error"])
            for method_idx, value in enumerate(option["score_sums"]):
                current_sums[method_idx] += float(value)

        current_eval = calibration_eval(current_sums, selected_count, target_means, current_cell_error)
        total_improvements = 0
        used_passes = 0

        for pass_idx in range(CALIBRATION_MAX_PASSES):
            improved = False
            used_passes = pass_idx + 1

            for cell_idx, cell in enumerate(cells):
                old_option = cell["options"][indices[cell_idx]]
                base_sums = [
                    current_sums[method_idx] - float(old_option["score_sums"][method_idx])
                    for method_idx in range(len(METHOD_KEYS))
                ]
                base_cell_error = current_cell_error - float(old_option["cell_error"])

                best_option_idx = indices[cell_idx]
                best_option_eval = current_eval
                best_option_sums = current_sums
                best_option_cell_error = current_cell_error

                for option_idx, option in enumerate(cell["options"]):
                    if option_idx == indices[cell_idx]:
                        continue
                    candidate_sums = [
                        base_sums[method_idx] + float(option["score_sums"][method_idx])
                        for method_idx in range(len(METHOD_KEYS))
                    ]
                    candidate_cell_error = base_cell_error + float(option["cell_error"])
                    candidate_eval = calibration_eval(
                        candidate_sums,
                        selected_count,
                        target_means,
                        candidate_cell_error,
                    )
                    if candidate_eval["objective"] < best_option_eval["objective"] - 1e-15:
                        best_option_idx = option_idx
                        best_option_eval = candidate_eval
                        best_option_sums = candidate_sums
                        best_option_cell_error = candidate_cell_error

                if best_option_idx != indices[cell_idx]:
                    indices[cell_idx] = best_option_idx
                    current_eval = best_option_eval
                    current_sums = best_option_sums
                    current_cell_error = best_option_cell_error
                    total_improvements += 1
                    improved = True

            if not improved:
                break

        result = {
            "indices": indices,
            "eval": current_eval,
            "passes": used_passes,
            "improvements": total_improvements,
            "starts": len(unique_starts),
        }
        if best_result is None or result["eval"]["objective"] < best_result["eval"]["objective"] - 1e-15:
            best_result = result

    if best_result is None:
        raise RuntimeError("Calibration search did not produce a result")
    return best_result


def select_fast_rows(rows: list[dict[str, Any]], target_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["scenario_type"])].append(row)

    scenario_types = sorted(by_type)
    type_targets = distribute(target_size, scenario_types)
    cells: list[dict[str, Any]] = []
    difficulty_targets_by_type: dict[str, dict[str, int]] = {}

    for scenario_type in scenario_types:
        type_rows = by_type[scenario_type]
        difficulty_targets = distribute(type_targets[scenario_type], DIFFICULTY_ORDER)
        difficulty_targets_by_type[scenario_type] = difficulty_targets

        for difficulty in DIFFICULTY_ORDER:
            candidates = [
                row for row in type_rows
                if row["difficulty_bucket"] == difficulty
            ]
            count = difficulty_targets[difficulty]
            if count > 0:
                cells.append(build_cell_options(scenario_type, difficulty, candidates, count))

    selected_count = sum(cell["target_count"] for cell in cells)
    if selected_count != target_size:
        raise ValueError(f"Internal target mismatch: cell total {selected_count}, target_size {target_size}")

    target_means = [
        mean([float(row[f"{method}_nr_cls"]) for row in rows])
        for method in METHOD_KEYS
    ]
    best = run_calibration_search(cells, target_size, target_means, seed)

    selected: list[dict[str, Any]] = []
    for cell, option_idx in zip(cells, best["indices"]):
        option = cell["options"][option_idx]
        for row in option["rows"]:
            token = str(row["scenario_name"])
            row["selection_strategy"] = "full_trend_calibrated_score_vector_representative"
            row["selection_cell_rank"] = cell["rank_by_token"][token]
            row["selection_cell_size"] = cell["candidate_count"]
            row["selection_cell_target_count"] = cell["target_count"]
            row["selection_combo_rank"] = option["cell_error_rank"]
            row["selection_combo_error"] = round(float(option["cell_error"]), 9)
            row["selection_calibration_cell"] = cell["key"]
            selected.append(row)

    selected = sorted(
        selected,
        key=lambda row: (
            str(row["scenario_type"]),
            DIFFICULTY_ORDER.index(str(row["difficulty_bucket"])),
            int(row.get("selection_cell_rank", 0)),
            str(row["scenario_name"]),
        ),
    )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    selected_method_means = best["eval"]["selected_means"]
    summary = {
        "target_size": target_size,
        "selected_size": len(selected),
        "seed": seed,
        "selection_strategy": "full_trend_calibrated_score_vector_representative",
        "scenario_type_targets": type_targets,
        "difficulty_targets_by_type": difficulty_targets_by_type,
        "difficulty_order": DIFFICULTY_ORDER,
        "candidate_count": len(rows),
        "full_test_method_means": {
            method: round(target_means[idx], 9)
            for idx, method in enumerate(METHOD_KEYS)
        },
        "selected_method_mean_errors": {
            method: round(selected_method_means[idx] - target_means[idx], 9)
            for idx, method in enumerate(METHOD_KEYS)
        },
        "calibration": {
            "score_weight": CALIBRATION_SCORE_WEIGHT,
            "cell_error_weight": CALIBRATION_CELL_ERROR_WEIGHT,
            "restarts": CALIBRATION_RESTARTS,
            "unique_starts": best["starts"],
            "max_passes": CALIBRATION_MAX_PASSES,
            "passes_used_for_best": best["passes"],
            "improvements_for_best": best["improvements"],
            "objective": round(float(best["eval"]["objective"]), 12),
            "score_sse": round(float(best["eval"]["score_sse"]), 12),
            "cell_error_total": round(float(best["eval"]["cell_error_total"]), 9),
        },
        "selection_notes": [
            "Candidate pool is the complete four-method Test14-hard intersection.",
            "Selection does not require any method-specific failure.",
            "Difficulty is assigned within each scenario type by average NR-CLS across zeroshot, lossbased, randombucket, and llm_guided.",
            "The default 84-scene filter targets 6 scenes per scenario type and 2 scenes per within-type difficulty bucket.",
            "Within each scenario-type-by-difficulty cell, candidate pairs are scored by how well they represent the cell method-score vector.",
            "A deterministic local search then chooses one pair per cell so the final subset closely matches the full Test14-hard method means.",
            "This filter is a calibrated fast proxy for full Test14-hard, not an independent unseen benchmark.",
        ],
    }
    return selected, summary


def method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for method in METHODS:
        scores = [float(row[f"{method}_nr_cls"]) for row in rows]
        entry: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n": len(scores),
            "score": round(mean(scores), 6),
            "score_std": round(std(scores), 6),
            "perfect_count": sum(1 for row in rows if bool(row[f"{method}_success"])),
            "zero_score_count": sum(1 for score in scores if score == 0.0),
        }
        for metric in METRICS:
            entry[metric] = round(mean([float(row[f"{method}_{metric}"]) for row in rows]), 6)
        summary.append(entry)
    return summary


def grouped_method_summary(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_field])].append(row)

    out = []
    for group, group_rows in sorted(groups.items()):
        for method in METHODS:
            scores = [float(row[f"{method}_nr_cls"]) for row in group_rows]
            out.append(
                {
                    group_field: group,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n": len(scores),
                    "score": round(mean(scores), 6),
                    "perfect_count": sum(1 for row in group_rows if bool(row[f"{method}_success"])),
                    "zero_score_count": sum(1 for score in scores if score == 0.0),
                }
            )
    return out


def pairwise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    method_keys = list(METHODS)
    for left in method_keys:
        for right in method_keys:
            if left == right:
                continue
            deltas = [float(row[f"{left}_nr_cls"]) - float(row[f"{right}_nr_cls"]) for row in rows]
            out.append(
                {
                    "left": left,
                    "right": right,
                    "left_label": METHOD_LABELS[left],
                    "right_label": METHOD_LABELS[right],
                    "left_better": sum(1 for delta in deltas if delta > 1e-9),
                    "tie": sum(1 for delta in deltas if abs(delta) <= 1e-9),
                    "right_better": sum(1 for delta in deltas if delta < -1e-9),
                    "mean_delta": round(mean(deltas), 6),
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
        "selected_avg_nr_cls_mean": round(mean([float(row["avg_nr_cls"]) for row in rows]), 6),
        "selected_avg_nr_cls_std": round(std([float(row["avg_nr_cls"]) for row in rows]), 6),
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
        "avg_nr_cls",
        "selection_strategy",
        "selection_cell_rank",
        "selection_cell_size",
        "selection_cell_target_count",
        "selection_combo_rank",
        "selection_combo_error",
        "selection_calibration_cell",
        "method_success_count",
        "method_failure_count",
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
    parser.add_argument("--output-filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--target-size", type=int, default=84)
    parser.add_argument("--seed", type=int, default=1405)
    args = parser.parse_args()

    data = load_metrics(args.exp_root)
    candidates = base_rows(data)
    rows, summary = select_fast_rows(candidates, args.target_size, args.seed)

    summary.update(selection_summary(rows))
    summary["method_summary"] = method_summary(rows)
    summary["pairwise_summary"] = pairwise_summary(rows)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_filter(args.output_filter, rows)
    write_csv(args.artifact_dir / "selection.csv", rows)
    write_named_csv(args.artifact_dir / "method_summary.csv", summary["method_summary"])
    write_named_csv(args.artifact_dir / "by_difficulty.csv", grouped_method_summary(rows, "difficulty_bucket"))
    write_named_csv(args.artifact_dir / "by_scenario_type.csv", grouped_method_summary(rows, "scenario_type"))
    write_named_csv(args.artifact_dir / "pairwise_summary.csv", summary["pairwise_summary"])
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
