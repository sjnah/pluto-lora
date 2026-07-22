#!/usr/bin/env python3
"""Extract deterministic target-control demand from nuPlan ego trajectories.

The score uses only the supervised 8 s ego target and is independent of model
predictions and semantic difficulty labels. Component ECDFs are fitted on the
selected training universe before applying the predeclared weighted score.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from omegaconf import DictConfig

from score_scenarios_by_loss import CONFIG_NAME, CONFIG_PATH, _resolve_path


COMPONENT_WEIGHTS = {
    "curvature_p90_percentile": 0.30,
    "max_lateral_displacement_percentile": 0.25,
    "speed_range_percentile": 0.20,
    "max_abs_acceleration_percentile": 0.15,
    "stop_go_transition_count_percentile": 0.10,
}


def _speed_mps(state: Any) -> float:
    velocity = state.dynamic_car_state.rear_axle_velocity_2d
    return math.hypot(float(velocity.x), float(velocity.y))


def _hysteresis_transition_count(
    speeds: Iterable[float], *, stopped_mps: float, moving_mps: float
) -> int:
    state: str | None = None
    transitions = 0
    for speed in speeds:
        next_state = (
            "stopped"
            if speed < stopped_mps
            else "moving"
            if speed > moving_mps
            else None
        )
        if next_state is None:
            continue
        if state is not None and next_state != state:
            transitions += 1
        state = next_state
    return transitions


def trajectory_metrics(
    scenario: Any,
    *,
    horizon_s: float,
    num_samples: int,
    min_curvature_step_m: float,
    stopped_mps: float,
    moving_mps: float,
) -> dict[str, Any]:
    initial = scenario.initial_ego_state
    future = list(
        scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=horizon_s,
            num_samples=num_samples,
        )
    )
    if len(future) != num_samples:
        raise ValueError(
            f"expected {num_samples} future states, received {len(future)}"
        )

    states = [initial, *future]
    positions = np.asarray([state.rear_axle.array for state in states], dtype=np.float64)
    headings = np.unwrap(
        np.asarray([state.rear_axle.heading for state in states], dtype=np.float64)
    )
    speeds = np.asarray([_speed_mps(state) for state in states], dtype=np.float64)

    anchor = positions[0]
    anchor_heading = headings[0]
    delta = positions - anchor
    cosine = math.cos(anchor_heading)
    sine = math.sin(anchor_heading)
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]

    segment_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    delta_heading = np.diff(headings)
    moving = segment_distance >= min_curvature_step_m
    curvature = np.abs(delta_heading[moving] / segment_distance[moving])
    curvature_p90 = float(np.percentile(curvature, 90)) if curvature.size else 0.0

    sample_interval_s = horizon_s / num_samples
    acceleration = np.diff(speeds) / sample_interval_s
    max_acceleration = max(0.0, float(acceleration.max()))
    max_deceleration = max(0.0, float((-acceleration).max()))

    return {
        "scenario_id": str(scenario.token),
        "log_name": str(scenario.log_name),
        "scenario_type": str(scenario.scenario_type),
        "status": "ok",
        "horizon_s": horizon_s,
        "num_samples": num_samples,
        "max_lateral_displacement_m": float(np.abs(local_y).max()),
        "absolute_heading_change_rad": float(abs(headings[-1] - headings[0])),
        "curvature_p90": curvature_p90,
        "speed_range_mps": float(speeds.max() - speeds.min()),
        "max_acceleration_mps2": max_acceleration,
        "max_deceleration_mps2": max_deceleration,
        "max_abs_acceleration_mps2": max(max_acceleration, max_deceleration),
        "stop_go_transition_count": _hysteresis_transition_count(
            speeds, stopped_mps=stopped_mps, moving_mps=moving_mps
        ),
    }


def average_ecdf(values: list[float]) -> list[float]:
    """Return average-tie empirical percentiles on [0, 1]."""
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    ordered = sorted(values)
    denominator = len(values) - 1
    percentiles = []
    for value in values:
        left = bisect.bisect_left(ordered, value)
        right = bisect.bisect_right(ordered, value)
        average_rank = (left + right - 1) / 2.0
        percentiles.append(average_rank / denominator)
    return percentiles


def add_control_demand_scores(rows: list[dict[str, Any]]) -> None:
    components = {
        "curvature_p90_percentile": "curvature_p90",
        "max_lateral_displacement_percentile": "max_lateral_displacement_m",
        "speed_range_percentile": "speed_range_mps",
        "max_abs_acceleration_percentile": "max_abs_acceleration_mps2",
        "stop_go_transition_count_percentile": "stop_go_transition_count",
    }
    for percentile_column, source_column in components.items():
        values = [float(row[source_column]) for row in rows]
        for row, percentile in zip(rows, average_ecdf(values)):
            row[percentile_column] = percentile

    for row in rows:
        row["target_control_demand_score"] = sum(
            weight * float(row[column])
            for column, weight in COMPONENT_WEIGHTS.items()
        )
    combined = [float(row["target_control_demand_score"]) for row in rows]
    for row, percentile in zip(rows, average_ecdf(combined)):
        row["target_control_demand_percentile"] = percentile


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    settings = cfg.get("target_control_demand", {})
    output_path = _resolve_path(
        str(
            settings.get(
                "output_path",
                "../llm-taxonomy/artifacts/target_control_demand/"
                "pluto_train_target_control_demand_v1.csv",
            )
        )
    )
    summary_path = output_path.with_suffix(".summary.json")
    horizon_s = float(settings.get("horizon_s", 8.0))
    num_samples = int(settings.get("num_samples", 80))
    min_curvature_step_m = float(settings.get("min_curvature_step_m", 0.05))
    stopped_mps = float(settings.get("stopped_mps", 0.3))
    moving_mps = float(settings.get("moving_mps", 1.0))

    worker = build_worker(cfg)
    scenarios = build_scenarios(cfg, worker, None)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, scenario in enumerate(scenarios, start=1):
        try:
            rows.append(
                trajectory_metrics(
                    scenario,
                    horizon_s=horizon_s,
                    num_samples=num_samples,
                    min_curvature_step_m=min_curvature_step_m,
                    stopped_mps=stopped_mps,
                    moving_mps=moving_mps,
                )
            )
        except Exception as exc:
            errors.append({"scenario_id": str(scenario.token), "error": str(exc)})
        if index % 250 == 0:
            print(f"Extracted {index}/{len(scenarios)} target trajectories")

    if errors:
        preview = "; ".join(
            f"{item['scenario_id']}: {item['error']}" for item in errors[:5]
        )
        raise RuntimeError(
            f"Target-control extraction failed for {len(errors)} scenarios: {preview}"
        )
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate scenario IDs in target-control extraction")

    add_control_demand_scores(rows)
    rows.sort(key=lambda row: str(row["scenario_id"]))
    write_csv(output_path, rows)
    summary = {
        "method": "trajectory_control_demand_v1",
        "scenario_filter": str(
            settings.get("reference_universe", "uniform_train_all")
        ),
        "scenario_count": len(rows),
        "horizon_s": horizon_s,
        "num_samples": num_samples,
        "min_curvature_step_m": min_curvature_step_m,
        "stopped_mps": stopped_mps,
        "moving_mps": moving_mps,
        "component_weights": COMPONENT_WEIGHTS,
        "percentile_reference_universe": "selected scenario_filter",
        "output_path": str(output_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} target-control rows to {output_path}")


if __name__ == "__main__":
    main()
