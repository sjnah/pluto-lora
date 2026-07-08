#!/usr/bin/env python3
"""Collect and summarize quick-test evaluation results.

Examples:
  python scripts/evaluation/collect_quick_test_results.py --tests all
  python scripts/evaluation/collect_quick_test_results.py --tests val14
  python scripts/evaluation/collect_quick_test_results.py --tests test14-hard,interplan10
  python scripts/evaluation/collect_quick_test_results.py --tests val14 --methods zeroshot,rulebased,lossbased
  python scripts/evaluation/collect_quick_test_results.py --tests val14 --methods curriculum_llm_guided_v2
  python scripts/evaluation/collect_quick_test_results.py --tests val14 --detail
  python scripts/evaluation/collect_quick_test_results.py --tests all --format csv --output artifacts/records/quick_test_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RECORDS_DIR = REPO_ROOT / "artifacts" / "records" / "scenario_records"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "records" / "batched_runs"


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def resolve_runtime_root() -> Path:
    if "NUPLAN_RUNTIME_ROOT" in os.environ:
        return Path(os.environ["NUPLAN_RUNTIME_ROOT"])
    if path_exists(Path("/root/vessl-nuplan")):
        return Path("/root/vessl-nuplan")
    return WORKSPACE_ROOT / "nuplan-devkit" / "nuplan"


def resolve_default_exp_root() -> Path:
    explicit = os.environ.get("NUPLAN_EXP_ROOT")
    if explicit and os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1":
        return Path(explicit)
    return resolve_runtime_root() / "exp"


DEFAULT_NUPLAN_EXP_ROOT = resolve_default_exp_root()
DEFAULT_EXP_ROOT = DEFAULT_NUPLAN_EXP_ROOT / "exp"


@dataclass(frozen=True)
class TestSpec:
    key: str
    label: str
    suffix: str
    kind: str
    expected: int | None = None


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


TEST_SPECS = {
    "val14_benchmark": TestSpec("val14_benchmark", "Val14 benchmark", "val14_benchmark", "nuplan", 50),
    "val14_fast": TestSpec("val14_fast", "Val14 fast", "val14_fast", "nuplan", 270),
    "val_easy": TestSpec("val_easy", "Val easy", "llm_guided_val_easy", "nuplan", 50),
    "val_medium": TestSpec("val_medium", "Val medium", "llm_guided_val_medium", "nuplan", 50),
    "val_hard": TestSpec("val_hard", "Val hard", "llm_guided_val_hard", "nuplan", 50),
    "test14_hard": TestSpec("test14_hard", "Test14-hard", "test14_hard", "nuplan", 286),
    "test14_hard_fast": TestSpec(
        "test14_hard_fast",
        "Test14-hard fast",
        "test14_hard_fast",
        "nuplan",
        84,
    ),
    "test14_hard_llm_failure": TestSpec(
        "test14_hard_llm_failure",
        "Test14-hard LLM-failure",
        "test14_hard_llm_failure",
        "nuplan",
        93,
    ),
    "interplan10": TestSpec("interplan10", "InterPlan interplan10", "interplan10", "interplan", 80),
    "interplan_benchmark": TestSpec(
        "interplan_benchmark",
        "InterPlan benchmark_scenarios",
        "benchmark_scenarios",
        "interplan",
        335,
    ),
}

TEST_ALIASES = {
    "all": list(TEST_SPECS),
    "val14": ["val14_benchmark", "val_easy", "val_medium", "val_hard"],
    "val": ["val14_benchmark", "val_easy", "val_medium", "val_hard"],
    "val14-benchmark": ["val14_benchmark"],
    "val14_benchmark": ["val14_benchmark"],
    "val14-fast": ["val14_fast"],
    "val14_fast": ["val14_fast"],
    "easy": ["val_easy"],
    "medium": ["val_medium"],
    "hard": ["val_hard"],
    "test14": ["test14_hard"],
    "test14-hard": ["test14_hard"],
    "test14_hard": ["test14_hard"],
    "test14-hard-fast": ["test14_hard_fast"],
    "test14_hard_fast": ["test14_hard_fast"],
    "fast": ["test14_hard_fast"],
    "test14-hard-llm-failure": ["test14_hard_llm_failure"],
    "test14_hard_llm_failure": ["test14_hard_llm_failure"],
    "llm-failure": ["test14_hard_llm_failure"],
    "llm_failure": ["test14_hard_llm_failure"],
    # Backward-compatible aliases for the former diagnostic-set name.
    "test14-hard-sentinel": ["test14_hard_llm_failure"],
    "test14_hard_sentinel": ["test14_hard_llm_failure"],
    "sentinel": ["test14_hard_llm_failure"],
    "interplan": ["interplan10", "interplan_benchmark"],
    "interplan10": ["interplan10"],
    "interplan-benchmark": ["interplan_benchmark"],
    "interplan_benchmark": ["interplan_benchmark"],
    "benchmark-scenarios": ["interplan_benchmark"],
    "benchmark_scenarios": ["interplan_benchmark"],
    "val-easy": ["val_easy"],
    "val-medium": ["val_medium"],
    "val-hard": ["val_hard"],
}

METHOD_SPECS = {
    "zeroshot": MethodSpec("zeroshot", "Zero-shot"),
    "rulebased": MethodSpec("rulebased", "Rule-based"),
    "lossbased": MethodSpec("lossbased", "Loss-based"),
    "curriculum_uniform": MethodSpec("curriculum_uniform", "Curriculum uniform"),
    "curriculum_randombucket": MethodSpec("curriculum_randombucket", "RandomBucket"),
    "curriculum_llm_guided_v2": MethodSpec("curriculum_llm_guided_v2", "Curriculum LLM-guided v2"),
    "curriculum_llmbased": MethodSpec("curriculum_llmbased", "Curriculum LLM-based (legacy)"),
    "curriculum_mpoc": MethodSpec("curriculum_mpoc", "Curriculum MPOC"),
}

DEFAULT_METHODS = [
    "zeroshot",
    "rulebased",
    "lossbased",
    "curriculum_uniform",
    "curriculum_randombucket",
    "curriculum_llm_guided_v2",
    "curriculum_llmbased",
    "curriculum_mpoc",
]

NRCLS_MULTIPLE_METRICS = [
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
]

NRCLS_WEIGHTED_METRICS = {
    "ego_progress_along_expert_route": 5.0,
    "time_to_collision_within_bound": 5.0,
    "speed_limit_compliance": 4.0,
    "ego_is_comfortable": 2.0,
}

NRCLS_DETAIL_COLUMNS = [
    ("no_ego_at_fault_collisions", "without_collision", "w/o Collision"),
    ("drivable_area_compliance", "drivable", "Drivable"),
    ("ego_is_making_progress", "progress", "Progress"),
    ("driving_direction_compliance", "direction", "Direction"),
    ("ego_progress_along_expert_route", "expert_route", "Expert route"),
    ("time_to_collision_within_bound", "ttc", "TTC"),
    ("speed_limit_compliance", "speed_limit", "in Speed limit"),
    ("ego_is_comfortable", "comfortable", "Comfortable"),
]


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def llm_guided_label(method_key: str) -> str:
    prefix = "curriculum_llm_guided_"
    version = method_key[len(prefix):] if method_key.startswith(prefix) else method_key
    return f"Curriculum LLM-guided {version}"


def uniform_label(method_key: str) -> str:
    prefix = "curriculum_uniform_"
    version = method_key[len(prefix):] if method_key.startswith(prefix) else method_key
    return f"Uniform FT {version}"


def is_llm_guided_version_method(method_key: str) -> bool:
    return re.fullmatch(r"curriculum_llm_guided_v[0-9][A-Za-z0-9._-]*", method_key) is not None


def is_uniform_version_method(method_key: str) -> bool:
    return re.fullmatch(r"curriculum_uniform_v[0-9][A-Za-z0-9._-]*", method_key) is not None


def is_auto_discovered_version_method(method_key: str) -> bool:
    return is_llm_guided_version_method(method_key) or is_uniform_version_method(method_key)


def method_spec_for_key(method_key: str) -> MethodSpec | None:
    spec = METHOD_SPECS.get(method_key)
    if spec is not None:
        return spec
    if is_llm_guided_version_method(method_key):
        return MethodSpec(method_key, llm_guided_label(method_key))
    if is_uniform_version_method(method_key):
        return MethodSpec(method_key, uniform_label(method_key))
    return None


def expand_tests(value: str) -> list[TestSpec]:
    keys: list[str] = []
    for token in parse_csv_arg(value):
        normalized = token.lower().replace("_", "-")
        if normalized not in TEST_ALIASES:
            valid = ", ".join(sorted(TEST_ALIASES))
            raise SystemExit(f"Unknown test '{token}'. Valid values: {valid}")
        keys.extend(TEST_ALIASES[normalized])

    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return [TEST_SPECS[key] for key in deduped]


def add_unique_method_key(keys: list[str], method_key: str) -> None:
    if method_key not in keys:
        keys.append(method_key)


def extract_method_from_experiment_name(test: TestSpec, name: str) -> str | None:
    if test.kind == "interplan":
        prefix = "quick_test_interplan_"
    else:
        prefix = "quick_test_"
    suffix = f"_{test.suffix}"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    method_key = name[len(prefix) : -len(suffix)]
    return method_key or None


def discover_versioned_methods(
    tests: Iterable[TestSpec],
    exp_root: Path,
    records_dir: Path,
    manifest_dir: Path,
) -> list[str]:
    discovered: list[str] = []
    search_roots = [exp_root, records_dir, manifest_dir]
    for test in tests:
        prefix = "quick_test_interplan_" if test.kind == "interplan" else "quick_test_"
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.glob(f"{prefix}*_{test.suffix}*"):
                name = path.stem if path.is_file() else path.name
                method_key = extract_method_from_experiment_name(test, name)
                if method_key and is_auto_discovered_version_method(method_key):
                    add_unique_method_key(discovered, method_key)
    return sorted(discovered)


def expand_methods(
    value: str,
    tests: Iterable[TestSpec],
    exp_root: Path,
    records_dir: Path,
    manifest_dir: Path,
) -> list[MethodSpec]:
    if value == "all":
        keys = list(DEFAULT_METHODS)
        for method_key in discover_versioned_methods(tests, exp_root, records_dir, manifest_dir):
            add_unique_method_key(keys, method_key)
    else:
        keys = []
        for token in parse_csv_arg(value):
            normalized = token.lower().replace("-", "_")
            if method_spec_for_key(normalized) is None:
                valid = ", ".join(sorted(METHOD_SPECS))
                raise SystemExit(f"Unknown method '{token}'. Valid values: {valid}")
            keys.append(normalized)
    return [spec for key in keys if (spec := method_spec_for_key(key)) is not None]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def experiment_name(test: TestSpec, method: MethodSpec) -> str:
    if test.kind == "interplan":
        return f"quick_test_interplan_{method.key}_{test.suffix}"
    return f"quick_test_{method.key}_{test.suffix}"


def load_record(exp_name: str, records_dir: Path) -> dict[str, Any]:
    return load_json(records_dir / f"{exp_name}.json")


def load_manifest(exp_name: str, manifest_dir: Path) -> dict[str, Any]:
    return load_json(manifest_dir / f"{exp_name}.json")


def metrics_from_record(record: dict[str, Any]) -> list[Path]:
    paths = []
    for raw_path in record.get("resolved_metrics_dirs", []):
        path = resolve_path(str(raw_path))
        if path.exists():
            paths.append(path)
    return paths


def metrics_from_manifest(manifest: dict[str, Any]) -> list[Path]:
    paths = []
    for batch in manifest.get("batches", []):
        if not batch.get("success", False):
            continue
        raw_path = batch.get("metrics_dir")
        if not raw_path:
            continue
        path = resolve_path(str(raw_path))
        if path.exists():
            paths.append(path)
    return paths


def find_batch_dirs(exp_root: Path, exp_name: str) -> list[Path]:
    def batch_num(path: Path) -> int:
        match = re.search(r"_batch(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(exp_root.glob(f"{exp_name}_batch*"), key=batch_num)


def find_nuplan_metrics_dirs(
    exp_root: Path,
    records_dir: Path,
    manifest_dir: Path,
    exp_name: str,
) -> tuple[list[Path], str, int | None]:
    record = load_record(exp_name, records_dir)
    record_count = record.get("count")
    record_metrics = metrics_from_record(record)
    if record_metrics:
        return record_metrics, "record", int(record_count) if isinstance(record_count, int) else None

    manifest = load_manifest(exp_name, manifest_dir)
    manifest_metrics = metrics_from_manifest(manifest)
    if manifest_metrics:
        total = manifest.get("total_scenarios")
        return manifest_metrics, "manifest", int(total) if isinstance(total, int) else None

    direct_metrics = exp_root / exp_name / "metrics"
    if direct_metrics.exists():
        return [direct_metrics], "direct", None

    batch_metrics = [
        batch_dir / "metrics"
        for batch_dir in find_batch_dirs(exp_root, exp_name)
        if (batch_dir / "metrics").exists()
    ]
    if batch_metrics:
        return batch_metrics, "glob", None

    return [], "missing", record_count if isinstance(record_count, int) else None


def find_interplan_metric_dir(exp_root: Path, exp_name: str) -> tuple[Path | None, str]:
    exp_dir = exp_root / exp_name
    benchmark_dir = exp_dir / "default_interplan_benchmark"
    if benchmark_dir.exists():
        timestamp_dirs = [
            path for path in benchmark_dir.iterdir()
            if path.is_dir() and re.match(r"\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}", path.name)
        ]
        if timestamp_dirs:
            latest = sorted(timestamp_dirs, key=lambda path: path.name)[-1]
            aggregator = latest / "aggregator_metric"
            if aggregator.exists():
                return aggregator, "interplan_aggregator"
            metrics = latest / "metrics"
            if metrics.exists():
                return metrics, "interplan_metrics"

        aggregator = benchmark_dir / "aggregator_metric"
        if aggregator.exists():
            return aggregator, "interplan_aggregator"

    direct_metrics = exp_dir / "metrics"
    if direct_metrics.exists():
        return direct_metrics, "direct"

    return None, "missing"


def import_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pandas is required to compute scores from parquet metrics. "
            "Run this inside the nuplan conda environment."
        ) from exc
    return pd


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def calculate_nr_cls(metrics_dirs: list[Path]) -> dict[str, Any]:
    pd = import_pandas()
    all_metrics = NRCLS_MULTIPLE_METRICS + list(NRCLS_WEIGHTED_METRICS)
    scenario_metrics: dict[str, dict[str, float]] = {}
    missing = set()
    invalid = []

    for metrics_dir in metrics_dirs:
        for metric_name in all_metrics:
            metric_file = metrics_dir / f"{metric_name}.parquet"
            if not metric_file.exists():
                missing.add(metric_name)
                continue
            try:
                frame = pd.read_parquet(metric_file)
            except Exception as exc:
                invalid.append(f"{metric_name}: {exc}")
                continue
            if "scenario_name" not in frame.columns or "metric_score" not in frame.columns:
                invalid.append(f"{metric_name}: missing scenario_name or metric_score")
                continue
            for _, row in frame.iterrows():
                scenario_name = str(row["scenario_name"])
                metric_score = row["metric_score"]
                if pd.notna(metric_score) and isinstance(metric_score, (int, float)):
                    scenario_metrics.setdefault(scenario_name, {})[metric_name] = float(metric_score)

    found_metrics = {metric for values in scenario_metrics.values() for metric in values}
    missing_required = [metric for metric in all_metrics if metric not in found_metrics]
    if missing_required:
        return {
            "ok": False,
            "error": f"Missing required metrics: {', '.join(missing_required)}",
            "missing_metrics": sorted(set(missing_required) | missing),
            "invalid_metrics": invalid,
        }

    scores = []
    perfect = 0
    metric_means: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    total_weight = sum(NRCLS_WEIGHTED_METRICS.values())

    for scenario_name, values in scenario_metrics.items():
        if any(metric not in values for metric in all_metrics):
            continue

        multiplier = 1.0
        for metric in NRCLS_MULTIPLE_METRICS:
            multiplier *= values[metric]

        weighted_score = sum(values[metric] * weight for metric, weight in NRCLS_WEIGHTED_METRICS.items()) / total_weight
        score = multiplier * weighted_score
        scores.append(score)

        if all(values[metric] == 1.0 for metric in all_metrics):
            perfect += 1

    for metric in all_metrics:
        vals = [values[metric] for values in scenario_metrics.values() if metric in values]
        if vals:
            metric_means[metric] = mean(vals)
            metric_counts[metric] = len(vals)

    if not scores:
        return {"ok": False, "error": "No valid scenario scores calculated"}

    return {
        "ok": True,
        "metric_type": "NR-CLS",
        "score": mean(scores),
        "score_std": std(scores),
        "scenario_count": len(scores),
        "perfect_count": perfect,
        "metric_means": metric_means,
        "metric_counts": metric_counts,
    }


def calculate_interplan_score(metric_dir: Path) -> dict[str, Any]:
    pd = import_pandas()
    files = sorted(metric_dir.glob("closed_loop_reactive_agents_weighted_average_metrics_*.parquet"))
    if not files:
        return {"ok": False, "error": f"No interPlan aggregate parquet in {metric_dir}"}

    metric_file = files[-1]
    try:
        frame = pd.read_parquet(metric_file)
    except Exception as exc:
        return {"ok": False, "error": f"Could not read {metric_file}: {exc}"}

    if "scenario" not in frame.columns or "score" not in frame.columns:
        return {"ok": False, "error": f"Unexpected interPlan aggregate schema in {metric_file}"}

    final_rows = frame[frame["scenario"] == "final_score"]
    if final_rows.empty:
        return {"ok": False, "error": f"No final_score row in {metric_file}"}

    final = final_rows.iloc[0]
    scenario_rows = frame[frame["scenario"] != "final_score"]
    scenario_scores = [float(value) for value in scenario_rows["score"].dropna().tolist()]
    scenario_count = int(final.get("num_scenarios", len(scenario_scores)))

    return {
        "ok": True,
        "metric_type": "InterPlan",
        "score": float(final["score"]),
        "score_std": std(scenario_scores),
        "scenario_count": scenario_count,
        "perfect_count": sum(1 for score in scenario_scores if score >= 1.0),
        "metric_file": str(metric_file),
    }


def summarize_one(
    test: TestSpec,
    method: MethodSpec,
    exp_root: Path,
    records_dir: Path,
    manifest_dir: Path,
    include_detail: bool,
) -> dict[str, Any]:
    exp_name = experiment_name(test, method)
    row: dict[str, Any] = {
        "test": test.key,
        "test_label": test.label,
        "method": method.key,
        "method_label": method.label,
        "experiment": exp_name,
        "expected_scenarios": test.expected,
        "status": "missing",
        "score": None,
        "score_std": None,
        "scenario_count": None,
        "perfect_count": None,
        "metric_type": None,
        "source": None,
        "metrics_dirs": [],
        "error": None,
    }

    if test.kind == "interplan":
        metric_dir, source = find_interplan_metric_dir(exp_root, exp_name)
        row["source"] = source
        if metric_dir is None:
            row["error"] = f"No metrics found for {exp_name}"
            return row
        row["metrics_dirs"] = [str(metric_dir)]
        try:
            score = calculate_interplan_score(metric_dir)
        except RuntimeError as exc:
            row["status"] = "found_no_score"
            row["error"] = str(exc)
            return row
    else:
        metrics_dirs, source, recorded_count = find_nuplan_metrics_dirs(exp_root, records_dir, manifest_dir, exp_name)
        row["source"] = source
        row["metrics_dirs"] = [str(path) for path in metrics_dirs]
        if recorded_count is not None:
            row["scenario_count"] = recorded_count
        if not metrics_dirs:
            row["error"] = f"No metrics found for {exp_name}"
            return row
        try:
            score = calculate_nr_cls(metrics_dirs)
        except RuntimeError as exc:
            row["status"] = "found_no_score"
            row["error"] = str(exc)
            return row

    if not score.get("ok"):
        row["status"] = "invalid"
        row["error"] = score.get("error", "Unknown scoring error")
        return row

    row.update(
        {
            "status": "ok",
            "score": score["score"],
            "score_std": score.get("score_std"),
            "scenario_count": score.get("scenario_count", row.get("scenario_count")),
            "perfect_count": score.get("perfect_count"),
            "metric_type": score.get("metric_type"),
        }
    )
    if include_detail:
        metric_means = score.get("metric_means", {})
        metric_counts = score.get("metric_counts", {})
        row["metric_means"] = metric_means
        row["metric_counts"] = metric_counts
        for metric_name, key, _ in NRCLS_DETAIL_COLUMNS:
            row[key] = metric_means.get(metric_name)
    return row


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def print_table(rows: list[dict[str, Any]], include_detail: bool = False) -> None:
    columns = [
        ("test_label", "Test"),
        ("method_label", "Method"),
        ("status", "Status"),
        ("metric_type", "Metric"),
        ("scenario_count", "N"),
        ("score", "Score"),
        ("score_std", "Std"),
        ("perfect_count", "Perfect"),
        ("source", "Source"),
    ]
    if include_detail:
        columns.extend((key, label) for _, key, label in NRCLS_DETAIL_COLUMNS)
    detail_keys = {detail_key for _, detail_key, _ in NRCLS_DETAIL_COLUMNS}

    rendered_rows = []
    for row in rows:
        rendered = {}
        for key, _ in columns:
            value = row.get(key)
            if key in {"score", "score_std"} or key in detail_keys:
                rendered[key] = format_float(value)
            elif value is None:
                rendered[key] = "-"
            else:
                rendered[key] = str(value)
        rendered_rows.append(rendered)

    widths = {
        key: max(len(title), *(len(row[key]) for row in rendered_rows)) if rendered_rows else len(title)
        for key, title in columns
    }
    header = "  ".join(title.ljust(widths[key]) for key, title in columns)
    print(header)
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rendered_rows:
        print("  ".join(row[key].ljust(widths[key]) for key, _ in columns))


def write_csv(rows: list[dict[str, Any]], output: Path, include_detail: bool = False) -> None:
    fields = [
        "test",
        "test_label",
        "method",
        "method_label",
        "experiment",
        "status",
        "metric_type",
        "scenario_count",
        "expected_scenarios",
        "score",
        "score_std",
        "perfect_count",
        "source",
        "error",
        "metrics_dirs",
    ]
    if include_detail:
        fields.extend(key for _, key, _ in NRCLS_DETAIL_COLUMNS)
        fields.extend(["metric_means", "metric_counts"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["metrics_dirs"] = ";".join(row.get("metrics_dirs", []))
            if include_detail:
                flat["metric_means"] = json.dumps(row.get("metric_means", {}), sort_keys=True)
                flat["metric_counts"] = json.dumps(row.get("metric_counts", {}), sort_keys=True)
            writer.writerow({field: flat.get(field) for field in fields})


def write_json(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect quick-test result summaries.")
    parser.add_argument(
        "--tests",
        default="all",
        help="Comma-separated tests: all, val14, val14-fast, test14-hard, test14-hard-fast, test14-hard-llm-failure, interplan, interplan10, benchmark_scenarios.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods or all. Methods: zeroshot, rulebased, lossbased, "
            "curriculum_uniform, curriculum_randombucket, curriculum_llm_guided_v2, "
            "curriculum_uniform_v*, curriculum_llm_guided_v*, curriculum_llmbased, "
            "curriculum_mpoc. Versioned Uniform/LLM-guided methods present in result "
            "directories are auto-discovered for all."
        ),
    )
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--output", type=Path, help="Optional output path for json/csv/table text.")
    parser.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file.")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Include the eight NR-CLS component metric means in table/csv/json output.",
    )
    parser.add_argument("--include-missing", action="store_true", help="Show rows with no result directory.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any selected row is missing or invalid.")
    args = parser.parse_args()

    tests = expand_tests(args.tests)
    methods = expand_methods(args.methods, tests, args.exp_root, args.records_dir, args.manifest_dir)
    if args.output and args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing output: {args.output}\n"
            "Use a versioned --output path or pass --overwrite explicitly."
        )

    rows: list[dict[str, Any]] = []
    for test in tests:
        for method in methods:
            row = summarize_one(test, method, args.exp_root, args.records_dir, args.manifest_dir, args.detail)
            if args.include_missing or row["status"] != "missing":
                rows.append(row)

    if args.format == "json":
        text = json.dumps(rows, indent=2, sort_keys=True)
        if args.output:
            write_json(rows, args.output)
        else:
            print(text)
    elif args.format == "csv":
        if args.output:
            write_csv(rows, args.output, args.detail)
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    else:
        if args.output:
            lines: list[str] = []
            original_stdout = sys.stdout
            try:
                from io import StringIO

                buffer = StringIO()
                sys.stdout = buffer
                print_table(rows, args.detail)
                lines = buffer.getvalue().splitlines()
            finally:
                sys.stdout = original_stdout
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            print_table(rows, args.detail)

    if args.strict and any(row["status"] != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
