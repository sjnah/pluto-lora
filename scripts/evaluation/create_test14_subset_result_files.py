#!/usr/bin/env python3
"""Create quick-test-compatible subset result directories from full Test14-hard metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLUTO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PLUTO_ROOT.parent

DEFAULT_EXP_ROOT = WORKSPACE_ROOT / "nuplan-devkit" / "nuplan" / "exp" / "exp"
DEFAULT_RECORDS_DIR = PLUTO_ROOT / "artifacts" / "records" / "scenario_records"
DEFAULT_ARTIFACT_ROOT = PLUTO_ROOT / "artifacts" / "records"

METHOD_EXPERIMENT_PREFIXES = {
    "zeroshot": "quick_test_zeroshot",
    "rulebased": "quick_test_rulebased",
    "lossbased": "quick_test_lossbased",
    "curriculum_uniform": "quick_test_curriculum_uniform",
    "curriculum_randombucket": "quick_test_curriculum_randombucket",
    "curriculum_llm_guided_v2": "quick_test_curriculum_llm_guided_v2",
    "curriculum_llmbased": "quick_test_curriculum_llmbased",
    "curriculum_mpoc": "quick_test_curriculum_mpoc",
}
DEFAULT_METHODS = [
    "zeroshot",
    "lossbased",
    "curriculum_randombucket",
    "curriculum_llm_guided_v2",
    "curriculum_llmbased",
]

SUBSETS = {
    "val14_fast": {
        "selection": DEFAULT_ARTIFACT_ROOT / "val14_fast" / "selection.csv",
        "source_suffix": "val14_benchmark",
        "dest_suffix": "val14_fast",
        "label": "Val14 fast",
    },
    "test14_hard_fast": {
        "selection": DEFAULT_ARTIFACT_ROOT / "test14_hard_fast" / "selection.csv",
        "source_suffix": "test14_hard",
        "dest_suffix": "test14_hard_fast",
        "label": "Test14-hard fast",
    },
    "test14_hard_llm_failure": {
        "selection": DEFAULT_ARTIFACT_ROOT / "test14_hard_llm_failure" / "selection.csv",
        "source_suffix": "test14_hard",
        "dest_suffix": "test14_hard_llm_failure",
        "label": "Test14-hard LLM-failure",
    },
}


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_tokens(selection_csv: Path) -> list[str]:
    if not selection_csv.exists():
        raise FileNotFoundError(f"Missing selection CSV: {selection_csv}")
    frame = pd.read_csv(selection_csv)
    if "scenario_name" not in frame.columns:
        raise ValueError(f"{selection_csv} missing scenario_name column")
    tokens = [str(token) for token in frame["scenario_name"].tolist()]
    unique_tokens = list(dict.fromkeys(tokens))
    if len(unique_tokens) != len(tokens):
        raise ValueError(f"{selection_csv} contains duplicate scenario_name values")
    return unique_tokens


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def subset_one_experiment(
    exp_root: Path,
    records_dir: Path,
    method: str,
    source_suffix: str,
    dest_suffix: str,
    tokens: list[str],
    label: str,
) -> dict[str, Any]:
    prefix = METHOD_EXPERIMENT_PREFIXES[method]
    source_exp = f"{prefix}_{source_suffix}"
    dest_exp = f"{prefix}_{dest_suffix}"
    direct_source_metrics = exp_root / source_exp / "metrics"
    source_metrics_dirs = [direct_source_metrics] if direct_source_metrics.exists() else []
    source_metrics_dirs.extend(sorted(exp_root.glob(f"{source_exp}_batch*/metrics")))
    dest_dir = exp_root / dest_exp
    dest_metrics = dest_dir / "metrics"

    if not source_metrics_dirs:
        raise FileNotFoundError(f"Missing source metrics for {method}: {direct_source_metrics}")

    token_set = set(tokens)
    metric_names = sorted({path.name for metrics_dir in source_metrics_dirs for path in metrics_dir.glob("*.parquet")})
    if not metric_names:
        raise FileNotFoundError(f"No parquet metric files under {source_metrics_dirs}")

    metric_counts: dict[str, int] = {}
    missing_by_metric: dict[str, list[str]] = {}
    subsets: dict[str, pd.DataFrame] = {}
    for metric_name in metric_names:
        frames = []
        for source_metrics in source_metrics_dirs:
            metric_file = source_metrics / metric_name
            if metric_file.exists():
                frames.append(pd.read_parquet(metric_file))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        if "scenario_name" not in frame.columns:
            continue
        frame = frame.drop_duplicates(subset=["scenario_name"], keep="last")
        subset = frame[frame["scenario_name"].astype(str).isin(token_set)].copy()
        found = set(subset["scenario_name"].astype(str).tolist())
        missing = sorted(token_set.difference(found))
        if missing:
            missing_by_metric[Path(metric_name).stem] = missing
        subsets[metric_name] = subset
        metric_counts[Path(metric_name).stem] = int(len(subset))

    if missing_by_metric:
        preview = {
            metric: missing[:5]
            for metric, missing in list(missing_by_metric.items())[:3]
        }
        raise ValueError(f"{dest_exp} missing selected tokens in source metrics: {preview}")

    dest_metrics.mkdir(parents=True, exist_ok=True)
    for metric_name, subset in subsets.items():
        subset.to_parquet(dest_metrics / metric_name, index=False)

    record = {
        "count": len(tokens),
        "source": "full_test_metric_subset",
        "source_experiment": source_exp,
        "source_metrics_dirs": [str(path) for path in source_metrics_dirs],
        "experiment": dest_exp,
        "label": label,
        "metrics_dir": str(dest_metrics),
        "resolved_metrics_dirs": [str(dest_metrics)],
        "scenario_tokens": tokens,
        "tokens": tokens,
    }
    write_json(records_dir / f"{dest_exp}.json", record)

    provenance = {
        "source": "full_test_metric_subset",
        "source_experiment": source_exp,
        "source_metrics_dirs": [str(path) for path in source_metrics_dirs],
        "dest_experiment": dest_exp,
        "dest_metrics_dir": str(dest_metrics),
        "label": label,
        "method": method,
        "scenario_count": len(tokens),
        "metric_counts": metric_counts,
    }
    write_json(dest_dir / "subset_from_full.json", provenance)
    (dest_dir / "log.txt").write_text(
        f"{dest_exp} was generated by slicing {source_exp} metric parquet files.\n",
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--subsets", default="test14_hard_fast,test14_hard_llm_failure")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()

    subset_keys = parse_csv_arg(args.subsets)
    method_keys = parse_csv_arg(args.methods)
    unknown_subsets = sorted(set(subset_keys).difference(SUBSETS))
    unknown_methods = sorted(set(method_keys).difference(METHOD_EXPERIMENT_PREFIXES))
    if unknown_subsets:
        raise SystemExit(f"Unknown subsets: {', '.join(unknown_subsets)}")
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {', '.join(unknown_methods)}")

    results = []
    for subset_key in subset_keys:
        subset = SUBSETS[subset_key]
        tokens = load_tokens(Path(subset["selection"]))
        for method in method_keys:
            results.append(
                subset_one_experiment(
                    args.exp_root,
                    args.records_dir,
                    method,
                    str(subset["source_suffix"]),
                    str(subset["dest_suffix"]),
                    tokens,
                    str(subset["label"]),
                )
            )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
