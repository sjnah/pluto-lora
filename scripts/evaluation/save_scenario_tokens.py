#!/usr/bin/env python3
"""Save scenario tokens observed in a nuPlan metrics directory.

This helper is intentionally best-effort: quick-test scripts use it only to
record the scenario set used by each run, so it should not fail a simulation
when a metric backend or optional parser is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

TOKEN_KEYS = {
    "scenario_token",
    "scenario_tokens",
    "token",
    "log_name",
    "scenario_name",
}


def _flatten_json(value) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TOKEN_KEYS and isinstance(item, (str, int)):
                yield str(item)
            yield from _flatten_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_json(item)


def _read_json(path: Path) -> Iterable[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            yield from _flatten_json(json.load(f))
    except Exception:
        return


def _read_csv(path: Path) -> Iterable[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in TOKEN_KEYS:
                    value = row.get(key)
                    if value:
                        yield value
    except Exception:
        return


def _read_parquet(path: Path) -> Iterable[str]:
    try:
        import pandas as pd
    except Exception:
        return

    try:
        frame = pd.read_parquet(path)
    except Exception:
        return

    for key in TOKEN_KEYS.intersection(frame.columns):
        for value in frame[key].dropna().astype(str).tolist():
            yield value


def _read_filter_tokens(filter_path: Path) -> Iterable[str]:
    if not filter_path.exists():
        return

    in_tokens = False
    try:
        with filter_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped == "scenario_tokens:":
                    in_tokens = True
                    continue
                if in_tokens and stripped and not stripped.startswith("-"):
                    break
                if in_tokens and stripped.startswith("-"):
                    token = stripped[1:].strip().strip("'\"")
                    if token:
                        yield token
    except Exception:
        return


def _read_tokens_from_batch_filter(metrics_dir: Path) -> Iterable[str]:
    overrides_path = metrics_dir.parent / "code" / "hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return

    try:
        lines = overrides_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- scenario_filter="):
            continue
        filter_name = stripped.split("=", 1)[1].strip()
        if not filter_name.startswith(".batch_filters/"):
            continue
        filter_path = REPO_ROOT / "config" / "scenario_filter" / f"{filter_name}.yaml"
        yield from _read_filter_tokens(filter_path)


def _manifest_path_for_metrics_dir(metrics_dir: Path) -> Path | None:
    if metrics_dir.name != "metrics":
        return None
    experiment_dir = metrics_dir.parent
    return REPO_ROOT / "artifacts" / "records" / "batched_runs" / f"{experiment_dir.name}.json"


def _read_tokens_from_manifest(metrics_dir: Path) -> Iterable[str]:
    manifest_path = _manifest_path_for_metrics_dir(metrics_dir)
    if manifest_path is None or not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for batch in manifest.get("batches", []):
        if not batch.get("success", False):
            continue
        for token in batch.get("tokens", []):
            if token:
                yield str(token)


def _resolve_metrics_dirs(metrics_dir: Path) -> list[Path]:
    if metrics_dir.exists():
        return [metrics_dir]

    # Batched quick tests write to sibling experiment directories:
    #   quick_test_name_batch0/metrics, quick_test_name_batch1/metrics, ...
    if metrics_dir.name == "metrics":
        experiment_dir = metrics_dir.parent
        manifest_path = _manifest_path_for_metrics_dir(metrics_dir)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_metrics = [
                    Path(batch["metrics_dir"])
                    for batch in manifest.get("batches", [])
                    if batch.get("success", False)
                ]
                existing_manifest_metrics = [path for path in manifest_metrics if path.exists()]
                if existing_manifest_metrics:
                    return existing_manifest_metrics
            except Exception:
                pass

        exp_root = experiment_dir.parent
        batch_metrics = sorted(exp_root.glob(f"{experiment_dir.name}_batch*/metrics"))
        existing_batch_metrics = [path for path in batch_metrics if path.exists()]
        if existing_batch_metrics:
            return existing_batch_metrics

    return []


def collect_tokens(metrics_dir: Path) -> list[str]:
    manifest_tokens = {token for token in _read_tokens_from_manifest(metrics_dir) if token}
    if manifest_tokens:
        return sorted(manifest_tokens)

    tokens: set[str] = set()
    metrics_dirs = _resolve_metrics_dirs(metrics_dir)
    if not metrics_dirs:
        return sorted(token for token in tokens if token)

    for resolved_metrics_dir in metrics_dirs:
        before_count = len(tokens)
        for path in resolved_metrics_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".json":
                tokens.update(_read_json(path))
            elif suffix == ".csv":
                tokens.update(_read_csv(path))
            elif suffix == ".parquet":
                tokens.update(_read_parquet(path))
        if len(tokens) == before_count:
            tokens.update(_read_tokens_from_batch_filter(resolved_metrics_dir))

    return sorted(token for token in tokens if token)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    resolved_metrics_dirs = _resolve_metrics_dirs(args.metrics_dir)
    tokens = collect_tokens(args.metrics_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metrics_dir": str(args.metrics_dir),
                "resolved_metrics_dirs": [str(path) for path in resolved_metrics_dirs],
                "count": len(tokens),
                "scenario_tokens": tokens,
                "tokens": tokens,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(tokens)} scenario tokens to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
