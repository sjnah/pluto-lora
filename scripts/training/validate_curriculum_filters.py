#!/usr/bin/env python3
"""Validate the scenario-filter contract for a bucketed curriculum method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


BUCKET_NAMES = ("easy", "medium", "hard")
EQUAL_CARDINALITY_QUANTILE = "equal_cardinality_quantile"


def load_scenario_tokens(path: Path) -> list[str]:
    """Load and validate a nuPlan scenario filter's token list."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    tokens = payload.get("scenario_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"Expected a non-empty scenario_tokens list: {path}")
    normalized = [str(token) for token in tokens]
    duplicate_count = len(normalized) - len(set(normalized))
    if duplicate_count:
        raise ValueError(
            f"Scenario filter contains {duplicate_count} duplicate token(s): {path}"
        )
    return normalized


def exact_quantile_counts(total: int, quantile_count: int = 3) -> list[int]:
    """Return exact rank-quantile sizes using integer floor boundaries."""
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if quantile_count <= 0:
        raise ValueError(f"quantile_count must be positive, got {quantile_count}")
    boundaries = [
        total * index // quantile_count for index in range(quantile_count + 1)
    ]
    return [
        boundaries[index + 1] - boundaries[index]
        for index in range(quantile_count)
    ]


def validate_equal_cardinality_filters(
    *,
    filter_dir: Path,
    filter_prefix: str,
    master_filter: str,
) -> dict[str, Any]:
    """Require disjoint rank terciles whose union matches all and master filters."""
    bucket_tokens = {
        bucket: load_scenario_tokens(
            filter_dir / f"{filter_prefix}_train_{bucket}.yaml"
        )
        for bucket in BUCKET_NAMES
    }
    all_tokens = load_scenario_tokens(filter_dir / f"{filter_prefix}_train_all.yaml")
    master_tokens = load_scenario_tokens(filter_dir / f"{master_filter}.yaml")

    token_to_bucket: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for bucket in BUCKET_NAMES:
        for token in bucket_tokens[bucket]:
            previous = token_to_bucket.setdefault(token, bucket)
            if previous != bucket:
                overlaps.append((token, previous, bucket))
    if overlaps:
        preview = ", ".join(
            f"{token}:{left}/{right}" for token, left, right in overlaps[:5]
        )
        raise ValueError(
            f"Curriculum buckets overlap ({len(overlaps)} token(s)): {preview}"
        )

    bucket_union = set(token_to_bucket)
    all_set = set(all_tokens)
    master_set = set(master_tokens)
    if bucket_union != all_set:
        raise ValueError(
            "Bucket union does not match the method all-filter: "
            f"missing={len(all_set - bucket_union)} extra={len(bucket_union - all_set)}"
        )
    if all_set != master_set:
        raise ValueError(
            "Method scenario universe does not match the declared master filter: "
            f"missing={len(master_set - all_set)} extra={len(all_set - master_set)}"
        )

    actual_counts = [len(bucket_tokens[bucket]) for bucket in BUCKET_NAMES]
    expected_counts = exact_quantile_counts(len(master_set), len(BUCKET_NAMES))
    if actual_counts != expected_counts:
        raise ValueError(
            "Buckets violate the equal-cardinality rank-quantile contract: "
            f"actual={actual_counts} expected={expected_counts}"
        )

    return {
        "bucket_cardinality_contract": EQUAL_CARDINALITY_QUANTILE,
        "filter_prefix": filter_prefix,
        "master_filter": master_filter,
        "total_scenarios": len(master_set),
        "bucket_counts": dict(zip(BUCKET_NAMES, actual_counts)),
    }


def validate_filter_contract(
    *,
    contract: str,
    filter_dir: Path,
    filter_prefix: str,
    master_filter: str,
) -> dict[str, Any]:
    """Dispatch validation for a declared curriculum bucket contract."""
    if contract == EQUAL_CARDINALITY_QUANTILE:
        return validate_equal_cardinality_filters(
            filter_dir=filter_dir,
            filter_prefix=filter_prefix,
            master_filter=master_filter,
        )
    raise ValueError(f"Unsupported bucket cardinality contract: {contract}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--filter-dir", type=Path, required=True)
    parser.add_argument("--filter-prefix", required=True)
    parser.add_argument("--master-filter", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_filter_contract(
        contract=args.contract,
        filter_dir=args.filter_dir,
        filter_prefix=args.filter_prefix,
        master_filter=args.master_filter,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
