"""Curriculum bucketization and sampling utilities.

These helpers are intentionally pure-Python so the quota math can be unit tested
without importing the full PLUTO/nuPlan training stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


BUCKET_NAMES = ("easy", "medium", "hard")


def stable_hash_fraction(value: str, seed: int = 0) -> float:
    """Return a deterministic [0, 1) fraction for tie-breaking."""
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def normalize_proportions(proportions: Sequence[float]) -> List[float]:
    if len(proportions) != 3:
        raise ValueError(f"Expected three bucket proportions, got {len(proportions)}")
    if any(float(p) < 0.0 for p in proportions):
        raise ValueError(f"Bucket proportions must be non-negative: {proportions}")
    total = float(sum(proportions))
    if total <= 0.0:
        raise ValueError("At least one bucket proportion must be positive")
    return [float(p) / total for p in proportions]


def largest_remainder_counts(total: int, proportions: Sequence[float]) -> List[int]:
    """Convert proportions to integer draw counts with largest remainder."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    normalized = normalize_proportions(proportions)
    raw = [total * p for p in normalized]
    floors = [int(math.floor(value)) for value in raw]
    remainder = total - sum(floors)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - floors[index], -index),
        reverse=True,
    )
    counts = list(floors)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def exact_tercile_counts(total: int) -> List[int]:
    """Return bottom/middle/top tercile sizes using floor boundaries.

    For 3059 this returns 1019/1020/1020.
    """
    cut1 = total // 3
    cut2 = (2 * total) // 3
    return [cut1, cut2 - cut1, total - cut2]


def build_shuffled_cyclic_bucket_indices(
    indices: Sequence[int],
    draw_count: int,
    *,
    seed: int,
) -> List[int]:
    """Draw from a bucket as evenly as possible using shuffled cycles."""
    if draw_count < 0:
        raise ValueError(f"draw_count must be non-negative, got {draw_count}")
    if draw_count == 0:
        return []
    if not indices:
        raise ValueError("Cannot draw from an empty bucket")

    rng = random.Random(seed)
    base = list(indices)
    selected: List[int] = []
    full_cycles, remainder = divmod(draw_count, len(base))
    for _ in range(full_cycles):
        cycle = list(base)
        rng.shuffle(cycle)
        selected.extend(cycle)
    if remainder:
        cycle = list(base)
        rng.shuffle(cycle)
        selected.extend(cycle[:remainder])
    return selected


def repeat_stats(indices: Sequence[int], bucket_indices: Sequence[int]) -> Dict[str, float | int]:
    counts = Counter(indices)
    values = [counts[index] for index in bucket_indices]
    if not values:
        return {
            "unique_scenarios": 0,
            "min_repeat": 0,
            "mean_repeat": 0.0,
            "max_repeat": 0,
            "repeat_cap_reached": 0,
        }
    return {
        "unique_scenarios": len(bucket_indices),
        "min_repeat": min(values),
        "mean_repeat": sum(values) / len(values),
        "max_repeat": max(values),
        "repeat_cap_reached": 0,
    }


def build_exact_bucket_quota_indices(
    bucket_sizes: Sequence[int],
    target_proportions: Sequence[float],
    *,
    max_repeat_per_scenario: int,
    seed: int,
    epoch: int,
    shuffle_output: bool = True,
) -> Tuple[List[int], Dict[str, object]]:
    """Build one epoch of concat-dataset indices with exact bucket quotas."""
    if len(bucket_sizes) != 3:
        raise ValueError(f"Expected three bucket sizes, got {len(bucket_sizes)}")
    if any(size <= 0 for size in bucket_sizes):
        raise ValueError(f"Curriculum bucket sizes must be positive: {bucket_sizes}")
    if max_repeat_per_scenario <= 0:
        raise ValueError("max_repeat_per_scenario must be positive for exact quota sampling")

    total_samples = int(sum(bucket_sizes))
    requested_counts = largest_remainder_counts(total_samples, target_proportions)
    for bucket_name, bucket_size, draw_count in zip(BUCKET_NAMES, bucket_sizes, requested_counts):
        if draw_count > bucket_size * max_repeat_per_scenario:
            raise ValueError(
                f"Impossible exact quota for {bucket_name}: requested {draw_count} draws "
                f"but bucket_size {bucket_size} * max_repeat_per_scenario {max_repeat_per_scenario} "
                f"= {bucket_size * max_repeat_per_scenario}"
            )

    offsets = [0]
    for size in bucket_sizes[:-1]:
        offsets.append(offsets[-1] + int(size))

    selected: List[int] = []
    bucket_metadata: Dict[str, Dict[str, float | int]] = {}
    for bucket_idx, (bucket_name, bucket_size, draw_count, offset) in enumerate(
        zip(BUCKET_NAMES, bucket_sizes, requested_counts, offsets)
    ):
        bucket_indices = list(range(offset, offset + int(bucket_size)))
        bucket_seed = int(seed) + int(epoch) * 1009 + bucket_idx * 104729
        bucket_draws = build_shuffled_cyclic_bucket_indices(
            bucket_indices,
            draw_count,
            seed=bucket_seed,
        )
        selected.extend(bucket_draws)
        stats = repeat_stats(bucket_draws, bucket_indices)
        stats["repeat_cap_reached"] = sum(
            1
            for count in Counter(bucket_draws).values()
            if count >= max_repeat_per_scenario
        )
        stats["target_proportion"] = normalize_proportions(target_proportions)[bucket_idx]
        stats["requested_draws"] = draw_count
        stats["actual_draws"] = len(bucket_draws)
        stats["actual_proportion"] = len(bucket_draws) / total_samples if total_samples else 0.0
        bucket_metadata[bucket_name] = stats

    if shuffle_output:
        rng = random.Random(int(seed) + int(epoch) * 9176 + 17)
        rng.shuffle(selected)

    metadata = {
        "sampler_mode": "exact_bucket_quota",
        "epoch": int(epoch),
        "seed": int(seed),
        "total_samples": total_samples,
        "bucket_sizes": {
            bucket_name: int(size)
            for bucket_name, size in zip(BUCKET_NAMES, bucket_sizes)
        },
        "target_proportions": {
            bucket_name: normalize_proportions(target_proportions)[idx]
            for idx, bucket_name in enumerate(BUCKET_NAMES)
        },
        "requested_draws": {
            bucket_name: int(count)
            for bucket_name, count in zip(BUCKET_NAMES, requested_counts)
        },
        "actual_draws": {
            bucket_name: int(bucket_metadata[bucket_name]["actual_draws"])
            for bucket_name in BUCKET_NAMES
        },
        "actual_proportions": {
            bucket_name: float(bucket_metadata[bucket_name]["actual_proportion"])
            for bucket_name in BUCKET_NAMES
        },
        "repeat_stats": bucket_metadata,
    }
    return selected, metadata


@dataclass(frozen=True)
class PercentileSplitResult:
    groups: Dict[str, List[str]]
    metadata: Dict[str, object]


def split_scores_into_terciles(
    score_rows: Sequence[Tuple[str, float]],
    *,
    seed: int,
) -> PercentileSplitResult:
    """Split score rows into exact easy/medium/hard terciles."""
    if not score_rows:
        raise ValueError("No score rows supplied")
    seen = set()
    duplicates = []
    invalid = []
    for scenario_id, score in score_rows:
        if scenario_id in seen:
            duplicates.append(scenario_id)
        seen.add(scenario_id)
        if score is None or not math.isfinite(float(score)):
            invalid.append(scenario_id)
    if duplicates or invalid:
        raise ValueError(
            f"Invalid score rows: duplicates={len(duplicates)}, invalid_scores={len(invalid)}"
        )

    sorted_rows = sorted(
        ((str(scenario_id), float(score)) for scenario_id, score in score_rows),
        key=lambda item: (item[1], stable_hash_fraction(item[0], seed)),
    )
    counts = exact_tercile_counts(len(sorted_rows))
    boundaries = [counts[0], counts[0] + counts[1]]
    groups = {
        "easy": [scenario_id for scenario_id, _ in sorted_rows[: boundaries[0]]],
        "medium": [scenario_id for scenario_id, _ in sorted_rows[boundaries[0] : boundaries[1]]],
        "hard": [scenario_id for scenario_id, _ in sorted_rows[boundaries[1] :]],
    }

    score_counts = Counter(score for _, score in sorted_rows)
    tied_scene_count = sum(count for count in score_counts.values() if count > 1)
    tie_groups = sum(1 for count in score_counts.values() if count > 1)
    boundary_tie_scenes = 0
    boundary_tie_groups = 0
    for boundary in boundaries:
        if 0 < boundary < len(sorted_rows):
            left_score = sorted_rows[boundary - 1][1]
            right_score = sorted_rows[boundary][1]
            if left_score == right_score:
                boundary_tie_groups += 1
                boundary_tie_scenes += score_counts[left_score]

    metadata = {
        "bucketization_mode": "percentile_tercile",
        "percentile_split_seed": int(seed),
        "tie_break_mode": "stable_hash",
        "total_scenarios": len(sorted_rows),
        "bucket_counts": {name: len(groups[name]) for name in BUCKET_NAMES},
        "unique_score_count": len(score_counts),
        "tie_group_count": tie_groups,
        "tied_scene_count": tied_scene_count,
        "boundary_tie_group_count": boundary_tie_groups,
        "boundary_tie_scene_count": boundary_tie_scenes,
    }
    return PercentileSplitResult(groups=groups, metadata=metadata)


def validate_master_score_coverage(
    master_tokens: Iterable[str],
    score_rows: Sequence[Tuple[str, float]],
    *,
    allow_extra_scores: bool = False,
) -> Dict[str, object]:
    """Validate exact score coverage against a master scenario universe."""
    master = [str(token) for token in master_tokens]
    master_set = set(master)
    if len(master) != len(master_set):
        duplicates = [token for token, count in Counter(master).items() if count > 1]
        raise ValueError(f"Master universe has duplicate scenario tokens: {duplicates[:10]}")

    row_ids = [str(scenario_id) for scenario_id, _ in score_rows]
    row_counts = Counter(row_ids)
    duplicate_scores = sorted(token for token, count in row_counts.items() if count > 1)
    row_set = set(row_ids)
    missing_scores = sorted(master_set - row_set)
    extra_scores = sorted(row_set - master_set)
    invalid_scores = sorted(
        str(scenario_id)
        for scenario_id, score in score_rows
        if score is None or not math.isfinite(float(score))
    )

    report = {
        "master_size": len(master_set),
        "score_row_count": len(score_rows),
        "score_unique_count": len(row_set),
        "missing_score_count": len(missing_scores),
        "extra_score_count": len(extra_scores),
        "duplicate_score_count": len(duplicate_scores),
        "invalid_score_count": len(invalid_scores),
        "missing_scores": missing_scores,
        "extra_scores": extra_scores,
        "duplicate_scores": duplicate_scores,
        "invalid_scores": invalid_scores,
    }
    if missing_scores or duplicate_scores or invalid_scores or (extra_scores and not allow_extra_scores):
        raise ValueError(json.dumps(report, indent=2, sort_keys=True))
    return report


def write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
