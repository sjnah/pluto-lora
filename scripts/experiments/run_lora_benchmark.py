#!/usr/bin/env python3
"""YAML-first PLUTO LoRA paper benchmark runner.

The benchmark YAML owns experiment semantics.  CLI arguments are optional
one-run overrides.  The initial migration intentionally keeps the canonical
training shell and benchmark-specific quick-test shells as leaf adapters, but
does not traverse the seeded/suite/unified wrapper stack.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


PLUTO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PLUTO_ROOT / "config/benchmark/paper_main_v1.yaml"
TRAINING_ADAPTER = PLUTO_ROOT / "scripts/training/run_lora_experiment.sh"
RESOLVER_DIR = PLUTO_ROOT / "scripts/training"
if str(RESOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(RESOLVER_DIR))
SAMPLING_DIR = PLUTO_ROOT / "src/custom_training"
if str(SAMPLING_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLING_DIR))

from resolve_lora_experiment_config import (  # noqa: E402
    digest as config_digest,
    load_yaml as resolver_load_yaml,
    resolve_protocol_method,
)
from curriculum_sampling import (  # noqa: E402
    largest_remainder_counts,
    scheduled_target_proportions,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
WORKFLOW_MODES = {"train_and_evaluate", "train_only", "evaluate_only"}
CHECKPOINT_POLICIES = {"reuse", "retrain", "error"}
CHECKPOINT_LINEAGE_FILENAME = ".checkpoint_lineage.json"
POST_TRAINING_DISCOVERY_ATTEMPTS = 30
POST_TRAINING_DISCOVERY_DELAY_SECONDS = 1.0
TRAINING_CONFIG_SNAPSHOT_ROOT = PLUTO_ROOT / "artifacts/benchmark_config_snapshots"
RESUME_POLICIES = {"auto", "fresh", "require_resume"}
SAMPLER_CONTRACTS = {"uniform", "exact_bucket_quota", "capped_weighted"}
BENCHMARKS = {
    "val14",
    "val14-fast",
    "test14-hard",
    "test14-hard-fast",
    "interplan10",
    "interplan-benchmark",
}
PRIORITY_BENCHMARK_ORDER = (
    "interplan10",
    "test14-hard-fast",
    "val14-fast",
    "test14-hard",
    "val14",
)
STAGE_CUDA_ENVIRONMENT = {
    "training": "PLUTO_TRAINING_CUDA_VISIBLE_DEVICES",
    "evaluation": "PLUTO_EVALUATION_CUDA_VISIBLE_DEVICES",
}


class ConfigError(ValueError):
    """Raised before any training or simulation for an invalid suite."""


@dataclass(frozen=True)
class ResolvedArm:
    arm_id: str
    kind: str
    label: str
    method: str
    artifact_version: str
    method_config: Optional[Path]
    sampler_contract: str
    routing_mode: str
    experiment_template: str
    slug_template: str
    method_values: Optional[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_yaml_snapshot(
    source_path: Path, *, expected_digest: str, category: str
) -> Path:
    """Create a content-addressed YAML snapshot and reject resolution races."""
    raw = source_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a YAML mapping: {source_path}")
    actual_digest = config_digest(payload)
    if actual_digest != expected_digest:
        raise ConfigError(
            f"{category} config changed while the benchmark was starting: "
            f"expected={expected_digest} actual={actual_digest} path={source_path}"
        )

    snapshot_dir = TRAINING_CONFIG_SNAPSHOT_ROOT / category
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{source_path.stem}-{expected_digest}.yaml"
    if snapshot_path.is_file():
        _, snapshot_payload = resolver_load_yaml(str(snapshot_path))
        snapshot_digest = config_digest(snapshot_payload)
        if snapshot_digest != expected_digest:
            raise ConfigError(
                f"Immutable {category} snapshot has unexpected content: {snapshot_path}"
            )
        return snapshot_path.resolve()

    temporary_path = snapshot_path.with_name(
        f"{snapshot_path.name}.tmp.{os.getpid()}"
    )
    temporary_path.write_text(raw, encoding="utf-8")
    temporary_path.replace(snapshot_path)
    return snapshot_path.resolve()


def freeze_training_configs(validated: dict[str, Any]) -> None:
    """Pin long-running training to the configs validated at suite startup."""
    if validated["mode"] == "evaluate_only":
        return
    protocol_path = immutable_yaml_snapshot(
        validated["protocol_path"],
        expected_digest=validated["protocol_sha256"],
        category="training_protocol",
    )
    frozen_arms: dict[str, ResolvedArm] = {}
    for arm_id, arm in validated["arms"].items():
        if arm.kind != "trainable":
            frozen_arms[arm_id] = arm
            continue
        assert arm.method_config is not None
        assert arm.method_values is not None
        method_path = immutable_yaml_snapshot(
            arm.method_config,
            expected_digest=arm.method_values["CFG_METHOD_SHA256"],
            category="curriculum_method",
        )
        frozen_arms[arm_id] = replace(arm, method_config=method_path)
    validated["protocol_path"] = protocol_path
    validated["arms"] = frozen_arms


def bundle_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "release_label": manifest["release_label"],
        "split": manifest["split"],
        "versions": manifest["versions"],
        "model": manifest["model"],
        "artifacts": {
            role: {
                key: value
                for key, value in artifact.items()
                if key != "source_path"
            }
            for role, artifact in sorted(manifest["artifacts"].items())
        },
        "contracts": manifest["contracts"],
    }


def validate_artifact_bundle(
    value: object,
    arms: dict[str, "ResolvedArm"],
    selected_arms: list[str],
) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = resolve_path(value, label="artifact_bundle_manifest")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid artifact bundle manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != 1 or manifest.get("state") != "immutable":
        raise ConfigError("Artifact bundle must use schema_version=1 and state=immutable")
    computed_id = canonical_digest(bundle_identity_payload(manifest))
    if manifest.get("bundle_id") != computed_id:
        raise ConfigError(
            "Artifact bundle identity mismatch: "
            f"manifest={manifest.get('bundle_id')} computed={computed_id}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ConfigError("Artifact bundle has no artifacts mapping")
    base = path.parent.resolve()
    for role, artifact in artifacts.items():
        if not isinstance(artifact, dict) or not artifact.get("path"):
            raise ConfigError(f"Invalid bundled artifact entry: {role}")
        artifact_path = (base / str(artifact["path"])).resolve()
        try:
            artifact_path.relative_to(base)
        except ValueError as exc:
            raise ConfigError(f"Bundled artifact escapes bundle directory: {role}") from exc
        if not artifact_path.is_file():
            raise ConfigError(f"Missing bundled artifact {role}: {artifact_path}")
        actual = file_sha256(artifact_path)
        if actual != artifact.get("sha256"):
            raise ConfigError(
                f"Bundled artifact drift for {role}: expected={artifact.get('sha256')} actual={actual}"
            )

    llm_arms = [arms[arm_id] for arm_id in selected_arms if arms[arm_id].method == "llm"]
    if not llm_arms:
        return {
            "path": str(path),
            "bundle_id": computed_id,
            "manifest_sha256": file_sha256(path),
            "release_label": str(manifest.get("release_label", "")),
            "routing_metadata_path": None,
        }
    contracts = manifest.get("contracts") or {}
    if contracts.get("filter_and_routing_share_source") is not True:
        raise ConfigError("Artifact bundle does not bind filter and routing to one score source")
    routing = artifacts.get("normalized_score")
    if not isinstance(routing, dict):
        raise ConfigError("Artifact bundle is missing normalized_score")
    routing_path = (base / str(routing["path"])).resolve()
    for arm in llm_arms:
        assert arm.method_values is not None
        prefix = str(arm.method_values["CFG_FILTER_PREFIX"])
        for bucket, role in (
            ("easy", "filter_easy"),
            ("medium", "filter_medium"),
            ("hard", "filter_hard"),
            ("all", "filter_all"),
        ):
            bundled = artifacts.get(role)
            if not isinstance(bundled, dict):
                raise ConfigError(f"Artifact bundle is missing {role}")
            active = PLUTO_ROOT / "config/scenario_filter" / f"{prefix}_train_{bucket}.yaml"
            if not active.is_file() or file_sha256(active) != bundled.get("sha256"):
                raise ConfigError(
                    f"Active PLUTO filter does not match bundle for {arm.arm_id}/{bucket}: {active}"
                )
    return {
        "path": str(path),
        "bundle_id": computed_id,
        "manifest_sha256": file_sha256(path),
        "release_label": str(manifest.get("release_label", "")),
        "routing_metadata_path": str(routing_path),
    }


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping at {key}")
    return value


def require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"Expected list at {key}")
    return value


def require_choice(value: object, choices: set[str], label: str) -> str:
    text = str(value or "")
    if text not in choices:
        raise ConfigError(f"Unsupported {label}={text!r}; choose from {sorted(choices)}")
    return text


def require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"Expected boolean at {label}")
    return value


def require_id(value: object, label: str) -> str:
    text = str(value or "")
    if not SAFE_ID.fullmatch(text):
        raise ConfigError(f"Invalid {label}: {text!r}")
    return text


def resolve_path(value: object, *, label: str, must_exist: bool = True) -> Path:
    text = str(value or "")
    if not text:
        raise ConfigError(f"Missing {label}")
    path = Path(text)
    if not path.is_absolute():
        path = PLUTO_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ConfigError(f"{label} does not exist: {path}")
    return path


def load_suite(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value)
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        path = cwd_candidate if cwd_candidate.is_file() else (PLUTO_ROOT / path).resolve()
    if not path.is_file():
        raise ConfigError(f"Benchmark config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid benchmark YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected YAML mapping: {path}")
    return path, payload


def parse_seed_override(value: str) -> list[int]:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise ConfigError(f"Invalid seed range: {value!r}")
        start, end = (int(part) for part in parts)
        if start < 0 or end < start:
            raise ConfigError(f"Invalid seed range: {value!r}")
        return list(range(start, end + 1))
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise ConfigError(f"Invalid seed list: {value!r}")
    return seeds


def unique_values(values: Iterable[Any], label: str) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ConfigError(f"Duplicate {label}: {value}")
        seen.add(value)
        result.append(value)
    return result


def resolve_benchmark_order(
    benchmarks: list[str], *, use_priority_order: bool
) -> list[str]:
    if not use_priority_order:
        return benchmarks
    priority = {
        benchmark: index
        for index, benchmark in enumerate(PRIORITY_BENCHMARK_ORDER)
    }
    return sorted(
        benchmarks,
        key=lambda benchmark: priority.get(benchmark, len(PRIORITY_BENCHMARK_ORDER)),
    )


def apply_cli_overrides(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(payload)
    workflow = require_mapping(resolved, "workflow")
    selection = require_mapping(resolved, "selection")
    training = require_mapping(resolved, "training")
    evaluation = require_mapping(resolved, "evaluation")
    if args.mode:
        workflow["mode"] = args.mode
    if args.arms:
        selection["arms"] = args.arms
    if args.seeds:
        selection["seeds"] = parse_seed_override(args.seeds)
    if args.benchmarks:
        selection["benchmarks"] = args.benchmarks
    if args.checkpoint_policy:
        training["existing_checkpoint"] = args.checkpoint_policy
    if args.dry_run is not None:
        workflow["dry_run"] = args.dry_run
    if args.skip_completed_evaluation is not None:
        evaluation["skip_completed_same_version_seed"] = (
            args.skip_completed_evaluation
        )
    if args.artifact_bundle_manifest:
        resolved["artifact_bundle_manifest"] = args.artifact_bundle_manifest
    return resolved


def render_template(template: str, *, artifact_version: str, protocol_id: str, seed: int) -> str:
    try:
        rendered = template.format(
            artifact_version=artifact_version,
            protocol_id=protocol_id,
            seed=seed,
        )
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"Invalid naming template {template!r}: {exc}") from exc
    return require_id(rendered, "rendered run name")


def validate_arm(
    arm_id: str,
    raw: dict[str, Any],
    protocol_path: Path,
) -> ResolvedArm:
    require_id(arm_id, "arm id")
    kind = str(raw.get("kind", "trainable"))
    if kind not in {"zero_shot", "trainable"}:
        raise ConfigError(f"Unsupported arms.{arm_id}.kind={kind!r}")
    label = str(raw.get("label") or arm_id)
    slug_template = str(raw.get("evaluation_slug_template") or "")
    if not slug_template:
        raise ConfigError(f"arms.{arm_id}.evaluation_slug_template is required")
    if kind == "zero_shot":
        return ResolvedArm(
            arm_id=arm_id,
            kind=kind,
            label=label,
            method="zero_shot",
            artifact_version="",
            method_config=None,
            sampler_contract="none",
            routing_mode="off",
            experiment_template="",
            slug_template=slug_template,
            method_values=None,
        )

    method = require_id(raw.get("method"), f"arms.{arm_id}.method")
    artifact_version = require_id(
        raw.get("artifact_version"), f"arms.{arm_id}.artifact_version"
    )
    method_path = resolve_path(raw.get("method_config"), label=f"arms.{arm_id}.method_config")
    sampler_contract = require_choice(
        raw.get("sampler_contract"), SAMPLER_CONTRACTS, f"arms.{arm_id}.sampler_contract"
    )
    routing_mode = require_choice(
        raw.get("routing_mode", "off"), {"off", "on"}, f"arms.{arm_id}.routing_mode"
    )
    experiment_template = str(raw.get("experiment_base_template") or "")
    if not experiment_template:
        raise ConfigError(f"arms.{arm_id}.experiment_base_template is required")

    values = resolve_protocol_method(str(protocol_path), str(method_path))
    if values["CFG_METHOD"] != method:
        raise ConfigError(
            f"Arm {arm_id} method={method} disagrees with {method_path}: "
            f"{values['CFG_METHOD']}"
        )
    mode = values["CFG_METHOD_MODE"]
    if sampler_contract == "uniform":
        if mode != "uniform":
            raise ConfigError(f"Arm {arm_id} declares uniform but method mode is {mode}")
    else:
        if mode != "bucketed":
            raise ConfigError(f"Arm {arm_id} declares curriculum but method mode is {mode}")
        expected_sampler = (
            "exact_bucket_quota"
            if sampler_contract == "exact_bucket_quota"
            else "exposure_capped_weighted"
        )
        if values["CFG_SAMPLER_MODE"] != expected_sampler:
            raise ConfigError(
                f"Arm {arm_id} contract={sampler_contract} requires sampler={expected_sampler}, "
                f"got {values['CFG_SAMPLER_MODE']}"
            )
        if sampler_contract == "exact_bucket_quota":
            if values["CFG_PERSISTENT_EXPOSURE"]:
                raise ConfigError(f"Exact comparison arm {arm_id} cannot use persistent exposure")
            if values["CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP"] > 0:
                raise ConfigError(f"Exact comparison arm {arm_id} cannot use group caps")
        else:
            if not values["CFG_PERSISTENT_EXPOSURE"]:
                raise ConfigError(f"Capped arm {arm_id} must use persistent exposure")
            if values["CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP"] <= 0:
                raise ConfigError(f"Capped arm {arm_id} must set a positive group cap")
            if not values["CFG_NEAR_DUPLICATE_GROUP_WEIGHTING"]:
                raise ConfigError(f"Capped arm {arm_id} must enable group weighting")

    if routing_mode == "on":
        if method != "llm" or not values["CFG_TYPE_ROUTING_SUPPORTED"]:
            raise ConfigError(f"Type routing is not supported for arm {arm_id}")
        expected_routing_sampler = (
            "exact_bucket_quota"
            if sampler_contract == "exact_bucket_quota"
            else "exposure_capped_weighted"
        )
        if values["CFG_TYPE_ROUTING_ENABLED_SAMPLER_MODE"] != expected_routing_sampler:
            raise ConfigError(
                f"Type-on arm {arm_id} must keep its matched sampler contract "
                f"({expected_routing_sampler})"
            )
        if (
            sampler_contract == "exact_bucket_quota"
            and values["CFG_TYPE_ROUTING_ALGORITHM"]
            != "paired_minimal_delta_v1"
        ):
            raise ConfigError(
                f"Exact type-on arm {arm_id} must declare "
                "type_routing.algorithm=paired_minimal_delta_v1"
            )
    elif method != "llm" and values["CFG_TYPE_ROUTING_SUPPORTED"]:
        raise ConfigError(f"Non-LLM arm {arm_id} unexpectedly advertises type routing")

    return ResolvedArm(
        arm_id=arm_id,
        kind=kind,
        label=label,
        method=method,
        artifact_version=artifact_version,
        method_config=method_path,
        sampler_contract=sampler_contract,
        routing_mode=routing_mode,
        experiment_template=experiment_template,
        slug_template=slug_template,
        method_values=values,
    )


def scenario_filter_tokens_from_path(path: Path) -> list[str]:
    if not path.is_file():
        raise ConfigError(f"Scenario filter does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Scenario filter is not a mapping: {path}")
    tokens = payload.get("scenario_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ConfigError(f"Scenario filter has no scenario_tokens: {path}")
    normalized = [str(token) for token in tokens]
    if len(normalized) != len(set(normalized)):
        raise ConfigError(f"Scenario filter contains duplicate tokens: {path}")
    return normalized


def scenario_filter_tokens(filter_name: str) -> list[str]:
    path = PLUTO_ROOT / "config/scenario_filter" / f"{filter_name}.yaml"
    return scenario_filter_tokens_from_path(path)


def exact_sampler_capacity_preflight(
    arm_id: str,
    arm: ResolvedArm,
    bucket_sizes: list[int],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    assert arm.method_values is not None
    phases = require_mapping(protocol, "phases")
    cumulative = require_mapping(phases, "cumulative_epochs")
    proportions = require_mapping(phases, "bucket_target_proportions")
    pacing = phases.get("pacing") or {}
    phase_alpha = pacing.get("phase_alpha") or {}
    boundaries = {
        "a": (0, int(cumulative["a"])),
        "b": (int(cumulative["a"]), int(cumulative["b"])),
        "c": (int(cumulative["b"]), int(cumulative["c"])),
    }
    total_samples = sum(bucket_sizes)
    max_repeat = int(arm.method_values["CFG_MAX_REPEAT_PER_SCENARIO"])
    summaries: list[dict[str, Any]] = []
    for phase, (start, stop) in boundaries.items():
        schedule = phase_alpha.get(phase) or {}
        for epoch in range(start, stop):
            resolved, pacing_meta = scheduled_target_proportions(
                proportions[phase], schedule, epoch=epoch, phase_start_epoch=start
            )
            counts = largest_remainder_counts(total_samples, resolved)
            for index, (count, size) in enumerate(zip(counts, bucket_sizes)):
                if max_repeat > 0 and count > size * max_repeat:
                    raise ConfigError(
                        f"Exact sampler infeasible for {arm_id} phase={phase} epoch={epoch}: "
                        f"bucket {index} requests {count}, capacity={size * max_repeat}"
                    )
            summaries.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "target_proportions": resolved,
                    "draw_counts": counts,
                    "pacing": pacing_meta,
                }
            )
    return summaries


def validate_filter_contracts(
    arms: dict[str, ResolvedArm],
    selected_arms: list[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    reference_tokens: Optional[set[str]] = None
    reference_arm: Optional[str] = None
    summary: dict[str, Any] = {}

    if "uniform" in selected_arms:
        uniform_arm = arms["uniform"]
        assert uniform_arm.method_values is not None
        uniform_name = str(uniform_arm.method_values["CFG_SCENARIO_FILTER_UNIFORM"])
        uniform_tokens = set(scenario_filter_tokens(uniform_name))
        reference_tokens = uniform_tokens
        reference_arm = "uniform"
        summary["uniform"] = {
            "filter": uniform_name,
            "unique_scenarios": len(uniform_tokens),
        }

    seen_prefixes: dict[str, tuple[set[str], list[int]]] = {}
    for arm_id in selected_arms:
        arm = arms[arm_id]
        if arm.kind != "trainable" or arm.sampler_contract == "uniform":
            continue
        assert arm.method_values is not None
        prefix = str(arm.method_values["CFG_FILTER_PREFIX"])
        if prefix in seen_prefixes:
            union, bucket_sizes = seen_prefixes[prefix]
        else:
            bucket_sets: list[set[str]] = []
            bucket_sizes = []
            for bucket in ("easy", "medium", "hard"):
                tokens = scenario_filter_tokens(f"{prefix}_train_{bucket}")
                token_set = set(tokens)
                bucket_sets.append(token_set)
                bucket_sizes.append(len(tokens))
            overlaps = {
                "easy_medium": len(bucket_sets[0] & bucket_sets[1]),
                "easy_hard": len(bucket_sets[0] & bucket_sets[2]),
                "medium_hard": len(bucket_sets[1] & bucket_sets[2]),
            }
            if any(overlaps.values()):
                raise ConfigError(f"Bucket overlap for {prefix}: {overlaps}")
            union = set().union(*bucket_sets)
            seen_prefixes[prefix] = (union, bucket_sizes)

        if reference_tokens is None:
            reference_tokens = union
            reference_arm = arm_id
        elif union != reference_tokens:
            missing = sorted(reference_tokens - union)
            extra = sorted(union - reference_tokens)
            raise ConfigError(
                f"Scenario universe mismatch: {arm_id} vs {reference_arm}; "
                f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
            )

        entry: dict[str, Any] = {
            "filter_prefix": prefix,
            "bucket_sizes": bucket_sizes,
            "unique_scenarios": len(union),
        }
        if arm.sampler_contract == "exact_bucket_quota":
            entry["epoch_plan"] = exact_sampler_capacity_preflight(
                arm_id, arm, bucket_sizes, protocol
            )
        summary[arm_id] = entry
    return summary


def validate_suite(payload: dict[str, Any]) -> dict[str, Any]:
    suite_id = require_id(payload.get("suite_id"), "suite_id")
    workflow = require_mapping(payload, "workflow")
    selection = require_mapping(payload, "selection")
    training = require_mapping(payload, "training")
    evaluation = require_mapping(payload, "evaluation")
    arms_payload = require_mapping(payload, "arms")

    mode = require_choice(workflow.get("mode"), WORKFLOW_MODES, "workflow.mode")
    interleave_evaluation_by_seed = require_bool(
        workflow.get("interleave_evaluation_by_seed", True),
        "workflow.interleave_evaluation_by_seed",
    )
    checkpoint_policy = require_choice(
        training.get("existing_checkpoint"), CHECKPOINT_POLICIES, "training.existing_checkpoint"
    )
    resume_policy = require_choice(
        training.get("resume_policy"), RESUME_POLICIES, "training.resume_policy"
    )
    selected_arms = unique_values(
        [str(value) for value in require_list(selection, "arms")], "selected arm"
    )
    seeds = unique_values([int(value) for value in require_list(selection, "seeds")], "seed")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ConfigError("selection.seeds must contain non-negative integers")
    benchmarks = unique_values(
        [str(value) for value in require_list(selection, "benchmarks")], "benchmark"
    )
    use_priority_benchmark_order = require_bool(
        selection.get("use_priority_benchmark_order", False),
        "selection.use_priority_benchmark_order",
    )
    unsupported = sorted(set(benchmarks) - BENCHMARKS)
    if unsupported:
        raise ConfigError(f"Unsupported benchmarks: {unsupported}")
    if mode != "train_only" and not benchmarks:
        raise ConfigError(f"workflow.mode={mode} requires at least one benchmark")
    benchmarks = resolve_benchmark_order(
        benchmarks, use_priority_order=use_priority_benchmark_order
    )

    protocol_path = resolve_path(payload.get("training_protocol"), label="training_protocol")
    _, protocol_payload = resolver_load_yaml(str(protocol_path))
    protocol_id = require_id(protocol_payload.get("protocol_id"), "protocol_id")
    protocol_sha256 = config_digest(protocol_payload)
    protocol_runtime = protocol_payload.get("runtime") or {}
    if not isinstance(protocol_runtime, dict):
        raise ConfigError("training_protocol.runtime must be a mapping when present")
    feature_cache_name = require_id(
        protocol_runtime.get("feature_cache_name"),
        "training_protocol.runtime.feature_cache_name",
    )
    pretrained_checkpoint = resolve_path(
        training.get("pretrained_checkpoint"), label="training.pretrained_checkpoint"
    )

    resolved_arms: dict[str, ResolvedArm] = {}
    for arm_id in selected_arms:
        raw = arms_payload.get(arm_id)
        if not isinstance(raw, dict):
            raise ConfigError(f"Selected arm is missing or not a mapping: {arm_id}")
        resolved_arms[arm_id] = validate_arm(arm_id, raw, protocol_path)

    skip_completed_evaluation = bool(
        evaluation.get("skip_completed_same_version_seed", False)
    )
    if skip_completed_evaluation:
        for arm_id, arm in resolved_arms.items():
            if arm.kind != "trainable":
                continue
            if (
                "{artifact_version}" not in arm.slug_template
                or "{seed}" not in arm.slug_template
            ):
                raise ConfigError(
                    "evaluation.skip_completed_same_version_seed requires "
                    f"arms.{arm_id}.evaluation_slug_template to contain "
                    "{artifact_version} and {seed}"
                )

    filter_contracts = validate_filter_contracts(
        resolved_arms, selected_arms, protocol_payload
    )
    artifact_bundle = validate_artifact_bundle(
        payload.get("artifact_bundle_manifest"), resolved_arms, selected_arms
    )

    capped_pair = [resolved_arms.get("llm_capped_off"), resolved_arms.get("llm_capped_on")]
    if all(capped_pair):
        off, on = capped_pair
        assert off is not None and on is not None
        if off.method_config != on.method_config:
            raise ConfigError("llm_capped_off/on must reference the same method config")
        if off.sampler_contract != on.sampler_contract:
            raise ConfigError("llm_capped_off/on must use the same sampler contract")
        if {off.routing_mode, on.routing_mode} != {"off", "on"}:
            raise ConfigError("llm_capped_off/on must differ in routing mode")

    exact_pair = [resolved_arms.get("llm_exact_off"), resolved_arms.get("llm_exact_on")]
    if all(exact_pair):
        off, on = exact_pair
        assert off is not None and on is not None
        if off.sampler_contract != on.sampler_contract or off.sampler_contract != "exact_bucket_quota":
            raise ConfigError("llm_exact_off/on must use exact_bucket_quota")
        if {off.routing_mode, on.routing_mode} != {"off", "on"}:
            raise ConfigError("llm_exact_off/on must differ in routing mode")
        matched_keys = (
            "CFG_FILTER_PREFIX",
            "CFG_SCORE_METHOD",
            "CFG_CURRICULUM_METHOD",
            "CFG_SAMPLER_MODE",
            "CFG_MAX_REPEAT_PER_SCENARIO",
            "CFG_PERSISTENT_EXPOSURE",
            "CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP",
            "CFG_NEAR_DUPLICATE_GROUP_WEIGHTING",
            "CFG_MAX_CUMULATIVE_EXPOSURE_PER_SCENARIO",
            "CFG_MAX_CUMULATIVE_EXPOSURE_PER_NEAR_DUPLICATE_GROUP",
        )
        mismatched = [
            key
            for key in matched_keys
            if off.method_values[key] != on.method_values[key]
        ]
        if mismatched:
            raise ConfigError(
                "llm_exact_off/on must differ only in routing policy fields; "
                f"mismatched contracts: {', '.join(mismatched)}"
            )
        strength = float(on.method_values["CFG_TYPE_ROUTING_STRENGTH"])
        if not 0.0 < strength <= 1.0:
            raise ConfigError("llm_exact_on routing strength must be in (0, 1]")

    overrides = payload.get("checkpoint_overrides") or {}
    if not isinstance(overrides, dict):
        raise ConfigError("checkpoint_overrides must be a mapping")
    for arm_id, per_seed in overrides.items():
        if arm_id not in arms_payload:
            raise ConfigError(f"checkpoint_overrides refers to unknown arm: {arm_id}")
        if not isinstance(per_seed, dict):
            raise ConfigError(f"checkpoint_overrides.{arm_id} must be a mapping")

    return {
        "suite_id": suite_id,
        "mode": mode,
        "interleave_evaluation_by_seed": interleave_evaluation_by_seed,
        "continue_on_failure": bool(workflow.get("continue_on_failure", False)),
        "dry_run": bool(workflow.get("dry_run", False)),
        "selected_arms": selected_arms,
        "seeds": seeds,
        "benchmarks": benchmarks,
        "use_priority_benchmark_order": use_priority_benchmark_order,
        "checkpoint_policy": checkpoint_policy,
        "resume_policy": resume_policy,
        "feature_cache_name": feature_cache_name,
        "pretrained_checkpoint": pretrained_checkpoint,
        "retrain_tag": training.get("retrain_tag"),
        "use_ema_checkpoint": require_bool(
            evaluation.get("use_ema_checkpoint", False),
            "evaluation.use_ema_checkpoint",
        ),
        "require_completed_checkpoint": bool(
            evaluation.get("require_completed_checkpoint", True)
        ),
        "skip_completed_same_version_seed": skip_completed_evaluation,
        "disable_simulation_log": bool(evaluation.get("disable_simulation_log", True)),
        "collect_results": bool(evaluation.get("collect_results", True)),
        "checkpoint_overrides": overrides,
        "protocol_path": protocol_path,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "filter_contracts": filter_contracts,
        "artifact_bundle": artifact_bundle,
        "arms": resolved_arms,
    }


def checkpoint_candidates(
    experiment_dir: Path, *, use_ema_checkpoint: bool = False
) -> list[Path]:
    if use_ema_checkpoint:
        return [experiment_dir / "lora_checkpoints/merged_final_ema.ckpt"]
    return [
        experiment_dir / "lora_checkpoints/merged_final.ckpt",
        experiment_dir / "checkpoints/last.ckpt",
    ]


def checkpoint_config_matches(
    experiment_dir: Path,
    *,
    seed: int,
    protocol_id: str,
    protocol_sha256: str,
    method_sha256: str,
    execution_mode: str,
    artifact_bundle_id: str = "",
) -> bool:
    lineage_config = experiment_dir / CHECKPOINT_LINEAGE_FILENAME
    if lineage_config.is_file():
        config_path = lineage_config
    else:
        try:
            config_path = experiment_dir.parents[1] / ".hydra/config.yaml"
        except IndexError:
            return False
    if not config_path.is_file():
        return False
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        lora = payload.get("lora") or {}
        matches = (
            int(payload.get("seed")) == seed
            and str(lora.get("training_protocol_id", "")) == protocol_id
            and str(lora.get("training_protocol_sha256", "")) == protocol_sha256
            and str(lora.get("curriculum_method_sha256", "")) == method_sha256
            and str(lora.get("execution_mode", "")) == execution_mode
        )
        if artifact_bundle_id:
            matches = matches and str(
                lora.get("curriculum_artifact_bundle_id", "")
            ) == artifact_bundle_id
        return matches
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return False


def discover_checkpoint(
    final_experiment: str,
    *,
    seed: int,
    protocol_id: str,
    protocol_sha256: str,
    method_sha256: str,
    execution_mode: str,
    artifact_bundle_id: str = "",
    use_ema_checkpoint: bool = False,
) -> Optional[Path]:
    output_root = PLUTO_ROOT / "outputs"
    if not output_root.is_dir():
        return None
    candidates: list[Path] = []
    for experiment_dir in output_root.glob(f"*/*/outputs/{final_experiment}"):
        if not checkpoint_config_matches(
            experiment_dir,
            seed=seed,
            protocol_id=protocol_id,
            protocol_sha256=protocol_sha256,
            method_sha256=method_sha256,
            execution_mode=execution_mode,
            artifact_bundle_id=artifact_bundle_id,
        ):
            continue
        candidates.extend(
            path
            for path in checkpoint_candidates(
                experiment_dir, use_ema_checkpoint=use_ema_checkpoint
            )
            if path.is_file()
        )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def discover_checkpoint_with_retry(
    final_experiment: str,
    *,
    attempts: int = POST_TRAINING_DISCOVERY_ATTEMPTS,
    delay_seconds: float = POST_TRAINING_DISCOVERY_DELAY_SECONDS,
    **discovery_kwargs: Any,
) -> Optional[Path]:
    """Retry post-training discovery while newly closed files become visible."""
    if attempts < 1:
        raise ValueError("checkpoint discovery attempts must be at least 1")
    for attempt in range(attempts):
        checkpoint = discover_checkpoint(final_experiment, **discovery_kwargs)
        if checkpoint is not None:
            return checkpoint
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return None


def has_partial_checkpoint(experiment_base: str) -> bool:
    output_root = PLUTO_ROOT / "outputs"
    if not output_root.is_dir():
        return False
    patterns = (
        f"*/*/outputs/{experiment_base}_phaseA_*/checkpoints/last.ckpt",
        f"*/*/outputs/{experiment_base}_phaseB_*/checkpoints/last.ckpt",
    )
    return any(any(output_root.glob(pattern)) for pattern in patterns)


def experiment_state_contract_status(
    experiment_base: str,
    arm: ResolvedArm,
    *,
    seed: int,
    protocol_id: str,
    protocol_sha256: str,
    method_sha256: str,
    artifact_bundle_id: str = "",
) -> str:
    """Return none, matching, or mismatch for state using an experiment base."""
    found = False
    snapshot_root = PLUTO_ROOT / "artifacts/training_protocols"
    snapshot_paths = list(snapshot_root.glob(f"{experiment_base}.json"))
    snapshot_paths.extend(snapshot_root.glob(f"{experiment_base}.*.json"))
    for snapshot_path in snapshot_paths:
        found = True
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "mismatch"
        if (
            str(payload.get("CFG_PROTOCOL_SHA256", "")) != protocol_sha256
            or str(payload.get("CFG_METHOD_SHA256", "")) != method_sha256
        ):
            return "mismatch"

    assert arm.method_values is not None
    if arm.method_values["CFG_METHOD_MODE"] == "uniform":
        experiment_names = [final_experiment_name(experiment_base, arm)]
        execution_mode = "continuous_uniform"
    else:
        experiment_names = [
            f"{experiment_base}_phaseA_{arm.method_values['CFG_PHASE_A_NAME']}",
            f"{experiment_base}_phaseB_{arm.method_values['CFG_PHASE_B_NAME']}",
            f"{experiment_base}_phaseC_{arm.method_values['CFG_PHASE_C_NAME']}",
        ]
        execution_mode = "staged_curriculum"

    output_root = PLUTO_ROOT / "outputs"
    for experiment_name in experiment_names:
        for experiment_dir in output_root.glob(f"*/*/outputs/{experiment_name}"):
            if not any(path.is_file() for path in checkpoint_candidates(experiment_dir)):
                continue
            found = True
            if not checkpoint_config_matches(
                experiment_dir,
                seed=seed,
                protocol_id=protocol_id,
                protocol_sha256=protocol_sha256,
                method_sha256=method_sha256,
                execution_mode=execution_mode,
                artifact_bundle_id=artifact_bundle_id,
            ):
                return "mismatch"
    return "matching" if found else "none"


def contract_isolation_suffix(
    protocol_sha256: str, method_sha256: str
) -> str:
    return f"contract_{protocol_sha256[:12]}_{method_sha256[:12]}"


def checkpoint_override(
    overrides: dict[str, Any], arm_id: str, seed: Optional[int]
) -> Optional[Path]:
    per_seed = overrides.get(arm_id) or {}
    if not isinstance(per_seed, dict):
        return None
    keys = [str(seed), seed] if seed is not None else ["default"]
    value = next((per_seed[key] for key in keys if key in per_seed), None)
    if value is None:
        return None
    return resolve_path(value, label=f"checkpoint_overrides.{arm_id}.{seed}")


def checkpoint_override_lineage_matches(
    checkpoint: Path,
    *,
    seed: int,
    protocol_id: str,
    protocol_sha256: str,
    method_sha256: str,
    execution_mode: str,
    artifact_bundle_id: str = "",
) -> bool:
    """Validate modern overrides while retaining support for legacy paths."""
    if checkpoint.parent.name not in {"checkpoints", "lora_checkpoints"}:
        return True
    experiment_dir = checkpoint.parent.parent
    if not (experiment_dir / CHECKPOINT_LINEAGE_FILENAME).is_file():
        return True
    return checkpoint_config_matches(
        experiment_dir,
        seed=seed,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        method_sha256=method_sha256,
        execution_mode=execution_mode,
        artifact_bundle_id=artifact_bundle_id,
    )


def final_experiment_name(base: str, arm: ResolvedArm) -> str:
    assert arm.method_values is not None
    phase_c = str(arm.method_values["CFG_PHASE_C_NAME"])
    if arm.method_values["CFG_METHOD_MODE"] == "uniform":
        return f"{base}_{phase_c}"
    return f"{base}_phaseC_{phase_c}"


def build_runs(validated: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    protocol_id = validated["protocol_id"]
    overrides = validated["checkpoint_overrides"]
    for arm_id in validated["selected_arms"]:
        arm: ResolvedArm = validated["arms"][arm_id]
        if arm.kind == "zero_shot":
            checkpoint = checkpoint_override(overrides, arm_id, None)
            if checkpoint is None:
                checkpoint = validated["pretrained_checkpoint"]
            runs.append(
                {
                    "arm_id": arm_id,
                    "kind": arm.kind,
                    "label": arm.label,
                    "method": arm.method,
                    "artifact_version": arm.artifact_version,
                    "sampler_contract": arm.sampler_contract,
                    "routing_mode": arm.routing_mode,
                    "seed": None,
                    "experiment_base": None,
                    "final_experiment": None,
                    "evaluation_slug": arm.slug_template,
                    "checkpoint": str(checkpoint),
                    "checkpoint_source": "override" if checkpoint_override(overrides, arm_id, None) else "pretrained",
                    "training_status": "not_applicable",
                    "evaluation_status": {},
                }
            )
            continue
        assert arm.method_values is not None
        for seed in validated["seeds"]:
            base = render_template(
                arm.experiment_template,
                artifact_version=arm.artifact_version,
                protocol_id=protocol_id,
                seed=seed,
            )
            slug = render_template(
                arm.slug_template,
                artifact_version=arm.artifact_version,
                protocol_id=protocol_id,
                seed=seed,
            )
            final_exp = final_experiment_name(base, arm)
            override = checkpoint_override(overrides, arm_id, seed)
            execution_mode = (
                "continuous_uniform"
                if arm.method_values["CFG_METHOD_MODE"] == "uniform"
                else "staged_curriculum"
            )
            artifact_bundle_id = (
                validated["artifact_bundle"]["bundle_id"]
                if arm.method == "llm" and validated.get("artifact_bundle")
                else ""
            )
            if override and not checkpoint_override_lineage_matches(
                override,
                seed=seed,
                protocol_id=protocol_id,
                protocol_sha256=validated["protocol_sha256"],
                method_sha256=arm.method_values["CFG_METHOD_SHA256"],
                execution_mode=execution_mode,
                artifact_bundle_id=artifact_bundle_id,
            ):
                raise ConfigError(
                    f"checkpoint_overrides.{arm_id}.{seed} lineage does not match "
                    "the active training contract"
                )
            discovered = override or discover_checkpoint(
                final_exp,
                seed=seed,
                protocol_id=protocol_id,
                protocol_sha256=validated["protocol_sha256"],
                method_sha256=arm.method_values["CFG_METHOD_SHA256"],
                execution_mode=execution_mode,
                artifact_bundle_id=artifact_bundle_id,
                use_ema_checkpoint=validated["use_ema_checkpoint"],
            )
            if validated["use_ema_checkpoint"]:
                slug = f"{slug}_ema"
            runs.append(
                {
                    "arm_id": arm_id,
                    "kind": arm.kind,
                    "label": arm.label,
                    "method": arm.method,
                    "artifact_version": arm.artifact_version,
                    "sampler_contract": arm.sampler_contract,
                    "routing_mode": arm.routing_mode,
                    "seed": seed,
                    "method_config": str(arm.method_config),
                    "method_sha256": arm.method_values["CFG_METHOD_SHA256"],
                    "experiment_base": base,
                    "final_experiment": final_exp,
                    "evaluation_slug": slug,
                    "checkpoint": str(discovered) if discovered else None,
                    "checkpoint_source": "override" if override else ("discovered" if discovered else None),
                    "checkpoint_variant": (
                        "ema" if validated["use_ema_checkpoint"] else "standard"
                    ),
                    "training_status": "pending",
                    "evaluation_status": {},
                }
            )
    return runs


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_experiment_environment() -> dict[str, str]:
    env = dict(os.environ)
    exact_keys = {
        "METHOD",
        "METHOD_CONFIG",
        "METHOD_LABEL",
        "CURRICULUM_VERSION",
        "CURRICULUM_BASE_EXP",
        "TRAINING_PROTOCOL_CONFIG",
        "TRAINING_SEED",
        "SAMPLER_SEED",
        "SAMPLER_MODE",
        "TYPE_ROUTING_MODE",
        "START_PHASE",
        "FRESH_START",
        "DRY_RUN",
        "RUN_LLM_TYPE_ROUTING_COMPARISON",
        "CURRICULUM_MODE",
        "FILTER_PREFIX",
        "PERCENTILE_SPLIT_SEED",
        "TIE_BREAK_MODE",
        "TYPE_ROUTING_METADATA_PATH",
        "TYPE_ROUTING_SNAPSHOT_PATH",
        "FEATURE_CACHE_NAME",
        "PRETRAINED_CKPT",
        "PLUTO_EVAL_CHECKPOINT",
        "FILTER_NAME",
        "EXPERIMENT_SUFFIX",
        "TEST_LABEL",
        "SCENARIO_BUILDER",
        "COLLECT_TEST",
        "SCENARIOS_PER_STAGE",
        "DISABLE_SIMULATION_LOG",
        "SKIP_RESULT_COLLECTION",
    }
    method_prefixes = (
        "RUN_",
        "LLM_CURRICULUM_",
        "RULE_CURRICULUM_",
        "RULE_RAW_CURRICULUM_",
        "LOSS_CURRICULUM_",
        "RANDOM_BUCKET_CURRICULUM_",
        "MPOC_CURRICULUM_",
        "UNIFORM_CURRICULUM_",
    )
    for key in list(env):
        if key in exact_keys or key.startswith(method_prefixes):
            env.pop(key, None)
    return env


def stage_cuda_environment(stage: str) -> dict[str, str]:
    """Apply an optional machine-specific GPU set to one benchmark stage."""
    variable = STAGE_CUDA_ENVIRONMENT[stage]
    value = os.environ.get(variable)
    if value is None:
        return {}
    value = value.strip()
    if not value:
        raise ConfigError(f"{variable} must not be empty when set")
    return {"CUDA_VISIBLE_DEVICES": value}


def print_command(command: list[str], env_updates: dict[str, str]) -> None:
    rendered_env = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(env_updates.items())
    )
    rendered_command = " ".join(shlex.quote(value) for value in command)
    print(f"DRY_RUN: {rendered_env} {rendered_command}".strip(), flush=True)


def retrain_suffix(validated: dict[str, Any]) -> str:
    configured = validated.get("retrain_tag")
    if configured:
        return require_id(configured, "training.retrain_tag")
    return datetime.now().strftime("rerun_%Y%m%d_%H%M%S")


def train_one(
    run: dict[str, Any],
    arm: ResolvedArm,
    validated: dict[str, Any],
) -> None:
    if run["kind"] == "zero_shot":
        return
    policy = validated["checkpoint_policy"]
    resume_policy = validated["resume_policy"]
    existing = Path(run["checkpoint"]) if run.get("checkpoint") else None
    if existing and policy == "reuse":
        run["training_status"] = "reused"
        return
    if existing and policy == "error":
        raise RuntimeError(
            f"Completed checkpoint already exists for {run['arm_id']} seed {run['seed']}: {existing}"
        )

    assert arm.method_values is not None
    base = str(run["experiment_base"])
    artifact_bundle_id = (
        validated["artifact_bundle"]["bundle_id"]
        if arm.method == "llm" and validated.get("artifact_bundle")
        else ""
    )
    state_status = experiment_state_contract_status(
        base,
        arm,
        seed=int(run["seed"]),
        protocol_id=validated["protocol_id"],
        protocol_sha256=validated["protocol_sha256"],
        method_sha256=arm.method_values["CFG_METHOD_SHA256"],
        artifact_bundle_id=artifact_bundle_id,
    )
    if existing is None and state_status == "mismatch":
        suffix = contract_isolation_suffix(
            validated["protocol_sha256"], arm.method_values["CFG_METHOD_SHA256"]
        )
        base = f"{base}_{suffix}"
        run["experiment_base"] = base
        run["final_experiment"] = final_experiment_name(base, arm)
        run["evaluation_slug"] = f"{run['evaluation_slug']}_{suffix}"
        print(
            "Isolating stale experiment state under a contract-specific base: "
            f"{base}",
            flush=True,
        )
    if existing and policy == "retrain":
        base = f"{base}_{retrain_suffix(validated)}"
        run["experiment_base"] = base
        run["final_experiment"] = final_experiment_name(base, arm)
        run["evaluation_slug"] = f"{run['evaluation_slug']}_{base.rsplit('_', 2)[-2]}_{base.rsplit('_', 1)[-1]}"
        run["checkpoint"] = None
        run["checkpoint_source"] = None
    elif resume_policy == "fresh" and has_partial_checkpoint(base):
        base = f"{base}_{retrain_suffix(validated).replace('rerun_', 'fresh_')}"
        run["experiment_base"] = base
        run["final_experiment"] = final_experiment_name(base, arm)
        run["evaluation_slug"] = f"{run['evaluation_slug']}_fresh"

    if resume_policy == "require_resume" and not has_partial_checkpoint(base):
        raise RuntimeError(
            f"resume_policy=require_resume but no Phase A/B checkpoint exists for {base}"
        )

    assert arm.method_config is not None
    env = clean_experiment_environment()
    updates = {
        "METHOD": arm.method,
        "METHOD_CONFIG": str(arm.method_config),
        "METHOD_LABEL": arm.label,
        "CURRICULUM_VERSION": arm.artifact_version,
        "CURRICULUM_BASE_EXP": base,
        "TRAINING_PROTOCOL_CONFIG": str(validated["protocol_path"]),
        "TRAINING_SEED": str(run["seed"]),
        "SAMPLER_SEED": str(run["seed"]),
        "FEATURE_CACHE_NAME": validated["feature_cache_name"],
        "PRETRAINED_CKPT": str(validated["pretrained_checkpoint"]),
        "START_PHASE": "auto",
        "FRESH_START": "true" if resume_policy == "fresh" else "false",
        "DRY_RUN": "false",
    }
    updates.update(stage_cuda_environment("training"))
    if arm.method == "llm":
        updates["TYPE_ROUTING_MODE"] = arm.routing_mode
        bundle = validated.get("artifact_bundle")
        if bundle:
            updates["CURRICULUM_ARTIFACT_BUNDLE_ID"] = bundle["bundle_id"]
            updates["CURRICULUM_ARTIFACT_BUNDLE_MANIFEST_SHA256"] = bundle[
                "manifest_sha256"
            ]
            if arm.routing_mode == "on":
                updates["TYPE_ROUTING_METADATA_PATH"] = bundle[
                    "routing_metadata_path"
                ]
    env.update(updates)
    command = ["bash", str(TRAINING_ADAPTER)]
    if validated["dry_run"]:
        print_command(command, updates)
        run["training_status"] = "dry_run"
        return
    subprocess.run(command, cwd=PLUTO_ROOT, env=env, check=True)
    checkpoint = discover_checkpoint_with_retry(
        str(run["final_experiment"]),
        seed=int(run["seed"]),
        protocol_id=validated["protocol_id"],
        protocol_sha256=validated["protocol_sha256"],
        method_sha256=arm.method_values["CFG_METHOD_SHA256"],
        execution_mode=(
            "continuous_uniform"
            if arm.method_values and arm.method_values["CFG_METHOD_MODE"] == "uniform"
            else "staged_curriculum"
        ),
        artifact_bundle_id=(
            validated["artifact_bundle"]["bundle_id"]
            if arm.method == "llm" and validated.get("artifact_bundle")
            else ""
        ),
        use_ema_checkpoint=validated["use_ema_checkpoint"],
    )
    if checkpoint is None:
        raise RuntimeError(
            f"Training finished but final checkpoint was not found: {run['final_experiment']}"
        )
    run["checkpoint"] = str(checkpoint)
    run["checkpoint_source"] = "trained"
    run["training_status"] = "completed"


def benchmark_command(benchmark: str) -> tuple[list[str], dict[str, str]]:
    evaluation_dir = PLUTO_ROOT / "scripts/evaluation"
    if benchmark == "val14":
        return ["bash", str(evaluation_dir / "quick_test_val14.sh")], {}
    if benchmark == "val14-fast":
        return ["bash", str(evaluation_dir / "quick_test_val14.sh")], {
            "FILTER_NAME": "val14-fast",
            "EXPERIMENT_SUFFIX": "val14_fast",
            "TEST_LABEL": "Val14 Fast",
            "SCENARIO_BUILDER": "nuplan_v1_1_val",
            "COLLECT_TEST": "val14-fast",
            "SCENARIOS_PER_STAGE": "auto",
        }
    if benchmark == "test14-hard":
        return ["bash", str(evaluation_dir / "quick_test_test14-hard.sh")], {}
    if benchmark == "test14-hard-fast":
        return ["bash", str(evaluation_dir / "quick_test_test14-hard.sh")], {
            "FILTER_NAME": "test14-hard-fast",
            "EXPERIMENT_SUFFIX": "test14_hard_fast",
            "TEST_LABEL": "Test14-Hard Fast",
            "SCENARIO_BUILDER": "nuplan_v1_1_test",
            "COLLECT_TEST": "test14-hard-fast",
            "SCENARIOS_PER_STAGE": "auto",
        }
    if benchmark == "interplan10":
        return ["bash", str(evaluation_dir / "quick_test_interplan.sh"), "interplan10"], {
            "EXPERIMENT_SUFFIX": "interplan10",
            "COLLECT_TEST": "interplan10",
        }
    if benchmark == "interplan-benchmark":
        return [
            "bash",
            str(evaluation_dir / "quick_test_interplan.sh"),
            "benchmark_scenarios",
        ], {
            "EXPERIMENT_SUFFIX": "interplan_benchmark",
            "COLLECT_TEST": "interplan-benchmark",
        }
    raise AssertionError(benchmark)


def evaluation_experiment_name(run: dict[str, Any], benchmark: str) -> str:
    suffixes = {
        "val14": "val14",
        "val14-fast": "val14_fast",
        "test14-hard": "test14_hard",
        "test14-hard-fast": "test14_hard_fast",
        "interplan10": "interplan10",
        "interplan-benchmark": "interplan_benchmark",
    }
    prefix = (
        "quick_test_interplan_" if benchmark.startswith("interplan") else "quick_test_"
    )
    return f"{prefix}{run['evaluation_slug']}_{suffixes[benchmark]}"


def evaluation_filter_path(benchmark: str) -> Path:
    names = {
        "val14": "val14",
        "val14-fast": "val14-fast",
        "test14-hard": "test14-hard",
        "test14-hard-fast": "test14-hard-fast",
    }
    if benchmark in names:
        return PLUTO_ROOT / "config/scenario_filter" / f"{names[benchmark]}.yaml"
    interplan_name = (
        "interplan10" if benchmark == "interplan10" else "benchmark_scenarios"
    )
    return (
        PLUTO_ROOT.parent
        / "interPlan/interplan/planning/script/config/common/scenario_filter"
        / f"{interplan_name}.yaml"
    )


def existing_evaluation_result(
    run: dict[str, Any], benchmark: str
) -> Optional[Path]:
    """Return a completed same-version/seed result record, if one is reusable."""
    version = str(run.get("artifact_version") or "")
    seed = run.get("seed")
    slug = str(run.get("evaluation_slug") or "")
    if not version or seed is None or version not in slug or f"seed{seed}" not in slug:
        return None

    experiment = evaluation_experiment_name(run, benchmark)
    record_path = (
        PLUTO_ROOT / "artifacts/records/scenario_records" / f"{experiment}.json"
    )
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_count = len(
            scenario_filter_tokens_from_path(evaluation_filter_path(benchmark))
        )
        if int(record.get("count", 0)) < expected_count:
            return None
        raw_metrics_dirs = record.get("resolved_metrics_dirs")
        if not isinstance(raw_metrics_dirs, list) or not raw_metrics_dirs:
            return None
        for raw_path in raw_metrics_dirs:
            metrics_path = Path(str(raw_path))
            if not metrics_path.is_absolute():
                metrics_path = PLUTO_ROOT / metrics_path
            if not metrics_path.exists() or not any(metrics_path.rglob("*.parquet")):
                return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return None
    return record_path.resolve()


def evaluation_method_environment(run: dict[str, Any]) -> dict[str, str]:
    updates = {
        "RUN_ZERO_SHOT": "false",
        "RUN_RULE_BASED": "false",
        "RUN_RULE_RAW": "false",
        "RUN_LOSS_BASED": "false",
        "RUN_UNIFORM": "false",
        "RUN_RANDOM_BUCKET": "false",
        "RUN_LLM_CURRICULUM": "false",
        "RUN_MPOC": "false",
    }
    if run.get("checkpoint"):
        updates["PLUTO_EVAL_CHECKPOINT"] = str(run["checkpoint"])
    method = run["method"]
    if method == "zero_shot":
        updates["RUN_ZERO_SHOT"] = "true"
        return updates
    mapping = {
        "rule": ("RUN_RULE_BASED", "RULE"),
        "rule_raw": ("RUN_RULE_RAW", "RULE_RAW"),
        "loss": ("RUN_LOSS_BASED", "LOSS"),
        "uniform": ("RUN_UNIFORM", "UNIFORM"),
        "random": ("RUN_RANDOM_BUCKET", "RANDOM_BUCKET"),
        "llm": ("RUN_LLM_CURRICULUM", "LLM"),
        "mpoc": ("RUN_MPOC", "MPOC"),
    }
    flag, prefix = mapping[method]
    updates[flag] = "true"
    updates[f"{prefix}_CURRICULUM_VERSION"] = str(run["artifact_version"])
    updates[f"{prefix}_CURRICULUM_SLUG"] = str(run["evaluation_slug"])
    updates[f"{prefix}_CURRICULUM_EXP"] = str(run["final_experiment"])
    return updates


def evaluate_one(
    run: dict[str, Any], benchmark: str, validated: dict[str, Any]
) -> None:
    if validated.get("skip_completed_same_version_seed", False):
        existing_result = existing_evaluation_result(run, benchmark)
        if existing_result is not None:
            print(
                "Skipping completed evaluation with matching version/seed: "
                f"{existing_result}",
                flush=True,
            )
            run["evaluation_status"][benchmark] = "skipped_existing"
            run.setdefault("evaluation_results", {})[benchmark] = str(existing_result)
            return
    checkpoint = run.get("checkpoint")
    if (
        not checkpoint
        and validated["require_completed_checkpoint"]
        and not validated["dry_run"]
    ):
        raise RuntimeError(
            f"No completed checkpoint for evaluation: {run['arm_id']} seed {run['seed']}"
        )
    if checkpoint and not validated["dry_run"] and not Path(checkpoint).is_file():
        raise RuntimeError(f"Evaluation checkpoint does not exist: {checkpoint}")
    command, benchmark_env = benchmark_command(benchmark)
    env = clean_experiment_environment()
    updates = evaluation_method_environment(run)
    updates.update(benchmark_env)
    updates["DISABLE_SIMULATION_LOG"] = (
        "true" if validated["disable_simulation_log"] else "false"
    )
    updates["SKIP_RESULT_COLLECTION"] = (
        "false" if validated["collect_results"] else "true"
    )
    updates.update(stage_cuda_environment("evaluation"))
    env.update(updates)
    if validated["dry_run"]:
        print_command(command, updates)
        run["evaluation_status"][benchmark] = "dry_run"
        return
    subprocess.run(command, cwd=PLUTO_ROOT, env=env, check=True)
    run["evaluation_status"][benchmark] = "completed"


def manifest_payload(
    suite_path: Path,
    source_payload: dict[str, Any],
    resolved_payload: dict[str, Any],
    validated: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_summaries = {}
    for arm_id, arm in validated["arms"].items():
        arm_summaries[arm_id] = {
            "kind": arm.kind,
            "label": arm.label,
            "method": arm.method,
            "artifact_version": arm.artifact_version,
            "method_config": str(arm.method_config) if arm.method_config else None,
            "method_sha256": (
                arm.method_values["CFG_METHOD_SHA256"] if arm.method_values else None
            ),
            "sampler_contract": arm.sampler_contract,
            "routing_mode": arm.routing_mode,
        }
    return {
        "schema_version": 1,
        "suite_id": validated["suite_id"],
        "suite_config": str(suite_path),
        "suite_source_sha256": canonical_digest(source_payload),
        "resolved_config_sha256": canonical_digest(resolved_payload),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "workflow": {
            "mode": validated["mode"],
            "dry_run": validated["dry_run"],
            "interleave_evaluation_by_seed": validated[
                "interleave_evaluation_by_seed"
            ],
            "checkpoint_policy": validated["checkpoint_policy"],
            "resume_policy": validated["resume_policy"],
            "use_ema_checkpoint": validated["use_ema_checkpoint"],
            "skip_completed_same_version_seed": validated[
                "skip_completed_same_version_seed"
            ],
        },
        "selection": {
            "arms": validated["selected_arms"],
            "seeds": validated["seeds"],
            "benchmarks": validated["benchmarks"],
            "use_priority_benchmark_order": validated[
                "use_priority_benchmark_order"
            ],
        },
        "protocol": {
            "path": str(validated["protocol_path"]),
            "id": validated["protocol_id"],
            "sha256": validated["protocol_sha256"],
        },
        "filter_contracts": validated["filter_contracts"],
        "artifact_bundle": validated.get("artifact_bundle"),
        "arms": arm_summaries,
        "runs": runs,
    }


def run_suite(
    manifest: dict[str, Any],
    manifest_path: Path,
    validated: dict[str, Any],
) -> int:
    runs: list[dict[str, Any]] = manifest["runs"]
    mode = validated["mode"]
    failures = 0

    def train_run(run: dict[str, Any]) -> None:
        nonlocal failures
        arm: ResolvedArm = validated["arms"][run["arm_id"]]
        print(
            f"\n=== train arm={run['arm_id']} seed={run['seed']} "
            f"contract={run['sampler_contract']} routing={run['routing_mode']} ===",
            flush=True,
        )
        try:
            train_one(run, arm, validated)
        except Exception as exc:
            failures += 1
            run["training_status"] = "failed"
            run["training_error"] = str(exc)
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            if not validated["continue_on_failure"]:
                raise
            print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)

    def evaluate_run(run: dict[str, Any]) -> None:
        nonlocal failures
        if run.get("training_status") == "failed":
            return
        for benchmark in validated["benchmarks"]:
            print(
                f"\n=== evaluate arm={run['arm_id']} seed={run['seed']} "
                f"benchmark={benchmark} ===",
                flush=True,
            )
            try:
                evaluate_one(run, benchmark, validated)
            except Exception as exc:
                failures += 1
                run["evaluation_status"][benchmark] = "failed"
                run.setdefault("evaluation_errors", {})[benchmark] = str(exc)
                manifest["updated_at"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                if not validated["continue_on_failure"]:
                    raise
                print(f"ERROR: {exc}", file=sys.stderr, flush=True)
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)

    if mode == "evaluate_only":
        for run in runs:
            if run["kind"] == "trainable":
                run["training_status"] = "not_requested"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)

    if mode == "train_and_evaluate" and validated["interleave_evaluation_by_seed"]:
        # build_runs() stores runs by arm. Regroup them by configured seed so
        # every seed's train/evaluate pipelines finish before the next seed.
        execution_runs = [run for run in runs if run.get("seed") is None]
        for seed in validated["seeds"]:
            execution_runs.extend(run for run in runs if run.get("seed") == seed)
        for run in execution_runs:
            train_run(run)
            evaluate_run(run)
    else:
        if mode in {"train_and_evaluate", "train_only"}:
            for run in runs:
                train_run(run)
        if mode in {"train_and_evaluate", "evaluate_only"}:
            for run in runs:
                evaluate_run(run)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=sorted(WORKFLOW_MODES))
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", help="Comma list (1,2,3) or inclusive range (1:3)")
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS))
    parser.add_argument("--checkpoint-policy", choices=sorted(CHECKPOINT_POLICIES))
    parser.add_argument(
        "--skip-completed-evaluation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip completed evaluation records whose slug matches artifact version and seed",
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--manifest", help="Optional resolved manifest output path")
    parser.add_argument(
        "--artifact-bundle-manifest",
        help="Immutable LLM artifact bundle manifest; overrides the suite YAML",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    suite_path, source_payload = load_suite(args.config)
    resolved_payload = apply_cli_overrides(source_payload, args)
    validated = validate_suite(resolved_payload)
    freeze_training_configs(validated)
    runs = build_runs(validated)
    manifest = manifest_payload(
        suite_path, source_payload, resolved_payload, validated, runs
    )
    if args.manifest:
        manifest_path = resolve_path(args.manifest, label="manifest", must_exist=False)
    else:
        digest_part = manifest["resolved_config_sha256"][:12]
        manifest_path = (
            PLUTO_ROOT
            / "artifacts/benchmark_runs"
            / validated["suite_id"]
            / f"{digest_part}.json"
        )
    atomic_write_json(manifest_path, manifest)
    print(f"Validated benchmark suite: {validated['suite_id']}")
    print(f"Workflow: {validated['mode']} (dry_run={validated['dry_run']})")
    print(f"Arms: {', '.join(validated['selected_arms'])}")
    print(f"Seeds: {validated['seeds']}")
    print(f"Benchmarks: {', '.join(validated['benchmarks']) or '-'}")
    print(f"Manifest: {manifest_path}")
    if args.validate_only:
        return 0
    return run_suite(manifest, manifest_path, validated)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ConfigError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
