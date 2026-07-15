#!/usr/bin/env python3
"""Resolve canonical LoRA protocol/method/suite YAML for shell runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_yaml(path_value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value)
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        path = cwd_candidate if cwd_candidate.is_file() else REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return path, payload


def digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_id(value: object, label: str) -> str:
    text = str(value or "")
    if not SAFE_ID.fullmatch(text):
        raise ValueError(f"Invalid {label}: {text!r}")
    return text


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {key}")
    return value


def list_literal(values: object, label: str) -> str:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    numbers = [float(value) for value in values]
    if abs(sum(numbers) - 1.0) > 1e-6:
        raise ValueError(f"{label} must sum to one: {numbers}")
    return "[" + ",".join(f"{value:.12g}" for value in numbers) + "]"


def json_literal(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_protocol_method(protocol_path: str, method_path: str) -> dict[str, Any]:
    protocol_file, protocol = load_yaml(protocol_path)
    method_file, method = load_yaml(method_path)

    protocol_id = require_id(protocol.get("protocol_id"), "protocol_id")
    method_id = require_id(method.get("method"), "method")
    schedule = require_mapping(protocol, "schedule")
    optimizer = require_mapping(protocol, "optimizer")
    phases = require_mapping(protocol, "phases")
    cumulative = require_mapping(phases, "cumulative_epochs")
    proportions = require_mapping(phases, "bucket_target_proportions")
    pacing = phases.get("pacing") or {}
    if not isinstance(pacing, dict):
        raise ValueError("phases.pacing must be a mapping when present")
    phase_alpha = pacing.get("phase_alpha") or {}
    if not isinstance(phase_alpha, dict):
        raise ValueError("phases.pacing.phase_alpha must be a mapping when present")
    lora = require_mapping(protocol, "lora")
    training = require_mapping(protocol, "training")
    checkpoint = require_mapping(protocol, "checkpoint")
    phase_names = require_mapping(method, "phase_names")
    exposure = method.get("exposure") or {}
    if not isinstance(exposure, dict):
        raise ValueError("method exposure must be a mapping")

    epoch_a = int(cumulative["a"])
    epoch_b = int(cumulative["b"])
    epoch_c = int(cumulative["c"])
    if not (0 < epoch_a < epoch_b < epoch_c):
        raise ValueError("cumulative epochs must be strictly increasing")
    reset_boundaries = optimizer.get("reset_boundaries") or []
    if not isinstance(reset_boundaries, list):
        raise ValueError("optimizer.reset_boundaries must be a list")

    persistent = bool(exposure.get("persistent", False))
    max_scenario = int(exposure.get("max_cumulative_exposure_per_scenario", 0))
    max_group = int(
        exposure.get("max_cumulative_exposure_per_near_duplicate_group", 0)
    )
    if persistent:
        max_scenario = int(
            exposure.get("max_cumulative_exposure_per_scenario_per_epoch", 0)
        ) * epoch_c
        max_group = int(
            exposure.get(
                "max_cumulative_exposure_per_near_duplicate_group_per_epoch", 0
            )
        ) * epoch_c

    metadata_path = ""
    type_routing = method.get("type_routing") or {}
    if not isinstance(type_routing, dict):
        raise ValueError("method type_routing must be a mapping")
    if type_routing.get("metadata_path"):
        metadata_path = str((REPO_ROOT / str(type_routing["metadata_path"])).resolve())

    return {
        "CFG_PROTOCOL_PATH": str(protocol_file),
        "CFG_PROTOCOL_ID": protocol_id,
        "CFG_PROTOCOL_SHA256": digest(protocol),
        "CFG_SCHEDULER_TYPE": require_id(schedule.get("type"), "schedule.type"),
        "CFG_SCHEDULER_HORIZON_EPOCHS": int(schedule["horizon_epochs"]),
        "CFG_WARMUP_STEPS": int(schedule["warmup_steps"]),
        "CFG_LORA_LR": float(schedule["lora_lr"]),
        "CFG_HEAD_LR": float(schedule["head_lr"]),
        "CFG_WEIGHT_DECAY": float(optimizer["weight_decay"]),
        "CFG_RESET_AT_PHASE_B": "phase_a_to_b" in reset_boundaries,
        "CFG_EPOCHS_PHASE_A": epoch_a,
        "CFG_EPOCHS_PHASE_B": epoch_b,
        "CFG_EPOCHS_PHASE_C": epoch_c,
        "CFG_PHASE_A_TARGET_PROPORTIONS": list_literal(
            proportions["a"], "phase A proportions"
        ),
        "CFG_PHASE_B_TARGET_PROPORTIONS": list_literal(
            proportions["b"], "phase B proportions"
        ),
        "CFG_PHASE_C_TARGET_PROPORTIONS": list_literal(
            proportions["c"], "phase C proportions"
        ),
        "CFG_PHASE_A_PACING_SCHEDULE": json_literal(phase_alpha.get("a") or {}),
        "CFG_PHASE_B_PACING_SCHEDULE": json_literal(phase_alpha.get("b") or {}),
        "CFG_PHASE_C_PACING_SCHEDULE": json_literal(phase_alpha.get("c") or {}),
        "CFG_LORA_ENABLED": bool(lora["enabled"]),
        "CFG_LORA_RANK": int(lora["rank"]),
        "CFG_LORA_ALPHA": float(lora["alpha"]),
        "CFG_LORA_DROPOUT": float(lora["dropout"]),
        "CFG_ULTRA_MINIMAL": bool(lora["ultra_minimal"]),
        "CFG_BASE_LR": float(training["base_lr"]),
        "CFG_BATCH_SIZE": int(training["batch_size"]),
        "CFG_ACCUMULATE_GRAD_BATCHES": int(
            training["accumulate_grad_batches"]
        ),
        "CFG_GRADIENT_CLIP_VAL": float(training["gradient_clip_val"]),
        "CFG_SKIP_NAN_STEPS": bool(training["skip_nan_steps"]),
        "CFG_REMOVE_INVALID_GOALS": bool(training["remove_invalid_goals"]),
        "CFG_NUM_SANITY_VAL_STEPS": int(training["num_sanity_val_steps"]),
        "CFG_REQUIRE_PROTOCOL_MATCH_ON_RESUME": bool(
            checkpoint["require_protocol_match_on_resume"]
        ),
        "CFG_METHOD_PATH": str(method_file),
        "CFG_METHOD_SHA256": digest(method),
        "CFG_METHOD": method_id,
        "CFG_METHOD_LABEL": str(method.get("label", method_id)),
        "CFG_METHOD_MODE": str(method["mode"]),
        "CFG_SCENARIO_FILTER_UNIFORM": str(method.get("scenario_filter", "")),
        "CFG_FILTER_PREFIX": str(method.get("filter_prefix", "")),
        "CFG_SCORE_METHOD": str(method.get("score_method", method_id)),
        "CFG_CURRICULUM_METHOD": str(method.get("curriculum_method", "legacy")),
        "CFG_SAMPLER_MODE": str(method.get("sampler_mode", "")),
        "CFG_MAX_REPEAT_PER_SCENARIO": int(
            method.get("max_repeat_per_scenario", 0)
        ),
        "CFG_HARD_SUBTYPE_BALANCE": bool(
            method.get("hard_subtype_balance", False)
        ),
        "CFG_TIE_BREAK_MODE": str(method.get("tie_break_mode", "stable_hash")),
        "CFG_BUCKETIZATION_MODE": str(
            method.get("bucketization_mode", "percentile_tercile")
        ),
        "CFG_PHASE_A_NAME": str(phase_names["a"]),
        "CFG_PHASE_B_NAME": str(phase_names["b"]),
        "CFG_PHASE_C_NAME": str(phase_names["c"]),
        "CFG_PERSISTENT_EXPOSURE": persistent,
        "CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP": int(
            exposure.get("max_repeat_per_near_duplicate_group", 0)
        ),
        "CFG_MAX_CUMULATIVE_EXPOSURE_PER_SCENARIO": max_scenario,
        "CFG_MAX_CUMULATIVE_EXPOSURE_PER_NEAR_DUPLICATE_GROUP": max_group,
        "CFG_TYPE_ROUTING_SUPPORTED": bool(type_routing.get("supported", False)),
        "CFG_TYPE_ROUTING_DEFAULT_MODE": str(
            type_routing.get("default_mode", "off")
        ),
        "CFG_TYPE_ROUTING_METADATA_PATH": metadata_path,
        "CFG_TYPE_ROUTING_ENABLED_SAMPLER_MODE": str(
            type_routing.get("enabled_sampler_mode", "")
        ),
    }


def resolve_suite(suite_path: str) -> dict[str, Any]:
    suite_file, suite = load_yaml(suite_path)
    suite_id = require_id(suite.get("suite_id"), "suite_id")
    methods = require_mapping(suite, "methods")
    seeds = require_mapping(suite, "seeds")
    evaluation = require_mapping(suite, "evaluation")
    runtime = require_mapping(suite, "runtime")
    protocol_value = str(suite["training_protocol"])
    protocol_path = Path(protocol_value)
    if not protocol_path.is_absolute():
        protocol_path = (REPO_ROOT / protocol_path).resolve()

    result: dict[str, Any] = {
        "CFG_SUITE_PATH": str(suite_file),
        "CFG_SUITE_ID": suite_id,
        "CFG_SUITE_SHA256": digest(suite),
        "CFG_SUITE_TRAINING_PROTOCOL": str(protocol_path),
        "CFG_SUITE_SEED_START": int(seeds["start"]),
        "CFG_SUITE_SEED_END": int(seeds["end"]),
        "CFG_SUITE_FEATURE_CACHE_NAME": require_id(
            runtime.get("feature_cache_name"), "runtime.feature_cache_name"
        ),
        "CFG_SUITE_TYPE_ROUTING_MODE": str(runtime["type_routing_mode"]),
        "CFG_SUITE_CONTINUE_ON_FAILURE": bool(runtime["continue_on_failure"]),
        "CFG_SUITE_DISABLE_SIMULATION_LOG": bool(
            runtime["disable_simulation_log"]
        ),
        "CFG_SUITE_SKIP_TRAINING_IF_CHECKPOINT_EXISTS": bool(
            runtime["skip_training_if_checkpoint_exists"]
        ),
    }
    for method in ("llm", "rule", "loss", "random", "mpoc", "uniform"):
        entry = methods.get(method) or {}
        if not isinstance(entry, dict):
            raise ValueError(f"suite methods.{method} must be a mapping")
        upper = method.upper()
        result[f"CFG_SUITE_RUN_{upper}"] = bool(entry.get("enabled", False))
        result[f"CFG_SUITE_{upper}_VERSION"] = str(
            entry.get("artifact_version", "")
        )
    for key in (
        "val14",
        "val14_fast",
        "test14_hard",
        "test14_hard_fast",
        "interplan10",
        "interplan_benchmark",
    ):
        result[f"CFG_SUITE_RUN_{key.upper()}"] = bool(evaluation[key])
    return result


def shell_value(value: object) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--method")
    parser.add_argument("--suite")
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.suite:
        if args.protocol or args.method:
            parser.error("--suite cannot be combined with --protocol/--method")
        resolved = resolve_suite(args.suite)
    else:
        if not args.protocol or not args.method:
            parser.error("--protocol and --method are required together")
        resolved = resolve_protocol_method(args.protocol, args.method)

    if args.format == "json":
        rendered = json.dumps(resolved, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and output.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(
                    f"Resolved configuration changed for existing snapshot: {output}"
                )
            output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return
    if args.output:
        parser.error("--output is supported only with --format json")
    for key in sorted(resolved):
        print(f"{key}={shell_value(resolved[key])}")


if __name__ == "__main__":
    main()
