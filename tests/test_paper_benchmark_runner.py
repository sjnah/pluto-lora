from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.modules.setdefault("wandb", None)


PLUTO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PLUTO_ROOT / "scripts/experiments/run_lora_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_lora_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def load_default():
    path = PLUTO_ROOT / "config/benchmark/paper_main_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return path, payload, runner.validate_suite(payload)


def test_default_suite_contains_all_paper_arms() -> None:
    _, _, validated = load_default()
    assert validated["selected_arms"] == [
        "zero_shot",
        "uniform",
        "random_exact",
        "rule_exact",
        "loss_exact",
        "mpoc_exact",
        "llm_exact_off",
        "llm_capped_off",
        "llm_capped_on",
    ]


def test_common_comparison_arms_are_exact_without_llm_caps() -> None:
    _, _, validated = load_default()
    for arm_id in (
        "random_exact",
        "rule_exact",
        "loss_exact",
        "mpoc_exact",
        "llm_exact_off",
    ):
        arm = validated["arms"][arm_id]
        assert arm.sampler_contract == "exact_bucket_quota"
        assert arm.routing_mode == "off"
        assert arm.method_values["CFG_SAMPLER_MODE"] == "exact_bucket_quota"
        assert arm.method_values["CFG_PERSISTENT_EXPOSURE"] is False
        assert arm.method_values["CFG_MAX_REPEAT_PER_NEAR_DUPLICATE_GROUP"] == 0


def test_capped_routing_pair_shares_sampler_and_exposure_config() -> None:
    _, _, validated = load_default()
    off = validated["arms"]["llm_capped_off"]
    on = validated["arms"]["llm_capped_on"]
    assert off.method_config == on.method_config
    assert off.method_values == on.method_values
    assert off.sampler_contract == on.sampler_contract == "capped_weighted"
    assert off.routing_mode == "off"
    assert on.routing_mode == "on"
    assert off.method_values["CFG_SAMPLER_MODE"] == "exposure_capped_weighted"
    assert off.method_values["CFG_PERSISTENT_EXPOSURE"] is True
    assert off.method_values["CFG_NEAR_DUPLICATE_GROUP_WEIGHTING"] is True


def test_near_duplicate_group_weighting_is_independent_of_type_routing() -> None:
    module_path = PLUTO_ROOT / "src/custom_training/curriculum_sampling.py"
    spec = importlib.util.spec_from_file_location("paper_group_weighting", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    records = [
        {"near_duplicate_groups": ["dense"]},
        {"near_duplicate_groups": ["dense"]},
        {"near_duplicate_groups": ["unique"]},
    ]
    assert module.apply_near_duplicate_group_inverse_weighting(
        [1.0, 1.0, 1.0], records
    ) == [0.5, 0.5, 1.0]


def test_run_expansion_does_not_repeat_zero_shot_per_seed() -> None:
    _, _, validated = load_default()
    runs = runner.build_runs(validated)
    zero_shot = [run for run in runs if run["arm_id"] == "zero_shot"]
    trainable = [run for run in runs if run["kind"] == "trainable"]
    assert len(zero_shot) == 1
    assert zero_shot[0]["seed"] is None
    assert len(trainable) == 8 * len(validated["seeds"])


def test_type_routing_is_not_exported_for_non_llm_training(monkeypatch) -> None:
    _, _, validated = load_default()
    arm = validated["arms"]["uniform"]
    run = next(run for run in runner.build_runs(validated) if run["arm_id"] == "uniform")
    validated = dict(validated)
    validated["dry_run"] = True
    captured = {}

    def capture(command, updates):
        captured.update(updates)

    monkeypatch.setattr(runner, "print_command", capture)
    runner.train_one(run, arm, validated)
    assert "TYPE_ROUTING_MODE" not in captured


def test_server_training_gpu_override_is_applied_only_when_configured(
    monkeypatch,
) -> None:
    _, _, validated = load_default()
    arm = validated["arms"]["uniform"]
    run = next(run for run in runner.build_runs(validated) if run["arm_id"] == "uniform")
    validated = dict(validated)
    validated["dry_run"] = True
    captured = {}

    monkeypatch.setenv("PLUTO_TRAINING_CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner, "print_command", lambda _command, updates: captured.update(updates))

    runner.train_one(run, arm, validated)

    assert captured["CUDA_VISIBLE_DEVICES"] == "0"


def test_server_evaluation_gpu_override_exposes_all_devices(monkeypatch) -> None:
    run = {
        "arm_id": "zero_shot",
        "seed": None,
        "method": "zero_shot",
        "checkpoint": "/tmp/dry-run-checkpoint.ckpt",
        "evaluation_status": {},
    }
    captured = {}
    monkeypatch.setenv("PLUTO_EVALUATION_CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(runner, "print_command", lambda _command, updates: captured.update(updates))

    runner.evaluate_one(
        run,
        "test14-hard-fast",
        {
            "require_completed_checkpoint": True,
            "dry_run": True,
            "disable_simulation_log": True,
            "collect_results": True,
        },
    )

    assert captured["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_evaluation_only_requires_a_checkpoint_before_simulation() -> None:
    run = {
        "arm_id": "uniform",
        "seed": 1,
        "method": "uniform",
        "checkpoint": None,
        "evaluation_status": {},
    }
    with pytest.raises(RuntimeError, match="No completed checkpoint"):
        runner.evaluate_one(
            run,
            "test14-hard-fast",
            {
                "require_completed_checkpoint": True,
                "dry_run": False,
            },
        )


def test_checkpoint_discovery_requires_seed_protocol_and_execution_mode(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = "paper_test_phaseC_hard_replay"
    run_root = tmp_path / "outputs/2026-07-15/12-00-00"
    experiment_dir = run_root / "outputs" / experiment
    checkpoint = experiment_dir / "lora_checkpoints/merged_final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    hydra = run_root / ".hydra/config.yaml"
    hydra.parent.mkdir(parents=True)
    hydra.write_text(
        yaml.safe_dump(
            {
                "seed": 3,
                "lora": {
                    "training_protocol_id": "protocol",
                    "training_protocol_sha256": "sha",
                    "execution_mode": "staged_curriculum",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PLUTO_ROOT", tmp_path)

    assert runner.discover_checkpoint(
        experiment,
        seed=3,
        protocol_id="protocol",
        protocol_sha256="sha",
        execution_mode="staged_curriculum",
    ) == checkpoint.resolve()
    assert runner.discover_checkpoint(
        experiment,
        seed=4,
        protocol_id="protocol",
        protocol_sha256="sha",
        execution_mode="staged_curriculum",
    ) is None


def test_cli_validate_only_writes_resolved_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    status = runner.main(
        [
            "--config",
            str(PLUTO_ROOT / "config/benchmark/paper_main_v1.yaml"),
            "--mode",
            "train_only",
            "--arms",
            "rule_exact",
            "loss_exact",
            "--seeds",
            "3:4",
            "--validate-only",
            "--manifest",
            str(manifest),
        ]
    )
    assert status == 0
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["workflow"]["mode"] == "train_only"
    assert payload["selection"]["arms"] == ["rule_exact", "loss_exact"]
    assert payload["selection"]["seeds"] == [3, 4]


def test_evaluation_only_uses_explicit_checkpoint_without_training(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(
        (PLUTO_ROOT / "config/benchmark/paper_main_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = tmp_path / "merged_final.ckpt"
    checkpoint.write_bytes(b"test")
    source["workflow"]["mode"] = "evaluate_only"
    source["workflow"]["dry_run"] = True
    source["selection"]["arms"] = ["uniform"]
    source["selection"]["seeds"] = [7]
    source["selection"]["benchmarks"] = ["test14-hard-fast"]
    source["checkpoint_overrides"] = {"uniform": {"7": str(checkpoint)}}
    config = tmp_path / "suite.yaml"
    config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    assert runner.main(
        ["--config", str(config), "--manifest", str(manifest)]
    ) == 0
    resolved = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert resolved["runs"][0]["training_status"] == "not_requested"
    assert resolved["runs"][0]["checkpoint"] == str(checkpoint)
    assert resolved["runs"][0]["evaluation_status"]["test14-hard-fast"] == "dry_run"
