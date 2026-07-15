from hydra.core.override_parser.overrides_parser import OverridesParser

from scripts.training.resolve_lora_experiment_config import (
    hydra_literal,
    normalize_snapshot_semantics,
)


def test_pacing_schedule_serializes_as_hydra_flow_mapping() -> None:
    schedule = {
        "type": "hard_replay_ramp",
        "alpha_start": 0.0,
        "alpha_end": 0.8,
        "ramp_epochs": 4,
        "uniform_prior": [1 / 3, 1 / 3, 1 / 3],
        "hard_prior": [0.0, 0.0, 1.0],
    }

    literal = hydra_literal(schedule)
    assert "\n" not in literal
    assert '"alpha_end"' not in literal

    override = OverridesParser.create().parse_overrides(
        [f"curriculum.pacing_schedule={literal}"]
    )[0]
    assert override.value() == schedule


def test_snapshot_normalization_treats_json_and_hydra_pacing_as_equivalent() -> None:
    json_snapshot = {
        "CFG_PROTOCOL_ID": "flat_area_matched_v1",
        "CFG_PHASE_B_PACING_SCHEDULE": (
            '{"alpha_end":0.8,"ramp_epochs":4,"type":"hard_replay_ramp"}'
        ),
    }
    hydra_snapshot = {
        "CFG_PROTOCOL_ID": "flat_area_matched_v1",
        "CFG_PHASE_B_PACING_SCHEDULE": (
            "{alpha_end: 0.8, ramp_epochs: 4, type: hard_replay_ramp}"
        ),
    }

    assert normalize_snapshot_semantics(json_snapshot) == normalize_snapshot_semantics(
        hydra_snapshot
    )
