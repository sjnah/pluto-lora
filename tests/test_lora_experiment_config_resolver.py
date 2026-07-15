from hydra.core.override_parser.overrides_parser import OverridesParser

from scripts.training.resolve_lora_experiment_config import hydra_literal


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
