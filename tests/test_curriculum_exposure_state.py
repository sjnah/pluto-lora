import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["WANDB_DISABLED"] = "true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["wandb"] = None

from src.custom_training.custom_datamodule import (
    ExactBucketQuotaSampler,
    ExposureCappedWeightedSampler,
)


class TestPersistentCurriculumExposureState(unittest.TestCase):
    def _sampler(
        self,
        state_path: Path,
        *,
        proportions=None,
        phase_start_epoch=0,
        sampling_log_path=None,
        pacing_schedule=None,
    ):
        records = [
            {
                "scenario_id": f"scene_{index}",
                "near_duplicate_groups": [f"log:cell_{index // 2}"],
            }
            for index in range(12)
        ]
        return ExactBucketQuotaSampler(
            bucket_sizes=[4, 4, 4],
            target_proportions=proportions or [1 / 3, 1 / 3, 1 / 3],
            max_repeat_per_scenario=2,
            random_seed=42,
            phase_name="validation_phase",
            phase_start_epoch=phase_start_epoch,
            sampling_log_path=sampling_log_path,
            scenario_records=records,
            max_repeat_per_group=4,
            cumulative_exposure_state_path=str(state_path),
            max_cumulative_exposure_per_scenario=4,
            max_cumulative_exposure_per_group=8,
            pacing_schedule=pacing_schedule,
        )

    def test_same_epoch_is_idempotent_and_next_epoch_accumulates(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "exposure.json"
            sampler = self._sampler(state_path)
            sampler.set_epoch(0)
            first = list(sampler)
            repeated = list(sampler)
            self.assertEqual(first, repeated)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(state["plans"]), 1)
            self.assertEqual(sum(state["cumulative_scenario_exposure"].values()), 12)

            sampler.set_epoch(1)
            second = list(sampler)
            self.assertEqual(len(second), 12)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(state["plans"]), 2)
            self.assertEqual(sum(state["cumulative_scenario_exposure"].values()), 24)
            self.assertLessEqual(max(state["cumulative_scenario_exposure"].values()), 4)
            self.assertLessEqual(
                max(state["cumulative_near_duplicate_group_exposure"].values()), 8
            )
            latest = state["plans"]["validation_phase:1"]["metadata"]
            self.assertEqual(latest["demonstration_type_exposure"], {"normal": 12})
            self.assertEqual(sum(latest["sampled_split_counts"].values()), 12)

    def test_existing_plan_rejects_changed_quota_config(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "exposure.json"
            sampler = self._sampler(state_path)
            sampler.set_epoch(0)
            list(sampler)

            changed = self._sampler(state_path, proportions=[0.5, 0.25, 0.25])
            changed.set_epoch(0)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                list(changed)

    def test_existing_plan_rejects_changed_pacing_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "exposure.json"
            first_schedule = {
                "type": "hard_replay_ramp",
                "alpha_start": 0.0,
                "alpha_end": 0.7,
                "ramp_epochs": 2,
            }
            second_schedule = {
                "type": "hard_replay_ramp",
                "alpha_start": 0.0,
                "alpha_end": 0.8,
                "ramp_epochs": 2,
            }
            sampler = self._sampler(state_path, pacing_schedule=first_schedule)
            sampler.set_epoch(0)
            list(sampler)

            changed = self._sampler(state_path, pacing_schedule=second_schedule)
            changed.set_epoch(0)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                list(changed)

    def test_weighted_sampler_accumulates_without_losing_type_exposure(self):
        records = [
            {
                "scenario_id": f"scene_{index}",
                "near_duplicate_groups": [f"log:cell_{index // 2}"],
                "split": "easy" if index < 3 else "hard",
                "log_name": f"log_{index // 2}",
                "demonstration_type": "necessary_exception" if index == 0 else "normal",
            }
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "weighted_exposure.json"
            sampler = ExposureCappedWeightedSampler(
                weights=[5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                scenario_records=records,
                num_samples=6,
                max_repeat_per_scenario=2,
                max_repeat_per_group=2,
                random_seed=11,
                cumulative_exposure_state_path=str(state_path),
                max_cumulative_exposure_per_scenario=4,
                max_cumulative_exposure_per_group=4,
                phase_name="weighted_phase",
                max_exposure_per_demonstration_type={"necessary_exception": 1},
            )
            sampler.set_epoch(0)
            self.assertEqual(len(list(sampler)), 6)
            sampler.set_epoch(1)
            self.assertEqual(len(list(sampler)), 6)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(sum(state["cumulative_scenario_exposure"].values()), 12)
            latest = state["plans"]["weighted_phase:1"]["metadata"]
            self.assertEqual(sum(latest["demonstration_type_exposure"].values()), 6)
            self.assertLessEqual(
                latest["demonstration_type_exposure"].get("necessary_exception", 0), 1
            )

    def test_pre_resume_preflight_does_not_persist_exposure_or_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "exposure.json"
            log_path = root / "sampling.json"
            sampler = self._sampler(
                state_path,
                phase_start_epoch=4,
                sampling_log_path=str(log_path),
            )

            sampler.set_epoch(0)
            self.assertEqual(len(list(sampler)), 12)
            self.assertFalse(state_path.exists())
            self.assertEqual(list(root.glob("sampling.epoch_*.json")), [])

            sampler.set_epoch(4)
            self.assertEqual(len(list(sampler)), 12)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(list(state["plans"]), ["validation_phase:4"])
            self.assertTrue((root / "sampling.epoch_0004.rank_000.json").is_file())


if __name__ == "__main__":
    unittest.main()
