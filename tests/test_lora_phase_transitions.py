import copy
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["WANDB_DISABLED"] = "true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["wandb"] = None

from src.models.pluto.pluto_lora_trainer import PLUTOLoRATrainer
import pytorch_lightning as pl
import torch


class FakeTrainer:
    def __init__(self, optimizers, *, scheduler=None, global_step=0):
        self.optimizers = optimizers
        self.global_step = global_step
        self.lr_scheduler_configs = (
            [SimpleNamespace(scheduler=scheduler)] if scheduler is not None else []
        )


class TestLoRAPhaseTransitionState(unittest.TestCase):
    @staticmethod
    def _checkpoint():
        return {
            "epoch": 4,
            "global_step": 268,
            "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 4}}}},
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": 17,
                            "exp_avg": "first_moment",
                            "exp_avg_sq": "second_moment",
                        }
                    },
                    "param_groups": [{"lr": 5e-5, "params": [0]}],
                }
            ],
            "lr_schedulers": [{"last_epoch": 268, "_step_count": 269}],
        }

    @staticmethod
    def _trainer(*, reset_optimizer_moments: bool):
        trainer = PLUTOLoRATrainer.__new__(PLUTOLoRATrainer)
        pl.LightningModule.__init__(trainer)
        trainer.is_curriculum_stage = False
        trainer.reset_optimizer_moments_on_resume = reset_optimizer_moments
        trainer.scheduler_horizon_epochs = 12
        trainer.scheduler_type = "warmup_constant"
        trainer.training_protocol_id = "flat_area_matched_v1"
        trainer.training_protocol_sha256 = "protocol-sha"
        trainer.curriculum_method_id = "llm"
        trainer.curriculum_method_sha256 = "method-sha"
        trainer.require_protocol_match_on_resume = False
        trainer.ultra_minimal = False
        trainer.ema_initialized = True
        trainer.model = torch.nn.Linear(1, 1)
        return trainer

    def test_a_to_b_clears_only_optimizer_tensor_state(self):
        checkpoint = self._checkpoint()
        expected_param_groups = copy.deepcopy(checkpoint["optimizer_states"][0]["param_groups"])
        expected_schedulers = copy.deepcopy(checkpoint["lr_schedulers"])
        expected_loops = copy.deepcopy(checkpoint["loops"])

        self._trainer(reset_optimizer_moments=True).on_load_checkpoint(checkpoint)

        self.assertEqual(checkpoint["optimizer_states"][0]["state"], {})
        self.assertEqual(checkpoint["optimizer_states"][0]["param_groups"], expected_param_groups)
        self.assertEqual(checkpoint["lr_schedulers"], expected_schedulers)
        self.assertEqual(checkpoint["loops"], expected_loops)
        self.assertEqual(checkpoint["epoch"], 4)
        self.assertEqual(checkpoint["global_step"], 268)

    def test_b_to_c_preserves_full_resume_state(self):
        checkpoint = self._checkpoint()
        expected = copy.deepcopy(checkpoint)

        self._trainer(reset_optimizer_moments=False).on_load_checkpoint(checkpoint)

        self.assertEqual(checkpoint, expected)

    def test_a_to_b_clears_live_adam_state_after_restore(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=5e-5)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        self.assertTrue(optimizer.state)
        expected_param_groups = copy.deepcopy(optimizer.param_groups)

        trainer = self._trainer(reset_optimizer_moments=True)
        fake_trainer = FakeTrainer([optimizer])
        trainer.trainer = fake_trainer
        trainer.on_fit_start()

        self.assertIn("train", trainer.metrics)
        self.assertIn("val", trainer.metrics)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(optimizer.param_groups, expected_param_groups)
        self.assertTrue(trainer._optimizer_moments_reset_applied)

    def test_b_to_c_does_not_clear_live_adam_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=5e-5)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        expected_state_keys = set(optimizer.state)

        trainer = self._trainer(reset_optimizer_moments=False)
        fake_trainer = FakeTrainer([optimizer])
        trainer.trainer = fake_trainer
        trainer.on_fit_start()

        self.assertIn("train", trainer.metrics)
        self.assertIn("val", trainer.metrics)
        self.assertEqual(set(optimizer.state), expected_state_keys)

    def test_scheduler_total_steps_scale_to_final_cumulative_horizon(self):
        resolve = PLUTOLoRATrainer._resolve_scheduler_total_steps

        self.assertEqual(
            resolve(
                estimated_steps=268,
                current_cumulative_epochs=4,
                scheduler_horizon_epochs=12,
            ),
            804,
        )
        self.assertEqual(
            resolve(
                estimated_steps=536,
                current_cumulative_epochs=8,
                scheduler_horizon_epochs=12,
            ),
            804,
        )
        self.assertEqual(
            resolve(
                estimated_steps=804,
                current_cumulative_epochs=12,
                scheduler_horizon_epochs=12,
            ),
            804,
        )

    def test_resume_rejects_scheduler_from_old_phase_local_horizon(self):
        trainer = self._trainer(reset_optimizer_moments=False)
        trainer._configured_scheduler_total_steps = 804
        scheduler = SimpleNamespace(total_steps=268, base_lrs=[5e-5, 1e-5])
        fake_trainer = FakeTrainer([], scheduler=scheduler, global_step=268)
        trainer.trainer = fake_trainer

        with self.assertRaisesRegex(RuntimeError, "checkpoint total_steps=268"):
            trainer._validate_scheduler_horizon()

    def test_resume_accepts_full_run_scheduler_horizon(self):
        trainer = self._trainer(reset_optimizer_moments=False)
        trainer._configured_scheduler_total_steps = 804
        scheduler = SimpleNamespace(total_steps=804, base_lrs=[5e-5, 1e-5])
        fake_trainer = FakeTrainer([], scheduler=scheduler, global_step=536)
        trainer.trainer = fake_trainer

        trainer._validate_scheduler_horizon()

    def test_checkpoint_persists_and_accepts_exact_protocol_identity(self):
        trainer = self._trainer(reset_optimizer_moments=False)
        checkpoint = {}
        trainer.on_save_checkpoint(checkpoint)
        trainer.require_protocol_match_on_resume = True

        trainer._validate_checkpoint_protocol(checkpoint)
        self.assertEqual(
            checkpoint["lora_training_protocol"]["protocol_id"],
            "flat_area_matched_v1",
        )

    def test_resume_rejects_protocol_digest_mismatch(self):
        trainer = self._trainer(reset_optimizer_moments=False)
        checkpoint = {}
        trainer.on_save_checkpoint(checkpoint)
        checkpoint["lora_training_protocol"]["protocol_sha256"] = "changed"
        trainer.require_protocol_match_on_resume = True

        with self.assertRaisesRegex(RuntimeError, "training protocol mismatch"):
            trainer._validate_checkpoint_protocol(checkpoint)

    def test_resume_rejects_legacy_checkpoint_without_protocol_marker(self):
        trainer = self._trainer(reset_optimizer_moments=False)
        trainer.require_protocol_match_on_resume = True

        with self.assertRaisesRegex(RuntimeError, "no lora_training_protocol"):
            trainer._validate_checkpoint_protocol({})


if __name__ == "__main__":
    unittest.main()
