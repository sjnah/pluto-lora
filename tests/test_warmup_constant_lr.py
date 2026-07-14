import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optim.warmup_constant_lr import WarmupConstantLR


class TestWarmupConstantLR(unittest.TestCase):
    def test_optimizer_integration_starts_at_first_warmup_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=3.2e-5)
        scheduler = WarmupConstantLR(
            optimizer,
            warmup_steps=32,
            total_steps=804,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-6)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2.0e-6)
        self.assertEqual(scheduler.total_steps, 804)

    def test_linear_warmup_then_constant(self):
        scheduler = WarmupConstantLR.__new__(WarmupConstantLR)
        scheduler.warmup_steps = 32

        self.assertAlmostEqual(scheduler._get_scheduled_lr(2.0, 1), 2.0 / 32)
        self.assertAlmostEqual(scheduler._get_scheduled_lr(2.0, 32), 2.0)
        self.assertAlmostEqual(scheduler._get_scheduled_lr(2.0, 33), 2.0)
        self.assertAlmostEqual(scheduler._get_scheduled_lr(2.0, 804), 2.0)

    def test_flat_protocol_matches_reference_lr_step_integral(self):
        scheduler = WarmupConstantLR.__new__(WarmupConstantLR)
        scheduler.warmup_steps = 32
        lora_lr = 2.593849080532655e-5
        head_lr = 5.18769816106531e-6

        lora_area = sum(
            scheduler._get_scheduled_lr(lora_lr, step)
            for step in range(1, 805)
        )
        head_area = sum(
            scheduler._get_scheduled_lr(head_lr, step)
            for step in range(1, 805)
        )

        self.assertAlmostEqual(lora_area, 0.0204525, places=12)
        self.assertAlmostEqual(head_area / lora_area, 0.2, places=12)


if __name__ == "__main__":
    unittest.main()
