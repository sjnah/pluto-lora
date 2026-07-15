import importlib.util
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/evaluation/collect_quick_test_results.py"
spec = importlib.util.spec_from_file_location("quick_test_collector_under_test", MODULE_PATH)
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)


class TestTypeRoutingMethodVariants(unittest.TestCase):
    def test_off_and_on_are_distinct_supported_method_keys(self):
        off = "curriculum_llm_percentile_ehu_v5.0.4.0.13_type_off"
        on = "curriculum_llm_percentile_ehu_v5.0.4.0.13_type_on"
        self.assertTrue(collector.is_percentile_ehu_version_method(off))
        self.assertTrue(collector.is_percentile_ehu_version_method(on))
        off_spec = collector.method_spec_for_key(off)
        on_spec = collector.method_spec_for_key(on)
        self.assertIsNotNone(off_spec)
        self.assertIsNotNone(on_spec)
        self.assertNotEqual(off_spec.key, on_spec.key)
        self.assertIn("type routing off", off_spec.label)
        self.assertIn("type routing on", on_spec.label)


class TestSeedAggregation(unittest.TestCase):
    @staticmethod
    def row(seed, score, method="curriculum_llm_percentile_ehu_v5.0.4.0.13"):
        return {
            "test": "test14_hard",
            "test_label": "Test14-hard",
            "method": f"{method}_seed{seed}",
            "method_label": f"LLM seed {seed}",
            "experiment": f"quick_test_{method}_seed{seed}_test14_hard",
            "status": "ok",
            "score": score,
            "score_std": 0.2 + seed / 100,
            "scenario_count": 286,
            "expected_scenarios": 286,
            "perfect_count": 10 + seed,
            "metric_type": "NR-CLS",
            "simulation_type": "nonreactive",
            "simulation_challenge": "closed_loop_nonreactive_agents",
            "source": "direct",
            "metrics_dirs": [f"/tmp/seed{seed}/metrics"],
            "error": None,
        }

    def test_multiple_seeds_collapse_to_mean_and_sample_std(self):
        rows = [self.row(1, 0.5), self.row(2, 0.7), self.row(3, 0.9)]
        result = collector.aggregate_seeded_rows(rows)
        self.assertEqual(len(result), 1)
        aggregate = result[0]
        self.assertEqual(aggregate["method"], "curriculum_llm_percentile_ehu_v5.0.4.0.13")
        self.assertEqual(aggregate["seeds"], [1, 2, 3])
        self.assertEqual(aggregate["seed_count"], 3)
        self.assertAlmostEqual(aggregate["score"], 0.7)
        self.assertAlmostEqual(aggregate["seed_score_std"], 0.2)
        self.assertEqual(collector.format_score(aggregate), "0.7000 +/- 0.2000")

    def test_one_seed_stays_as_original_row(self):
        row = self.row(7, 0.6)
        self.assertEqual(collector.aggregate_seeded_rows([row]), [row])

    def test_incomparable_simulation_modes_do_not_collapse(self):
        first = self.row(1, 0.5)
        second = self.row(2, 0.7)
        second["simulation_type"] = "reactive"
        result = collector.aggregate_seeded_rows([first, second])
        self.assertEqual(len(result), 2)


class TestCsvOutput(unittest.TestCase):
    def test_detail_csv_uses_stable_fields_when_later_rows_have_extra_keys(self):
        rows = [
            {
                "test": "val14_fast",
                "test_label": "Val14 fast",
                "method": "zeroshot",
                "method_label": "Zero-shot",
                "experiment": "quick_test_zeroshot_val14_fast",
                "status": "invalid",
                "source": "glob",
                "metrics_dirs": ["/tmp/invalid/metrics"],
                "error": "Missing required metrics",
            },
            {
                "test": "val14_fast",
                "test_label": "Val14 fast",
                "method": "rulebased",
                "method_label": "Rule-based",
                "experiment": "quick_test_rulebased_val14_fast",
                "status": "ok",
                "score": 0.5,
                "source": "glob",
                "metrics_dirs": ["/tmp/ok/metrics"],
                "metric_means": {
                    "no_ego_at_fault_collisions": 1.0,
                    "drivable_area_compliance": 0.75,
                },
                "metric_counts": {
                    "no_ego_at_fault_collisions": 270,
                    "drivable_area_compliance": 270,
                },
                "without_collision": 1.0,
                "drivable": 0.75,
            },
        ]
        buffer = io.StringIO()
        collector.write_csv_rows(rows, buffer, include_detail=True)
        output = buffer.getvalue()
        self.assertIn("without_collision,drivable,progress", output.splitlines()[0])
        self.assertIn('"{""drivable_area_compliance"": 270', output)


if __name__ == "__main__":
    unittest.main()
