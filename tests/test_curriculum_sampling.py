import math
import sys
import unittest
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/custom_training/curriculum_sampling.py"
spec = importlib.util.spec_from_file_location("curriculum_sampling_under_test", MODULE_PATH)
curriculum_sampling = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = curriculum_sampling
spec.loader.exec_module(curriculum_sampling)

build_exact_bucket_quota_indices = curriculum_sampling.build_exact_bucket_quota_indices
build_exposure_capped_bucket_quota_indices = (
    curriculum_sampling.build_exposure_capped_bucket_quota_indices
)
build_exposure_capped_weighted_indices = (
    curriculum_sampling.build_exposure_capped_weighted_indices
)
exact_tercile_counts = curriculum_sampling.exact_tercile_counts
largest_remainder_counts = curriculum_sampling.largest_remainder_counts
split_scores_into_terciles = curriculum_sampling.split_scores_into_terciles
validate_master_score_coverage = curriculum_sampling.validate_master_score_coverage
validate_demonstration_type_routing = curriculum_sampling.validate_demonstration_type_routing


class TestPercentileTerciles(unittest.TestCase):
    def test_3059_bucket_sizes_are_exact_terciles(self):
        rows = [(f"scene_{idx:04d}", float(idx)) for idx in range(3059)]
        result = split_scores_into_terciles(rows, seed=42)
        self.assertEqual([len(result.groups[name]) for name in ("easy", "medium", "hard")], [1019, 1020, 1020])
        union = set().union(*(set(result.groups[name]) for name in ("easy", "medium", "hard")))
        self.assertEqual(len(union), 3059)
        self.assertEqual(sum(len(result.groups[name]) for name in ("easy", "medium", "hard")), 3059)

    def test_reproducible_with_same_seed(self):
        rows = [(f"scene_{idx:04d}", 1.0) for idx in range(3059)]
        first = split_scores_into_terciles(rows, seed=42)
        second = split_scores_into_terciles(rows, seed=42)
        self.assertEqual(first.groups, second.groups)

    def test_tie_split_changes_only_with_seed(self):
        rows = [(f"scene_{idx:04d}", 1.0) for idx in range(3059)]
        first = split_scores_into_terciles(rows, seed=42)
        second = split_scores_into_terciles(rows, seed=43)
        self.assertEqual([len(first.groups[name]) for name in ("easy", "medium", "hard")], [1019, 1020, 1020])
        self.assertEqual([len(second.groups[name]) for name in ("easy", "medium", "hard")], [1019, 1020, 1020])
        self.assertNotEqual(first.groups, second.groups)

    def test_coverage_rejects_missing_duplicate_nan_and_extra(self):
        with self.assertRaises(ValueError):
            validate_master_score_coverage(
                ["a", "b", "c"],
                [("a", 1.0), ("b", math.nan), ("b", 2.0), ("x", 3.0)],
            )


class TestExactQuotaSampler(unittest.TestCase):
    def _bucket_counts(self, indices, bucket_sizes):
        counts = []
        start = 0
        for size in bucket_sizes:
            stop = start + size
            counts.append(sum(1 for index in indices if start <= index < stop))
            start = stop
        return counts

    def test_largest_remainder_counts(self):
        self.assertEqual(largest_remainder_counts(3059, [0.50, 0.40, 0.10]), [1529, 1224, 306])
        self.assertEqual(largest_remainder_counts(3059, [0.267, 0.333, 0.400]), [817, 1019, 1223])
        self.assertEqual(exact_tercile_counts(3059), [1019, 1020, 1020])

    def test_exact_quota_stage2(self):
        bucket_sizes = [1019, 1020, 1020]
        indices, metadata = build_exact_bucket_quota_indices(
            bucket_sizes,
            [0.50, 0.40, 0.10],
            max_repeat_per_scenario=4,
            seed=42,
            epoch=0,
        )
        self.assertEqual(len(indices), 3059)
        self.assertEqual(self._bucket_counts(indices, bucket_sizes), [1529, 1224, 306])
        for stats in metadata["repeat_stats"].values():
            self.assertLessEqual(stats["max_repeat"], 4)

    def test_exact_quota_mild_hard_focus(self):
        bucket_sizes = [1019, 1020, 1020]
        indices, _ = build_exact_bucket_quota_indices(
            bucket_sizes,
            [0.267, 0.333, 0.400],
            max_repeat_per_scenario=4,
            seed=42,
            epoch=0,
        )
        self.assertEqual(self._bucket_counts(indices, bucket_sizes), [817, 1019, 1223])

    def test_uniform_phase_is_near_permutation(self):
        bucket_sizes = [1019, 1020, 1020]
        indices, metadata = build_exact_bucket_quota_indices(
            bucket_sizes,
            [1 / 3, 1 / 3, 1 / 3],
            max_repeat_per_scenario=4,
            seed=42,
            epoch=0,
        )
        self.assertEqual(self._bucket_counts(indices, bucket_sizes), [1020, 1020, 1019])
        self.assertLessEqual(max(stats["max_repeat"] for stats in metadata["repeat_stats"].values()), 2)

    def test_impossible_repeat_cap_fails(self):
        with self.assertRaises(ValueError):
            build_exact_bucket_quota_indices(
                [1, 100, 100],
                [0.90, 0.05, 0.05],
                max_repeat_per_scenario=4,
                seed=42,
                epoch=0,
            )

    def test_seed_epoch_reproducibility_and_epoch_change(self):
        kwargs = dict(
            bucket_sizes=[1019, 1020, 1020],
            target_proportions=[0.50, 0.40, 0.10],
            max_repeat_per_scenario=4,
            seed=42,
        )
        first, _ = build_exact_bucket_quota_indices(epoch=0, **kwargs)
        second, _ = build_exact_bucket_quota_indices(epoch=0, **kwargs)
        third, _ = build_exact_bucket_quota_indices(epoch=1, **kwargs)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


class TestDemonstrationTypeRoutingGuard(unittest.TestCase):
    def test_observe_only_never_requires_llm_method(self):
        self.assertEqual(
            validate_demonstration_type_routing("observe_only", "rule"),
            "observe_only",
        )

    def test_enabled_is_llm_v4_only(self):
        self.assertEqual(
            validate_demonstration_type_routing("enabled", "llm_guided_v4"),
            "enabled",
        )
        with self.assertRaises(ValueError):
            validate_demonstration_type_routing("enabled", "mpoc")


class TestExposureCappedExactQuota(unittest.TestCase):
    def _metadata(self):
        scenario_ids = [f"scene_{index}" for index in range(12)]
        groups = [[f"cell_{index // 2}"] for index in range(12)]
        return scenario_ids, groups

    def test_exact_quota_respects_temporal_cell_cap(self):
        scenario_ids, groups = self._metadata()
        indices, metadata = build_exposure_capped_bucket_quota_indices(
            [4, 4, 4],
            [1 / 3, 1 / 3, 1 / 3],
            scenario_ids=scenario_ids,
            near_duplicate_groups=groups,
            max_repeat_per_scenario=2,
            max_repeat_per_group=2,
            seed=42,
            epoch=0,
        )
        self.assertEqual(len(indices), 12)
        self.assertEqual(metadata["actual_draws"], {"easy": 4, "medium": 4, "hard": 4})
        self.assertLessEqual(metadata["max_near_duplicate_group_exposure"], 2)

    def test_cumulative_cap_excludes_exhausted_scenario(self):
        scenario_ids = [f"scene_{index}" for index in range(6)]
        groups = [[f"cell_{index}"] for index in range(6)]
        indices, metadata = build_exposure_capped_bucket_quota_indices(
            [2, 2, 2],
            [1 / 3, 1 / 3, 1 / 3],
            scenario_ids=scenario_ids,
            near_duplicate_groups=groups,
            max_repeat_per_scenario=2,
            max_repeat_per_group=2,
            prior_scenario_exposure={"scene_0": 2},
            max_cumulative_exposure_per_scenario=2,
            seed=7,
            epoch=1,
        )
        self.assertNotIn(0, indices)
        self.assertEqual(metadata["actual_draws"]["easy"], 2)

    def test_impossible_group_cap_fails_instead_of_silent_oversampling(self):
        scenario_ids = [f"scene_{index}" for index in range(6)]
        groups = [["one_cell"] for _ in scenario_ids]
        with self.assertRaisesRegex(ValueError, "Exposure caps make exact quota impossible"):
            build_exposure_capped_bucket_quota_indices(
                [2, 2, 2],
                [1 / 3, 1 / 3, 1 / 3],
                scenario_ids=scenario_ids,
                near_duplicate_groups=groups,
                max_repeat_per_scenario=2,
                max_repeat_per_group=2,
                seed=0,
                epoch=0,
            )

    def test_weighted_draw_respects_scenario_and_group_caps(self):
        scenario_ids = [f"scene_{index}" for index in range(6)]
        groups = [[f"cell_{index}"] for index in range(6)]
        indices, metadata = build_exposure_capped_weighted_indices(
            [10.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            scenario_ids=scenario_ids,
            near_duplicate_groups=groups,
            num_samples=6,
            max_repeat_per_scenario=2,
            max_repeat_per_group=2,
            category_ids=["special", "special", "normal", "normal", "normal", "normal"],
            max_exposure_per_category={"special": 1},
            seed=3,
            epoch=0,
        )
        self.assertEqual(len(indices), 6)
        self.assertLessEqual(metadata["max_scenario_exposure"], 2)
        self.assertLessEqual(metadata["max_near_duplicate_group_exposure"], 2)
        self.assertLessEqual(metadata["category_exposure"].get("special", 0), 1)


if __name__ == "__main__":
    unittest.main()
