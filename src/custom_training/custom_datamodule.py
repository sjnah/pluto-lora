import logging
import random
import json
import csv
import fcntl
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data.sampler import WeightedRandomSampler

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_loader.distributed_sampler_wrapper import (
    DistributedSamplerWrapper,
)
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    move_features_type_to_device,
)
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from src.custom_training.curriculum_sampling import (
    apply_near_duplicate_group_inverse_weighting,
    build_exact_bucket_quota_indices,
    build_exposure_capped_bucket_quota_indices,
    build_exposure_capped_weighted_indices,
    scheduled_target_proportions,
    validate_demonstration_type_routing,
)

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError('Data module has not been setup, call "setup()"')


def _plain_config(value: Any) -> Any:
    """Recursively detach Hydra/OmegaConf containers from runtime state."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


class FixedIndexSampler(torch.utils.data.Sampler[int]):
    """Sampler that yields a precomputed index list."""

    def __init__(self, indices: List[int]):
        self._indices = list(indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class ExactBucketQuotaSampler(torch.utils.data.Sampler[int]):
    """Sampler that enforces exact bucket-level draw quotas per epoch."""

    def __init__(
        self,
        bucket_sizes: List[int],
        target_proportions: List[float],
        max_repeat_per_scenario: int,
        random_seed: int,
        sampling_log_path: Optional[str] = None,
        score_method: str = "",
        phase_name: str = "",
        phase_start_epoch: int = 0,
        scenario_records: Optional[List[Dict[str, Any]]] = None,
        max_repeat_per_group: int = 0,
        cumulative_exposure_state_path: Optional[str] = None,
        max_cumulative_exposure_per_scenario: int = 0,
        max_cumulative_exposure_per_group: int = 0,
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        pacing_schedule: Optional[Dict[str, Any]] = None,
    ):
        self._bucket_sizes = [int(size) for size in bucket_sizes]
        self._target_proportions = [float(value) for value in target_proportions]
        self._max_repeat_per_scenario = int(max_repeat_per_scenario)
        self._random_seed = int(random_seed)
        self._sampling_log_path = sampling_log_path
        self._score_method = score_method
        self._phase_name = phase_name
        self._phase_start_epoch = int(phase_start_epoch)
        if self._phase_start_epoch < 0:
            raise ValueError("phase_start_epoch must be non-negative")
        self._scenario_records = list(scenario_records or [])
        self._max_repeat_per_group = int(max_repeat_per_group)
        self._cumulative_exposure_state_path = cumulative_exposure_state_path
        self._max_cumulative_exposure_per_scenario = int(max_cumulative_exposure_per_scenario)
        self._max_cumulative_exposure_per_group = int(max_cumulative_exposure_per_group)
        self._batch_size = max(1, int(batch_size))
        self._accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self._pacing_schedule = dict(_plain_config(pacing_schedule or {}))
        self._epoch: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _current_epoch(self) -> int:
        if self._epoch is not None:
            return self._epoch
        # DistributedSamplerWrapper calls torch.manual_seed(self.epoch) before
        # iterating the wrapped sampler, so torch.initial_seed() carries the
        # epoch in the existing nuPlan wrapper path.
        return int(torch.initial_seed())

    def _pool_fingerprint(self) -> str:
        payload = [
            {
                "scenario_id": record["scenario_id"],
                "near_duplicate_groups": record["near_duplicate_groups"],
            }
            for record in self._scenario_records
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _plan_fingerprint(self, epoch: int) -> str:
        payload = {
            "phase_name": self._phase_name,
            "epoch": int(epoch),
            "bucket_sizes": self._bucket_sizes,
            "target_proportions": self._target_proportions,
            "pacing_schedule": self._pacing_schedule,
            "max_repeat_per_scenario": self._max_repeat_per_scenario,
            "max_repeat_per_group": self._max_repeat_per_group,
            "max_cumulative_exposure_per_scenario": self._max_cumulative_exposure_per_scenario,
            "max_cumulative_exposure_per_group": self._max_cumulative_exposure_per_group,
            "random_seed": self._random_seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_indices(self, epoch: int, prior_scenario=None, prior_group=None):
        target_proportions, pacing_metadata = scheduled_target_proportions(
            self._target_proportions,
            self._pacing_schedule,
            epoch=epoch,
            phase_start_epoch=self._phase_start_epoch,
        )
        if self._scenario_records and self._max_repeat_per_group > 0:
            indices, metadata = build_exposure_capped_bucket_quota_indices(
                self._bucket_sizes,
                target_proportions,
                scenario_ids=[record["scenario_id"] for record in self._scenario_records],
                near_duplicate_groups=[
                    record["near_duplicate_groups"] for record in self._scenario_records
                ],
                max_repeat_per_scenario=self._max_repeat_per_scenario,
                max_repeat_per_group=self._max_repeat_per_group,
                prior_scenario_exposure=prior_scenario,
                prior_group_exposure=prior_group,
                max_cumulative_exposure_per_scenario=self._max_cumulative_exposure_per_scenario,
                max_cumulative_exposure_per_group=self._max_cumulative_exposure_per_group,
                seed=self._random_seed,
                epoch=epoch,
            )
        else:
            indices, metadata = build_exact_bucket_quota_indices(
                self._bucket_sizes,
                target_proportions,
                max_repeat_per_scenario=self._max_repeat_per_scenario,
                seed=self._random_seed,
                epoch=epoch,
            )
        if self._pacing_schedule:
            metadata["base_target_proportions"] = self._target_proportions
            metadata["pacing_schedule"] = self._pacing_schedule
            metadata["pacing"] = pacing_metadata
        if self._scenario_records:
            sampled_records = [self._scenario_records[index] for index in indices]
            metadata.update(
                {
                    "sampled_split_counts": dict(
                        Counter(record.get("split", "unknown") for record in sampled_records)
                    ),
                    "log_exposure": dict(
                        Counter(record.get("log_name", "unknown") for record in sampled_records)
                    ),
                    "demonstration_type_exposure": dict(
                        Counter(
                            record.get("demonstration_type", "normal")
                            for record in sampled_records
                        )
                    ),
                    "scenario_id_list": [
                        record["scenario_id"] for record in sampled_records
                    ],
                    "actual_sampled_indices_logged": True,
                }
            )
        return indices, metadata

    def _persistent_plan(self, epoch: int):
        if not self._cumulative_exposure_state_path:
            return self._build_indices(epoch)

        state_path = Path(self._cumulative_exposure_state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")
        plan_key = f"{self._phase_name or 'phase'}:{epoch}"
        fingerprint = self._pool_fingerprint()
        plan_fingerprint = self._plan_fingerprint(epoch)
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if state_path.exists():
                with state_path.open("r", encoding="utf-8") as stream:
                    state = json.load(stream)
            else:
                state = {
                    "schema_version": 1,
                    "pool_fingerprint": fingerprint,
                    "cumulative_scenario_exposure": {},
                    "cumulative_near_duplicate_group_exposure": {},
                    "plans": {},
                }
            if state.get("pool_fingerprint") != fingerprint:
                raise ValueError(
                    "Cumulative exposure state belongs to a different scenario pool: "
                    f"{state_path}"
                )
            existing = state.get("plans", {}).get(plan_key)
            if existing is not None:
                if existing.get("plan_fingerprint") != plan_fingerprint:
                    raise ValueError(
                        "Cumulative exposure plan configuration changed for existing key "
                        f"{plan_key}: {state_path}"
                    )
                return list(existing["indices"]), dict(existing["metadata"])

            indices, metadata = self._build_indices(
                epoch,
                prior_scenario=state.get("cumulative_scenario_exposure", {}),
                prior_group=state.get("cumulative_near_duplicate_group_exposure", {}),
            )
            state["cumulative_scenario_exposure"] = metadata.get(
                "cumulative_scenario_exposure", {}
            )
            state["cumulative_near_duplicate_group_exposure"] = metadata.get(
                "cumulative_near_duplicate_group_exposure", {}
            )
            state.setdefault("plans", {})[plan_key] = {
                "plan_fingerprint": plan_fingerprint,
                "indices": indices,
                "metadata": metadata,
            }
            temporary_path = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, sort_keys=True)
            os.replace(temporary_path, state_path)
            return indices, metadata

    def __iter__(self):
        epoch = self._current_epoch()
        if epoch < self._phase_start_epoch:
            # Lightning may iterate the dataloader while computing
            # estimated_stepping_batches, before the resume checkpoint restores
            # the real epoch. That preflight must not consume cumulative
            # exposure or create an epoch-0 sampling artifact for later phases.
            logger.info(
                "Building non-persistent pre-resume sampling plan for phase %s: "
                "observed epoch=%d, phase_start_epoch=%d",
                self._phase_name,
                epoch,
                self._phase_start_epoch,
            )
            indices, _metadata = self._build_indices(epoch)
            return iter(indices)
        indices, metadata = self._persistent_plan(epoch)
        metadata["score_method"] = self._score_method
        metadata["phase_name"] = self._phase_name
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
        metadata["distributed_rank"] = rank
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = int(torch.distributed.get_world_size())
        per_rank_batches = math.ceil(len(indices) / world_size / self._batch_size)
        metadata["total_sampled_items"] = len(indices)
        metadata["estimated_optimizer_updates"] = math.ceil(
            per_rank_batches / self._accumulate_grad_batches
        )
        metadata["cumulative_exposure_state_path"] = self._cumulative_exposure_state_path
        if self._sampling_log_path:
            base_path = Path(self._sampling_log_path)
            log_path = base_path.with_name(
                f"{base_path.stem}.epoch_{epoch:04d}.rank_{rank:03d}{base_path.suffix or '.json'}"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            logger.info("Saved exact quota curriculum sampling log to %s", log_path)
        return iter(indices)

    def __len__(self) -> int:
        return sum(self._bucket_sizes)


class ExposureCappedWeightedSampler(ExactBucketQuotaSampler):
    """Epoch-varying weighted sampler with persistent exposure caps."""

    def __init__(
        self,
        *,
        weights: List[float],
        scenario_records: List[Dict[str, Any]],
        num_samples: int,
        max_repeat_per_scenario: int,
        max_repeat_per_group: int,
        random_seed: int,
        cumulative_exposure_state_path: Optional[str],
        max_cumulative_exposure_per_scenario: int,
        max_cumulative_exposure_per_group: int,
        sampling_log_path: Optional[str] = None,
        score_method: str = "",
        phase_name: str = "",
        phase_start_epoch: int = 0,
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        max_exposure_per_demonstration_type: Optional[Dict[str, int]] = None,
        pacing_schedule: Optional[Dict[str, Any]] = None,
        split_names: Optional[List[str]] = None,
        base_target_proportions: Optional[List[float]] = None,
    ):
        if (
            max_cumulative_exposure_per_scenario > 0
            or max_cumulative_exposure_per_group > 0
        ) and not cumulative_exposure_state_path:
            raise ValueError("Cumulative exposure caps require cumulative_exposure_state_path")
        self._weights = [float(weight) for weight in weights]
        self._num_samples = int(num_samples)
        self._max_exposure_per_demonstration_type = dict(
            max_exposure_per_demonstration_type or {}
        )
        self._split_names = list(split_names or [])
        self._base_target_proportions = [
            float(value) for value in (base_target_proportions or [])
        ]
        if pacing_schedule:
            if not self._split_names or not self._base_target_proportions:
                raise ValueError(
                    "Weighted pacing requires split_names and base_target_proportions"
                )
            if len(self._split_names) != len(self._base_target_proportions):
                raise ValueError(
                    "Weighted pacing split names and target proportions must align"
                )
        super().__init__(
            bucket_sizes=[self._num_samples],
            target_proportions=[1.0],
            max_repeat_per_scenario=max_repeat_per_scenario,
            random_seed=random_seed,
            sampling_log_path=sampling_log_path,
            score_method=score_method,
            phase_name=phase_name,
            phase_start_epoch=phase_start_epoch,
            scenario_records=scenario_records,
            max_repeat_per_group=max_repeat_per_group,
            cumulative_exposure_state_path=cumulative_exposure_state_path,
            max_cumulative_exposure_per_scenario=max_cumulative_exposure_per_scenario,
            max_cumulative_exposure_per_group=max_cumulative_exposure_per_group,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            pacing_schedule=pacing_schedule,
        )

    def _plan_fingerprint(self, epoch: int) -> str:
        payload = {
            "base": super()._plan_fingerprint(epoch),
            "weights": self._weights,
            "num_samples": self._num_samples,
            "max_exposure_per_demonstration_type": self._max_exposure_per_demonstration_type,
        }
        if self._split_names:
            payload["split_names"] = self._split_names
            payload["base_target_proportions"] = self._base_target_proportions
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_indices(self, epoch: int, prior_scenario=None, prior_group=None):
        weights = self._weights
        pacing_metadata: Dict[str, Any] = {}
        resolved_proportions: List[float] = []
        if self._split_names and self._base_target_proportions:
            resolved_proportions, pacing_metadata = scheduled_target_proportions(
                self._base_target_proportions,
                self._pacing_schedule,
                epoch=epoch,
                phase_start_epoch=self._phase_start_epoch,
            )
            split_masses = {
                split_name: sum(
                    weight
                    for weight, record in zip(self._weights, self._scenario_records)
                    if record.get("split") == split_name
                )
                for split_name in self._split_names
            }
            if any(mass <= 0.0 for mass in split_masses.values()):
                raise ValueError(
                    f"Weighted pacing encountered an empty split mass: {split_masses}"
                )
            target_by_split = dict(zip(self._split_names, resolved_proportions))
            weights = [
                weight
                * target_by_split[str(record.get("split"))]
                / split_masses[str(record.get("split"))]
                for weight, record in zip(self._weights, self._scenario_records)
            ]

        indices, metadata = build_exposure_capped_weighted_indices(
            weights,
            scenario_ids=[record["scenario_id"] for record in self._scenario_records],
            near_duplicate_groups=[
                record["near_duplicate_groups"] for record in self._scenario_records
            ],
            num_samples=self._num_samples,
            max_repeat_per_scenario=self._max_repeat_per_scenario,
            max_repeat_per_group=self._max_repeat_per_group,
            prior_scenario_exposure=prior_scenario,
            prior_group_exposure=prior_group,
            max_cumulative_exposure_per_scenario=self._max_cumulative_exposure_per_scenario,
            max_cumulative_exposure_per_group=self._max_cumulative_exposure_per_group,
            category_ids=[
                record.get("demonstration_type", "normal")
                for record in self._scenario_records
            ],
            max_exposure_per_category=self._max_exposure_per_demonstration_type,
            seed=self._random_seed,
            epoch=epoch,
        )
        sampled_records = [self._scenario_records[index] for index in indices]
        metadata.update(
            {
                "sampled_split_counts": dict(
                    Counter(record["split"] for record in sampled_records)
                ),
                "log_exposure": dict(
                    Counter(record["log_name"] for record in sampled_records)
                ),
                "demonstration_type_exposure": dict(
                    Counter(
                        record.get("demonstration_type", "normal")
                        for record in sampled_records
                    )
                ),
                "scenario_id_list": [record["scenario_id"] for record in sampled_records],
                "actual_sampled_indices_logged": True,
                "base_target_proportions": self._base_target_proportions,
                "resolved_target_proportions": resolved_proportions,
                "pacing_schedule": self._pacing_schedule,
                "pacing": pacing_metadata,
            }
        )
        return indices, metadata


def _scenario_id(scenario: AbstractScenario) -> str:
    return str(getattr(scenario, "token", None) or getattr(scenario, "scenario_name", "unknown"))


def _near_duplicate_groups(scenario: AbstractScenario, cell_width_s: float = 10.0) -> List[str]:
    log_name = str(getattr(scenario, "log_name", "unknown_log"))
    try:
        start_s = float(scenario.get_time_point(0).time_s)
        iteration_count = int(scenario.get_number_of_iterations())
        end_s = float(scenario.get_time_point(max(0, iteration_count - 1)).time_s)
        if end_s < start_s:
            end_s = start_s
    except Exception:
        return [f"{log_name}:unavailable"]
    first_cell = int(math.floor(start_s / cell_width_s))
    last_cell = int(math.floor(max(start_s, end_s - 1e-9) / cell_width_s))
    return [f"{log_name}:{cell}" for cell in range(first_cell, last_cell + 1)]


def _sample_indices_with_repeat_cap(
    weights: List[float],
    num_samples: int,
    max_repeat_per_scenario: int,
    random_seed: int,
) -> List[int]:
    if max_repeat_per_scenario <= 0:
        raise ValueError("max_repeat_per_scenario must be positive for capped sampling")
    if len(weights) * max_repeat_per_scenario < num_samples:
        raise ValueError(
            "max_repeat_per_scenario is too small for the requested sample count: "
            f"{len(weights)} scenarios * {max_repeat_per_scenario} < {num_samples}"
        )

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    base_weights = torch.as_tensor(weights, dtype=torch.double)
    selected: List[int] = []
    counts: Counter[int] = Counter()

    for _ in range(num_samples):
        effective_weights = base_weights.clone()
        for index, count in counts.items():
            if count >= max_repeat_per_scenario:
                effective_weights[index] = 0.0
        if float(effective_weights.sum().item()) <= 0.0:
            raise RuntimeError("No sampling weight remains before reaching num_samples")
        sampled = int(torch.multinomial(effective_weights, 1, replacement=True, generator=generator).item())
        selected.append(sampled)
        counts[sampled] += 1

    return selected


class RobustScenarioDataset(torch.utils.data.Dataset):
    """
    Wrapper around ScenarioDataset that gracefully handles RuntimeError
    from invalid scenarios (e.g., scenarios with invalid route data).
    When an error occurs, it retries with a different scenario index.
    """
    def __init__(self, base_dataset: ScenarioDataset, max_retries: int = 10):
        """
        Initialize the robust dataset wrapper.
        :param base_dataset: The underlying ScenarioDataset to wrap.
        :param max_retries: Maximum number of retries when encountering errors.
        """
        self.base_dataset = base_dataset
        self.max_retries = max_retries
        self._scenarios = base_dataset._scenarios
        self._feature_preprocessor = base_dataset._feature_preprocessor
        self._augmentors = base_dataset._augmentors
        self._invalid_indices = set()
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from dataset, retrying with different indices if RuntimeError occurs.
        """
        dataset_len = len(self.base_dataset)
        if dataset_len == 0:
            raise RuntimeError("Cannot sample from an empty dataset")

        max_attempts = min(max(self.max_retries, 1), dataset_len)
        candidate_indices = []

        # Prefer nearby samples first to keep deterministic behavior for the common case.
        for offset in range(max_attempts):
            candidate_indices.append((idx + offset) % dataset_len)

        # If nearby samples are invalid, spread retries across the dataset instead of
        # repeatedly failing on a cluster of bad route annotations.
        remaining_indices = [
            candidate_idx
            for candidate_idx in range(dataset_len)
            if candidate_idx not in self._invalid_indices
            and candidate_idx not in candidate_indices
        ]
        random.shuffle(remaining_indices)
        candidate_indices.extend(remaining_indices[:max_attempts])

        last_error_msg = None
        for attempt, current_idx in enumerate(candidate_indices, start=1):
            if current_idx in self._invalid_indices:
                continue

            try:
                return self.base_dataset[current_idx]
            except RuntimeError as e:
                error_msg = str(e)
                last_error_msg = error_msg
                # Check if it's a route computation error
                if "Failed to compute route" in error_msg or "Failed to compute features" in error_msg:
                    self._invalid_indices.add(current_idx)
                    logger.warning(
                        f"Skipping scenario at index {current_idx} due to route computation error "
                        f"(attempt {attempt}/{len(candidate_indices)}, "
                        f"known invalid: {len(self._invalid_indices)}): {error_msg}"
                    )
                    continue
                else:
                    # For other RuntimeErrors, re-raise immediately
                    raise

        logger.error(
            f"Failed to get a valid scenario after trying {len(candidate_indices)} candidates. "
            f"Known invalid scenarios: {len(self._invalid_indices)}/{dataset_len}. "
            f"Last error: {last_error_msg}"
        )
        raise RuntimeError(
            f"Failed to get a valid scenario after trying {len(candidate_indices)} candidates"
        )


def create_dataset(
    samples: List[AbstractScenario],
    feature_preprocessor: FeaturePreprocessor,
    dataset_fraction: float,
    dataset_name: str,
    augmentors: Optional[List[AbstractAugmentor]] = None,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    :param samples: List of dataset candidate samples.
    :param feature_preprocessor: Feature preprocessor object.
    :param dataset_fraction: Fraction of the dataset to load.
    :param dataset_name: Set name (train/val/test).
    :param scenario_type_loss_weights: Dictionary of scenario type loss weights.
    :param augmentors: List of augmentor objects for providing data augmentation to data samples.
    :return: The instantiated torch dataset.
    """
    # Sample the desired fraction from the total samples
    num_keep = int(len(samples) * dataset_fraction)
    selected_scenarios = random.sample(samples, num_keep)

    logger.info(f"Number of samples in {dataset_name} set: {len(selected_scenarios)}")
    base_dataset = ScenarioDataset(
        scenarios=selected_scenarios,
        feature_preprocessor=feature_preprocessor,
        augmentors=augmentors,
    )
    # Wrap with RobustScenarioDataset to handle invalid scenarios gracefully
    return RobustScenarioDataset(base_dataset, max_retries=10)


def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,
    scenario_sampling_weights: Dict[str, float],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Initiliazes WeightedSampler object with sampling weights for each scenario_type and returns it.
    :param scenario_dataset: ScenarioDataset object
    :param replacement: Samples with replacement if True. By default set to True.
    return: Initialized Weighted sampler
    """
    scenarios = scenario_dataset._scenarios
    if (
        not replacement
    ):  # If we don't sample with replacement, then all sample weights must be nonzero
        assert all(
            w > 0 for w in scenario_sampling_weights.values()
        ), "All scenario sampling weights must be positive when sampling without replacement."

    default_scenario_sampling_weight = 1.0

    scenario_sampling_weights_per_idx = [
        scenario_sampling_weights[scenario.scenario_type]
        if scenario.scenario_type in scenario_sampling_weights
        else default_scenario_sampling_weight
        for scenario in scenarios
    ]

    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=scenario_sampling_weights_per_idx,
        num_samples=len(scenarios),
        replacement=replacement,
    )

    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
    return distributed_weighted_sampler


def distributed_curriculum_sampler_init(
    scenario_datasets: List[ScenarioDataset],
    split_weights: List[float],
    replacement: bool = True,
    split_names: Optional[List[str]] = None,
    max_repeat_per_scenario: int = 0,
    random_seed: int = 42,
    sampling_log_path: Optional[str] = None,
    score_method: str = "",
    filter_file_path: str = "",
    hard_subtype_balance: bool = False,
    sampler_mode: str = "exposure_capped_weighted",
    phase_name: str = "",
    phase_start_epoch: int = 0,
    curriculum_method: str = "",
    demonstration_type_mode: str = "observe_only",
    demonstration_type_metadata_path: Optional[str] = None,
    demonstration_type_policy: Optional[Dict[str, Any]] = None,
    max_repeat_per_near_duplicate_group: int = 0,
    near_duplicate_group_weighting: bool = False,
    cumulative_exposure_state_path: Optional[str] = None,
    max_cumulative_exposure_per_scenario: int = 0,
    max_cumulative_exposure_per_near_duplicate_group: int = 0,
    batch_size: int = 1,
    accumulate_grad_batches: int = 1,
    pacing_schedule: Optional[Dict[str, Any]] = None,
) -> WeightedRandomSampler:
    """
    Initialize WeightedSampler for curriculum learning with multiple splits.
    Each split gets a weight, and scenarios within each split are sampled according to that weight.
    
    :param scenario_datasets: List of ScenarioDataset objects, one per split
    :param split_weights: List of weights for each split (e.g., [0.7, 0.3] for 70% split1, 30% split2)
    :param replacement: Samples with replacement if True. By default set to True.
    :return: Initialized Weighted sampler
    """
    assert len(scenario_datasets) == len(split_weights), \
        f"Number of datasets ({len(scenario_datasets)}) must match number of weights ({len(split_weights)})"
    
    assert all(w > 0 for w in split_weights), "All split weights must be positive"
    
    # Normalize weights
    total_weight = sum(split_weights)
    normalized_weights = [w / total_weight for w in split_weights]

    type_mode = validate_demonstration_type_routing(
        demonstration_type_mode, curriculum_method
    )
    if type_mode == "enabled" and sampler_mode == "exact_bucket_quota":
        raise ValueError(
            "Demonstration-type routing currently requires exposure_capped_weighted so its "
            "stage-level expected exposure weights remain explicit"
        )
    
    split_names = split_names or [f"split_{idx}" for idx in range(len(scenario_datasets))]

    sampler_mode = str(sampler_mode or "exposure_capped_weighted")
    if sampler_mode not in {
        "exposure_capped_weighted",
        "legacy_weighted",  # Compatibility alias for pre-refactor snapshots.
        "exact_bucket_quota",
    }:
        raise ValueError(f"Unsupported curriculum sampler_mode: {sampler_mode}")
    if (
        pacing_schedule
        and sampler_mode in {"exposure_capped_weighted", "legacy_weighted"}
        and max_repeat_per_near_duplicate_group <= 0
    ):
        raise ValueError(
            "weighted pacing requires the exposure-capped epoch-aware sampler"
        )

    scenario_records: List[Dict[str, Any]] = []
    for split_name, dataset in zip(split_names, scenario_datasets):
        for scenario in dataset._scenarios:
            groups = _near_duplicate_groups(scenario)
            scenario_records.append(
                {
                    "split": split_name,
                    "scenario_id": _scenario_id(scenario),
                    "near_duplicate_group": groups[0],
                    "near_duplicate_groups": groups,
                    "log_name": str(getattr(scenario, "log_name", "unknown_log")),
                }
            )

    if sampler_mode == "exact_bucket_quota":
        bucket_sizes = [len(dataset._scenarios) for dataset in scenario_datasets]
        sampler = ExactBucketQuotaSampler(
            bucket_sizes=bucket_sizes,
            target_proportions=normalized_weights,
            max_repeat_per_scenario=max_repeat_per_scenario,
            random_seed=random_seed,
            sampling_log_path=sampling_log_path,
            score_method=score_method,
            phase_name=phase_name,
            phase_start_epoch=phase_start_epoch,
            scenario_records=scenario_records,
            max_repeat_per_group=max_repeat_per_near_duplicate_group,
            cumulative_exposure_state_path=cumulative_exposure_state_path,
            max_cumulative_exposure_per_scenario=max_cumulative_exposure_per_scenario,
            max_cumulative_exposure_per_group=max_cumulative_exposure_per_near_duplicate_group,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            pacing_schedule=pacing_schedule,
        )
        distributed_weighted_sampler = DistributedSamplerWrapper(sampler)
        return distributed_weighted_sampler

    # Legacy mode: create weights for each scenario. Divide by split size so
    # split_weights represent split-level sampling probabilities, not per-sample
    # weights.
    all_weights = []
    for split_name, dataset, weight in zip(split_names, scenario_datasets, normalized_weights):
        num_scenarios = len(dataset._scenarios)
        assert num_scenarios > 0, "Curriculum split dataset must not be empty"
        all_weights.extend([weight / num_scenarios] * num_scenarios)

    type_metadata: Dict[str, Dict[str, Any]] = {}
    type_draw_caps: Dict[str, int] = {}
    if demonstration_type_metadata_path:
        metadata_path = Path(demonstration_type_metadata_path)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    scene_id = str(row.get("scene_id") or row.get("scenario_id") or "")
                    if scene_id:
                        type_metadata[scene_id] = row
        elif type_mode == "enabled":
            raise FileNotFoundError(f"Demonstration-type metadata not found: {metadata_path}")

    if type_mode == "enabled":
        policy = dict(demonstration_type_policy or {})
        role = str(policy.get("stage_role", phase_name or "all_consolidation"))
        presets = policy.get("stage_presets", {})
        preset = dict(presets.get(role, presets.get("all_consolidation", {})))
        multipliers = {
            "normal": 1.0,
            "necessary_exception": float(preset.get("necessary_exception_multiplier", 1.0)),
            "expert_error": float(preset.get("expert_error_multiplier", 0.1)),
            "uncertain": float(preset.get("uncertain_multiplier", 0.5)),
        }
        for index, record in enumerate(scenario_records):
            meta = type_metadata.get(record["scenario_id"], {})
            eligible = str(meta.get("type_routing_eligible", "")).strip().lower() in {"1", "true", "yes"}
            demo_type = str(meta.get("demonstration_type", "normal")).strip().lower()
            if not eligible:
                demo_type = "normal"
            all_weights[index] *= multipliers.get(demo_type, multipliers["uncertain"])
            record["demonstration_type"] = demo_type
            record["type_routing_eligible"] = eligible

    if type_mode == "enabled" or near_duplicate_group_weighting:
        all_weights = apply_near_duplicate_group_inverse_weighting(
            all_weights, scenario_records
        )

    if type_mode == "enabled":
        absolute_caps = {
            "necessary_exception": float(preset.get("necessary_exception_absolute_cap", 1.0)),
            "expert_error": float(preset.get("expert_error_absolute_cap", 1.0)),
            "uncertain": float(preset.get("uncertain_absolute_cap", 0.05)),
        }
        for demo_type, cap in absolute_caps.items():
            cap = min(1.0, max(0.0, cap))
            indices = [
                index
                for index, record in enumerate(scenario_records)
                if record.get("demonstration_type") == demo_type
            ]
            type_mass = sum(all_weights[index] for index in indices)
            total_mass = sum(all_weights)
            if indices and total_mass > 0 and type_mass / total_mass > cap:
                other_mass = total_mass - type_mass
                target_mass = (cap / max(1e-9, 1.0 - cap)) * other_mass if cap < 1.0 else type_mass
                scale = target_mass / type_mass if type_mass > 0 else 0.0
                for index in indices:
                    all_weights[index] *= scale

        # Normalize after orthogonal type weighting. Absolute caps are enforced
        # conservatively by clipping per-scene weights relative to normal mass.
        weight_sum = sum(all_weights)
        if weight_sum <= 0:
            raise ValueError("Demonstration-type policy produced zero sampling mass")
        all_weights = [value / weight_sum for value in all_weights]

    num_samples = sum(len(d) for d in scenario_datasets)
    if type_mode == "enabled":
        type_draw_caps = {
            demonstration_type: int(math.floor(num_samples * min(1.0, max(0.0, cap))))
            for demonstration_type, cap in absolute_caps.items()
        }
    if max_repeat_per_near_duplicate_group > 0:
        sampler = ExposureCappedWeightedSampler(
            weights=all_weights,
            scenario_records=scenario_records,
            num_samples=num_samples,
            max_repeat_per_scenario=max_repeat_per_scenario,
            max_repeat_per_group=max_repeat_per_near_duplicate_group,
            random_seed=random_seed,
            cumulative_exposure_state_path=cumulative_exposure_state_path,
            max_cumulative_exposure_per_scenario=max_cumulative_exposure_per_scenario,
            max_cumulative_exposure_per_group=max_cumulative_exposure_per_near_duplicate_group,
            sampling_log_path=sampling_log_path,
            score_method=score_method,
            phase_name=phase_name,
            phase_start_epoch=phase_start_epoch,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            max_exposure_per_demonstration_type=type_draw_caps,
            pacing_schedule=pacing_schedule,
            split_names=split_names,
            base_target_proportions=normalized_weights,
        )
        return DistributedSamplerWrapper(sampler)
    if (
        cumulative_exposure_state_path
        or max_cumulative_exposure_per_scenario > 0
        or max_cumulative_exposure_per_near_duplicate_group > 0
    ):
        raise ValueError(
            "Cumulative exposure controls require max_repeat_per_near_duplicate_group > 0"
        )
    sampled_indices: Optional[List[int]] = None
    if max_repeat_per_scenario > 0:
        sampled_indices = _sample_indices_with_repeat_cap(
            all_weights,
            num_samples=num_samples,
            max_repeat_per_scenario=max_repeat_per_scenario,
            random_seed=random_seed,
        )
        sampler = FixedIndexSampler(sampled_indices)
    else:
        sampler = WeightedRandomSampler(
            weights=all_weights,
            num_samples=num_samples,
            replacement=replacement,
        )

    if sampling_log_path:
        split_pool_counts = Counter(record["split"] for record in scenario_records)
        if sampled_indices is not None:
            sampled_records = [scenario_records[index] for index in sampled_indices]
            sampled_split_counts = Counter(record["split"] for record in sampled_records)
            scenario_counts = Counter(record["scenario_id"] for record in sampled_records)
            group_counts = Counter(
                group
                for record in sampled_records
                for group in record["near_duplicate_groups"]
            )
            log_counts = Counter(record["log_name"] for record in sampled_records)
            type_counts = Counter(record.get("demonstration_type", "normal") for record in sampled_records)
            duplicate_count = sum(1 for count in scenario_counts.values() if count > 1)
            max_repeat_count = max(scenario_counts.values()) if scenario_counts else 0
            effective_sample_size = (
                (sum(scenario_counts.values()) ** 2) / sum(count * count for count in scenario_counts.values())
                if scenario_counts
                else 0.0
            )
            scenario_ids = [record["scenario_id"] for record in sampled_records]
        else:
            sampled_split_counts = {}
            group_counts = {}
            log_counts = {}
            type_counts = {}
            effective_sample_size = None
            duplicate_count = None
            max_repeat_count = None
            scenario_ids = []

        log_payload = {
            "score_method": score_method,
            "sampler_mode": sampler_mode,
            "phase_name": phase_name,
            "filter_file_path": filter_file_path,
            "random_seed": random_seed,
            "replacement": replacement,
            "max_repeat_per_scenario": max_repeat_per_scenario,
            "max_repeat_per_near_duplicate_group": max_repeat_per_near_duplicate_group,
            "max_cumulative_exposure_per_scenario": max_cumulative_exposure_per_scenario,
            "max_cumulative_exposure_per_near_duplicate_group": max_cumulative_exposure_per_near_duplicate_group,
            "cumulative_exposure_state_path": cumulative_exposure_state_path,
            "hard_subtype_balance": hard_subtype_balance,
            "curriculum_method": curriculum_method,
            "demonstration_type_mode": type_mode,
            "demonstration_type_metadata_path": demonstration_type_metadata_path,
            "demonstration_type_pool_counts": dict(
                Counter(record.get("demonstration_type", "normal") for record in scenario_records)
            ),
            "unique_log_count": len({record["log_name"] for record in scenario_records}),
            "near_duplicate_group_count": len(
                {
                    group
                    for record in scenario_records
                    for group in record["near_duplicate_groups"]
                }
            ),
            "split_names": split_names,
            "sampling_weights": normalized_weights,
            "split_pool_counts": dict(split_pool_counts),
            "sampled_split_counts": dict(sampled_split_counts),
            "hard_subtype_counts": {},
            "duplicated_scenario_count": duplicate_count,
            "max_repeat_count": max_repeat_count,
            "near_duplicate_group_exposure": dict(group_counts),
            "log_exposure": dict(log_counts),
            "demonstration_type_exposure": dict(type_counts),
            "effective_sample_size": effective_sample_size,
            "scenario_id_list": scenario_ids,
            "actual_sampled_indices_logged": sampled_indices is not None,
        }
        path = Path(sampling_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2, sort_keys=True)
        logger.info("Saved curriculum sampling log to %s", path)
    
    distributed_weighted_sampler = DistributedSamplerWrapper(sampler)
    return distributed_weighted_sampler


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        curriculum_splits: Optional[List[str]] = None,
        curriculum_weights: Optional[List[float]] = None,
        all_scenarios_list: Optional[List[List[AbstractScenario]]] = None,
        curriculum_replacement: bool = True,
        curriculum_max_repeat_per_scenario: int = 0,
        curriculum_random_seed: int = 42,
        curriculum_sampling_log_path: Optional[str] = None,
        curriculum_score_method: str = "",
        curriculum_filter_file_path: str = "",
        hard_subtype_balance: bool = False,
        curriculum_sampler_mode: str = "exposure_capped_weighted",
        curriculum_phase_name: str = "",
        curriculum_phase_start_epoch: int = 0,
        curriculum_method: str = "",
        demonstration_type_mode: str = "observe_only",
        demonstration_type_metadata_path: Optional[str] = None,
        demonstration_type_policy: Optional[Dict[str, Any]] = None,
        curriculum_max_repeat_per_near_duplicate_group: int = 0,
        curriculum_near_duplicate_group_weighting: bool = False,
        curriculum_cumulative_exposure_state_path: Optional[str] = None,
        curriculum_max_cumulative_exposure_per_scenario: int = 0,
        curriculum_max_cumulative_exposure_per_near_duplicate_group: int = 0,
        curriculum_accumulate_grad_batches: int = 1,
        curriculum_pacing_schedule: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        assert train_fraction > 0.0, "Train fraction has to be larger than 0!"
        assert val_fraction > 0.0, "Validation fraction has to be larger than 0!"
        assert test_fraction >= 0.0, "Test fraction has to be larger/equal than 0!"

        # Datasets
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

        # Feature computation
        self._feature_preprocessor = feature_preprocessor

        # Data splitter train/test/val
        self._splitter = splitter

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # Extract all samples
        self._all_samples = all_scenarios
        assert len(self._all_samples) > 0, "No samples were passed to the datamodule"

        # Scenario sampling weights
        self._scenario_type_sampling_weights = scenario_type_sampling_weights

        # Augmentation setup
        self._augmentors = augmentors

        # Worker for multiprocessing to speed up initialization of datasets
        self._worker = worker
        
        # Curriculum learning: multiple splits with weights
        self._curriculum_splits = curriculum_splits
        self._curriculum_weights = curriculum_weights
        self._all_scenarios_list = all_scenarios_list  # List of scenario lists, one per split
        self._curriculum_replacement = curriculum_replacement
        self._curriculum_max_repeat_per_scenario = curriculum_max_repeat_per_scenario
        self._curriculum_random_seed = curriculum_random_seed
        self._curriculum_sampling_log_path = curriculum_sampling_log_path
        self._curriculum_score_method = curriculum_score_method
        self._curriculum_filter_file_path = curriculum_filter_file_path
        self._hard_subtype_balance = hard_subtype_balance
        self._curriculum_sampler_mode = curriculum_sampler_mode
        self._curriculum_phase_name = curriculum_phase_name
        self._curriculum_phase_start_epoch = int(curriculum_phase_start_epoch)
        self._curriculum_method = curriculum_method
        self._demonstration_type_mode = demonstration_type_mode
        self._demonstration_type_metadata_path = demonstration_type_metadata_path
        self._demonstration_type_policy = demonstration_type_policy
        self._curriculum_max_repeat_per_near_duplicate_group = int(
            curriculum_max_repeat_per_near_duplicate_group
        )
        self._curriculum_near_duplicate_group_weighting = bool(
            curriculum_near_duplicate_group_weighting
        )
        self._curriculum_cumulative_exposure_state_path = curriculum_cumulative_exposure_state_path
        self._curriculum_max_cumulative_exposure_per_scenario = int(
            curriculum_max_cumulative_exposure_per_scenario
        )
        self._curriculum_max_cumulative_exposure_per_near_duplicate_group = int(
            curriculum_max_cumulative_exposure_per_near_duplicate_group
        )
        self._curriculum_accumulate_grad_batches = max(
            1, int(curriculum_accumulate_grad_batches)
        )
        self._curriculum_pacing_schedule = dict(
            _plain_config(curriculum_pacing_schedule or {})
        )

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        """Get feature and target builders."""
        return self._feature_preprocessor

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return

        if stage == "fit":
            # Training Dataset
            if self._curriculum_splits is not None and self._all_scenarios_list is not None:
                # Curriculum learning: create separate datasets for each split
                logger.info(f"🔄 Curriculum learning: Creating datasets for {len(self._curriculum_splits)} splits")
                train_datasets = []
                for split_idx, split_scenarios in enumerate(self._all_scenarios_list):
                    train_samples = self._splitter.get_train_samples(
                        split_scenarios, self._worker
                    )
                    assert len(train_samples) > 0, f"Splitter returned no training samples for split {self._curriculum_splits[split_idx]}"
                    
                    split_dataset = create_dataset(
                        train_samples,
                        self._feature_preprocessor,
                        self._train_fraction,
                        f"train_split_{split_idx}",
                        self._augmentors,
                    )
                    train_datasets.append(split_dataset)
                    logger.info(f"  ✓ Split {split_idx} ({self._curriculum_splits[split_idx]}): {len(train_samples)} samples")
                
                # Store datasets and weights for use in train_dataloader
                self._train_datasets = train_datasets
            else:
                # Normal mode: single dataset
                train_samples = self._splitter.get_train_samples(
                    self._all_samples, self._worker
                )
                assert len(train_samples) > 0, "Splitter returned no training samples"

                self._train_set = create_dataset(
                    train_samples,
                    self._feature_preprocessor,
                    self._train_fraction,
                    "train",
                    self._augmentors,
                )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )
        elif stage == "validate":
            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
            )
        elif stage == "test":
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(
                self._all_samples, self._worker
            )
            assert len(test_samples) > 0, "Splitter returned no test samples"

            self._test_set = create_dataset(
                test_samples, self._feature_preprocessor, self._test_fraction, "test"
            )
        else:
            raise ValueError(f'Stage must be one of ["fit", "test"], got ${stage}.')

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after a training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        :raises RuntimeError: If this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        # Check if curriculum learning mode
        if hasattr(self, '_train_datasets') and self._train_datasets is not None:
            # Curriculum learning: use weighted sampler across splits
            logger.info(f"🔄 Curriculum learning: Using sampling weights {self._curriculum_weights}")
            weighted_sampler = distributed_curriculum_sampler_init(
                scenario_datasets=self._train_datasets,
                split_weights=self._curriculum_weights,
                replacement=self._curriculum_replacement,
                split_names=list(self._curriculum_splits),
                max_repeat_per_scenario=int(self._curriculum_max_repeat_per_scenario or 0),
                random_seed=int(self._curriculum_random_seed),
                sampling_log_path=self._curriculum_sampling_log_path,
                score_method=self._curriculum_score_method,
                filter_file_path=self._curriculum_filter_file_path,
                hard_subtype_balance=bool(self._hard_subtype_balance),
                sampler_mode=str(self._curriculum_sampler_mode or "exposure_capped_weighted"),
                phase_name=str(self._curriculum_phase_name or ""),
                phase_start_epoch=self._curriculum_phase_start_epoch,
                curriculum_method=str(self._curriculum_method or ""),
                demonstration_type_mode=str(self._demonstration_type_mode or "observe_only"),
                demonstration_type_metadata_path=self._demonstration_type_metadata_path,
                demonstration_type_policy=self._demonstration_type_policy,
                max_repeat_per_near_duplicate_group=self._curriculum_max_repeat_per_near_duplicate_group,
                near_duplicate_group_weighting=self._curriculum_near_duplicate_group_weighting,
                cumulative_exposure_state_path=self._curriculum_cumulative_exposure_state_path,
                max_cumulative_exposure_per_scenario=self._curriculum_max_cumulative_exposure_per_scenario,
                max_cumulative_exposure_per_near_duplicate_group=self._curriculum_max_cumulative_exposure_per_near_duplicate_group,
                batch_size=int(self._dataloader_params.get("batch_size", 1)),
                accumulate_grad_batches=self._curriculum_accumulate_grad_batches,
                pacing_schedule=self._curriculum_pacing_schedule,
            )
            
            # Combine all datasets into one
            from torch.utils.data import ConcatDataset
            combined_dataset = ConcatDataset(self._train_datasets)
            
            return torch.utils.data.DataLoader(
                dataset=combined_dataset,
                shuffle=False,  # Use sampler instead
                collate_fn=FeatureCollate(),
                sampler=weighted_sampler,
                **self._dataloader_params,
            )
        
        # Normal mode: single dataset
        if self._train_set is None:
            raise DataModuleNotSetupError

        # Initialize weighted sampler
        if self._scenario_type_sampling_weights.enable:
            weighted_sampler = distributed_weighted_sampler_init(
                scenario_dataset=self._train_set,
                scenario_sampling_weights=self._scenario_type_sampling_weights.scenario_type_weights,
            )
        else:
            weighted_sampler = None

        return torch.utils.data.DataLoader(
            dataset=self._train_set,
            shuffle=weighted_sampler is None,
            collate_fn=FeatureCollate(),
            sampler=weighted_sampler,
            **self._dataloader_params,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the validation dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._val_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._val_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the test dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._test_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._test_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    # ! Modified to adapt to newer version of pytorch-lightning
    def transfer_batch_to_device(
        self, batch: Tuple[FeaturesType, ...], device: torch.device, dataloader_idx: int
    ) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :return: Batch in new device.
        """
        return tuple(
            (
                move_features_type_to_device(batch[0], device),
                move_features_type_to_device(batch[1], device),
                batch[2],
            )
        )
