import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
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

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError('Data module has not been setup, call "setup()"')


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
    
    # Create weights for each scenario
    all_weights = []
    for dataset, weight in zip(scenario_datasets, normalized_weights):
        # Each scenario in this split gets the same weight
        num_scenarios = len(dataset._scenarios)
        all_weights.extend([weight] * num_scenarios)
    
    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=all_weights,
        num_samples=sum(len(d) for d in scenario_datasets),
        replacement=replacement,
    )
    
    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
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
