import logging
import os
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import cast

import pytorch_lightning as pl
from hydra.utils import instantiate
from nuplan.planning.script.builders.data_augmentation_builder import (
    build_agent_augmentor,
)
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_metrics_builder import (
    build_training_metrics,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from .custom_datamodule import CustomDataModule

logger = logging.getLogger(__name__)


class LocalTensorBoardLogger(TensorBoardLogger):
    """TensorBoard logger with hparams summary disabled for local protobuf compatibility."""

    def log_hyperparams(self, params, metrics=None) -> None:  # type: ignore[override]
        if os.environ.get("PLUTO_TENSORBOARD_LOG_HPARAMS") == "1":
            super().log_hyperparams(params=params, metrics=metrics)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.cache_path is None:
        logger.warning("Parameter cache_path is not set, caching is disabled")
    else:
        if not str(cfg.cache.cache_path).startswith("s3://"):
            if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
                rmtree(cfg.cache.cache_path)

            Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)
        logger.info("Feature cache enabled: %s", cfg.cache.cache_path)

    if cfg.lightning.trainer.overfitting.enable:
        cfg.data_loader.params.num_workers = 0

    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(
            f"Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config..."
        )
        logger.info("\n" + OmegaConf.to_yaml(cfg))


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""

    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = (
        build_agent_augmentor(cfg.data_augmentation)
        if "data_augmentation" in cfg
        else None
    )

    # Check if this is a curriculum learning stage with multiple splits
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_splits = curriculum_cfg.get("splits", None)
    curriculum_weights = curriculum_cfg.get("sampling_weights", None)
    all_scenarios_list = None
    
    if curriculum_splits is not None and curriculum_weights is not None:
        # Curriculum learning: load multiple splits and combine with weights
        logger.info(f"🔄 Curriculum learning mode: Loading {len(curriculum_splits)} splits with weights {curriculum_weights}")
        
        all_scenarios_list = []
        for split_name in curriculum_splits:
            # Create a temporary config with this split's scenario_filter
            temp_cfg = OmegaConf.create(cfg)
            OmegaConf.set_struct(temp_cfg, False)
            
            # Load scenario_filter config file directly
            import yaml
            # Try to find config directory
            config_dir = None
            
            # First, try to get from hydra runtime config_dir
            if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'runtime') and hasattr(cfg.hydra.runtime, 'config_dir'):
                config_dir = Path(cfg.hydra.runtime.config_dir)
                logger.info(f"  Using config_dir from hydra.runtime: {config_dir}")
            
            # Second, try to extract from searchpath
            if config_dir is None or not config_dir.exists():
                if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'searchpath'):
                    for searchpath in cfg.hydra.searchpath:
                        if 'file://' in str(searchpath):
                            candidate_dir = Path(str(searchpath).replace('file://', ''))
                            if candidate_dir.exists():
                                config_dir = candidate_dir
                                logger.info(f"  Using config_dir from searchpath: {config_dir}")
                                break
            
            # Third, try relative path from current working directory
            if config_dir is None or not config_dir.exists():
                candidate_dir = Path("./config")
                if candidate_dir.exists():
                    config_dir = candidate_dir.resolve()
                    logger.info(f"  Using relative config_dir: {config_dir}")
            
            # Fourth, try absolute path based on script location
            if config_dir is None or not config_dir.exists():
                # Try to find pluto/config directory
                script_dir = Path(__file__).parent.parent.parent  # Go up from src/custom_training to pluto
                candidate_dir = script_dir / "config"
                if candidate_dir.exists():
                    config_dir = candidate_dir
                    logger.info(f"  Using script-based config_dir: {config_dir}")
            
            if config_dir is None or not config_dir.exists():
                raise FileNotFoundError(
                    f"Could not find config directory. Tried:\n"
                    f"  - hydra.runtime.config_dir: {getattr(getattr(getattr(cfg, 'hydra', None), 'runtime', None), 'config_dir', None)}\n"
                    f"  - searchpath: {getattr(getattr(cfg, 'hydra', None), 'searchpath', None)}\n"
                    f"  - ./config (relative)\n"
                    f"  - {Path(__file__).parent.parent.parent / 'config'}"
                )
            
            filter_config_path = config_dir / "scenario_filter" / f"{split_name}.yaml"
            
            if not filter_config_path.exists():
                raise FileNotFoundError(
                    f"Could not find scenario_filter config for {split_name} at {filter_config_path}. "
                    f"Config directory: {config_dir} (exists: {config_dir.exists()})"
                )
            
            logger.info(f"  Loading scenario_filter from: {filter_config_path}")
            
            # Load the scenario_filter config
            with open(filter_config_path, 'r') as f:
                filter_config = yaml.safe_load(f)
            
            # Convert to OmegaConf DictConfig
            temp_cfg.scenario_filter = OmegaConf.create(filter_config)
            OmegaConf.set_struct(temp_cfg, True)
            
            # Load scenarios for this split
            split_scenarios = build_scenarios(temp_cfg, worker, model)
            all_scenarios_list.append(split_scenarios)
            logger.info(f"  ✓ Loaded {len(split_scenarios)} scenarios from split: {split_name}")
        
        # Combine all scenarios (will be weighted in datamodule)
        scenarios = []
        for split_scenarios in all_scenarios_list:
            scenarios.extend(split_scenarios)
        
        logger.info(f"  ✓ Total scenarios: {len(scenarios)}")
    else:
        # Normal mode: single split
        scenarios = build_scenarios(cfg, worker, model)

    # Create datamodule
    datamodule: pl.LightningDataModule = CustomDataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        curriculum_splits=curriculum_splits,
        curriculum_weights=curriculum_weights,
        all_scenarios_list=all_scenarios_list,
        curriculum_replacement=bool(curriculum_cfg.get("replacement", True)),
        curriculum_max_repeat_per_scenario=int(curriculum_cfg.get("max_repeat_per_scenario", 0)),
        curriculum_random_seed=int(curriculum_cfg.get("random_seed", cfg.get("seed", 42))),
        curriculum_sampling_log_path=curriculum_cfg.get("sampling_log_path", None),
        curriculum_score_method=str(curriculum_cfg.get("score_method", "")),
        curriculum_filter_file_path=str(curriculum_cfg.get("filter_file_path", "")),
        hard_subtype_balance=bool(curriculum_cfg.get("hard_subtype_balance", False)),
        curriculum_sampler_mode=str(curriculum_cfg.get("sampler_mode", "legacy_weighted")),
        curriculum_phase_name=str(curriculum_cfg.get("phase_name", "")),
        curriculum_phase_start_epoch=int(curriculum_cfg.get("phase_start_epoch", 0)),
        curriculum_method=str(curriculum_cfg.get("method", "")),
        demonstration_type_mode=str(curriculum_cfg.get("demonstration_type_mode", "observe_only")),
        demonstration_type_metadata_path=curriculum_cfg.get("demonstration_type_metadata_path", None),
        demonstration_type_policy=curriculum_cfg.get("demonstration_type_policy", None),
        curriculum_max_repeat_per_near_duplicate_group=int(
            curriculum_cfg.get("max_repeat_per_near_duplicate_group", 0)
        ),
        curriculum_cumulative_exposure_state_path=curriculum_cfg.get(
            "cumulative_exposure_state_path", None
        ),
        curriculum_max_cumulative_exposure_per_scenario=int(
            curriculum_cfg.get("max_cumulative_exposure_per_scenario", 0)
        ),
        curriculum_max_cumulative_exposure_per_near_duplicate_group=int(
            curriculum_cfg.get("max_cumulative_exposure_per_near_duplicate_group", 0)
        ),
        curriculum_accumulate_grad_batches=int(
            cfg.lightning.trainer.params.get("accumulate_grad_batches", 1)
        ),
        curriculum_pacing_schedule=curriculum_cfg.get("pacing_schedule", None),
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(
    cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper
) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Create the complete Module
    if "custom_trainer" in cfg:
        model = instantiate(
            cfg.custom_trainer,
            model=torch_module_wrapper,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
        )
    else:
        objectives = build_objectives(cfg)
        metrics = build_training_metrics(cfg)
        model = LightningModuleWrapper(
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
            if "warm_up_lr_scheduler" in cfg
            else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )

    return cast(pl.LightningModule, model)


def build_custom_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    # callbacks = build_callbacks(cfg)
    # Import NaN protection callback
    from src.models.pluto.nan_protection import NaNProtectionCallback
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            filename="{epoch}-{val_minFDE:.3f}",
            monitor=cfg.lightning.trainer.checkpoint.monitor,
            mode=cfg.lightning.trainer.checkpoint.mode,
            save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        NaNProtectionCallback(check_frequency=1),  # Check every step for NaN
    ]

    if cfg.wandb.mode in ["disable", "disabled", "offline"]:
        training_logger = LocalTensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        )
    else:
        if cfg.wandb.artifact is not None:
            os.system(f"wandb artifact get {cfg.wandb.artifact}")
            _, _, artifact = cfg.wandb.artifact.split("/")
            checkpoint = os.path.join(os.getcwd(), f"artifacts/{artifact}/model.ckpt")
            run_id = artifact.split(":")[0][-8:]
            cfg.checkpoint = checkpoint
            cfg.wandb.run_id = run_id

        training_logger = WandbLogger(
            save_dir=cfg.group,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            resume=cfg.checkpoint is not None,
            id=cfg.wandb.run_id,
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=training_logger,
        **params,
    )

    return trainer


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info("Building training engine...")

    trainer = build_custom_trainer(cfg)

    # Create model
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    # Build lightning module
    model = build_lightning_module(cfg, torch_module_wrapper)

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine
