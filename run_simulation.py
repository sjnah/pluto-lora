import logging
import os
import pprint
import sys
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union


REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent


def resolve_exp_root(runtime_root: Path) -> Path:
    explicit = os.environ.get("NUPLAN_EXP_ROOT")
    candidate = Path(explicit) if explicit else runtime_root / "exp"

    if os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1" and explicit:
        return candidate

    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".write_test_{os.getpid()}"
        probe.write_text("")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError:
        pass

    fallback = Path(os.environ.get("NUPLAN_FALLBACK_EXP_ROOT", WORKSPACE_ROOT / "nuplan-exp"))
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def bootstrap_workspace_paths() -> None:
    """Expose sibling workspace checkouts before importing nuPlan modules."""
    nuplan_devkit_root = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", WORKSPACE_ROOT / "nuplan-devkit"))
    if "NUPLAN_RUNTIME_ROOT" in os.environ:
        nuplan_runtime_root = Path(os.environ["NUPLAN_RUNTIME_ROOT"])
    elif Path("/root/vessl-nuplan").exists():
        nuplan_runtime_root = Path("/root/vessl-nuplan")
    else:
        nuplan_runtime_root = nuplan_devkit_root / "nuplan"
    nuplan_exp_root = resolve_exp_root(nuplan_runtime_root)
    candidate_roots = [
        REPO_ROOT,
        nuplan_devkit_root,
        WORKSPACE_ROOT / "interPlan",
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    prepend_paths = [str(path) for path in candidate_roots if path.exists()]

    for path in reversed(prepend_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    merged_paths = prepend_paths + [path for path in existing_pythonpath if path]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(merged_paths))

    os.environ.setdefault("NUPLAN_DEVKIT_ROOT", str(nuplan_devkit_root))
    os.environ.setdefault("NUPLAN_RUNTIME_ROOT", str(nuplan_runtime_root))
    if os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1":
        os.environ.setdefault("NUPLAN_DATA_ROOT", str(nuplan_runtime_root / "database"))
        os.environ.setdefault("NUPLAN_MAPS_ROOT", str(Path(os.environ["NUPLAN_DATA_ROOT"]) / "maps"))
        os.environ.setdefault("NUPLAN_EXP_ROOT", str(nuplan_runtime_root / "exp"))
    else:
        os.environ["NUPLAN_DATA_ROOT"] = str(nuplan_runtime_root / "database")
        os.environ["NUPLAN_MAPS_ROOT"] = str(Path(os.environ["NUPLAN_DATA_ROOT"]) / "maps")
        os.environ["NUPLAN_EXP_ROOT"] = str(nuplan_exp_root)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    if not Path(os.environ["NUPLAN_DATA_ROOT"]).exists():
        raise FileNotFoundError(
            "NUPLAN_DATA_ROOT does not exist: "
            f"{os.environ['NUPLAN_DATA_ROOT']}. "
            f"Resolved NUPLAN_DEVKIT_ROOT={os.environ['NUPLAN_DEVKIT_ROOT']}, "
            f"NUPLAN_RUNTIME_ROOT={os.environ['NUPLAN_RUNTIME_ROOT']}. "
            "Set NUPLAN_RUNTIME_ROOT=/root/vessl-nuplan or source the current .env.server."
        )


bootstrap_workspace_paths()

import hydra
import pandas as pd
import pytorch_lightning as pl
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.utils import (
    run_runners,
    set_default_path,
    set_up_common_builder,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/simulation")


def print_simulation_results(file=None):
    try:
        if file is not None:
            df = pd.read_parquet(file)
        else:
            root = Path(os.getcwd()) / "aggregator_metric"
            result = list(root.glob("*.parquet"))
            if not result:
                logger.warning("No aggregator metric files found. Metrics may still be processing.")
                return
            result = max(result, key=lambda item: item.stat().st_ctime)
            df = pd.read_parquet(result)
        final_score = df[df["scenario"] == "final_score"]
        if len(final_score) > 0:
            final_score = final_score.to_dict(orient="records")[0]
            pprint.PrettyPrinter(indent=4).pprint(final_score)
        else:
            logger.info("Simulation completed. Final score not yet aggregated.")
    except Exception as e:
        logger.warning(f"Could not print simulation results: {e}")


def run_simulation(
    cfg: DictConfig,
    planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None,
) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = "building_simulation"
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(
        cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool
    )

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and "planner" in cfg.keys():
        logger.info("Using pre-instantiated planner. Ignoring planner in config")
        OmegaConf.set_struct(cfg, False)
        cfg.pop("planner")
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    runners = build_simulations(
        cfg=cfg,
        callbacks=callbacks,
        worker=common_builder.worker,
        pre_built_planners=planners,
        callbacks_worker=callbacks_worker_pool,
    )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info("Running simulation...")
    run_runners(
        runners=runners,
        common_builder=common_builder,
        cfg=cfg,
        profiler_name="running_simulation",
    )
    logger.info("Finished running simulation!")


def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path="./config", config_name="default_simulation")
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert (
        cfg.simulation_log_main_path is None
    ), "Simulation_log_main_path must not be set when running simulation."

    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()

    print_simulation_results()


if __name__ == "__main__":
    main()
