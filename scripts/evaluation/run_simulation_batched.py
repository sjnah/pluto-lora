#!/usr/bin/env python3
"""
Run simulation in batches to avoid OOM (Out Of Memory) errors.
This script automatically splits scenarios into smaller batches and processes them sequentially.
"""

import sys
import os
import json
import argparse
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add pluto directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
NUPLAN_DEVKIT_ROOT = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", WORKSPACE_ROOT / "nuplan-devkit"))


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


if "NUPLAN_RUNTIME_ROOT" in os.environ:
    NUPLAN_RUNTIME_ROOT = Path(os.environ["NUPLAN_RUNTIME_ROOT"])
elif path_exists(Path("/root/vessl-nuplan")):
    NUPLAN_RUNTIME_ROOT = Path("/root/vessl-nuplan")
else:
    NUPLAN_RUNTIME_ROOT = NUPLAN_DEVKIT_ROOT / "nuplan"


def resolve_exp_root(runtime_root: Path) -> Path:
    explicit = os.environ.get("NUPLAN_EXP_ROOT")
    preserve_explicit = os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1"
    candidate = Path(explicit) if explicit and preserve_explicit else runtime_root / "exp"

    if preserve_explicit and explicit:
        return candidate

    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".write_test_{os.getpid()}"
        probe.write_text("")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError as error:
        if preserve_explicit and explicit:
            raise RuntimeError(
                f"Explicit NUPLAN_EXP_ROOT is not writable: {candidate}. "
                "Fix the mount permissions or choose a writable shared exp root."
            ) from error

    fallback = Path(os.environ.get("NUPLAN_FALLBACK_EXP_ROOT", WORKSPACE_ROOT / "nuplan-exp"))
    print(f"Warning: {candidate} is not writable; falling back to {fallback}", file=sys.stderr)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


NUPLAN_EXP_ROOT = resolve_exp_root(NUPLAN_RUNTIME_ROOT)
EXP_ROOT = NUPLAN_EXP_ROOT / "exp"
MANIFEST_DIR = REPO_ROOT / "artifacts" / "records" / "batched_runs"
WANDB_SHIM_ROOT = SCRIPT_DIR / "wandb_disabled_shim"
EXTRA_PYTHONPATHS = [
    WANDB_SHIM_ROOT,
    REPO_ROOT,
    NUPLAN_DEVKIT_ROOT,
    WORKSPACE_ROOT / "interPlan",
]


def configure_pythonpath() -> None:
    """Expose sibling checkout packages used by Hydra config search paths."""
    existing_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    extra_paths = list(EXTRA_PYTHONPATHS)
    if os.environ.get("PLUTO_EVAL_ALLOW_WANDB") == "1":
        extra_paths = [path for path in extra_paths if path != WANDB_SHIM_ROOT]

    prepend_paths = [str(path) for path in extra_paths if path.exists()]

    for path in reversed(prepend_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    merged_paths = prepend_paths + [path for path in existing_paths if path]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(merged_paths))

    os.environ.setdefault("NUPLAN_DEVKIT_ROOT", str(NUPLAN_DEVKIT_ROOT))
    os.environ.setdefault("NUPLAN_RUNTIME_ROOT", str(NUPLAN_RUNTIME_ROOT))
    if os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1":
        os.environ.setdefault("NUPLAN_DATA_ROOT", str(NUPLAN_RUNTIME_ROOT / "database"))
        os.environ.setdefault("NUPLAN_MAPS_ROOT", str(Path(os.environ["NUPLAN_DATA_ROOT"]) / "maps"))
        os.environ.setdefault("NUPLAN_EXP_ROOT", str(NUPLAN_RUNTIME_ROOT / "exp"))
    else:
        os.environ["NUPLAN_DATA_ROOT"] = str(NUPLAN_RUNTIME_ROOT / "database")
        os.environ["NUPLAN_MAPS_ROOT"] = str(Path(os.environ["NUPLAN_DATA_ROOT"]) / "maps")
        os.environ["NUPLAN_EXP_ROOT"] = str(NUPLAN_EXP_ROOT)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    if not Path(os.environ["NUPLAN_DATA_ROOT"]).exists():
        raise FileNotFoundError(
            "NUPLAN_DATA_ROOT does not exist: "
            f"{os.environ['NUPLAN_DATA_ROOT']}. "
            f"Resolved NUPLAN_DEVKIT_ROOT={os.environ['NUPLAN_DEVKIT_ROOT']}, "
            f"NUPLAN_RUNTIME_ROOT={os.environ['NUPLAN_RUNTIME_ROOT']}. "
            "Set NUPLAN_RUNTIME_ROOT=/root/vessl-nuplan or source the current .env.server."
        )


configure_pythonpath()

def extract_scenario_tokens(filter_name: str, scenario_builder: Optional[str] = None, 
                           limit: Optional[int] = None) -> List[str]:
    """
    Extract scenario tokens from a scenario filter without running simulation.
    
    Args:
        filter_name: Name of the scenario filter config
        scenario_builder: Optional scenario builder name
        limit: Optional limit on number of scenarios
    
    Returns:
        List of scenario tokens
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    # Set up config directory
    config_dir = REPO_ROOT / "config"
    nuplan_simulation_config = NUPLAN_DEVKIT_ROOT / "nuplan" / "planning" / "script" / "config" / "simulation"
    
    print(f"🔍 Extracting scenario tokens from filter: {filter_name}")
    
    # Initialize hydra (without version_base for older Hydra versions)
    with initialize_config_dir(config_dir=str(config_dir)):
        # Compose config with proper Hydra override format
        overrides = [
            "hydra.searchpath="
            "[pkg://nuplan.planning.script.config.common,"
            "pkg://nuplan.planning.script.config.simulation,"
            "pkg://nuplan.planning.script.experiments,"
            f"file://{nuplan_simulation_config},"
            f"file://{config_dir / 'scenario_filter'},"
            f"file://{config_dir / 'scenario_builder'}]",
            f"scenario_filter={filter_name}",
        ]
        if scenario_builder:
            overrides.append(f"scenario_builder={scenario_builder}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"provider=hydra\.searchpath in main, path=config/simulation is not available\.",
                category=UserWarning,
            )
            cfg = compose(config_name="default_simulation", overrides=overrides)
        
        # Override limit if specified
        if limit is not None:
            cfg.scenario_filter.limit_total_scenarios = limit
        
        # Build scenario builder and load scenarios directly (no model needed)
        from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
        from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
        
        print(f"   Loading scenarios (this may take a moment)...")
        scenario_builder = build_scenario_builder(cfg)
        scenario_filter = build_scenario_filter(cfg.scenario_filter)
        scenarios = scenario_builder.get_scenarios(scenario_filter, Sequential())
        
        # Extract tokens
        tokens = [s.token for s in scenarios]
        
        print(f"✅ Extracted {len(tokens)} scenario tokens")
        return tokens


def split_into_batches(items: List, batch_size: int) -> List[List]:
    """Split a list into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def strip_msgpack_token(path: Path) -> Optional[str]:
    """Extract a scenario token from a nuPlan simulation log filename."""
    suffix = ".msgpack.xz"
    name = path.name
    if not name.endswith(suffix):
        return None
    token = name[: -len(suffix)]
    return token or None


def completed_tokens_from_simulation_log(experiment_name: str) -> Set[str]:
    """Return tokens with completed simulation logs for one experiment directory."""
    simulation_log_dir = EXP_ROOT / experiment_name / "simulation_log"
    if not simulation_log_dir.exists():
        return set()

    tokens: Set[str] = set()
    for path in simulation_log_dir.rglob("*.msgpack.xz"):
        token = strip_msgpack_token(path)
        if token:
            tokens.add(token)
    return tokens


def completed_tokens_from_metrics(experiment_name: str) -> Set[str]:
    """Return tokens present in any metric parquet for one experiment directory."""
    metrics_dir = EXP_ROOT / experiment_name / "metrics"
    if not metrics_dir.exists():
        return set()

    tokens: Set[str] = set()
    try:
        import pandas as pd

        preferred_files = [
            metrics_dir / "no_ego_at_fault_collisions.parquet",
            metrics_dir / "drivable_area_compliance.parquet",
            metrics_dir / "ego_is_making_progress.parquet",
        ]
        candidate_files = [path for path in preferred_files if path.exists()]
        if not candidate_files:
            candidate_files = sorted(metrics_dir.glob("*.parquet"))

        for metric_file in candidate_files:
            df = pd.read_parquet(metric_file, columns=["scenario_name"])
            tokens.update(str(token) for token in df["scenario_name"].dropna().unique())
            if tokens:
                break
    except Exception as error:
        print(f"⚠️  Could not read completed tokens from metrics for {experiment_name}: {error}")
        return set()

    return tokens


def completed_tokens_from_existing_outputs(experiment_name: str) -> Set[str]:
    """Return completed tokens from metrics first, then simulation logs."""
    tokens = completed_tokens_from_metrics(experiment_name)
    if tokens:
        return tokens
    return completed_tokens_from_simulation_log(experiment_name)


def metrics_dir_has_parquet(experiment_name: str) -> bool:
    """Check whether an experiment has metric parquet outputs to aggregate."""
    metrics_dir = EXP_ROOT / experiment_name / "metrics"
    return metrics_dir.exists() and any(metrics_dir.glob("*.parquet"))


def load_manifest(experiment_name: str) -> Dict[str, Any]:
    manifest_path = MANIFEST_DIR / f"{experiment_name}.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_resume_index(experiment_name: str, base_experiment: str) -> Optional[int]:
    prefix = f"{base_experiment}_resume"
    if not experiment_name.startswith(prefix):
        return None
    remainder = experiment_name[len(prefix):]
    index_text = remainder.split("_batch", 1)[0]
    try:
        return int(index_text)
    except ValueError:
        return None


def batch_sort_key(experiment_name: str, base_experiment: str) -> tuple[int, int, str]:
    resume_idx = parse_resume_index(experiment_name, base_experiment)
    if resume_idx is None:
        resume_order = -1
    else:
        resume_order = resume_idx

    batch_idx = -1
    if "_batch" in experiment_name:
        try:
            batch_idx = int(experiment_name.rsplit("_batch", 1)[1])
        except ValueError:
            batch_idx = -1
    return (resume_order, batch_idx, experiment_name)


def discover_candidate_experiments(base_experiment: str, manifest: Dict[str, Any]) -> List[str]:
    """Find existing batch/resume experiment directories for the same logical run."""
    candidates: Set[str] = set()

    for batch in manifest.get("batches", []):
        experiment = batch.get("experiment")
        if isinstance(experiment, str):
            candidates.add(experiment)

    for path in EXP_ROOT.glob(f"{base_experiment}_batch*"):
        if path.is_dir():
            candidates.add(path.name)
    for path in EXP_ROOT.glob(f"{base_experiment}_resume*_batch*"):
        if path.is_dir():
            candidates.add(path.name)

    return sorted(candidates, key=lambda name: batch_sort_key(name, base_experiment))


def reusable_existing_batches(base_experiment: str, target_tokens: List[str]) -> tuple[List[Dict[str, Any]], Set[str]]:
    """Build manifest entries for completed existing batches that are safe to reuse."""
    target_set = {str(token) for token in target_tokens}
    manifest = load_manifest(base_experiment)
    entries: List[Dict[str, Any]] = []
    completed_tokens: Set[str] = set()

    for experiment_name in discover_candidate_experiments(base_experiment, manifest):
        tokens = completed_tokens_from_existing_outputs(experiment_name)
        if not tokens:
            continue
        if not metrics_dir_has_parquet(experiment_name):
            print(f"⚠️  Ignoring {experiment_name}: simulation logs exist but metrics parquet files are missing")
            continue
        extra_tokens = tokens - target_set
        if extra_tokens:
            print(
                f"⚠️  Ignoring {experiment_name}: {len(extra_tokens)} completed token(s) are outside "
                "the current target set"
            )
            continue

        new_tokens = tokens - completed_tokens
        if not new_tokens:
            continue

        completed_tokens.update(new_tokens)
        entries.append(
            {
                "batch_idx": None,
                "batch_size": len(tokens),
                "experiment": experiment_name,
                "metrics_dir": str(EXP_ROOT / experiment_name / "metrics"),
                "success": True,
                "tokens": sorted(tokens),
                "source": "existing_simulation_log",
            }
        )

    return entries, completed_tokens


def next_resume_index(base_experiment: str) -> int:
    """Return the next resume index, scanning directories and the manifest."""
    manifest = load_manifest(base_experiment)
    indices = []
    for experiment_name in discover_candidate_experiments(base_experiment, manifest):
        resume_idx = parse_resume_index(experiment_name, base_experiment)
        if resume_idx is not None:
            indices.append(resume_idx)
    return max(indices, default=0) + 1


def batch_filter_name_for_experiment(experiment_name: str) -> str:
    """Use a stable, experiment-specific temporary filter path."""
    return f".batch_filters/{experiment_name}"


def create_batch_filter_config(batch_tokens: List[str], output_path: Path) -> None:
    """Create a temporary scenario filter config file for a batch of tokens."""
    # Ensure all tokens are strings
    batch_tokens_str = [str(token) for token in batch_tokens]
    
    # Write YAML file manually to ensure all tokens are quoted as strings
    # This prevents YAML from parsing them as numbers
    with open(output_path, 'w') as f:
        f.write("_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter\n")
        f.write("_convert_: all\n")
        f.write("scenario_types: null\n")
        f.write("scenario_tokens:\n")
        for token in batch_tokens_str:
            # Always quote tokens as strings to prevent YAML from parsing as numbers
            f.write(f"  - '{token}'\n")
        f.write("log_names: null\n")
        f.write("map_names: null\n")
        f.write("num_scenarios_per_type: null\n")
        f.write("limit_total_scenarios: null\n")
        f.write("timestamp_threshold_s: null\n")
        f.write("ego_displacement_minimum_m: null\n")
        f.write("expand_scenarios: false\n")
        f.write("remove_invalid_goals: false\n")
        f.write("shuffle: false\n")


def write_manifest(
    experiment: str,
    filter_name: str,
    scenario_builder: Optional[str],
    limit: Optional[int],
    batch_size: int,
    target_tokens: List[str],
    completed_tokens: Set[str],
    batch_manifest: List[Dict[str, Any]],
    successful_batches: int,
    failed_batches: int,
    resume_enabled: bool,
) -> None:
    """Write the canonical manifest for one logical batched experiment."""
    target_set = {str(token) for token in target_tokens}
    completed_target_tokens = sorted(target_set & {str(token) for token in completed_tokens})
    remaining_tokens = sorted(target_set - set(completed_target_tokens))

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{experiment}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "filter": filter_name,
                "scenario_builder": scenario_builder,
                "limit": limit,
                "batch_size": batch_size,
                "resume_enabled": resume_enabled,
                "total_scenarios": len(target_tokens),
                "target_scenarios": len(target_tokens),
                "completed_scenarios": len(completed_target_tokens),
                "remaining_scenarios": len(remaining_tokens),
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "target_tokens": [str(token) for token in target_tokens],
                "completed_tokens": completed_target_tokens,
                "remaining_tokens": remaining_tokens,
                "batches": batch_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"📝 Wrote batch manifest: {manifest_path}")


def hydra_override_value(cmd: List[str], key: str) -> Optional[str]:
    """Return the value for a Hydra override in the simulation command."""
    prefix = f"{key}="
    for arg in cmd:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def log_worker_plan(cmd: List[str], batch_size: int) -> None:
    """Print the expected scenario-level worker concurrency for this batch."""
    worker = hydra_override_value(cmd, "worker") or "default"
    max_workers = hydra_override_value(cmd, "worker.max_workers")
    threads_per_node = hydra_override_value(cmd, "worker.threads_per_node")
    num_gpus = hydra_override_value(cmd, "number_of_gpus_allocated_per_simulation")
    num_cpus = hydra_override_value(cmd, "number_of_cpus_allocated_per_simulation")

    if worker == "sequential":
        concurrency = 1
        detail = "sequential"
    elif worker == "ray_distributed":
        if threads_per_node in (None, "", "null", "None"):
            concurrency = os.cpu_count() or 1
            detail = "all logical CPUs via Ray"
        else:
            try:
                concurrency = int(threads_per_node)
                detail = f"worker.threads_per_node={threads_per_node}"
            except ValueError:
                concurrency = None
                detail = f"worker.threads_per_node={threads_per_node}"
    elif worker == "single_machine_thread_pool":
        if max_workers in (None, "", "null", "None"):
            concurrency = os.cpu_count() or 1
            detail = "all logical CPUs"
        else:
            try:
                concurrency = int(max_workers)
                detail = f"worker.max_workers={max_workers}"
            except ValueError:
                concurrency = None
                detail = f"worker.max_workers={max_workers}"
    else:
        concurrency = None
        detail = f"worker.max_workers={max_workers}" if max_workers is not None else "configured by Hydra defaults"

    print("🧵 Worker plan:")
    print(f"   worker: {worker}")
    print(f"   batch scenarios: {batch_size}")
    if concurrency is None:
        print(f"   scenario concurrency: unknown ({detail})")
    else:
        print(f"   scenario concurrency: up to {min(concurrency, batch_size)} simultaneous scenario(s) ({detail})")
    if num_gpus is not None or num_cpus is not None:
        print(f"   per-simulation resources: num_gpus={num_gpus or 'config default'}, num_cpus={num_cpus or 'config default'}")


def run_simulation_batch(
    batch_idx: int,
    total_batches: int,
    batch_tokens: List[str],
    experiment_name: str,
    ckpt_path: Optional[str],
    scenario_builder: Optional[str],
    script_dir: Path,
    batch_filter_name: str,
    planner: str,
    planner_overrides: Optional[List[str]],
    worker: str,
    worker_threads: Optional[int],
    worker_max_workers: Optional[int],
    ray_log_to_driver: Optional[str],
    num_gpus: Optional[str],
    num_cpus: Optional[str],
    disable_simulation_log: bool,
    simulation_verbose: bool,
    quiet: bool,
) -> bool:
    """Run simulation for a single batch."""
    print(f"\n{'='*70}")
    print(f"📦 Batch {batch_idx + 1}/{total_batches} ({len(batch_tokens)} scenarios)")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        'python', '-X', 'faulthandler',
        str(script_dir / 'run_simulation.py'),
        '+simulation=closed_loop_nonreactive_agents',
        'observation=box_observation',
        'ego_controller=two_stage_controller',
        f'planner={planner}',
        f'scenario_filter={batch_filter_name}',
        f'experiment={experiment_name}',
        f'worker={worker}',
        f'verbose={str(simulation_verbose).lower()}',
    ]

    if planner == "pluto_planner" and ckpt_path is not None:
        cmd.append(f'+planner.pluto_planner.planner_ckpt={ckpt_path}')
    if num_gpus is not None:
        cmd.append(f'number_of_gpus_allocated_per_simulation={num_gpus}')
    if num_cpus is not None:
        cmd.append(f'number_of_cpus_allocated_per_simulation={num_cpus}')
    if worker == "ray_distributed":
        if worker_threads is not None:
            cmd.append(f'worker.threads_per_node={worker_threads}')
        if ray_log_to_driver is not None:
            cmd.append(f'worker.log_to_driver={ray_log_to_driver}')
    elif worker == "single_machine_thread_pool" and worker_max_workers is not None:
        cmd.append(f'worker.max_workers={worker_max_workers}')
    if disable_simulation_log:
        cmd.append('callback=no_simulation_log')
    if quiet:
        cmd.extend([
            'logger_level=warning',
            'enable_simulation_progress_bar=false',
        ])
    if planner_overrides:
        cmd.extend(planner_overrides)
    
    if scenario_builder:
        cmd.append(f'scenario_builder={scenario_builder}')

    log_worker_plan(cmd, len(batch_tokens))
    
    # Run simulation
    try:
        result = subprocess.run(cmd, cwd=str(script_dir), check=True)
        print(f"✅ Batch {batch_idx + 1}/{total_batches} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch {batch_idx + 1}/{total_batches} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation in batches to avoid OOM errors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/evaluation/run_simulation_batched.py \\
    --filter val14_benchmark \\
    --ckpt checkpoints/pluto_1M_aux_cil.ckpt \\
    --experiment quick_test_zeroshot_val14_benchmark \\
    --batch-size 200 \\
    --scenario-builder nuplan_v1_1_val
        """
    )
    
    parser.add_argument('--filter', required=True, help='Scenario filter config name')
    parser.add_argument('--ckpt', help='Path to checkpoint file; required for pluto_planner')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--planner', default='pluto_planner', help='Hydra planner config name (default: pluto_planner)')
    parser.add_argument(
        '--planner-override',
        action='append',
        default=[],
        help='Additional Hydra override for planner setup; can be repeated',
    )
    parser.add_argument(
        '--disable-simulation-log',
        action='store_true',
        help='Disable nuPlan simulation_log_callback and avoid writing .msgpack.xz logs',
    )
    parser.add_argument(
        '--simulation-verbose',
        choices=['true', 'false'],
        default='true',
        help='Set nuPlan verbose flag for simulation execution (default: true)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce console/progress logging while keeping metric outputs',
    )
    parser.add_argument('--batch-size', type=int, default=200, 
                       help='Number of scenarios per batch (default: 200)')
    parser.add_argument(
        '--worker',
        default='sequential',
        choices=['sequential', 'single_machine_thread_pool', 'ray_distributed'],
        help='nuPlan worker for scenarios inside each batch (default: sequential)',
    )
    parser.add_argument(
        '--worker-threads',
        type=int,
        help='Ray threads_per_node. Omit to let Ray use all logical CPUs.',
    )
    parser.add_argument(
        '--worker-max-workers',
        type=int,
        help='single_machine_thread_pool max_workers.',
    )
    parser.add_argument(
        '--ray-log-to-driver',
        choices=['true', 'false'],
        help='Ray worker.log_to_driver override.',
    )
    parser.add_argument(
        '--num-gpus',
        help='Hydra number_of_gpus_allocated_per_simulation override.',
    )
    parser.add_argument(
        '--num-cpus',
        help='Hydra number_of_cpus_allocated_per_simulation override.',
    )
    parser.add_argument('--limit', type=int, help='Total limit on scenarios')
    parser.add_argument('--scenario-builder', help='Scenario builder name')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only extract tokens, do not run simulation')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not skip scenarios that already have reusable simulation logs and metrics')
    
    args = parser.parse_args()
    
    ckpt_path: Optional[Path] = None
    if args.ckpt:
        if args.planner != "pluto_planner":
            print("❌ Error: --ckpt is only wired for planner=pluto_planner. Use --planner-override for custom planner checkpoint fields.")
            sys.exit(1)
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt_path = ckpt_path.resolve()

        if not ckpt_path.exists():
            print(f"❌ Error: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
    elif args.planner == "pluto_planner":
        print("❌ Error: --ckpt is required when planner=pluto_planner")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🚀 BATCHED SIMULATION RUNNER")
    print(f"{'='*70}")
    print(f"Filter: {args.filter}")
    print(f"Planner: {args.planner}")
    if ckpt_path is not None:
        print(f"Checkpoint: {ckpt_path}")
    if args.disable_simulation_log:
        print("Simulation logs: disabled (.msgpack.xz will not be written)")
    if args.quiet:
        print("Console logging: quiet")
    if args.planner_override:
        print(f"Planner overrides: {args.planner_override}")
    print(f"Experiment: {args.experiment}")
    print(f"Batch size: {args.batch_size} scenarios")
    print(f"Worker: {args.worker}")
    if args.worker_threads is not None:
        print(f"Ray threads per node: {args.worker_threads}")
    if args.worker_max_workers is not None:
        print(f"Thread-pool max workers: {args.worker_max_workers}")
    if args.num_gpus is not None or args.num_cpus is not None:
        print(f"Per-simulation resources: num_gpus={args.num_gpus or 'config default'}, num_cpus={args.num_cpus or 'config default'}")
    if args.limit:
        print(f"Total limit: {args.limit} scenarios")
    print(f"{'='*70}\n")
    
    # Extract scenario tokens
    try:
        tokens = extract_scenario_tokens(
            filter_name=args.filter,
            scenario_builder=args.scenario_builder,
            limit=args.limit
        )
    except Exception as e:
        print(f"❌ Failed to extract scenario tokens: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if len(tokens) == 0:
        print("❌ No scenarios found!")
        sys.exit(1)
    
    existing_entries: List[Dict[str, Any]] = []
    completed_tokens: Set[str] = set()
    if not args.no_resume:
        existing_entries, completed_tokens = reusable_existing_batches(args.experiment, tokens)

    remaining_tokens = [str(token) for token in tokens if str(token) not in completed_tokens]

    print(f"\n📊 Target scenarios: {len(tokens)}")
    if args.no_resume:
        print("⏭️  Resume skip disabled (--no-resume)")
    else:
        print(f"⏭️  Reusable completed scenarios: {len(completed_tokens)}")
        print(f"▶️  Scenarios remaining to run: {len(remaining_tokens)}")

    if args.dry_run:
        print(f"\n✅ Dry run complete. Would run {len(remaining_tokens)} scenarios.")
        return

    if not remaining_tokens:
        print("\n✅ All target scenarios already have reusable results. Skipping simulation.")
        write_manifest(
            experiment=args.experiment,
            filter_name=args.filter,
            scenario_builder=args.scenario_builder,
            limit=args.limit,
            batch_size=args.batch_size,
            target_tokens=[str(token) for token in tokens],
            completed_tokens=completed_tokens,
            batch_manifest=existing_entries,
            successful_batches=len(existing_entries),
            failed_batches=0,
            resume_enabled=not args.no_resume,
        )
        return
    
    # Split into batches
    batches = split_into_batches(remaining_tokens, args.batch_size)
    num_batches = len(batches)
    
    print(f"📦 Number of batches: {num_batches}")
    print(f"   Batch sizes: {[len(b) for b in batches]}\n")
    
    # Create temporary directory for batch filter configs
    temp_dir = REPO_ROOT / 'config' / 'scenario_filter' / '.batch_filters'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each batch
    successful_batches = 0
    failed_batches = 0
    batch_manifest = list(existing_entries)
    current_run_entries: List[Dict[str, Any]] = []
    resume_idx = next_resume_index(args.experiment) if completed_tokens and not args.no_resume else None
    
    try:
        for batch_idx, batch_tokens in enumerate(batches):
            # Use unique experiment name per batch to avoid overwriting metrics
            if resume_idx is None:
                batch_experiment_name = f"{args.experiment}_batch{batch_idx}"
            else:
                batch_experiment_name = f"{args.experiment}_resume{resume_idx}_batch{batch_idx}"

            batch_filter_name = batch_filter_name_for_experiment(batch_experiment_name)
            batch_filter_path = temp_dir / f"{batch_experiment_name}.yaml"

            # Create batch filter config
            create_batch_filter_config(batch_tokens, batch_filter_path)
            
            # Run simulation for this batch
            success = run_simulation_batch(
                batch_idx=batch_idx,
                total_batches=num_batches,
                batch_tokens=batch_tokens,
                experiment_name=batch_experiment_name,
                ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
                scenario_builder=args.scenario_builder,
                script_dir=REPO_ROOT,
                batch_filter_name=batch_filter_name,
                planner=args.planner,
                planner_overrides=args.planner_override,
                worker=args.worker,
                worker_threads=args.worker_threads,
                worker_max_workers=args.worker_max_workers,
                ray_log_to_driver=args.ray_log_to_driver,
                num_gpus=args.num_gpus,
                num_cpus=args.num_cpus,
                disable_simulation_log=args.disable_simulation_log,
                simulation_verbose=args.simulation_verbose == 'true',
                quiet=args.quiet,
            )
            
            if success:
                successful_batches += 1
            else:
                failed_batches += 1

            entry = {
                "batch_idx": batch_idx,
                "batch_size": len(batch_tokens),
                "experiment": batch_experiment_name,
                "metrics_dir": str(EXP_ROOT / batch_experiment_name / "metrics"),
                "success": success,
                "tokens": [str(token) for token in batch_tokens],
                "source": "current_run",
            }
            batch_manifest.append(entry)
            current_run_entries.append(entry)
            
            # Clean up batch filter (optional - we can keep them for debugging)
            # batch_filter_path.unlink()
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total batches: {num_batches}")
        print(f"Successful: {successful_batches}")
        print(f"Failed: {failed_batches}")
        print(f"Experiment: {args.experiment}")
        print(f"{'='*70}\n")

        current_success_tokens = {
            token
            for entry in current_run_entries
            if entry.get("success")
            for token in entry.get("tokens", [])
        }
        all_completed_tokens = completed_tokens | current_success_tokens
        write_manifest(
            experiment=args.experiment,
            filter_name=args.filter,
            scenario_builder=args.scenario_builder,
            limit=args.limit,
            batch_size=args.batch_size,
            target_tokens=[str(token) for token in tokens],
            completed_tokens=all_completed_tokens,
            batch_manifest=batch_manifest,
            successful_batches=sum(1 for entry in batch_manifest if entry.get("success")),
            failed_batches=failed_batches,
            resume_enabled=not args.no_resume,
        )
        
        if failed_batches > 0:
            print(f"⚠️  Warning: {failed_batches} batch(es) failed!")
            sys.exit(1)
        else:
            print("✅ All batches completed successfully!")
    
    finally:
        # Optional: cleanup temp directory
        # import shutil
        # if temp_dir.exists():
        #     shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    main()
