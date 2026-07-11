#!/usr/bin/env python3
"""
Score PLUTO scenarios by frozen-model open-loop imitation loss.

This script does not run closed-loop simulation. It loads a frozen PLUTO
checkpoint, builds the configured nuPlan scenarios, runs a no-grad forward pass,
and writes one loss row per scenario for loss-ranked curriculum generation.
"""

import csv
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def disable_wandb_for_loss_scoring() -> None:
    """Avoid importing broken optional wandb while running local loss scoring."""
    if os.environ.get("PLUTO_TRAINING_ALLOW_WANDB") == "1":
        return

    os.environ.setdefault("WANDB_DISABLED", "true")
    sys.modules.setdefault("wandb", None)


disable_wandb_for_loss_scoring()
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
NUPLAN_DEVKIT_ROOT = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", WORKSPACE_ROOT / "nuplan-devkit"))
# if "NUPLAN_RUNTIME_ROOT" in os.environ:
#     NUPLAN_RUNTIME_ROOT = Path(os.environ["NUPLAN_RUNTIME_ROOT"])
# elif Path("/root/vessl-nuplan").exists():
#     NUPLAN_RUNTIME_ROOT = Path("/root/vessl-nuplan")
# else:
#     NUPLAN_RUNTIME_ROOT = NUPLAN_DEVKIT_ROOT / "nuplan"
NUPLAN_RUNTIME_ROOT = NUPLAN_DEVKIT_ROOT / "nuplan"


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
    except OSError as error:
        if explicit:
            raise RuntimeError(
                f"Explicit NUPLAN_EXP_ROOT is not writable: {candidate}. "
                "Fix the mount permissions or choose a writable shared exp root."
            ) from error

    fallback = Path(os.environ.get("NUPLAN_FALLBACK_EXP_ROOT", WORKSPACE_ROOT / "nuplan-exp"))
    print(f"Warning: {candidate} is not writable; falling back to {fallback}", file=sys.stderr)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def configure_workspace_environment() -> None:
    """Pin nuPlan paths to this workspace before Hydra composes env-based configs."""
    nuplan_exp_root = resolve_exp_root(NUPLAN_RUNTIME_ROOT)
    os.environ.setdefault("NUPLAN_DEVKIT_ROOT", str(NUPLAN_DEVKIT_ROOT))
    os.environ.setdefault("NUPLAN_RUNTIME_ROOT", str(NUPLAN_RUNTIME_ROOT))
    if os.environ.get("NUPLAN_PRESERVE_EXPLICIT_PATHS") == "1":
        os.environ.setdefault("NUPLAN_DATA_ROOT", str(NUPLAN_RUNTIME_ROOT / "database"))
        os.environ.setdefault("NUPLAN_MAPS_ROOT", str(Path(os.environ["NUPLAN_DATA_ROOT"]) / "maps"))
        os.environ.setdefault("NUPLAN_EXP_ROOT", str(NUPLAN_RUNTIME_ROOT / "exp"))
    else:
        os.environ["NUPLAN_DATA_ROOT"] = str(NUPLAN_RUNTIME_ROOT / "database")
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


configure_workspace_environment()

for path in (REPO_ROOT, NUPLAN_DEVKIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd, instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.modeling.types import move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from omegaconf import DictConfig, OmegaConf

from src.models.pluto.loss.esdf_collision_loss import ESDFCollisionLoss

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
set_default_path()

CONFIG_PATH = "../../config"
CONFIG_NAME = "training/train_pluto_lora"


def _original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path.cwd()


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _original_cwd() / candidate


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[6:] if key.startswith("model.") else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def _model_config_without_loss(cfg: DictConfig) -> DictConfig:
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    if isinstance(model_cfg_dict, dict):
        model_cfg_dict.pop("loss", None)
    return OmegaConf.create(model_cfg_dict)


def _loss_config(cfg: DictConfig) -> Dict[str, float]:
    defaults = {
        "weight_reg_loss": 1.0,
        "weight_cls_loss": 1.0,
        "weight_prediction_loss": 1.0,
        "weight_collision_loss": 1.0,
        "weight_ref_free_reg_loss": 1.0,
        "weight_auxiliary": 1.0,
    }
    if "model" in cfg and "loss" in cfg.model:
        configured = OmegaConf.to_container(cfg.model.loss, resolve=True)
        if isinstance(configured, dict):
            defaults.update({k: float(v) for k, v in configured.items() if k in defaults})
    return defaults


def _scenario_error_row(scenario: Any, error: Exception) -> Dict[str, Any]:
    return {
        "scene_id": getattr(scenario, "token", ""),
        "log_name": getattr(scenario, "log_name", ""),
        "scenario_type": getattr(scenario, "scenario_type", ""),
        "status": "feature_error",
        "error": str(error),
    }


def _iter_valid_batches(
    dataset: ScenarioDataset,
    batch_size: int,
) -> Iterable[Tuple[Optional[Tuple[Any, Any, List[Any]]], Optional[Dict[str, Any]]]]:
    collate = FeatureCollate()
    pending = []

    for idx, scenario in enumerate(dataset._scenarios):
        try:
            pending.append(dataset[idx])
        except Exception as exc:  # Keep scoring moving and record the failed token.
            yield None, _scenario_error_row(scenario, exc)
            continue

        if len(pending) == batch_size:
            yield collate(pending), None
            pending = []

    if pending:
        yield collate(pending), None


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    denominator = denominator.float()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def _collision_loss_per_sample(
    collision_loss: ESDFCollisionLoss,
    trajectory: torch.Tensor,
    sdf: torch.Tensor,
) -> torch.Tensor:
    bs, height, width = sdf.shape
    origin_offset = torch.tensor([width // 2, height // 2], device=sdf.device)
    offset = collision_loss.offset.to(sdf.device).view(1, 1, collision_loss.N, 1)

    centers = trajectory[..., None, :2] + offset * trajectory[..., None, 2:4]
    pixel_coord = torch.stack(
        [centers[..., 0] / collision_loss.resolution, -centers[..., 1] / collision_loss.resolution],
        dim=-1,
    )
    grid_xy = pixel_coord / origin_offset
    valid_mask = (grid_xy < 0.95).all(-1) & (grid_xy > -0.95).all(-1)
    on_road_mask = sdf[:, height // 2, width // 2] > 0

    distance = F.grid_sample(
        sdf.unsqueeze(1), grid_xy, mode="bilinear", padding_mode="zeros"
    ).squeeze(1)

    cost = collision_loss.radius - distance
    valid_mask = valid_mask & (cost > 0) & on_road_mask[:, None, None]
    cost = cost.masked_fill(~valid_mask, 0)

    loss = F.l1_loss(cost, torch.zeros_like(cost), reduction="none").sum(-1)
    return loss.sum(-1) / (valid_mask.sum(dim=(-1, -2)).float() + 1e-6)


def _compute_loss_rows(
    model: torch.nn.Module,
    data: Dict[str, Any],
    scenarios: List[Any],
    loss_weights: Dict[str, float],
    collision_loss: ESDFCollisionLoss,
    include_collision_loss: bool,
    rank_score: str,
) -> List[Dict[str, Any]]:
    res = model(data)
    bs, _, horizon, _ = res["prediction"].shape

    prediction = res["prediction"]
    trajectory = res["trajectory"]
    probability = res["probability"]
    ref_free_trajectory = res.get("ref_free_trajectory")

    targets_pos = data["agent"]["target"][:, :, -horizon:]
    valid_mask = data["agent"]["valid_mask"][:, :, -horizon:]
    targets_vel = data["agent"]["velocity"][:, :, -horizon:]

    target = torch.cat(
        [
            targets_pos[..., :2],
            torch.stack([targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1),
            targets_vel,
        ],
        dim=-1,
    )

    if trajectory is not None and probability is not None and probability.numel() > 0:
        ego_valid_mask = valid_mask[:, 0]
        num_valid_points = ego_valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs, device=trajectory.device), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        mode_interval = model.radius / model.num_modes
        target_m_index = (
            future_projection[torch.arange(bs, device=trajectory.device), target_r_index, 0]
            / mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=model.num_modes - 1)

        best_trajectory = trajectory[
            torch.arange(bs, device=trajectory.device), target_r_index, target_m_index
        ]

        reg_loss_raw = F.smooth_l1_loss(
            best_trajectory, target[:, 0], reduction="none"
        ).sum(-1)
        reg_loss = _safe_divide(
            (reg_loss_raw * ego_valid_mask).sum(-1), ego_valid_mask.sum(-1)
        )

        logits = probability.clone()
        logits.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)
        cls_target = target_r_index * model.num_modes + target_m_index
        cls_loss = F.cross_entropy(logits.reshape(bs, -1), cls_target, reduction="none")

        if include_collision_loss:
            collision = _collision_loss_per_sample(
                collision_loss, best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision = reg_loss.new_zeros(bs)
    else:
        reg_loss = prediction.new_zeros(bs)
        cls_loss = prediction.new_zeros(bs)
        collision = prediction.new_zeros(bs)

    if ref_free_trajectory is not None:
        ref_target = target[:, 0, :, : ref_free_trajectory.shape[-1]]
        ref_raw = F.smooth_l1_loss(
            ref_free_trajectory, ref_target, reduction="none"
        ).sum(-1)
        ref_free_reg_loss = _safe_divide(
            (ref_raw * valid_mask[:, 0]).sum(-1), valid_mask[:, 0].sum(-1)
        )
    else:
        ref_free_reg_loss = reg_loss.new_zeros(bs)

    pred_raw = F.smooth_l1_loss(
        prediction, target[:, 1:, :, : prediction.shape[-1]], reduction="none"
    ).sum(-1)
    pred_mask = valid_mask[:, 1:]
    prediction_loss = _safe_divide(
        (pred_raw * pred_mask).sum(dim=(1, 2)), pred_mask.sum(dim=(1, 2))
    )

    planning_loss = (
        loss_weights["weight_reg_loss"] * reg_loss
        + loss_weights["weight_cls_loss"] * cls_loss
        + loss_weights["weight_ref_free_reg_loss"] * ref_free_reg_loss
    )
    auxiliary_loss = loss_weights["weight_auxiliary"] * (
        loss_weights["weight_prediction_loss"] * prediction_loss
        + loss_weights["weight_collision_loss"] * collision
    )
    total_loss = planning_loss + auxiliary_loss

    score_sources = {
        "planning_loss": planning_loss,
        "total_loss": total_loss,
        "reg_loss": reg_loss,
        "cls_loss": cls_loss,
        "ref_free_reg_loss": ref_free_reg_loss,
        "prediction_loss": prediction_loss,
        "collision_loss": collision,
    }
    if rank_score not in score_sources:
        raise ValueError(
            f"Unknown rank_score '{rank_score}'. Expected one of {sorted(score_sources)}"
        )
    loss_rank_score = score_sources[rank_score]

    rows = []
    for i, scenario in enumerate(scenarios):
        row = {
            "scene_id": scenario.token,
            "log_name": scenario.log_name,
            "scenario_type": scenario.scenario_type,
            "status": "ok",
            "loss_rank_score": float(loss_rank_score[i].detach().cpu()),
            "difficulty_score": float(loss_rank_score[i].detach().cpu()),
            "rank_score_source": rank_score,
            "planning_loss": float(planning_loss[i].detach().cpu()),
            "total_loss": float(total_loss[i].detach().cpu()),
            "reg_loss": float(reg_loss[i].detach().cpu()),
            "cls_loss": float(cls_loss[i].detach().cpu()),
            "ref_free_reg_loss": float(ref_free_reg_loss[i].detach().cpu()),
            "prediction_loss": float(prediction_loss[i].detach().cpu()),
            "collision_loss": float(collision[i].detach().cpu()),
        }
        rows.append(row)

    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool = True) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nonfinite_decoder_warning_messages(caught_warnings: List[warnings.WarningMessage]) -> List[str]:
    """Return warning messages that indicate decoder non-finite values."""
    messages = []
    warning_patterns = (
        "Non-finite values in planning decoder query tensor",
        "NaN detected immediately after q_proj",
    )
    for warning in caught_warnings:
        message = str(warning.message)
        if any(pattern in message for pattern in warning_patterns):
            messages.append(message)
    return messages


def _annotate_forward_warnings(
    rows: List[Dict[str, Any]],
    warning_messages: List[str],
) -> None:
    """Attach batch-level forward warning metadata to each row in the batch."""
    if not warning_messages:
        return

    warning_summary = " | ".join(message.splitlines()[0] for message in warning_messages)
    for row in rows:
        row["forward_warning"] = "decoder_nonfinite"
        row["forward_warning_count"] = len(warning_messages)
        row["forward_warning_messages"] = warning_summary


def _annotate_nonfinite_losses(rows: List[Dict[str, Any]]) -> int:
    """Mark rows with non-finite scalar loss fields so downstream filters can exclude them."""
    numeric_columns = (
        "loss_rank_score",
        "difficulty_score",
        "planning_loss",
        "total_loss",
        "reg_loss",
        "cls_loss",
        "ref_free_reg_loss",
        "prediction_loss",
        "collision_loss",
    )
    nonfinite_count = 0
    for row in rows:
        bad_columns = [
            column
            for column in numeric_columns
            if column in row
            and isinstance(row[column], (int, float))
            and not math.isfinite(float(row[column]))
        ]
        if bad_columns:
            row["status"] = "nonfinite_loss"
            row["error"] = f"Non-finite loss columns: {','.join(bad_columns)}"
            nonfinite_count += 1
    return nonfinite_count


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    scoring_cfg = cfg.get("loss_scoring", {})
    output_path = _resolve_path(
        scoring_cfg.get(
            "output_path",
            "artifacts/loss_scores/pluto_train_loss_scores.jsonl",
        )
    )
    csv_output_path = _resolve_path(
        scoring_cfg.get(
            "csv_output_path",
            str(output_path.with_suffix(".csv")),
        )
    )
    batch_size = int(scoring_cfg.get("batch_size", cfg.data_loader.params.batch_size))
    rank_score = str(scoring_cfg.get("rank_score", "planning_loss"))
    progress_every = int(scoring_cfg.get("progress_every", 50))
    include_collision_loss = bool(scoring_cfg.get("include_collision_loss", False))
    device_name = str(scoring_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    logger.info("=" * 80)
    logger.info("Scoring PLUTO scenarios by frozen open-loop imitation loss")
    logger.info("=" * 80)
    logger.info("Scenario filter: %s", cfg.scenario_filter)
    logger.info("Output JSONL: %s", output_path)
    logger.info("Output CSV: %s", csv_output_path)
    logger.info("Rank score: %s", rank_score)
    logger.info("Device: %s", device)

    worker = build_worker(cfg)
    model_cfg = _model_config_without_loss(cfg)
    model = instantiate(model_cfg)

    checkpoint_path = _resolve_path(str(cfg.pretrained_ckpt))
    state_dict = _load_state_dict(checkpoint_path, torch.device("cpu"))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning("Missing checkpoint keys: %s", missing_keys[:10])
    if unexpected_keys:
        logger.warning("Unexpected checkpoint keys: %s", unexpected_keys[:10])

    model.to(device)
    model.eval()

    scenarios = build_scenarios(cfg, worker, model)
    logger.info("Loaded %d scenarios for scoring", len(scenarios))

    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=model.get_list_of_required_feature(),
        target_builders=model.get_list_of_computed_target(),
    )
    dataset = ScenarioDataset(
        scenarios=scenarios,
        feature_preprocessor=feature_preprocessor,
        augmentors=None,
    )

    loss_weights = _loss_config(cfg)
    collision_loss = ESDFCollisionLoss().to(device)

    all_rows: List[Dict[str, Any]] = []
    scored = 0
    failed = 0
    decoder_warning_rows = 0
    nonfinite_loss_rows = 0

    with torch.no_grad():
        for batch, error_row in _iter_valid_batches(dataset, batch_size=batch_size):
            if error_row is not None:
                failed += 1
                all_rows.append(error_row)
                _write_jsonl(output_path, [error_row], append=True)
                continue

            assert batch is not None
            features, _targets, batch_scenarios = batch
            features = move_features_type_to_device(features, device)
            data = features["feature"].data

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                rows = _compute_loss_rows(
                    model=model,
                    data=data,
                    scenarios=batch_scenarios,
                    loss_weights=loss_weights,
                    collision_loss=collision_loss,
                    include_collision_loss=include_collision_loss,
                    rank_score=rank_score,
                )

            warning_messages = _nonfinite_decoder_warning_messages(caught_warnings)
            if warning_messages:
                _annotate_forward_warnings(rows, warning_messages)
                decoder_warning_rows += len(rows)

            nonfinite_loss_rows += _annotate_nonfinite_losses(rows)

            scored += len(rows)
            all_rows.extend(rows)
            _write_jsonl(output_path, rows, append=True)

            if progress_every > 0 and scored % progress_every < len(rows):
                logger.info("Scored %d/%d scenarios (failed=%d)", scored, len(scenarios), failed)

    _write_csv(csv_output_path, all_rows)

    ok_rows = [row for row in all_rows if row.get("status") == "ok"]
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "scenario_filter": str(cfg.scenario_filter),
        "checkpoint": str(checkpoint_path),
        "rank_score": rank_score,
        "include_collision_loss": include_collision_loss,
        "num_scenarios_loaded": len(scenarios),
        "num_scored": len(ok_rows),
        "num_failed": failed,
        "num_decoder_warning_rows": decoder_warning_rows,
        "num_nonfinite_loss_rows": nonfinite_loss_rows,
        "output_path": str(output_path),
        "csv_output_path": str(csv_output_path),
    }
    if ok_rows:
        scores = [float(row["loss_rank_score"]) for row in ok_rows]
        summary.update(
            {
                "min_loss_rank_score": min(scores),
                "max_loss_rank_score": max(scores),
                "mean_loss_rank_score": sum(scores) / len(scores),
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Done. Summary: %s", summary_path)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
