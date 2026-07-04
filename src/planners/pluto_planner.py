import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import numpy.typing as npt
import shapely
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from scipy.special import softmax

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender

from ..post_processing.emergency_brake import EmergencyBrake
from ..post_processing.trajectory_evaluator import TrajectoryEvaluator
from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states, load_checkpoint

logger = logging.getLogger(__name__)


class PlutoPlanner(AbstractPlanner):
    requires_scenario: bool = True

    def __init__(
        self,
        planner: TorchModuleWrapper,
        scenario: AbstractScenario = None,
        planner_ckpt: str = None,
        render: bool = False,
        use_gpu=True,
        save_dir=None,
        candidate_subsample_ratio: int = 0.5,
        candidate_min_num: int = 1,
        candidate_max_num: int = 20,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        learning_based_score_weight: float = 0.25,
        use_prediction: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._render = render
        self._imgs = []
        self._scenario = scenario
        if use_gpu:
            # Check if CUDA is actually available (may not be in Ray workers)
            if torch.cuda.is_available():
                try:
                    # Try to create a tensor on CUDA to verify it works
                    _ = torch.zeros(1).cuda()
                    self.device = torch.device("cuda")
                except RuntimeError:
                    # CUDA not actually available (e.g., in Ray worker)
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self._use_prediction = use_prediction

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
        self._candidate_subsample_ratio = candidate_subsample_ratio
        self._candidate_min_num = candidate_min_num
        self._topk = candidate_max_num

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._scenario_type = scenario.scenario_type
        self._profile_enabled = self._env_flag_enabled("PLUTO_PLANNER_PROFILE")
        self._profile_sync_cuda = self._env_flag_enabled(
            "PLUTO_PLANNER_PROFILE_SYNC_CUDA", default="1"
        )
        self._profile_totals: Dict[str, float] = {}
        self._profile_counts: Dict[str, int] = {}
        self._profile_steps = 0
        self._profile_emitted = False

        # post-processing
        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()
        self._learning_based_score_weight = learning_based_score_weight

        if render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _env_flag_enabled(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}

    def _profile_start(self) -> Optional[float]:
        return time.perf_counter() if self._profile_enabled else None

    def _profile_stop(self, name: str, start_time: Optional[float]) -> None:
        if start_time is None:
            return
        elapsed = time.perf_counter() - start_time
        self._profile_totals[name] = self._profile_totals.get(name, 0.0) + elapsed
        self._profile_counts[name] = self._profile_counts.get(name, 0) + 1

    def _profile_cuda_barrier(self) -> None:
        if (
            self._profile_enabled
            and self._profile_sync_cuda
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def _emit_profile_summary(self) -> None:
        if not self._profile_enabled or self._profile_emitted:
            return

        means_ms = {
            name: (self._profile_totals[name] / self._profile_counts[name]) * 1000.0
            for name in self._profile_totals
        }
        payload = {
            "event": "pluto_planner_profile",
            "device": str(self.device),
            "scenario_log_name": getattr(self._scenario, "log_name", None),
            "scenario_name": getattr(self._scenario, "scenario_name", None),
            "scenario_token": getattr(self._scenario, "token", None),
            "scenario_type": getattr(self._scenario, "scenario_type", None),
            "steps": self._profile_steps,
            "totals_s": self._profile_totals,
            "means_ms": means_ms,
            "counts": self._profile_counts,
            "sync_cuda": self._profile_sync_cuda,
        }
        line = json.dumps(payload, sort_keys=True)
        profile_path = os.environ.get("PLUTO_PLANNER_PROFILE_PATH")
        if profile_path:
            output_path = Path(profile_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as profile_file:
                profile_file.write(line + "\n")
        else:
            print(f"PLUTO_PLANNER_PROFILE {line}", flush=True)

        self._profile_emitted = True

    @torch.no_grad()
    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        action = self._model_loader.infer(features)
        action = action[0].cpu().numpy()[0]

        return action.astype(np.float64)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            profile_start = self._profile_start()
            checkpoint_state = load_checkpoint(self._planner_ckpt, device=self.device)
            self._profile_stop("initialize.checkpoint_load", profile_start)
            
            # Debug: Log checkpoint info
            logger.debug(f"Loading checkpoint from: {self._planner_ckpt}")
            logger.debug(f"Checkpoint state dict has {len(checkpoint_state)} keys")
            logger.debug(f"Model state dict has {len(self._planner.state_dict())} keys")
            
            # Use strict=False to allow missing keys (e.g., from LoRA merged checkpoints)
            profile_start = self._profile_start()
            missing_keys, unexpected_keys = self._planner.load_state_dict(checkpoint_state, strict=False)
            self._profile_stop("initialize.load_state_dict", profile_start)
            
            # Filter out expected missing/unexpected keys for LoRA merged checkpoints
            # These are typically from model structure differences (e.g., pos_emb vs lora_base_pos_emb)
            # or from training-time only parameters that don't exist in inference model
            if missing_keys:
                # Check if this looks like a LoRA merged checkpoint issue
                # Missing encoder_blocks keys are concerning, but other missing keys might be expected
                encoder_missing = [k for k in missing_keys if 'encoder_blocks' in k]
                other_missing = [k for k in missing_keys if 'encoder_blocks' not in k]
                
                if encoder_missing:
                    # This is a real problem - encoder keys should be in merged checkpoint
                    logger.error(f"❌ Missing encoder keys when loading checkpoint: {len(encoder_missing)} keys")
                    logger.error(f"  This indicates a checkpoint format mismatch!")
                    logger.error(f"  Missing keys (first 10): {encoder_missing[:10]}")
                    logger.error(f"  Checkpoint path: {self._planner_ckpt}")
                    logger.error(f"  This may indicate the checkpoint was not properly merged or has a different structure.")
                elif other_missing:
                    # Other missing keys (e.g., from model structure differences) are usually fine
                    logger.debug(f"Missing non-encoder keys (likely expected): {len(other_missing)} keys")
            
            if unexpected_keys:
                # Filter out expected unexpected keys (e.g., lora_base_pos_emb from training)
                # These are typically from training-time only parameters
                lora_base_keys = [k for k in unexpected_keys if 'lora_base' in k or ('pos_emb' in k and 'lora_' not in k)]
                lora_prefixed_keys = [k for k in unexpected_keys if k.startswith('lora_')]
                other_unexpected = [k for k in unexpected_keys if 'lora_base' not in k and not k.startswith('lora_')]
                
                if lora_prefixed_keys:
                    # Keys with lora_ prefix suggest checkpoint was saved from a model with LoRA layers still active
                    logger.warning(f"⚠️  Checkpoint contains LoRA-prefixed keys (suggests unmerged LoRA checkpoint): {len(lora_prefixed_keys)} keys")
                    logger.warning(f"  First 5: {lora_prefixed_keys[:5]}")
                    logger.warning(f"  This checkpoint may not be properly merged. Expected merged_final.ckpt from LoRA training.")
                elif lora_base_keys:
                    logger.debug(f"Unexpected LoRA/training keys (expected for merged checkpoints): {len(lora_base_keys)} keys")
                
                if other_unexpected:
                    logger.warning(f"⚠️  Unexpected keys in checkpoint (will be ignored): {len(other_unexpected)} keys")
                    if len(other_unexpected) <= 10:
                        logger.warning(f"  Unexpected keys: {other_unexpected}")
                    else:
                        logger.warning(f"  Unexpected keys (first 10): {other_unexpected[:10]}...")

        profile_start = self._profile_start()
        self._planner.eval()
        # Move model to device, but fallback to CPU if CUDA fails (e.g., in Ray workers)
        try:
            self._planner = self._planner.to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e) or "No CUDA GPUs" in str(e):
                # Use debug level to avoid spam in Ray workers (this is expected behavior)
                logger.debug(f"CUDA not available in Ray worker, using CPU instead: {e}")
                self.device = torch.device("cpu")
                self._planner = self._planner.to(self.device)
            else:
                raise
        self._profile_stop("initialize.model_to_device", profile_start)

        self._initialization = initialization

        profile_start = self._profile_start()
        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager

        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager
        self._profile_stop("initialize.scenario_manager", profile_start)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        step_start = self._profile_start()
        if self._profile_enabled:
            self._profile_steps += 1

        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        ego_state = current_input.history.ego_states[-1]
        profile_start = self._profile_start()
        self._scenario_manager.update_ego_state(ego_state)
        self._profile_stop("step.update_ego_state", profile_start)

        profile_start = self._profile_start()
        self._scenario_manager.update_drivable_area_map()
        self._profile_stop("step.update_drivable_area_map", profile_start)

        planning_trajectory = self._run_planning_once(current_input)

        self._inference_runtimes.append(time.perf_counter() - start_time)
        self._profile_stop("step.total_compute_planner_trajectory", step_start)

        return planning_trajectory

    def _run_planning_once(self, current_input: PlannerInput):
        ego_state = current_input.history.ego_states[-1]

        profile_start = self._profile_start()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        self._profile_stop("step.feature_build", profile_start)

        profile_start = self._profile_start()
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor()]
        ).to_device(self.device)
        self._profile_stop("step.feature_tensor_to_device", profile_start)

        profile_start = self._profile_start()
        self._profile_cuda_barrier()
        out = self._planner.forward(planner_feature_torch.data)
        self._profile_cuda_barrier()
        self._profile_stop("step.model_forward", profile_start)

        profile_start = self._profile_start()
        candidate_trajectories = (
            out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
        )
        probability = out["probability"][0].cpu().numpy()

        if self._use_prediction:
            predictions = out["output_prediction"][0].cpu().numpy()
        else:
            predictions = None

        ref_free_trajectory = (
            (out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in out
            else None
        )
        self._profile_stop("step.output_to_numpy", profile_start)

        profile_start = self._profile_start()
        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            current_input.history.ego_states[-1],
            ref_free_trajectory,
        )
        self._profile_stop("step.trim_candidates", profile_start)

        profile_start = self._profile_start()
        agents_info = self._get_agent_info(planner_feature.data, predictions, ego_state)
        self._profile_stop("step.agent_info", profile_start)

        profile_start = self._profile_start()
        baseline_path = self._get_baseline_path_safe(ego_state)
        self._profile_stop("step.baseline_path", profile_start)

        profile_start = self._profile_start()
        rule_based_scores = self._trajectory_evaluator.evaluate(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=current_input.history.ego_states[-1],
            detections=current_input.history.observations[-1],
            traffic_light_data=current_input.traffic_light_data,
            agents_info=agents_info,
            route_lane_dict=self._scenario_manager.get_route_lane_dicts(),
            drivable_area_map=self._scenario_manager.drivable_area_map,
            baseline_path=baseline_path,
        )
        self._profile_stop("step.trajectory_evaluation", profile_start)

        profile_start = self._profile_start()
        final_scores = (
            rule_based_scores + self._learning_based_score_weight * learning_based_score
        )

        best_candidate_idx = final_scores.argmax()
        self._profile_stop("step.score_selection", profile_start)

        profile_start = self._profile_start()
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state,
            self._trajectory_evaluator.time_to_at_fault_collision(best_candidate_idx),
            candidate_trajectories[best_candidate_idx],
        )
        self._profile_stop("step.emergency_brake", profile_start)

        # no emergency
        profile_start = self._profile_start()
        if trajectory is None:
            trajectory = candidate_trajectories[best_candidate_idx, 1:]
            trajectory = InterpolatedTrajectory(
                global_trajectory_to_states(
                    global_trajectory=trajectory,
                    ego_history=current_input.history.ego_states,
                    future_horizon=len(trajectory) * self._step_interval,
                    step_interval=self._step_interval,
                    include_ego_state=False,
                )
            )
        self._profile_stop("step.trajectory_conversion", profile_start)

        if self._render:
            profile_start = self._profile_start()
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=self._global_to_local(trajectory, ego_state),
                    candidate_trajectories=self._global_to_local(
                        candidate_trajectories[rule_based_scores > 0], ego_state
                    ),
                    candidate_index=best_candidate_idx,
                    predictions=predictions,
                    return_img=True,
                )
            )
            self._profile_stop("step.render", profile_start)

        return trajectory

    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state: EgoState,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        candidate_trajectories: (n_ref, n_mode, 80, 3)
        probability: (n_ref, n_mode)
        """
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
            probability = probability.reshape(-1)

        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = probability[sorted_idx][: self._topk]
        
        # Handle empty candidate array
        if len(sorted_probability) == 0:
            # No candidates - use only ref_free_trajectory if available
            if ref_free_trajectory is not None:
                sorted_candidate_trajectories = ref_free_trajectory[None, ...]
                sorted_probability = np.array([1.0])
            else:
                # Fallback: return a simple straight trajectory
                T, C = candidate_trajectories.shape[1], candidate_trajectories.shape[2]
                fallback_traj = np.zeros((1, T, C))
                # Simple forward motion
                for t in range(T):
                    fallback_traj[0, t, 0] = t * 0.5  # 0.5m per step forward
                sorted_candidate_trajectories = fallback_traj
                sorted_probability = np.array([1.0])
        else:
            sorted_probability = softmax(sorted_probability)
            
            # Add ref_free_trajectory if available and not already used as fallback
            if ref_free_trajectory is not None:
                sorted_candidate_trajectories = np.concatenate(
                    [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                    axis=0,
                )
                sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    def _get_agent_info(self, data, predictions, ego_state: EgoState):
        """
        predictions: (n_agent, 80, 2 or 3)
        """
        current_velocity = np.linalg.norm(data["agent"]["velocity"][1:, -1], axis=-1)
        current_state = np.concatenate(
            [data["agent"]["position"][1:, -1], data["agent"]["heading"][1:, -1, None]],
            axis=-1,
        )
        velocity = None

        if predictions is None:  # constant velocity
            timesteps = np.linspace(0.1, 8, 80).reshape(1, 80, 1)
            displacement = data["agent"]["velocity"][1:, None, -1] * timesteps
            positions = current_state[:, None, :2] + displacement
            angles = current_state[:, None, 2:3].repeat(80, axis=1)
            predictions = np.concatenate([positions, angles], axis=-1)
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
            velocity = current_velocity[:, None].repeat(81, axis=1)
        elif predictions.shape[-1] == 2:
            predictions = np.concatenate(
                [current_state[:, None, :2], predictions], axis=1
            )
            diff = predictions[:, 1:] - predictions[:, :-1]
            start_end_dist = np.linalg.norm(
                predictions[:, -1, :2] - predictions[:, 0, :2], axis=-1
            )
            near_stop_mask = start_end_dist < 1.0
            angle = np.arctan2(diff[..., 1], diff[..., 0])
            angle = np.concatenate([current_state[:, None, -1], angle], axis=1)
            angle = np.where(
                near_stop_mask[:, None], current_state[:, 2:3].repeat(81, axis=1), angle
            )
            predictions = np.concatenate(
                [predictions[..., :2], angle[..., None]], axis=-1
            )
        elif predictions.shape[-1] == 3:
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
        elif predictions.shape[-1] == 5:
            velocity = np.linalg.norm(predictions[..., 3:5], axis=-1)
            predictions = np.concatenate(
                [current_state[:, None], predictions[..., :3]], axis=1
            )
            velocity = np.concatenate([current_velocity[:, None], velocity], axis=-1)
        else:
            raise ValueError("Invalid prediction shape")

        # to global
        predictions_global = self._local_to_global(predictions, ego_state)

        if velocity is None:
            velocity = (
                np.linalg.norm(np.diff(predictions_global[..., :2], axis=-2), axis=-1)
                / 0.1
            )
            velocity = np.concatenate([current_velocity[..., None], velocity], axis=-1)

        return {
            "tokens": data["agent_tokens"][1:],
            "shape": data["agent"]["shape"][1:, -1],
            "category": data["agent"]["category"][1:],
            "velocity": velocity,
            "predictions": predictions_global,
        }

    def _get_baseline_path_safe(self, ego_state: EgoState):
        """
        Safely get baseline path, with fallback if reference lines are not cached.
        """
        try:
            reference_lines = self._scenario_manager.get_cached_reference_lines()
            return self._get_ego_baseline_path(reference_lines, ego_state)
        except (ValueError, AttributeError):
            # Reference lines not available - create simple straight baseline
            ego_pos = ego_state.rear_axle.array
            ego_heading = ego_state.rear_axle.heading
            
            # Create a simple straight line ahead of ego
            length = 100.0  # 100m ahead
            end_x = ego_pos[0] + length * np.cos(ego_heading)
            end_y = ego_pos[1] + length * np.sin(ego_heading)
            
            baseline_path = shapely.LineString([
                [ego_pos[0], ego_pos[1]],
                [end_x, end_y]
            ])
            return baseline_path
    
    def _get_ego_baseline_path(self, reference_lines, ego_state: EgoState):
        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    def _local_to_global(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        self._emit_profile_summary()

        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio

            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report
