from copy import deepcopy
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def global_trajectory_to_states(
    global_trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
):
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

    velocities, accelerations = _get_velocity_and_acceleration(
        global_states, ego_history, timesteps
    )
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            global_states, velocities, accelerations, timesteps
        )
    ]

    if include_ego_state:
        agent_states.insert(0, ego_state)
    else:
        init_state = deepcopy(agent_states[0])
        init_state._time_point = ego_state.time_point
        agent_states.insert(0, init_state)

    return agent_states


def load_checkpoint(checkpoint: str, device=None):
    """
    Load checkpoint from file.
    :param checkpoint: path to checkpoint file (can be relative or absolute)
    :param device: device to load checkpoint to (None = default behavior)
    :return: state dict
    """
    import os
    # Convert to absolute path if relative
    if not os.path.isabs(checkpoint):
        # Try relative to current working directory first
        if os.path.exists(checkpoint):
            checkpoint = os.path.abspath(checkpoint)
        else:
            # Try relative to pluto directory
            pluto_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt_path = os.path.join(pluto_dir, checkpoint)
            if os.path.exists(alt_path):
                checkpoint = alt_path
            else:
                # Last resort: try as-is (will raise FileNotFoundError if doesn't exist)
                checkpoint = os.path.abspath(checkpoint)
    
    if device is None:
        # Auto-detect: use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # Always load to CPU first to avoid CUDA issues in Ray workers
    # Then move to target device if needed
    ckpt = torch.load(checkpoint, map_location="cpu")
    
    # Extract state dict and move to target device if it's CUDA and available
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        key = k.replace("model.", "")
        # Only move to CUDA if device is CUDA and CUDA is actually available
        if device.type == "cuda" and torch.cuda.is_available():
            state_dict[key] = v.to(device)
        else:
            state_dict[key] = v
    
    return state_dict
