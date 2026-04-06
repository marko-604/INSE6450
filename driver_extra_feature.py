import numpy as np
from typing import Callable, Dict


def _ensure_recording_array(recording) -> np.ndarray:
    """
    recording: whatever Driver.get_recording(all_info=False) returned.

    For the driver task, we expect shape (T, 2, 4):
      - time steps T
      - index 0: robot car, index 1: human car
      - state = [x, y, heading, speed]
    """
    rec = np.asarray(recording, dtype=float)
    if rec.ndim != 3 or rec.shape[1] < 2 or rec.shape[2] < 4:
        raise ValueError(
            f"Unexpected recording shape {rec.shape}, "
            "expected (T, 2, 4) for the driver task."
        )
    return rec


def extra_none(recording) -> float:
    return 0.0


def extra_progress(recording) -> float:
    rec = _ensure_recording_array(recording)
    robot_y = rec[:, 0, 1]
    return float(robot_y[-1] - robot_y[0])


def extra_speed_smoothness(recording) -> float:
    rec = _ensure_recording_array(recording)
    robot_speed = rec[:, 0, 3]
    if len(robot_speed) < 2:
        return 0.0
    dv = np.diff(robot_speed)
    return float(np.mean(dv ** 2))


def extra_relative_speed(recording) -> float:
    rec = _ensure_recording_array(recording)
    robot_speed = rec[:, 0, 3]
    human_speed = rec[:, 1, 3]
    return float(np.mean((robot_speed - human_speed) ** 2))


def extra_keeping_speed(recording) -> float:
    rec = _ensure_recording_array(recording)
    robot_speed = rec[:, 0, 3]
    return float(np.mean(np.square(robot_speed - 1.0)))


EXTRA_FEATURE_BANK: Dict[str, Dict[str, object]] = {
    "none": {
        "name": "No extra feature",
        "description": "No additional feature; use only the reduced fixed driver feature set.",
        "compute": extra_none,
    },
    "keeping_speed": {
        "name": "Keeping desired speed",
        "description": "Penalizes deviation from the desired speed of 1.0. This is the removed original speed feature.",
        "compute": extra_keeping_speed,
    },
    "progress": {
        "name": "Forward progress",
        "description": "Total forward progress of the robot car along the lane (final y-position minus initial y-position).",
        "compute": extra_progress,
    },
    "speed_smoothness": {
        "name": "Speed smoothness / jerk",
        "description": "Penalizes large changes in speed over time, capturing how smooth the robot's acceleration profile is.",
        "compute": extra_speed_smoothness,
    },
    "relative_speed": {
        "name": "Relative speed to human",
        "description": "Penalizes the squared difference between robot speed and human speed, encouraging the robot to match the human driver’s speed profile.",
        "compute": extra_relative_speed,
    },
}


def compute_extra_feature(recording, feature_id: str) -> float:
    if feature_id not in EXTRA_FEATURE_BANK:
        feature_id = "none"
    fn: Callable = EXTRA_FEATURE_BANK[feature_id]["compute"]  # type: ignore
    return float(fn(recording))
