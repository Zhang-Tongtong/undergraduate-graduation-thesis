import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_quat_continuous_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to keep temporal continuity."""
    q = np.asarray(quat_xyzw, dtype=np.float32).copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def normalize_quat_xyzw(quat_xyzw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).copy()
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return q / norm


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[:, 3:], quat_xyzw[:, :3]], axis=1)


def _quat_candidate_quality(quat_xyzw: np.ndarray) -> Dict[str, float]:
    """Smaller score is better for humanoid root orientation consistency."""
    q = normalize_quat_xyzw(make_quat_continuous_xyzw(quat_xyzw))
    euler = R.from_quat(q).as_euler("xyz", degrees=False)

    roll_abs = np.abs(euler[:, 0])
    pitch_abs = np.abs(euler[:, 1])

    # Penalize excessive roll/pitch and abrupt frame-to-frame rotation.
    rp_median = float(np.median(roll_abs) + np.median(pitch_abs))
    rp_p95 = float(np.percentile(roll_abs, 95) + np.percentile(pitch_abs, 95))

    dots = np.sum(q[1:] * q[:-1], axis=1)
    dots = np.clip(np.abs(dots), 1e-8, 1.0)
    delta_angle = 2.0 * np.arccos(dots)
    delta_p95 = float(np.percentile(delta_angle, 95)) if len(delta_angle) > 0 else 0.0

    score = rp_median + 0.3 * rp_p95 + 0.2 * delta_p95
    return {
        "score": float(score),
        "rp_median": rp_median,
        "rp_p95": rp_p95,
        "delta_angle_p95": delta_p95,
    }


def infer_root_rot_xyzw(
    root_rot: np.ndarray, root_rot_format: str = "auto"
) -> Tuple[np.ndarray, str, Dict[str, Dict[str, float]]]:
    """
    Convert root_rot into xyzw format with optional automatic format detection.

    Returns:
      quat_xyzw, inferred_format, scores
    """
    root_rot = np.asarray(root_rot, dtype=np.float32)
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must be [T,4], got {root_rot.shape}")

    if root_rot_format not in {"auto", "xyzw", "wxyz"}:
        raise ValueError(f"Unsupported root_rot_format={root_rot_format}")

    if root_rot_format == "xyzw":
        quat_xyzw = root_rot
        inferred = "xyzw"
        scores: Dict[str, Dict[str, float]] = {"xyzw": _quat_candidate_quality(quat_xyzw)}
    elif root_rot_format == "wxyz":
        quat_xyzw = wxyz_to_xyzw(root_rot)
        inferred = "wxyz"
        scores = {"wxyz": _quat_candidate_quality(quat_xyzw)}
    else:
        cand_xyzw = root_rot
        cand_wxyz = wxyz_to_xyzw(root_rot)
        score_xyzw = _quat_candidate_quality(cand_xyzw)
        score_wxyz = _quat_candidate_quality(cand_wxyz)
        scores = {"xyzw": score_xyzw, "wxyz": score_wxyz}
        if score_xyzw["score"] <= score_wxyz["score"]:
            quat_xyzw = cand_xyzw
            inferred = "xyzw"
        else:
            quat_xyzw = cand_wxyz
            inferred = "wxyz"

    quat_xyzw = normalize_quat_xyzw(make_quat_continuous_xyzw(quat_xyzw))
    return quat_xyzw, inferred, scores


def write_diag_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
