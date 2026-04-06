import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F

# ---------- 6D rotation -> rotation matrix ----------
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    d6: (..., 6)
    return: (..., 3, 3)
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)


# ---------- rotation matrix -> axis-angle ----------
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    matrix: (..., 3, 3)
    return: (..., 3)
    """
    eps = 1e-6
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    rx = matrix[..., 2, 1] - matrix[..., 1, 2]
    ry = matrix[..., 0, 2] - matrix[..., 2, 0]
    rz = matrix[..., 1, 0] - matrix[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta).unsqueeze(-1)
    axis = axis / (2.0 * sin_theta + eps)

    aa = axis * theta.unsqueeze(-1)
    aa = torch.nan_to_num(aa, nan=0.0, posinf=0.0, neginf=0.0)
    return aa


def main(in_npy, out_pkl, fps=30):
    x = np.load(in_npy).astype(np.float32)
    assert x.ndim == 2 and x.shape[1] == 139, f"expect [T,139], got {x.shape}"

    T = x.shape[0]

    # 139 = 4 contacts + 3 root_pos + 22*6 rot6d
    contacts = x[:, 0:4]            # [T,4]
    transl = x[:, 4:7]              # [T,3]
    rot6d = x[:, 7:]                # [T,132]
    rot6d = rot6d.reshape(T, 22, 6) # [T,22,6]

    rot6d_t = torch.from_numpy(rot6d)
    rotmat = rotation_6d_to_matrix(rot6d_t)      # [T,22,3,3]
    aa = matrix_to_axis_angle(rotmat).numpy()    # [T,22,3]

    # 约定：第 0 个 joint 是 global orient，后 21 个是 body pose
    global_orient = aa[:, 0, :]      # [T,3]
    body_pose = aa[:, 1:, :].reshape(T, 63)  # [T,21*3]

    out = {
        "transl": transl.astype(np.float32),
        "trans": transl.astype(np.float32),

        "global_orient": global_orient.astype(np.float32),
        "root_orient": global_orient.astype(np.float32),

        "body_pose": body_pose.astype(np.float32),
        "pose_body": body_pose.astype(np.float32),

        "betas": np.zeros((T, 10), dtype=np.float32),
        "gender": "neutral",
        "mocap_frame_rate": np.array(fps, dtype=np.int32),

        "left_hand_pose": np.zeros((T, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((T, 45), dtype=np.float32),
        "jaw_pose": np.zeros((T, 3), dtype=np.float32),
        "leye_pose": np.zeros((T, 3), dtype=np.float32),
        "reye_pose": np.zeros((T, 3), dtype=np.float32),
        "expression": np.zeros((T, 10), dtype=np.float32),

        "contacts": contacts.astype(np.float32),
        "raw_lodge_139": x.astype(np.float32),
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)

    print("saved to:", out_pkl)
    print("transl:", out["transl"].shape)
    print("global_orient:", out["global_orient"].shape)
    print("body_pose:", out["body_pose"].shape)


if __name__ == "__main__":
    in_npy = sys.argv[1]
    out_pkl = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    main(in_npy, out_pkl, fps)