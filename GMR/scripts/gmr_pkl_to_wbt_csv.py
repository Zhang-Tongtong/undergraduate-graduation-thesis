import argparse
import os
import pickle
import numpy as np


def make_quat_continuous(quat_xyzw: np.ndarray) -> np.ndarray:
    q = quat_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def main(input_pkl, output_csv, z_offset=0.0):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # [T,3], 现在假设已经是 Z-up
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # [T,4], 假设是 wxyz
    dof_pos  = np.asarray(data["dof_pos"], dtype=np.float32)    # [T,29]

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3, root_pos.shape
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4, root_rot.shape
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29, dof_pos.shape
    assert len(root_pos) == len(root_rot) == len(dof_pos)

    # 1) 坐标轴不再变换
    root_pos_out = root_pos.copy()
    root_pos_out[:, 2] += float(z_offset)

    # 2) 只把四元数从 wxyz -> xyzw
    root_rot_xyzw = np.concatenate([root_rot[:, 1:], root_rot[:, :1]], axis=1)

    # 3) 连续化
    root_rot_xyzw = make_quat_continuous(root_rot_xyzw)

    motion = np.concatenate([root_pos_out, root_rot_xyzw, dof_pos], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.savetxt(output_csv, motion, delimiter=",")

    print("saved:", output_csv)
    print("shape:", motion.shape)
    print("root_pos z min/max:", root_pos_out[:, 2].min(), root_pos_out[:, 2].max())
    print("first row:", motion[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--z_offset", type=float, default=0.0)
    args = parser.parse_args()
    main(args.input_pkl, args.output_csv, args.z_offset)