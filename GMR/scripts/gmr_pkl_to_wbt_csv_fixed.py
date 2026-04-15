import argparse
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

def make_quat_continuous(quat_xyzw: np.ndarray) -> np.ndarray:
    """让相邻帧四元数符号连续，避免 slerp / 角速度求导时 360° 翻转。"""
    q = quat_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q

def main(input_pkl, output_csv):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # (T, 3)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # (T, 4), 已经是 xyzw
    dof_pos  = np.asarray(data["dof_pos"],  dtype=np.float32)   # (T, 29)

    assert root_pos.shape[1] == 3
    assert root_rot.shape[1] == 4
    assert dof_pos.shape[1] == 29
    assert len(root_pos) == len(root_rot) == len(dof_pos)

    # -------------------------------------------------
    # 1) 世界坐标系变换
    #
    # 从你的数据看，第2列像“高度”，而 whole_body_tracking 要第3列是 z。
    # 因此把 root_pos 从源系变到 Isaac 系：
    #   [x, y, z]_src -> [x, -z, y]_tgt
    #
    # 这是一个绕 x 轴 +90° 的保右手系旋转。
    # -------------------------------------------------
    root_pos_fixed = np.stack(
        [root_pos[:, 0], -root_pos[:, 2], root_pos[:, 1]],
        axis=1
    )

    # -------------------------------------------------
    # 2) root_rot 已经是 xyzw，不要再重排分量顺序
    #    直接做同一个世界系旋转：R_tgt = R_fix * R_src
    # -------------------------------------------------
    r_src = R.from_quat(root_rot)  # 这里直接把 GMR pkl 里的 root_rot 当 xyzw
    r_fix = R.from_euler("x", 90, degrees=True)
    r_tgt = r_fix * r_src
    root_rot_fixed = r_tgt.as_quat().astype(np.float32)  # 仍然输出 xyzw

    # -------------------------------------------------
    # 3) 四元数符号连续化
    #    否则 whole_body_tracking 的 slerp / 角速度计算会出现 360° 翻转
    # -------------------------------------------------
    root_rot_fixed = make_quat_continuous(root_rot_fixed)

    # 拼成 whole_body_tracking 需要的 CSV
    motion = np.concatenate([root_pos_fixed, root_rot_fixed, dof_pos], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.savetxt(output_csv, motion, delimiter=",")

    print("saved:", output_csv)
    print("shape:", motion.shape)
    print("first row:", motion[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.input_pkl, args.output_csv)