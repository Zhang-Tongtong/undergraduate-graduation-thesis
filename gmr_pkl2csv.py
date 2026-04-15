import os
import sys
import pickle
import numpy as np

def main(in_pkl, out_csv):
    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # (T, 3)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # (T, 4)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)     # (T, 29)

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4
    assert dof_pos.ndim == 2

    T = min(len(root_pos), len(root_rot), len(dof_pos))
    root_pos = root_pos[:T]
    root_rot = root_rot[:T]
    dof_pos = dof_pos[:T]

    # -------- 关键修正 1：位置从 x,z,y 改成 x,y,z --------
    # 你当前数据里第二列明显像高度，所以交换第 2、3 列
    root_pos_fixed = root_pos[:, [0, 2, 1]]

    # -------- 关键修正 2：四元数从 wxyz 改成 xyzw --------
    # whole_body_tracking 的 csv_to_npz.py 会把输入的 xyzw 转成 wxyz
    # 所以这里先把 GMR 的 root_rot 变成 xyzw
    root_rot_fixed = root_rot[:, [1, 2, 3, 0]]

    motion = np.concatenate([root_pos_fixed, root_rot_fixed, dof_pos], axis=1)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, motion, delimiter=",")

    print("saved:", out_csv)
    print("shape:", motion.shape)
    print("first row:", motion[0, :10])
    print("root_pos min:", root_pos_fixed.min(axis=0))
    print("root_pos max:", root_pos_fixed.max(axis=0))

if __name__ == "__main__":
    in_pkl = sys.argv[1]
    out_csv = sys.argv[2]
    main(in_pkl, out_csv)