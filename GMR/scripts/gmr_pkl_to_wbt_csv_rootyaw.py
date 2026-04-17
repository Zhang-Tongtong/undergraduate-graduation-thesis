import argparse
import os
import pickle
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


def make_quat_continuous(quat_xyzw: np.ndarray) -> np.ndarray:
    q = quat_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def ensure_odd_window(window: int, length: int) -> int:
    if length < 3:
        return 1
    if window > length:
        window = length
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 1
    return window


def smooth_1d(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = ensure_odd_window(window, len(x))
    if window <= polyorder or window < 3:
        return x.copy()
    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([q_wxyz[:, 1:], q_wxyz[:, :1]], axis=1)


def main(
    input_pkl: str,
    output_csv: str,
    z_offset: float = 0.0,
    smooth_yaw: bool = True,
    yaw_window: int = 9,
    yaw_polyorder: int = 2,
    keep_raw_root_xy: bool = True,
):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # [T, 3], Z-up
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # [T, 4], 假定是 wxyz
    dof_pos  = np.asarray(data["dof_pos"], dtype=np.float32)    # [T, 29]

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3, root_pos.shape
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4, root_rot.shape
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29, dof_pos.shape
    assert len(root_pos) == len(root_rot) == len(dof_pos)

    # 1) root_pos: 只加 z_offset
    root_pos_out = root_pos.copy()
    root_pos_out[:, 2] += float(z_offset)

    if not keep_raw_root_xy:
        root_pos_out[:, 0] = root_pos_out[0, 0]
        root_pos_out[:, 1] = root_pos_out[0, 1]

    # 2) root_rot: wxyz -> xyzw
    root_rot_xyzw = wxyz_to_xyzw(root_rot)

    # 3) 连续化，避免 +q/-q 跳变
    root_rot_xyzw = make_quat_continuous(root_rot_xyzw)

    # 4) 提取 euler，注意 scipy 输入 xyzw
    euler_xyz = R.from_quat(root_rot_xyzw).as_euler("xyz", degrees=False)
    roll = euler_xyz[:, 0]
    pitch = euler_xyz[:, 1]
    yaw = euler_xyz[:, 2]

    # 5) 只保留 yaw，且对 yaw 做 unwrap + 平滑
    yaw_unwrap = np.unwrap(yaw)

    if smooth_yaw:
        yaw_proc = smooth_1d(yaw_unwrap, yaw_window, yaw_polyorder)
    else:
        yaw_proc = yaw_unwrap.copy()

    # 6) 重建 root_rot：roll=0, pitch=0, yaw=yaw_proc
    #    这一步是关键：去掉 root 的翻滚和俯仰，只保留朝向
    root_rot_xyzw_out = R.from_euler(
        "xyz",
        np.stack([np.zeros_like(yaw_proc), np.zeros_like(yaw_proc), yaw_proc], axis=1),
        degrees=False,
    ).as_quat().astype(np.float32)

    root_rot_xyzw_out = make_quat_continuous(root_rot_xyzw_out)

    # 7) 拼输出
    motion = np.concatenate([root_pos_out, root_rot_xyzw_out, dof_pos], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    np.savetxt(output_csv, motion, delimiter=",")

    print("=" * 80)
    print("saved:", output_csv)
    print("shape:", motion.shape)
    print("root_pos z min/max:", root_pos_out[:, 2].min(), root_pos_out[:, 2].max())
    print("raw yaw range   :", float(yaw_unwrap.min()), float(yaw_unwrap.max()))
    print("proc yaw range  :", float(yaw_proc.min()), float(yaw_proc.max()))
    print("first row:", motion[0])
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--z_offset", type=float, default=0.0)

    parser.add_argument("--smooth_yaw", action="store_true")
    parser.add_argument("--yaw_window", type=int, default=9)
    parser.add_argument("--yaw_polyorder", type=int, default=2)

    parser.add_argument(
        "--keep_raw_root_xy",
        action="store_true",
        help="保留原始 root x/y 平移；不加时默认 False，所以这里建议命令里显式写上",
    )

    args = parser.parse_args()

    main(
        input_pkl=args.input_pkl,
        output_csv=args.output_csv,
        z_offset=args.z_offset,
        smooth_yaw=args.smooth_yaw,
        yaw_window=args.yaw_window,
        yaw_polyorder=args.yaw_polyorder,
        keep_raw_root_xy=args.keep_raw_root_xy,
    )