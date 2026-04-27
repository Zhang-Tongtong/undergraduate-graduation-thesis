import argparse
import os
import pickle

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from gmr_pkl_to_wbt_csv_common import (
    ensure_parent_dir,
    infer_root_rot_xyzw,
    make_quat_continuous_xyzw,
    write_diag_json,
)


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


def main(
    input_pkl: str,
    output_csv: str,
    z_offset: float = 0.0,
    root_rot_format: str = "auto",
    smooth_yaw: bool = True,
    yaw_window: int = 9,
    yaw_polyorder: int = 2,
    root_attitude_blend: float = 0.35,
    max_abs_roll_pitch_rad: float = 0.6,
    lock_root_xy: bool = False,
):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # [T, 3], Z-up
    root_rot_raw = np.asarray(data["root_rot"], dtype=np.float32)   # [T,4], may be xyzw/wxyz
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)    # [T, 29]

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3, root_pos.shape
    assert root_rot_raw.ndim == 2 and root_rot_raw.shape[1] == 4, root_rot_raw.shape
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29, dof_pos.shape
    assert len(root_pos) == len(root_rot_raw) == len(dof_pos)

    root_rot_format_meta = str(data.get("root_rot_format", "")).lower()
    root_rot_format_use = root_rot_format
    if root_rot_format == "auto" and root_rot_format_meta in {"xyzw", "wxyz"}:
        root_rot_format_use = root_rot_format_meta

    # 1) root_pos: 只加 z_offset，可选锁定 x/y
    root_pos_out = root_pos.copy()
    root_pos_out[:, 2] += float(z_offset)
    if lock_root_xy:
        root_pos_out[:, 0] = root_pos_out[0, 0]
        root_pos_out[:, 1] = root_pos_out[0, 1]

    # 2) root_rot: 自动识别格式 -> xyzw
    root_rot_xyzw_in, inferred_format, format_scores = infer_root_rot_xyzw(
        root_rot_raw,
        root_rot_format=root_rot_format_use,
    )

    # 3) 提取 euler
    euler_xyz = R.from_quat(root_rot_xyzw_in).as_euler("xyz", degrees=False)
    roll_raw = euler_xyz[:, 0]
    pitch_raw = euler_xyz[:, 1]
    yaw_raw = euler_xyz[:, 2]

    # 4) yaw 平滑
    yaw_unwrap = np.unwrap(yaw_raw)
    yaw_proc = smooth_1d(yaw_unwrap, yaw_window, yaw_polyorder) if smooth_yaw else yaw_unwrap.copy()

    # 5) 仅部分压制 roll/pitch，避免把参考动作改得过头
    blend = float(np.clip(root_attitude_blend, 0.0, 1.0))
    roll_proc = (1.0 - blend) * roll_raw
    pitch_proc = (1.0 - blend) * pitch_raw
    if max_abs_roll_pitch_rad > 0:
        roll_proc = np.clip(roll_proc, -max_abs_roll_pitch_rad, max_abs_roll_pitch_rad)
        pitch_proc = np.clip(pitch_proc, -max_abs_roll_pitch_rad, max_abs_roll_pitch_rad)

    root_rot_xyzw_out = R.from_euler(
        "xyz",
        np.stack([roll_proc, pitch_proc, yaw_proc], axis=1),
        degrees=False,
    ).as_quat().astype(np.float32)
    root_rot_xyzw_out = make_quat_continuous_xyzw(root_rot_xyzw_out)

    # 6) 输出 CSV
    motion = np.concatenate([root_pos_out, root_rot_xyzw_out, dof_pos], axis=1)
    ensure_parent_dir(output_csv)
    np.savetxt(output_csv, motion, delimiter=",")

    # 7) 诊断
    r_in = R.from_quat(root_rot_xyzw_in)
    r_out = R.from_quat(root_rot_xyzw_out)
    delta_angle = np.linalg.norm((r_out * r_in.inv()).as_rotvec(), axis=1)
    diag = {
        "input_pkl": input_pkl,
        "output_csv": output_csv,
        "num_frames": int(len(root_pos)),
        "root_rot_format_arg": root_rot_format,
        "root_rot_format_meta": root_rot_format_meta,
        "root_rot_format_used": root_rot_format_use,
        "root_rot_format_inferred": inferred_format,
        "root_rot_format_scores": format_scores,
        "z_offset": float(z_offset),
        "lock_root_xy": bool(lock_root_xy),
        "smooth_yaw": bool(smooth_yaw),
        "yaw_window": int(yaw_window),
        "yaw_polyorder": int(yaw_polyorder),
        "root_attitude_blend": float(blend),
        "max_abs_roll_pitch_rad": float(max_abs_roll_pitch_rad),
        "raw_yaw_min": float(yaw_unwrap.min()),
        "raw_yaw_max": float(yaw_unwrap.max()),
        "proc_yaw_min": float(yaw_proc.min()),
        "proc_yaw_max": float(yaw_proc.max()),
        "mean_abs_roll_raw": float(np.mean(np.abs(roll_raw))),
        "mean_abs_pitch_raw": float(np.mean(np.abs(pitch_raw))),
        "mean_abs_roll_proc": float(np.mean(np.abs(roll_proc))),
        "mean_abs_pitch_proc": float(np.mean(np.abs(pitch_proc))),
        "mean_delta_root_rot_angle_rad": float(np.mean(delta_angle)),
        "p95_delta_root_rot_angle_rad": float(np.percentile(delta_angle, 95)),
    }
    diag_path = os.path.splitext(output_csv)[0] + "_diag.json"
    write_diag_json(diag_path, diag)

    print("=" * 80)
    print("saved:", output_csv)
    print("diag :", diag_path)
    print("shape:", motion.shape)
    print("root_rot format (arg/meta/used -> inferred):", root_rot_format, root_rot_format_meta, root_rot_format_use, "->", inferred_format)
    print("root_pos z min/max:", root_pos_out[:, 2].min(), root_pos_out[:, 2].max())
    print("raw yaw range :", float(yaw_unwrap.min()), float(yaw_unwrap.max()))
    print("proc yaw range:", float(yaw_proc.min()), float(yaw_proc.max()))
    print("mean root rot delta (rad):", float(np.mean(delta_angle)))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--z_offset", type=float, default=0.0)
    parser.add_argument(
        "--root_rot_format",
        type=str,
        default="auto",
        choices=["auto", "xyzw", "wxyz"],
        help="输入 pkl 里 root_rot 的格式。建议用 auto。",
    )

    parser.add_argument("--smooth_yaw", action="store_true")
    parser.add_argument("--yaw_window", type=int, default=9)
    parser.add_argument("--yaw_polyorder", type=int, default=2)
    parser.add_argument(
        "--root_attitude_blend",
        type=float,
        default=0.35,
        help="roll/pitch 压制比例，0=不改，1=完全清零。",
    )
    parser.add_argument(
        "--max_abs_roll_pitch_rad",
        type=float,
        default=0.6,
        help="处理后 roll/pitch 绝对值上限（弧度），0 表示不裁剪。",
    )

    parser.add_argument("--lock_root_xy", action="store_true", help="将 root x/y 锁定为首帧。")
    parser.add_argument(
        "--keep_raw_root_xy",
        dest="lock_root_xy",
        action="store_false",
        help="兼容旧参数，等价于不锁定 root x/y。",
    )
    parser.set_defaults(lock_root_xy=False)

    args = parser.parse_args()

    main(
        input_pkl=args.input_pkl,
        output_csv=args.output_csv,
        z_offset=args.z_offset,
        root_rot_format=args.root_rot_format,
        smooth_yaw=args.smooth_yaw,
        yaw_window=args.yaw_window,
        yaw_polyorder=args.yaw_polyorder,
        root_attitude_blend=args.root_attitude_blend,
        max_abs_roll_pitch_rad=args.max_abs_roll_pitch_rad,
        lock_root_xy=args.lock_root_xy,
    )
