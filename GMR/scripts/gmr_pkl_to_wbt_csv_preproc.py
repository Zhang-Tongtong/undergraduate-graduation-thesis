import argparse
import json
import os
import pickle
from typing import List

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        "需要安装 mujoco。请先执行: pip install mujoco"
    ) from e


# =========================
# 基础工具函数
# =========================

def ensure_odd_window(window: int, length: int) -> int:
    """把窗口修正成合法的奇数，并且不超过序列长度。"""
    if length < 3:
        return 1
    window = min(window, length if length % 2 == 1 else length - 1)
    if window < 3:
        return 1
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def make_quat_continuous_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """让相邻帧四元数符号连续，避免 +q/-q 跳变。"""
    q = quat_xyzw.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[:, 3:], quat_xyzw[:, :3]], axis=1)


def smooth_signal(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    对 T×D 的时序信号逐维做 Savitzky-Golay 平滑。
    如果长度太短，直接返回原信号。
    """
    if x.ndim != 2:
        raise ValueError(f"smooth_signal expects 2D array, got {x.shape}")

    T = x.shape[0]
    window = ensure_odd_window(window, T)
    if window <= polyorder or window < 3:
        return x.copy()

    y = savgol_filter(x, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    return y.astype(np.float32)


def smooth_quat_wxyz(quat_wxyz: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    四元数平滑做法：
    1) wxyz -> xyzw
    2) 四元数连续化
    3) 转 rotvec
    4) 对 rotvec 做 savgol 平滑
    5) 转回 quat
    6) 再连续化
    """
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    quat_xyzw = make_quat_continuous_xyzw(quat_xyzw)

    rotvec = R.from_quat(quat_xyzw).as_rotvec()
    rotvec_smooth = smooth_signal(rotvec, window=window, polyorder=polyorder)

    quat_xyzw_smooth = R.from_rotvec(rotvec_smooth).as_quat().astype(np.float32)
    quat_xyzw_smooth = make_quat_continuous_xyzw(quat_xyzw_smooth)

    return xyzw_to_wxyz(quat_xyzw_smooth).astype(np.float32)


# =========================
# MuJoCo 足底高度计算
# =========================

def get_body_ids(model, body_names: List[str]) -> List[int]:
    ids = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            ids.append(body_id)
    return ids


def prepare_qpos(model, root_pos_xyz, root_quat_wxyz, dof_pos):
    """
    GMR pkl 通常包含：
    root_pos: [3]
    root_rot: [4] wxyz
    dof_pos: [29]

    MuJoCo G1 模型一般 qpos = 3 + 4 + 29 = 36。
    这里做一个安全兼容：
    - 如果模型 qpos 更多，就后面补 0
    - 如果更少，就截断
    """
    q = np.concatenate([root_pos_xyz, root_quat_wxyz, dof_pos], axis=0).astype(np.float64)
    if len(q) < model.nq:
        q = np.concatenate([q, np.zeros(model.nq - len(q), dtype=np.float64)], axis=0)
    elif len(q) > model.nq:
        q = q[: model.nq]
    return q


def compute_min_foot_height_per_frame(
    model,
    data,
    root_pos,
    root_rot_wxyz,
    dof_pos,
    foot_body_names=("left_toe_link", "right_toe_link", "left_ankle_roll_link", "right_ankle_roll_link"),
):
    """
    对每一帧做 FK，取足部相关 body 的最小 z。
    默认优先 toe，其次 ankle_roll。
    """
    foot_ids = get_body_ids(model, list(foot_body_names))
    if len(foot_ids) == 0:
        raise RuntimeError(
            f"在 MuJoCo 模型里找不到这些 body: {foot_body_names}，"
            f"请检查 robot xml 是否是 G1 的 g1_mocap_29dof.xml"
        )

    T = len(root_pos)
    min_foot_z = np.zeros(T, dtype=np.float32)

    for t in range(T):
        qpos = prepare_qpos(model, root_pos[t], root_rot_wxyz[t], dof_pos[t])
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        z_list = []
        for bid in foot_ids:
            z_list.append(float(data.xpos[bid][2]))
        min_foot_z[t] = np.min(z_list)

    return min_foot_z


# =========================
# 主流程
# =========================

def main(args):
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # [T,3], GMR 当前输出按你的流程已是 Z-up
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # [T,4], GMR 当前输出是 wxyz
    dof_pos  = np.asarray(data["dof_pos"], dtype=np.float32)    # [T,29]

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3, root_pos.shape
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4, root_rot.shape
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29, dof_pos.shape
    assert len(root_pos) == len(root_rot) == len(dof_pos)

    T = len(root_pos)

    # ---------- Step 1: 四元数连续化 ----------
    root_rot_xyzw = wxyz_to_xyzw(root_rot)
    root_rot_xyzw = make_quat_continuous_xyzw(root_rot_xyzw)
    root_rot = xyzw_to_wxyz(root_rot_xyzw).astype(np.float32)

    # ---------- Step 2: 时序平滑 ----------
    root_pos_proc = root_pos.copy()
    root_rot_proc = root_rot.copy()
    dof_pos_proc = dof_pos.copy()

    if args.smooth_root_pos:
        root_pos_proc = smooth_signal(root_pos_proc, args.smooth_window, args.smooth_polyorder)

    if args.smooth_root_rot:
        root_rot_proc = smooth_quat_wxyz(root_rot_proc, args.smooth_window, args.smooth_polyorder)

    if args.smooth_dof:
        dof_pos_proc = smooth_signal(dof_pos_proc, args.smooth_window, args.smooth_polyorder)

    # ---------- Step 3: 用 MuJoCo 自动算足底贴地偏移 ----------
    model = mujoco.MjModel.from_xml_path(args.robot_xml)
    mj_data = mujoco.MjData(model)

    foot_body_names = [x.strip() for x in args.foot_bodies.split(",") if x.strip()]
    min_foot_z_before = compute_min_foot_height_per_frame(
        model, mj_data, root_pos_proc, root_rot_proc, dof_pos_proc, foot_body_names=foot_body_names
    )

    # 用百分位数比直接 min 更稳，不容易被单帧异常值带偏
    ref_height = float(np.percentile(min_foot_z_before, args.align_percentile))
    auto_z_shift = args.target_foot_height - ref_height
    total_z_shift = auto_z_shift + args.manual_z_offset

    root_pos_proc[:, 2] += total_z_shift

    min_foot_z_after = compute_min_foot_height_per_frame(
        model, mj_data, root_pos_proc, root_rot_proc, dof_pos_proc, foot_body_names=foot_body_names
    )

    # ---------- Step 4: 输出 CSV ----------
    # whole_body_tracking 的 csv_to_npz.py 需要 root_rot 为 xyzw
    root_rot_xyzw_out = wxyz_to_xyzw(root_rot_proc)
    root_rot_xyzw_out = make_quat_continuous_xyzw(root_rot_xyzw_out)

    motion = np.concatenate([root_pos_proc, root_rot_xyzw_out, dof_pos_proc], axis=1)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    np.savetxt(args.output_csv, motion, delimiter=",")

    # ---------- Step 5: 保存诊断 ----------
    diag = {
        "input_pkl": args.input_pkl,
        "output_csv": args.output_csv,
        "robot_xml": args.robot_xml,
        "num_frames": int(T),
        "smooth_root_pos": bool(args.smooth_root_pos),
        "smooth_root_rot": bool(args.smooth_root_rot),
        "smooth_dof": bool(args.smooth_dof),
        "smooth_window": int(args.smooth_window),
        "smooth_polyorder": int(args.smooth_polyorder),
        "target_foot_height": float(args.target_foot_height),
        "align_percentile": float(args.align_percentile),
        "auto_z_shift": float(auto_z_shift),
        "manual_z_offset": float(args.manual_z_offset),
        "total_z_shift": float(total_z_shift),
        "min_foot_z_before_min": float(np.min(min_foot_z_before)),
        "min_foot_z_before_p5": float(np.percentile(min_foot_z_before, 5)),
        "min_foot_z_before_p50": float(np.percentile(min_foot_z_before, 50)),
        "min_foot_z_after_min": float(np.min(min_foot_z_after)),
        "min_foot_z_after_p5": float(np.percentile(min_foot_z_after, 5)),
        "min_foot_z_after_p50": float(np.percentile(min_foot_z_after, 50)),
        "root_z_min_after": float(np.min(root_pos_proc[:, 2])),
        "root_z_max_after": float(np.max(root_pos_proc[:, 2])),
        "foot_bodies": foot_body_names,
    }

    diag_path = os.path.splitext(args.output_csv)[0] + "_diag.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("saved csv:", args.output_csv)
    print("saved diag:", diag_path)
    print("motion shape:", motion.shape)
    print("auto_z_shift:", auto_z_shift)
    print("manual_z_offset:", args.manual_z_offset)
    print("total_z_shift:", total_z_shift)
    print("foot_z before  min / p5 / p50:",
          np.min(min_foot_z_before),
          np.percentile(min_foot_z_before, 5),
          np.percentile(min_foot_z_before, 50))
    print("foot_z after   min / p5 / p50:",
          np.min(min_foot_z_after),
          np.percentile(min_foot_z_after, 5),
          np.percentile(min_foot_z_after, 50))
    print("root_z after min/max:", np.min(root_pos_proc[:, 2]), np.max(root_pos_proc[:, 2]))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GMR pkl into whole_body_tracking CSV.")
    parser.add_argument("--input_pkl", type=str, required=True, help="GMR 输出的 pkl 路径")
    parser.add_argument("--output_csv", type=str, required=True, help="输出给 whole_body_tracking 的 CSV 路径")
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="GMR/assets/unitree_g1/g1_mocap_29dof.xml",
        help="用于计算足底高度的 MuJoCo robot xml"
    )

    # 平滑开关
    parser.add_argument("--smooth_root_pos", action="store_true", help="对 root_pos 做时序平滑")
    parser.add_argument("--smooth_root_rot", action="store_true", help="对 root_rot 做时序平滑")
    parser.add_argument("--smooth_dof", action="store_true", help="对 dof_pos 做时序平滑")
    parser.add_argument("--smooth_window", type=int, default=9, help="Savitzky-Golay 窗口大小，建议 7/9/11")
    parser.add_argument("--smooth_polyorder", type=int, default=2, help="Savitzky-Golay 多项式阶数，建议 2")

    # 贴地参数
    parser.add_argument("--target_foot_height", type=float, default=0.01,
                        help="希望脚底最低点落在的高度，0.01 比 0.0 更稳一点")
    parser.add_argument("--align_percentile", type=float, default=5.0,
                        help="用脚底高度的哪个百分位做对齐，建议 5 或 10")
    parser.add_argument("--manual_z_offset", type=float, default=0.0,
                        help="在自动贴地基础上再额外手动加减的 z 偏移")
    parser.add_argument(
        "--foot_bodies",
        type=str,
        default="left_toe_link,right_toe_link,left_ankle_roll_link,right_ankle_roll_link",
        help="用于估计足底高度的 body 名，按优先顺序逗号分隔"
    )

    args = parser.parse_args()
    main(args)