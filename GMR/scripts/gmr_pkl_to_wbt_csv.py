import argparse
import os
import pickle
import numpy as np

from gmr_pkl_to_wbt_csv_common import ensure_parent_dir, infer_root_rot_xyzw, write_diag_json

def main(input_pkl, output_csv, z_offset=0.0, root_rot_format="auto", diag_json=""):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)   # [T,3]
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)   # [T,4], may be xyzw or wxyz
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)     # [T,29]

    assert root_pos.ndim == 2 and root_pos.shape[1] == 3, root_pos.shape
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4, root_rot.shape
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29, dof_pos.shape
    assert len(root_pos) == len(root_rot) == len(dof_pos)

    # 1) 坐标轴不再变换
    root_pos_out = root_pos.copy()
    root_pos_out[:, 2] += float(z_offset)

    root_rot_format_meta = str(data.get("root_rot_format", "")).lower()
    root_rot_format_use = root_rot_format
    if root_rot_format == "auto" and root_rot_format_meta in {"xyzw", "wxyz"}:
        root_rot_format_use = root_rot_format_meta

    # 2) 自动识别 root_rot 格式并统一输出 xyzw
    root_rot_xyzw, inferred_format, format_scores = infer_root_rot_xyzw(
        root_rot,
        root_rot_format=root_rot_format_use,
    )

    motion = np.concatenate([root_pos_out, root_rot_xyzw, dof_pos], axis=1)

    ensure_parent_dir(output_csv)
    np.savetxt(output_csv, motion, delimiter=",")

    diag = {
        "input_pkl": input_pkl,
        "output_csv": output_csv,
        "num_frames": int(len(root_pos)),
        "z_offset": float(z_offset),
        "root_rot_format_arg": root_rot_format,
        "root_rot_format_meta": root_rot_format_meta,
        "root_rot_format_used": root_rot_format_use,
        "root_rot_format_inferred": inferred_format,
        "root_rot_format_scores": format_scores,
        "root_pos_z_min": float(root_pos_out[:, 2].min()),
        "root_pos_z_max": float(root_pos_out[:, 2].max()),
    }
    diag_path = diag_json if diag_json else os.path.splitext(output_csv)[0] + "_diag.json"
    write_diag_json(diag_path, diag)

    print("=" * 80)
    print("saved:", output_csv)
    print("diag :", diag_path)
    print("shape:", motion.shape)
    print("root_rot format (arg/meta/used -> inferred):", root_rot_format, root_rot_format_meta, root_rot_format_use, "->", inferred_format)
    print("root_pos z min/max:", root_pos_out[:, 2].min(), root_pos_out[:, 2].max())
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
        help="GMR pkl 中 root_rot 的格式。建议用 auto。",
    )
    parser.add_argument(
        "--diag_json",
        type=str,
        default="",
        help="诊断 json 路径。不填则自动写到 output_csv 同名 _diag.json。",
    )
    args = parser.parse_args()
    main(args.input_pkl, args.output_csv, args.z_offset, args.root_rot_format, args.diag_json)
