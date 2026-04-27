import argparse
import pickle
import os

import numpy as np

from gmr_pkl_to_wbt_csv_common import ensure_parent_dir, infer_root_rot_xyzw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="csv",
        help="输出子目录名。",
    )
    parser.add_argument(
        "--root_rot_format",
        type=str,
        default="auto",
        choices=["auto", "xyzw", "wxyz"],
        help="输入 pkl 中 root_rot 格式，建议 auto。",
    )
    parser.add_argument("--z_offset", type=float, default=0.0, help="写 csv 前给 root z 统一偏移。")
    parser.add_argument(
        "--swap_yz",
        action="store_true",
        help="兼容旧流程：把 root_pos 从 [x,y,z] 变成 [x,z,y]。",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, args.output_folder)
    os.makedirs(out_folder, exist_ok=True)

    pkl_files = [f for f in os.listdir(args.folder) if f.endswith(".pkl")]
    for i, file in enumerate(sorted(pkl_files), start=1):
        if file.endswith(".pkl"):
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
        else:
            continue

        root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
        frame_rate = float(motion_data.get("fps", 30))

        if args.swap_yz:
            root_pos = root_pos[:, [0, 2, 1]]
        root_pos[:, 2] += float(args.z_offset)

        root_rot_format_meta = str(motion_data.get("root_rot_format", "")).lower()
        root_rot_format_use = args.root_rot_format
        if args.root_rot_format == "auto" and root_rot_format_meta in {"xyzw", "wxyz"}:
            root_rot_format_use = root_rot_format_meta
        root_rot_xyzw, inferred_format, _ = infer_root_rot_xyzw(root_rot, root_rot_format=root_rot_format_use)
        motion = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)

        if frame_rate > 30.0:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")

        output_csv = os.path.join(out_folder, file.replace(".pkl", ".csv"))
        ensure_parent_dir(output_csv)
        np.savetxt(output_csv, motion, delimiter=",")
        print(
            f"({i}/{len(pkl_files)}) Saved {output_csv} | "
            f"quat arg/meta/used -> {args.root_rot_format}/{root_rot_format_meta}/{root_rot_format_use} -> {inferred_format}"
        )
