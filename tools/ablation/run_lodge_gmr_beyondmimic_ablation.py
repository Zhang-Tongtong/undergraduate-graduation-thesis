#!/usr/bin/env python3
"""
Run end-to-end ablations for:
GMR pkl -> CSV -> NPZ artifact -> tracking training.

This script is intended to run on a machine with Isaac Sim / IsaacLab runtime ready.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {pretty}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_csv_to_npz_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    auto_interrupt_after_upload: bool = True,
    upload_grace_s: float = 5.0,
    hard_timeout_s: float = 3600.0,
) -> None:
    """Run csv_to_npz with robust auto-continue behavior.

    Some csv_to_npz versions keep Isaac Sim alive after artifact upload.
    This runner watches stdout and, once upload marker appears, will terminate
    the process after a short grace period so the pipeline can continue.
    """
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {pretty}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    upload_done = False
    upload_time = 0.0
    start_time = time.time()
    requested_stop = False

    try:
        for line in proc.stdout:
            print(line, end="")
            if "[INFO]: Motion saved to wandb registry:" in line:
                upload_done = True
                upload_time = time.time()

            if hard_timeout_s > 0 and (time.time() - start_time) > hard_timeout_s:
                print(f"[WARN] csv_to_npz exceeded hard timeout ({hard_timeout_s}s), terminating.")
                proc.terminate()
                requested_stop = True
                break

            if (
                upload_done
                and auto_interrupt_after_upload
                and (time.time() - upload_time) >= max(upload_grace_s, 0.0)
                and proc.poll() is None
            ):
                print("[INFO] csv_to_npz upload marker detected, stopping process to continue pipeline.")
                proc.terminate()
                requested_stop = True
                break
    finally:
        if proc.stdout:
            proc.stdout.close()

    try:
        rc = proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("[WARN] csv_to_npz did not exit in time, killing process.")
        proc.kill()
        rc = proc.wait(timeout=30)

    if rc != 0 and not upload_done:
        raise subprocess.CalledProcessError(rc, cmd)
    if rc != 0 and upload_done:
        # Process can be non-zero when we explicitly terminate after successful upload.
        reason = "auto-stop" if requested_stop else "post-upload non-zero exit"
        print(f"[WARN] csv_to_npz exited with code {rc} ({reason}), but upload marker was detected. Continuing.")


def build_variants_cmds(
    workspace: Path,
    input_pkl: Path,
    robot_xml: Path,
    output_root: Path,
    name_prefix: str,
    root_rot_format: str,
) -> dict[str, list[str]]:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = input_pkl.stem
    base_name = f"{name_prefix}_{stem}"

    variants: dict[str, list[str]] = {}

    csv_baseline = output_root / f"{base_name}_baseline_auto.csv"
    variants["baseline_auto"] = [
        "python",
        "GMR/scripts/gmr_pkl_to_wbt_csv.py",
        "--input_pkl",
        str(input_pkl),
        "--output_csv",
        str(csv_baseline),
        "--root_rot_format",
        root_rot_format,
    ]

    csv_align = output_root / f"{base_name}_align_auto.csv"
    variants["align_auto"] = [
        "python",
        "GMR/scripts/gmr_pkl_to_wbt_csv_preproc.py",
        "--input_pkl",
        str(input_pkl),
        "--output_csv",
        str(csv_align),
        "--robot_xml",
        str(robot_xml),
        "--root_rot_format",
        root_rot_format,
        "--target_foot_height",
        "0.01",
        "--align_percentile",
        "5",
    ]

    csv_smooth = output_root / f"{base_name}_align_smooth_clip.csv"
    variants["align_smooth_clip"] = [
        "python",
        "GMR/scripts/gmr_pkl_to_wbt_csv_preproc.py",
        "--input_pkl",
        str(input_pkl),
        "--output_csv",
        str(csv_smooth),
        "--robot_xml",
        str(robot_xml),
        "--root_rot_format",
        root_rot_format,
        "--smooth_root_pos",
        "--smooth_root_rot",
        "--smooth_dof",
        "--smooth_window",
        "9",
        "--smooth_polyorder",
        "2",
        "--root_pos_smooth_clip_m",
        "0.02",
        "--root_rot_smooth_clip_rad",
        "0.08",
        "--dof_smooth_clip_rad",
        "0.06",
        "--target_foot_height",
        "0.01",
        "--align_percentile",
        "5",
    ]

    csv_rootyaw = output_root / f"{base_name}_rootyaw_blend.csv"
    variants["rootyaw_blend"] = [
        "python",
        "GMR/scripts/gmr_pkl_to_wbt_csv_rootyaw.py",
        "--input_pkl",
        str(input_pkl),
        "--output_csv",
        str(csv_rootyaw),
        "--root_rot_format",
        root_rot_format,
        "--smooth_yaw",
        "--yaw_window",
        "9",
        "--yaw_polyorder",
        "2",
        "--root_attitude_blend",
        "0.35",
        "--max_abs_roll_pitch_rad",
        "0.6",
    ]
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LODGE->GMR->BeyondMimic ablations.")
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Project root path.",
    )
    parser.add_argument("--input_pkl", type=str, required=True, help="Input GMR pkl path.")
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="GMR/assets/unitree_g1/g1_mocap_29dof.xml",
        help="MuJoCo robot xml for preproc conversion.",
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default="GMR/test_outputs/ablation_csv",
        help="Where to store generated CSV files.",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="abl",
        help="Prefix used in csv/output_name/run_name.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_auto,align_auto,align_smooth_clip,rootyaw_blend",
        help="Comma-separated variants. Available: baseline_auto,align_auto,align_smooth_clip,rootyaw_blend",
    )
    parser.add_argument(
        "--root_rot_format",
        type=str,
        default="auto",
        choices=["auto", "xyzw", "wxyz"],
        help="Root quaternion format hint for converters.",
    )
    parser.add_argument(
        "--skip_npz",
        action="store_true",
        help="Only generate CSV, skip csv_to_npz and training.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Generate CSV+NPZ only, skip training.",
    )
    parser.add_argument("--entity", type=str, default="", help="WANDB entity. If empty, use WANDB_ENTITY env.")
    parser.add_argument("--project", type=str, default="lodge_gmr_beyondmimic", help="W&B project name.")
    parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Training task.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs for training.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="Training iterations for each variant.",
    )
    parser.add_argument(
        "--input_fps",
        type=int,
        default=30,
        help="FPS of input CSV motion for csv_to_npz.",
    )
    parser.add_argument(
        "--train_stability_profile",
        action="store_true",
        help="Use stability-oriented runtime overrides in train.py.",
    )
    parser.add_argument(
        "--csv_to_npz_auto_interrupt_after_upload",
        action="store_true",
        default=True,
        help="Auto-stop csv_to_npz after upload marker to avoid hanging before training.",
    )
    parser.add_argument(
        "--csv_to_npz_no_auto_interrupt_after_upload",
        dest="csv_to_npz_auto_interrupt_after_upload",
        action="store_false",
        help="Do not auto-stop csv_to_npz after upload marker.",
    )
    parser.add_argument(
        "--csv_to_npz_upload_grace_s",
        type=float,
        default=5.0,
        help="Seconds to wait after upload marker before stopping csv_to_npz.",
    )
    parser.add_argument(
        "--csv_to_npz_hard_timeout_s",
        type=float,
        default=3600.0,
        help="Hard timeout for csv_to_npz process.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    input_pkl = (workspace / args.input_pkl).resolve() if not Path(args.input_pkl).is_absolute() else Path(args.input_pkl)
    robot_xml = (workspace / args.robot_xml).resolve() if not Path(args.robot_xml).is_absolute() else Path(args.robot_xml)
    output_csv_dir = (workspace / args.output_csv_dir).resolve()
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    if not input_pkl.exists():
        raise FileNotFoundError(f"input_pkl not found: {input_pkl}")
    if not robot_xml.exists():
        raise FileNotFoundError(f"robot_xml not found: {robot_xml}")

    env = os.environ.copy()
    entity = args.entity if args.entity else env.get("WANDB_ENTITY", "")
    if not args.skip_npz and not entity:
        raise ValueError("WANDB entity is required for csv_to_npz/training. Set --entity or WANDB_ENTITY.")
    if entity:
        env["WANDB_ENTITY"] = entity

    all_variants = build_variants_cmds(
        workspace=workspace,
        input_pkl=input_pkl,
        robot_xml=robot_xml,
        output_root=output_csv_dir,
        name_prefix=args.name_prefix,
        root_rot_format=args.root_rot_format,
    )
    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    for name in requested:
        if name not in all_variants:
            raise ValueError(f"Unknown variant: {name}")

    print(f"[INFO] Workspace: {workspace}")
    print(f"[INFO] Input pkl: {input_pkl}")
    print(f"[INFO] Variants: {requested}")
    print(f"[INFO] WANDB_ENTITY: {entity}")

    for variant in requested:
        # 1) pkl -> csv
        convert_cmd = all_variants[variant]
        run_cmd(convert_cmd, cwd=workspace, env=env)

        # infer csv path from command
        csv_idx = convert_cmd.index("--output_csv") + 1
        csv_path = Path(convert_cmd[csv_idx]).resolve()
        output_name = csv_path.stem

        if args.skip_npz:
            continue

        # 2) csv -> npz artifact
        csv_to_npz_cmd = [
            "python",
            "whole_body_tracking/scripts/csv_to_npz.py",
            "--input_file",
            str(csv_path),
            "--input_fps",
            str(args.input_fps),
            "--output_name",
            output_name,
            "--headless",
        ]
        run_csv_to_npz_cmd(
            csv_to_npz_cmd,
            cwd=workspace,
            env=env,
            auto_interrupt_after_upload=args.csv_to_npz_auto_interrupt_after_upload,
            upload_grace_s=args.csv_to_npz_upload_grace_s,
            hard_timeout_s=args.csv_to_npz_hard_timeout_s,
        )

        if args.skip_train:
            continue

        # 3) training
        train_cmd = [
            "python",
            "whole_body_tracking/scripts/rsl_rl/train.py",
            "--task",
            args.task,
            "--registry_name",
            f"{entity}-org/wandb-registry-motions/{output_name}",
            "--headless",
            "--logger",
            "wandb",
            "--log_project_name",
            "lodge_gmr_beyondmimic",
            "--run_name",
            f"{output_name}_run",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.max_iterations),
            "--disable_command_debug_vis",
        ]

        if args.train_stability_profile:
            train_cmd.extend(
                [
                    "--command_noise_scale",
                    "0.7",
                    "--obs_noise_scale",
                    "0.8",
                    "--anchor_pos_threshold",
                    "0.30",
                    "--ee_body_pos_threshold",
                    "0.30",
                    "--push_warmup_s",
                    "120",
                    "--push_ramp_s",
                    "120",
                    "--push_velocity_scale",
                    "0.7",
                ]
            )

        run_cmd(train_cmd, cwd=workspace, env=env)

    print("\n[INFO] Ablation pipeline finished.")


if __name__ == "__main__":
    main()
