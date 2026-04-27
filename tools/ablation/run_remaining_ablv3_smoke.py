#!/usr/bin/env python3
"""
One-click runner for remaining ablation trainings after baseline.

Default remaining variants:
  align_auto, align_smooth_clip, rootyaw_blend
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {pretty}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remaining ablv3_smoke trainings and collect metrics.")
    parser.add_argument("--workspace", type=str, default=".", help="Repository root path.")
    parser.add_argument(
        "--input_pkl",
        type=str,
        default="GMR/test_outputs/dod_0_109g000g_l000_g1_3.pkl",
        help="Input GMR pkl.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "1838740093-nanjing-university"),
        help="W&B entity.",
    )
    parser.add_argument("--name_prefix", type=str, default="ablv3_smoke", help="Run-name prefix.")
    parser.add_argument(
        "--variants",
        type=str,
        default="align_auto,align_smooth_clip,rootyaw_blend",
        help="Comma-separated variants to run.",
    )
    parser.add_argument("--max_iterations", type=int, default=2000, help="Training iterations per variant.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Training env count.")
    parser.add_argument("--project", type=str, default="lodge_gmr_beyondmimic", help="W&B training project.")
    parser.add_argument(
        "--csv_to_npz_upload_grace_s",
        type=float,
        default=2.0,
        help="Auto-stop grace seconds after csv_to_npz upload marker.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output metrics CSV path (default: tools/ablation/<name_prefix>_metrics.csv).",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    env = os.environ.copy()
    env["WANDB_ENTITY"] = args.entity

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else (workspace / "tools" / "ablation" / f"{args.name_prefix}_metrics.csv").resolve()
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    run_ablation_cmd = [
        "python",
        "tools/ablation/run_lodge_gmr_beyondmimic_ablation.py",
        "--workspace",
        str(workspace),
        "--input_pkl",
        args.input_pkl,
        "--entity",
        args.entity,
        "--name_prefix",
        args.name_prefix,
        "--variants",
        args.variants,
        "--max_iterations",
        str(args.max_iterations),
        "--num_envs",
        str(args.num_envs),
        "--project",
        args.project,
        "--train_stability_profile",
        "--csv_to_npz_upload_grace_s",
        str(args.csv_to_npz_upload_grace_s),
    ]
    run_cmd(run_ablation_cmd, cwd=workspace, env=env)

    collect_cmd = [
        "python",
        "tools/ablation/collect_tracking_ablation_metrics.py",
        "--wandb_dir",
        "wandb",
        "--run_name_prefix",
        f"{args.name_prefix}_",
        "--dedup_by_run_name",
        "--verbose",
        "--output_csv",
        str(output_csv),
    ]
    run_cmd(collect_cmd, cwd=workspace, env=env)

    print(f"\n[DONE] Metrics CSV: {output_csv}")


if __name__ == "__main__":
    main()
