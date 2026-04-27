#!/usr/bin/env python3
"""
Collect local wandb metrics for tracking ablation runs.

Reads:
  wandb/run-*/files/config.yaml
  wandb/run-*/files/wandb-summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def unwrap_wandb_value(x: Any) -> Any:
    """Unwrap wandb config wrappers like {'desc': ..., 'value': ...}."""
    cur = x
    while isinstance(cur, dict) and "value" in cur and set(cur.keys()).issubset({"value", "desc", "_type"}):
        cur = cur["value"]
    return cur


def nested_get(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = unwrap_wandb_value(obj)
    for p in path:
        cur = unwrap_wandb_value(cur)
        if not isinstance(cur, dict):
            return default
        if p not in cur:
            # Some wandb configs insert one more wrapper level under "value".
            if "value" in cur:
                cur = unwrap_wandb_value(cur["value"])
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                    continue
            return default
        cur = cur[p]
    return unwrap_wandb_value(cur)


def find_key_values(obj: Any, target_key: str, out: list[Any], max_hits: int = 20) -> None:
    if len(out) >= max_hits:
        return
    obj = unwrap_wandb_value(obj)
    if isinstance(obj, dict):
        if target_key in obj:
            out.append(unwrap_wandb_value(obj[target_key]))
            if len(out) >= max_hits:
                return
        for v in obj.values():
            find_key_values(v, target_key, out, max_hits=max_hits)
            if len(out) >= max_hits:
                return
    elif isinstance(obj, list):
        for v in obj:
            find_key_values(v, target_key, out, max_hits=max_hits)
            if len(out) >= max_hits:
                return


def extract_run_name(cfg: dict[str, Any]) -> str:
    candidates = [
        nested_get(cfg, ["train_cfg", "run_name"], ""),
        nested_get(cfg, ["run_name"], ""),
        nested_get(cfg, ["args", "run_name"], ""),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()

    found: list[Any] = []
    find_key_values(cfg, "run_name", found)
    for c in found:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return ""


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local W&B metrics for ablation runs.")
    parser.add_argument("--wandb_dir", type=str, default="wandb", help="Path to local wandb directory.")
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="",
        help="Filter by train_cfg.run_name prefix.",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default="",
        help="Filter by substring in run_name.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="tools/ablation/ablation_metrics.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--dedup_by_run_name",
        action="store_true",
        help="Keep only one row per run_name (prefer larger _step, then larger _runtime).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug stats and sample missing runs.")
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir).resolve()
    run_dirs = sorted(list(wandb_dir.glob("run-*")) + list(wandb_dir.glob("offline-run-*")))
    if args.verbose:
        print(f"[DEBUG] run_dirs: {len(run_dirs)} at {wandb_dir}")

    rows: list[dict[str, Any]] = []
    all_run_names: list[str] = []
    missing_run_name_samples: list[str] = []
    num_with_cfg_summary = 0
    for run_dir in run_dirs:
        cfg_path = run_dir / "files" / "config.yaml"
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not cfg_path.exists() or not summary_path.exists():
            continue
        num_with_cfg_summary += 1

        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_name = extract_run_name(cfg)
        if not isinstance(run_name, str):
            run_name = str(run_name)
        run_name = run_name.strip()
        if run_name:
            all_run_names.append(run_name)
        if not run_name and len(missing_run_name_samples) < 5:
            missing_run_name_samples.append(str(run_dir))

        if args.run_name_prefix and not run_name.startswith(args.run_name_prefix):
            continue
        if args.contains and args.contains not in run_name:
            continue

        motion_file = nested_get(cfg, ["env_cfg", "commands", "motion", "motion_file"], "")
        experiment_name = nested_get(cfg, ["train_cfg", "experiment_name"], "")
        max_iterations = nested_get(cfg, ["train_cfg", "max_iterations"], "")
        num_envs = nested_get(cfg, ["env_cfg", "scene", "num_envs"], "")

        row = {
            "run_id": run_dir.name.split("-")[-1],
            "run_dir": str(run_dir),
            "run_name": run_name,
            "experiment_name": experiment_name,
            "motion_file": motion_file,
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "Train/mean_reward": safe_float(summary.get("Train/mean_reward")),
            "Train/mean_episode_length": safe_float(summary.get("Train/mean_episode_length")),
            "Episode_Termination/time_out": safe_float(summary.get("Episode_Termination/time_out")),
            "Episode_Termination/anchor_pos": safe_float(summary.get("Episode_Termination/anchor_pos")),
            "Episode_Termination/ee_body_pos": safe_float(summary.get("Episode_Termination/ee_body_pos")),
            "Metrics/motion/error_anchor_pos": safe_float(summary.get("Metrics/motion/error_anchor_pos")),
            "Metrics/motion/error_body_pos": safe_float(summary.get("Metrics/motion/error_body_pos")),
            "Metrics/motion/error_body_rot": safe_float(summary.get("Metrics/motion/error_body_rot")),
            "_step": safe_float(summary.get("_step")),
            "_runtime": safe_float(summary.get("_runtime")),
        }
        rows.append(row)

    if args.dedup_by_run_name:
        best: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row["run_name"]
            step = row["_step"] if row["_step"] is not None else -1.0
            runtime = row["_runtime"] if row["_runtime"] is not None else -1.0
            old = best.get(key)
            if old is None:
                best[key] = row
                continue
            old_step = old["_step"] if old["_step"] is not None else -1.0
            old_runtime = old["_runtime"] if old["_runtime"] is not None else -1.0
            if (step, runtime) > (old_step, old_runtime):
                best[key] = row
        rows = list(best.values())

    rows.sort(key=lambda x: (x["run_name"], x["run_id"]))
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "run_id",
        "run_name",
        "experiment_name",
        "motion_file",
        "num_envs",
        "max_iterations",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Episode_Termination/time_out",
        "Episode_Termination/anchor_pos",
        "Episode_Termination/ee_body_pos",
        "Metrics/motion/error_anchor_pos",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_body_rot",
        "_step",
        "_runtime",
        "run_dir",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.verbose:
        print(f"[DEBUG] runs with cfg+summary: {num_with_cfg_summary}")
        if missing_run_name_samples:
            print("[DEBUG] sample runs with empty run_name:")
            for p in missing_run_name_samples:
                print(f"  {p}")
        if all_run_names:
            uniq_names = sorted(set(all_run_names))
            print(f"[DEBUG] discovered run_name count: {len(uniq_names)}")
            print("[DEBUG] sample discovered run_names:")
            for name in uniq_names[:20]:
                print(f"  {name}")

    print(f"[INFO] Collected {len(rows)} runs -> {output_csv}")
    if len(rows) == 0 and (args.run_name_prefix or args.contains):
        print("[WARN] No runs matched current filter.")
        if args.run_name_prefix:
            print(f"[WARN] run_name_prefix={args.run_name_prefix}")
        if args.contains:
            print(f"[WARN] contains={args.contains}")
    if rows:
        print("[INFO] Preview:")
        for row in rows[:10]:
            print(
                f"  {row['run_name']:<45} "
                f"reward={row['Train/mean_reward']} "
                f"ep_len={row['Train/mean_episode_length']} "
                f"timeout={row['Episode_Termination/time_out']}"
            )


if __name__ == "__main__":
    main()
