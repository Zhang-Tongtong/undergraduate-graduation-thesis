# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument("--disable_push_robot", action="store_true", help="Disable interval push disturbance.")
parser.add_argument(
    "--push_velocity_scale",
    type=float,
    default=1.0,
    help="Scale push velocity range. 1.0 keeps default.",
)
parser.add_argument(
    "--push_warmup_s",
    type=float,
    default=0.0,
    help="Warmup seconds before enabling push disturbance.",
)
parser.add_argument(
    "--push_ramp_s",
    type=float,
    default=0.0,
    help="Linear ramp seconds for push disturbance after warmup.",
)
parser.add_argument(
    "--anchor_pos_threshold",
    type=float,
    default=None,
    help="Override termination threshold for anchor z error.",
)
parser.add_argument(
    "--ee_body_pos_threshold",
    type=float,
    default=None,
    help="Override termination threshold for end-effector z error.",
)
parser.add_argument(
    "--anchor_ori_threshold",
    type=float,
    default=None,
    help="Override termination threshold for anchor orientation.",
)
parser.add_argument(
    "--command_noise_scale",
    type=float,
    default=1.0,
    help="Scale command reset noise (pose/velocity/joint_position_range).",
)
parser.add_argument(
    "--obs_noise_scale",
    type=float,
    default=1.0,
    help="Scale observation additive uniform noise magnitude.",
)
parser.add_argument(
    "--disable_command_debug_vis",
    action="store_true",
    help="Disable command debug visualization for faster, cleaner training logs.",
)
parser.add_argument(
    "--reward_std_scale",
    type=float,
    default=1.0,
    help="Global multiplier for motion reward std (anchor/body pos/ori/vel).",
)
parser.add_argument("--reward_anchor_pos_std", type=float, default=None, help="Override std for motion_global_anchor_pos.")
parser.add_argument("--reward_anchor_ori_std", type=float, default=None, help="Override std for motion_global_anchor_ori.")
parser.add_argument("--reward_body_pos_std", type=float, default=None, help="Override std for motion_body_pos.")
parser.add_argument("--reward_body_ori_std", type=float, default=None, help="Override std for motion_body_ori.")
parser.add_argument("--reward_body_lin_vel_std", type=float, default=None, help="Override std for motion_body_lin_vel.")
parser.add_argument("--reward_body_ang_vel_std", type=float, default=None, help="Override std for motion_body_ang_vel.")
parser.add_argument(
    "--action_rate_weight_scale",
    type=float,
    default=1.0,
    help="Scale negative penalty weight for action_rate_l2. <1.0 relaxes action smoothness penalty.",
)
parser.add_argument(
    "--undesired_contacts_weight_scale",
    type=float,
    default=1.0,
    help="Scale penalty weight for undesired_contacts. <1.0 relaxes collision penalty.",
)
parser.add_argument("--algo_learning_rate", type=float, default=None, help="Override PPO learning rate.")
parser.add_argument("--algo_entropy_coef", type=float, default=None, help="Override PPO entropy coefficient.")
parser.add_argument("--algo_desired_kl", type=float, default=None, help="Override PPO desired KL target.")
parser.add_argument("--algo_num_learning_epochs", type=int, default=None, help="Override PPO learning epochs.")
parser.add_argument("--algo_num_mini_batches", type=int, default=None, help="Override PPO mini-batch count.")
parser.add_argument("--policy_init_noise_std", type=float, default=None, help="Override policy init action noise std.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
import os
import pickle
import yaml

def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def dump_yaml(filename, data, sort_keys=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _scale_range_tuple(v: tuple[float, float], scale: float) -> tuple[float, float]:
    return float(v[0]) * scale, float(v[1]) * scale


def _scale_range_dict(ranges: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    return {k: _scale_range_tuple(v, scale) for k, v in ranges.items()}


def _scale_obs_noise(env_cfg, scale: float):
    if abs(scale - 1.0) < 1e-8:
        return
    policy_cfg = env_cfg.observations.policy
    for term_name in dir(policy_cfg):
        if term_name.startswith("_"):
            continue
        term = getattr(policy_cfg, term_name, None)
        if term is None or not hasattr(term, "noise"):
            continue
        noise = term.noise
        if noise is None:
            continue
        if hasattr(noise, "n_min") and hasattr(noise, "n_max"):
            noise.n_min = float(noise.n_min) * scale
            noise.n_max = float(noise.n_max) * scale


def _apply_runtime_overrides(env_cfg):
    # 1) Command reset noise
    if abs(args_cli.command_noise_scale - 1.0) > 1e-8:
        scale = float(args_cli.command_noise_scale)
        env_cfg.commands.motion.pose_range = _scale_range_dict(env_cfg.commands.motion.pose_range, scale)
        env_cfg.commands.motion.velocity_range = _scale_range_dict(env_cfg.commands.motion.velocity_range, scale)
        env_cfg.commands.motion.joint_position_range = _scale_range_tuple(env_cfg.commands.motion.joint_position_range, scale)

    # 2) Observation noise
    _scale_obs_noise(env_cfg, float(args_cli.obs_noise_scale))

    # 3) Termination thresholds
    if args_cli.anchor_pos_threshold is not None:
        env_cfg.terminations.anchor_pos.params["threshold"] = float(args_cli.anchor_pos_threshold)
    if args_cli.ee_body_pos_threshold is not None:
        env_cfg.terminations.ee_body_pos.params["threshold"] = float(args_cli.ee_body_pos_threshold)
    if args_cli.anchor_ori_threshold is not None:
        env_cfg.terminations.anchor_ori.params["threshold"] = float(args_cli.anchor_ori_threshold)

    # 4) Push disturbance: disable / scale / warmup-ramp
    if args_cli.disable_push_robot:
        env_cfg.events.push_robot = None
    else:
        push_params = dict(env_cfg.events.push_robot.params)
        push_params["velocity_range"] = _scale_range_dict(push_params["velocity_range"], float(args_cli.push_velocity_scale))

        if args_cli.push_warmup_s > 0.0 or args_cli.push_ramp_s > 0.0:
            import whole_body_tracking.tasks.tracking.mdp as tracking_mdp

            step_dt = float(env_cfg.decimation * env_cfg.sim.dt)
            push_params["warmup_steps"] = int(max(args_cli.push_warmup_s, 0.0) / step_dt)
            push_params["ramp_steps"] = int(max(args_cli.push_ramp_s, 0.0) / step_dt)
            push_params["velocity_scale"] = 1.0
            env_cfg.events.push_robot.func = tracking_mdp.push_by_setting_velocity_with_warmup

        env_cfg.events.push_robot.params = push_params

    # 5) Command debug visualization
    if args_cli.disable_command_debug_vis:
        env_cfg.commands.motion.debug_vis = False

    # 6) Reward std shaping (keeps reward semantics, smooths learning landscape).
    reward_std_map = {
        "motion_global_anchor_pos": args_cli.reward_anchor_pos_std,
        "motion_global_anchor_ori": args_cli.reward_anchor_ori_std,
        "motion_body_pos": args_cli.reward_body_pos_std,
        "motion_body_ori": args_cli.reward_body_ori_std,
        "motion_body_lin_vel": args_cli.reward_body_lin_vel_std,
        "motion_body_ang_vel": args_cli.reward_body_ang_vel_std,
    }

    # Global std multiplier first.
    std_scale = float(args_cli.reward_std_scale)
    if abs(std_scale - 1.0) > 1e-8:
        for term_name in reward_std_map:
            term = getattr(env_cfg.rewards, term_name, None)
            if term is not None and hasattr(term, "params") and "std" in term.params:
                term.params["std"] = float(term.params["std"]) * std_scale

    # Then per-term explicit overrides.
    for term_name, maybe_std in reward_std_map.items():
        if maybe_std is None:
            continue
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None and hasattr(term, "params"):
            term.params["std"] = float(maybe_std)

    # 7) Penalty weight scaling.
    if abs(args_cli.action_rate_weight_scale - 1.0) > 1e-8:
        env_cfg.rewards.action_rate_l2.weight = float(env_cfg.rewards.action_rate_l2.weight) * float(
            args_cli.action_rate_weight_scale
        )
    if abs(args_cli.undesired_contacts_weight_scale - 1.0) > 1e-8:
        env_cfg.rewards.undesired_contacts.weight = float(env_cfg.rewards.undesired_contacts.weight) * float(
            args_cli.undesired_contacts_weight_scale
        )


def _apply_agent_overrides(agent_cfg):
    if args_cli.algo_learning_rate is not None:
        agent_cfg.algorithm.learning_rate = float(args_cli.algo_learning_rate)
    if args_cli.algo_entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = float(args_cli.algo_entropy_coef)
    if args_cli.algo_desired_kl is not None:
        agent_cfg.algorithm.desired_kl = float(args_cli.algo_desired_kl)
    if args_cli.algo_num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = int(args_cli.algo_num_learning_epochs)
    if args_cli.algo_num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = int(args_cli.algo_num_mini_batches)
    if args_cli.policy_init_noise_std is not None:
        agent_cfg.policy.init_noise_std = float(args_cli.policy_init_noise_std)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _apply_agent_overrides(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_runtime_overrides(env_cfg)
    print(
        "[runtime-overrides] "
        f"disable_push_robot={args_cli.disable_push_robot}, "
        f"push_velocity_scale={args_cli.push_velocity_scale}, "
        f"push_warmup_s={args_cli.push_warmup_s}, "
        f"push_ramp_s={args_cli.push_ramp_s}, "
        f"command_noise_scale={args_cli.command_noise_scale}, "
        f"obs_noise_scale={args_cli.obs_noise_scale}, "
        f"reward_std_scale={args_cli.reward_std_scale}, "
        f"action_rate_weight_scale={args_cli.action_rate_weight_scale}, "
        f"undesired_contacts_weight_scale={args_cli.undesired_contacts_weight_scale}, "
        f"algo_learning_rate={agent_cfg.algorithm.learning_rate}, "
        f"algo_entropy_coef={agent_cfg.algorithm.entropy_coef}, "
        f"algo_desired_kl={agent_cfg.algorithm.desired_kl}, "
        f"algo_num_learning_epochs={agent_cfg.algorithm.num_learning_epochs}, "
        f"algo_num_mini_batches={agent_cfg.algorithm.num_mini_batches}, "
        f"policy_init_noise_std={agent_cfg.policy.init_noise_std}"
    )

    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
