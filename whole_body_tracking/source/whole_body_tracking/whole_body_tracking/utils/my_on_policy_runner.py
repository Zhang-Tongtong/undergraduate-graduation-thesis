# import os

# from rsl_rl.env import VecEnv
# from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# from isaaclab_rl.rsl_rl import export_policy_as_onnx

# import wandb
# from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


# class MyOnPolicyRunner(OnPolicyRunner):
#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


# class MotionOnPolicyRunner(OnPolicyRunner):
#     def __init__(
#         self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
#     ):
#         super().__init__(env, train_cfg, log_dir, device)
#         self.registry_name = registry_name

#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_motion_policy_as_onnx(
#                 self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
#             )
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

#             # link the artifact registry to this run
#             if self.registry_name is not None:
#                 wandb.run.use_artifact(self.registry_name)
#                 self.registry_name = None

import os

import rsl_rl
import wandb

from packaging import version

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def _maybe_to_dict(cfg):
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return cfg


def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _manual_patch_train_cfg(train_cfg):
    """
    兼容旧版 whole_body_tracking / IsaacLab 的 PPO policy 配置，
    手工补出 rsl_rl 新版需要的 actor / critic / algorithm.class_name。
    """
    train_cfg = _maybe_to_dict(train_cfg)

    # algorithm 兜底
    if "algorithm" not in train_cfg or not isinstance(train_cfg["algorithm"], dict):
        train_cfg["algorithm"] = {}
    train_cfg["algorithm"].setdefault("class_name", "PPO")

    policy = train_cfg.get("policy", None)
    policy = _maybe_to_dict(policy) if policy is not None else {}

    activation = _cfg_get(policy, "activation", "elu")
    actor_hidden_dims = list(_cfg_get(policy, "actor_hidden_dims", [512, 256, 128]))
    critic_hidden_dims = list(_cfg_get(policy, "critic_hidden_dims", [512, 256, 128]))
    init_noise_std = float(_cfg_get(policy, "init_noise_std", 1.0))
    noise_std_type = _cfg_get(policy, "noise_std_type", "scalar")
    actor_obs_normalization = bool(_cfg_get(policy, "actor_obs_normalization", False))
    critic_obs_normalization = bool(_cfg_get(policy, "critic_obs_normalization", False))

    actor_cfg = train_cfg.get("actor", {})
    critic_cfg = train_cfg.get("critic", {})

    if not isinstance(actor_cfg, dict):
        actor_cfg = {}
    if not isinstance(critic_cfg, dict):
        critic_cfg = {}

    # Actor: 需要 distribution_cfg，否则 PPO 里没有 log_prob
    actor_cfg.setdefault("class_name", "MLPModel")
    actor_cfg.setdefault("hidden_dims", actor_hidden_dims)
    actor_cfg.setdefault("activation", activation)
    actor_cfg.setdefault("obs_normalization", actor_obs_normalization)
    actor_cfg.setdefault(
        "distribution_cfg",
        {
            "class_name": "GaussianDistribution",
            "init_std": init_noise_std,
            "std_type": noise_std_type,
        },
    )

    # Critic: 不需要 distribution_cfg
    critic_cfg.setdefault("class_name", "MLPModel")
    critic_cfg.setdefault("hidden_dims", critic_hidden_dims)
    critic_cfg.setdefault("activation", activation)
    critic_cfg.setdefault("obs_normalization", critic_obs_normalization)

    # 清掉旧字段，避免新版 rsl_rl 误用
    actor_cfg.pop("init_noise_std", None)
    actor_cfg.pop("noise_std_type", None)
    actor_cfg.pop("state_dependent_std", None)

    critic_cfg.pop("init_noise_std", None)
    critic_cfg.pop("noise_std_type", None)
    critic_cfg.pop("state_dependent_std", None)

    train_cfg["actor"] = actor_cfg
    train_cfg["critic"] = critic_cfg

    # 顺手把 obs_groups 也补明白，避免 warning
    if "obs_groups" not in train_cfg or not isinstance(train_cfg["obs_groups"], dict):
        train_cfg["obs_groups"] = {}
    train_cfg["obs_groups"].setdefault("actor", ["policy"])
    train_cfg["obs_groups"].setdefault("critic", ["critic"])

    return train_cfg


def _patch_train_cfg(train_cfg):
    """
    先尝试使用 IsaacLab 官方兼容函数 handle_deprecated_rsl_rl_cfg。
    如果导入失败或运行失败，再退回手工补丁。
    """
    train_cfg = _maybe_to_dict(train_cfg)

    try:
        # 注意：延迟导入，避免在 train.py 顶层 import 阶段触发 pxr 问题
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        installed_version = version.parse(getattr(rsl_rl, "__version__", "0.0.0"))
        train_cfg = handle_deprecated_rsl_rl_cfg(train_cfg, installed_version)
        train_cfg = _maybe_to_dict(train_cfg)

        print("[patch-rsl] used handle_deprecated_rsl_rl_cfg")
    except Exception as e:
        print("[patch-rsl] handle_deprecated_rsl_rl_cfg failed, fallback to manual patch:", repr(e))
        train_cfg = _manual_patch_train_cfg(train_cfg)

    # 再兜底一次，保证 algorithm.class_name 存在
    if "algorithm" in train_cfg and isinstance(train_cfg["algorithm"], dict):
        train_cfg["algorithm"].setdefault("class_name", "PPO")

    print("[patch-rsl] rsl_rl version =", getattr(rsl_rl, "__version__", "unknown"))
    print("[patch-rsl] train_cfg keys =", list(train_cfg.keys()))
    if "algorithm" in train_cfg and isinstance(train_cfg["algorithm"], dict):
        print("[patch-rsl] algorithm keys =", list(train_cfg["algorithm"].keys()))
    if "actor" in train_cfg and isinstance(train_cfg["actor"], dict):
        print("[patch-rsl] actor keys =", list(train_cfg["actor"].keys()))
    if "critic" in train_cfg and isinstance(train_cfg["critic"], dict):
        print("[patch-rsl] critic keys =", list(train_cfg["critic"].keys()))

    return train_cfg


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        train_cfg = _patch_train_cfg(train_cfg)
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        return


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        train_cfg = _patch_train_cfg(train_cfg)
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        return