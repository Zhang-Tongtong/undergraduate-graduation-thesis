# LODGE + GMR + BeyondMimic 消融实验脚本

## 1. 运行全链路消融（pkl -> csv -> npz -> train）

```bash
python tools/ablation/run_lodge_gmr_beyondmimic_ablation.py \
  --workspace . \
  --input_pkl GMR/test_outputs/dod_0_109g000g_l000_g1_3.pkl \
  --entity 1838740093-nanjing-university \
  --name_prefix ablv2 \
  --max_iterations 30000 \
  --train_stability_profile
```

默认会跑 4 个变体：
- `baseline_auto`
- `align_auto`
- `align_smooth_clip`
- `rootyaw_blend`

如需只跑部分变体：

```bash
python tools/ablation/run_lodge_gmr_beyondmimic_ablation.py \
  --workspace . \
  --input_pkl GMR/test_outputs/dod_0_109g000g_l000_g1_3.pkl \
  --entity 1838740093-nanjing-university \
  --name_prefix ablv2 \
  --variants baseline_auto,align_auto
```

如需只做 CSV 转换（不做 npz 与训练）：

```bash
python tools/ablation/run_lodge_gmr_beyondmimic_ablation.py \
  --workspace . \
  --input_pkl GMR/test_outputs/dod_0_109g000g_l000_g1_3.pkl \
  --name_prefix ablv2 \
  --skip_npz
```

## 2. 收集并汇总本地 wandb 训练指标

```bash
python tools/ablation/collect_tracking_ablation_metrics.py \
  --wandb_dir wandb \
  --run_name_prefix ablv2_ \
  --output_csv tools/ablation/ablv2_metrics.csv
```

脚本会输出：
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Termination/time_out`
- `Episode_Termination/anchor_pos`
- `Episode_Termination/ee_body_pos`
- 若干 motion error 指标

