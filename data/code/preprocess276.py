import argparse
import os
import numpy as np
import torch
import sys
from tqdm import tqdm

# 确保能找到 smplxfk 工具
sys.path.append(os.getcwd()) 
try:
    from dld.data.render_joints.smplfk import ax_to_6v
except ImportError:
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
    def ax_to_6v(q):
        # q: [..., 3]
        mat = axis_angle_to_matrix(q)
        return matrix_to_rotation_6d(mat)

def preprocess_to_dart_276(
    data_dir="./finedance/motion", 
    output_dir="./finedance/mofea276"
):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    print(f"正在生成 DartControl 标准的 276 维特征...")

    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        data = np.load(path) # LODGE 格式: [T, 159]
        
        # 1. 拆分数据
        # 0:3 是 root_pos (位移) -> DartControl 276维通常不包含显式位移，或者将其合在旋转里
        # 但标准的 276 维定义通常是 46 个关节的 Rot6D
        # 3:159 是 52 个关节的轴角 (Axis-Angle)
        local_q_axis = torch.from_numpy(data[:, 3:159]).float() # [T, 156]
        length = local_q_axis.shape[0]

        # 2. Reshape 并选择前 46 个关节
        # SMPL-X 关节顺序：0 是 Root, 1-21 是 Body, 22-24 是 Jaw/Eyes, 25-51 是 Hands
        # 选择前 46 个可以覆盖 Root + Body + 绝大部分手指，刚好 46 * 6 = 276
        local_q_axis = local_q_axis.view(length, 52, 3)
        selected_joints = local_q_axis[:, :46, :] # 选择前 46 个关节

        # 3. 转换为 Rot6D
        # [T, 46, 3] -> [T, 46, 6]
        local_q_6d = ax_to_6v(selected_joints)
        
        # 4. 展平为 276 维
        mofeats = local_q_6d.reshape(length, 276).detach().cpu().numpy()

        # 5. 保存
        np.save(os.path.join(output_dir, f), mofeats)

    print(f"处理完成！特征维度: {mofeats.shape}")

if __name__ == "__main__":
    preprocess_to_dart_276()