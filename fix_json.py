"""
fix_json.py - 修复JSON保存问题
不需要重新训练，直接从pickle文件重新生成JSON
"""

import pickle
import json
import os
from config.env_config import env_config
from config.train_config import train_config
from config.model_config import model_config


def make_serializable(obj):
    """将配置对象转换为可序列化的字典"""
    if hasattr(obj, '__dict__'):
        return {
            k: v for k, v in obj.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
    return obj


# 加载训练数据
pkl_path = 'Backup of experimental data/2000_Test20251013/generated_data/all_training_data.pkl'
print(f"Loading training data from {pkl_path}...")

with open(pkl_path, 'rb') as f:
    training_data = pickle.load(f)

# 提取episode奖励
episode_rewards = [ep['episode_reward'] for ep in training_data]

print(f"✅ Loaded {len(episode_rewards)} episodes")

# 构建可序列化的数据
data = {
    'episode_rewards': episode_rewards,
    'config': {
        'env_config': make_serializable(env_config),
        'train_config': make_serializable(train_config),
        'model_config': make_serializable(model_config)
    }
}

# 保存为JSON
json_path = 'Backup of experimental data/2000_Test20251013/logs/training_data.json'
os.makedirs(os.path.dirname(json_path), exist_ok=True)

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Training data saved to {json_path}")
print(f"\n统计信息:")
print(f"  总轮数: {len(episode_rewards)}")
print(f"  平均奖励: {sum(episode_rewards)/len(episode_rewards):.2f}")
print(f"  最大奖励: {max(episode_rewards):.2f}")
print(f"  最小奖励: {min(episode_rewards):.2f}")