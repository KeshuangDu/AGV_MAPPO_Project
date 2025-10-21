"""
实验2评估脚本：水平布局 + 单向路由
Horizontal Layout + Unidirectional Routing - Evaluation

运行方法：
    python evaluate_h_uni.py --checkpoint ./data/checkpoints_h_uni/mappo_final_1000ep.pt --episodes 100

✨ 关键差异：
- 使用 train_config_h_uni 配置
- 设置 BIDIRECTIONAL = False
- 不记录后退使用率（单向模式特征）
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config_h_uni import train_config  # ✨ 使用水平单向配置
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Evaluator:
    """评估器类 - 水平单向"""

    def __init__(self, checkpoint_path: str):
        """初始化评估器"""
        # ✨ 确保使用单向模式
        env_config.BIDIRECTIONAL = False  # ✨✨ 关键差异
        env_config.VERBOSE = False

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and train_config.USE_CUDA
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        print(f"✅ Evaluation Mode: 水平布局 + 单向路由 (h_uni)")
        print(f"⚠️  BIDIRECTIONAL = {env_config.BIDIRECTIONAL}")
        print(f"⚠️  AGV只能前进，不能后退")

        # 初始化环境
        self.env = PortEnvironment(env_config)
        self.num_agents = env_config.NUM_AGVS

        # 计算观察维度
        self.obs_dim = 7 + 5 * 4 + 6 + env_config.NUM_HORIZONTAL_LANES

        # 初始化模型
        self.actor_critic = ActorCritic(
            obs_dim=self.obs_dim,
            actor_hidden_dims=model_config.ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=model_config.CRITIC_HIDDEN_DIMS,
            num_lanes=env_config.NUM_HORIZONTAL_LANES,
            num_directions=2,
            use_centralized_critic=model_config.USE_CENTRALIZED_CRITIC
        )

        # 初始化MAPPO
        self.mappo = MAPPO(
            actor_critic=self.actor_critic,
            num_agents=self.num_agents,
            device=self.device
        )

        # 加载模型
        if os.path.exists(checkpoint_path):
            self.mappo.load(checkpoint_path)
            print(f"✅ 模型已加载: {checkpoint_path}")
        else:
            print(f"❌ 找不到检查点: {checkpoint_path}")
            sys.exit(1)

        # 评估结果（不记录后退使用率）
        self.eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'tasks_completed': [],
            'collisions': [],
            'direction_changes': [],
            # ✨ 单向模式不记录后退使用率
            'task_completion_times': []  # ✨ 新增：任务完成时间列表
        }

    def flatten_observation(self, obs_dict: dict) -> np.ndarray:
        """展平观察"""
        obs_list = []
        obs_list.append(obs_dict['own_state'])
        obs_list.append(obs_dict['nearby_agvs'].flatten())
        obs_list.append(obs_dict['task_info'])
        obs_list.append(obs_dict['path_occupancy'])
        return np.concatenate(obs_list, axis=0)

    def evaluate_episode(self, render: bool = False) -> dict:
        """评估单个episode"""
        obs_dict, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # 记录方向变化
        direction_changes = [0] * self.num_agents
        prev_directions = [agv.moving_forward for agv in self.env.agvs]

        while not done:
            # 收集观察
            obs_batch = []
            for i in range(self.num_agents):
                obs = self.flatten_observation(obs_dict[f'agent_{i}'])
                obs_batch.append(obs)

            obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)

            # 选择动作(确定性)
            actions_tensor, _, _ = self.mappo.select_action(
                obs_tensor, deterministic=True
            )

            # 转换为环境格式
            env_actions = {}
            for i in range(self.num_agents):
                env_actions[f'agent_{i}'] = {
                    'lane': actions_tensor['lane'][i].cpu().item(),
                    'direction': actions_tensor['direction'][i].cpu().item(),
                    'motion': actions_tensor['motion'][i].cpu().numpy()
                }

            # 步进
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(env_actions)

            # 统计方向变化（单向模式理论上不应该变化）
            for i, agv in enumerate(self.env.agvs):
                if agv.moving_forward != prev_directions[i]:
                    direction_changes[i] += 1
                prev_directions[i] = agv.moving_forward

            reward_batch = np.array([
                reward_dict[f'agent_{i}'] for i in range(self.num_agents)
            ])
            episode_reward += reward_batch.mean()
            episode_length += 1

            done = terminated_dict['__all__'] or truncated_dict['__all__']

            if render:
                self.env.render()

        # 收集统计信息
        info = info_dict.get('agent_0', {})

        return {
            'reward': episode_reward,
            'length': episode_length,
            'tasks_completed': info.get('completed_tasks', 0),
            'collisions': info.get('collisions', 0),
            'direction_changes': np.mean(direction_changes)
        }

    def evaluate(self, num_episodes: int = 100, render: bool = False):
        """运行完整评估"""
        print("\n" + "=" * 60)
        print("🎯 开始评估：水平单向 (h_uni)")
        print("=" * 60)
        print(f"评估轮数: {num_episodes}")
        print(f"AGV数量: {self.num_agents}")
        print(f"双向路由: 禁用（单向模式）")
        print("=" * 60 + "\n")

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            result = self.evaluate_episode(render=render)

            self.eval_results['episode_rewards'].append(result['reward'])
            self.eval_results['episode_lengths'].append(result['length'])
            self.eval_results['tasks_completed'].append(result['tasks_completed'])
            self.eval_results['collisions'].append(result['collisions'])
            self.eval_results['direction_changes'].append(result['direction_changes'])

        # 打印统计
        self.print_statistics()

        # 可视化结果
        self.visualize_results()

        # 保存结果
        self.save_results()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 评估结果统计 - 水平单向 (h_uni)")
        print("=" * 60)

        for key, values in self.eval_results.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)

                display_name = key.replace('_', ' ').title()
                print(f"\n{display_name}:")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}")
                print(f"  Min:  {min_val:.2f}")
                print(f"  Max:  {max_val:.2f}")

        # 特别强调关键指标
        tasks = self.eval_results['tasks_completed']
        if tasks:
            print(f"\n{'=' * 60}")
            print(f"⭐ 关键指标 - 单向路由限制")
            print(f"{'=' * 60}")
            print(f"  平均任务完成数: {np.mean(tasks):.2f}")
            print(f"  任务完成率: {sum(1 for t in tasks if t > 0) / len(tasks) * 100:.1f}%")
            print(f"  ⚠️  注意：AGV只能前进，无法后退避障")
            print(f"{'=' * 60}\n")

    def visualize_results(self):
        """可视化评估结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Evaluation Results - 水平单向 (h_uni)',
                     fontsize=16, fontweight='bold')

        # 1. Episode Rewards
        ax = axes[0, 0]
        ax.plot(self.eval_results['episode_rewards'], alpha=0.6)
        ax.axhline(np.mean(self.eval_results['episode_rewards']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Tasks Completed
        ax = axes[0, 1]
        ax.plot(self.eval_results['tasks_completed'], alpha=0.6,
                color='green', marker='o')
        ax.axhline(np.mean(self.eval_results['tasks_completed']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Tasks')
        ax.set_title('Tasks Completed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Direction Changes (应该接近0)
        ax = axes[0, 2]
        ax.plot(self.eval_results['direction_changes'], alpha=0.6,
                color='purple', marker='s')
        ax.axhline(np.mean(self.eval_results['direction_changes']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Direction Changes')
        ax.set_title('Direction Changes (Should be ~0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Episode Lengths
        ax = axes[1, 0]
        ax.plot(self.eval_results['episode_lengths'], alpha=0.6, color='orange')
        ax.axhline(np.mean(self.eval_results['episode_lengths']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length (steps)')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Collisions
        ax = axes[1, 1]
        ax.plot(self.eval_results['collisions'], alpha=0.6, color='red')
        ax.axhline(np.mean(self.eval_results['collisions']),
                   color='darkred', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Collisions')
        ax.set_title('Collision Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Task Distribution
        ax = axes[1, 2]
        ax.hist(self.eval_results['tasks_completed'], bins=20,
                alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(self.eval_results['tasks_completed']),
                   color='r', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Tasks Completed')
        ax.set_ylabel('Frequency')
        ax.set_title('Task Completion Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(train_config.LOG_DIR, 'evaluation_results_h_uni.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化结果已保存: {save_path}")

        plt.show()

    def save_results(self):
        """保存评估结果"""
        results = {
            'experiment': 'h_uni',
            'description': '水平布局 + 单向路由',
            'statistics': {
                key: {
                    'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
                    'std': float(np.std(values)) if len(values) > 0 else 0.0,
                    'min': float(np.min(values)) if len(values) > 0 else 0.0,
                    'max': float(np.max(values)) if len(values) > 0 else 0.0
                }
                for key, values in self.eval_results.items()
            },
            'raw_data': {
                key: [float(v) for v in values]
                for key, values in self.eval_results.items()
            },
            'config': {
                'num_agents': self.num_agents,
                'bidirectional': False,  # ✨ 单向
                'num_lanes': env_config.NUM_HORIZONTAL_LANES,
            }
        }

        save_path = os.path.join(train_config.LOG_DIR, 'evaluation_results_h_uni.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ 评估数据已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate h_uni Model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./data/checkpoints_h_uni/mappo_final_1000ep.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment'
    )

    args = parser.parse_args()

    # 运行评估
    evaluator = Evaluator(args.checkpoint)
    evaluator.evaluate(num_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()