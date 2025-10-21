"""
实验评估脚本 v2.2 - 添加任务完成时间追踪
适用于 h_bi 和 h_uni

✨ 新增功能：
1. 收集任务完成时间
2. 统计平均完成时间、最短/最长时间
3. 可视化任务时间分布
4. 对比任务完成效率

使用方法：
    # 双向评估
    python evaluate_h_bi_v2.py --checkpoint xxx.pt --episodes 50

    # 单向评估（复制并改名为 evaluate_h_uni_v2.py）
    python evaluate_h_uni_v2.py --checkpoint xxx.pt --episodes 50
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
from config.train_config_h_bi import train_config  # ✨ 改为 h_uni 时修改这里
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class EvaluatorV2:
    """评估器 v2.2 - 带任务时间追踪"""

    def __init__(self, checkpoint_path: str, experiment_name: str = "h_bi"):
        """初始化评估器"""
        self.experiment_name = experiment_name

        # 设置双向/单向模式
        env_config.BIDIRECTIONAL = (experiment_name == "h_bi")
        env_config.VERBOSE = False

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and train_config.USE_CUDA
            else 'cpu'
        )

        mode_name = "双向路由" if env_config.BIDIRECTIONAL else "单向路由"
        print(f"Using device: {self.device}")
        print(f"✅ Evaluation Mode: 水平布局 + {mode_name} ({experiment_name})")
        print(f"✅ BIDIRECTIONAL = {env_config.BIDIRECTIONAL}")

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

        # ✨ 评估结果（新增任务时间）
        self.eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'tasks_completed': [],
            'collisions': [],
            'direction_changes': [],
            'backward_usage': [] if env_config.BIDIRECTIONAL else None,
            'task_completion_times': []  # ✨ 新增：所有任务的完成时间
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

        # 记录方向变化和后退使用
        direction_changes = [0] * self.num_agents
        backward_steps = [0] * self.num_agents if env_config.BIDIRECTIONAL else None
        total_steps = 0
        prev_directions = [agv.moving_forward for agv in self.env.agvs]

        # ✨ 新增：记录本episode的任务完成时间
        episode_task_times = []
        recorded_task_ids = set()  # 防止重复记录

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

            # ✨ 新增：收集已完成任务的时间
            for task in self.env.completed_tasks:
                if (hasattr(task, 'completion_time') and
                        task.completion_time is not None and
                        task.id not in recorded_task_ids):
                    episode_task_times.append(task.completion_time)
                    recorded_task_ids.add(task.id)

            # 统计方向变化和后退使用
            for i, agv in enumerate(self.env.agvs):
                if agv.moving_forward != prev_directions[i]:
                    direction_changes[i] += 1
                if env_config.BIDIRECTIONAL and not agv.moving_forward:
                    backward_steps[i] += 1
                prev_directions[i] = agv.moving_forward

            reward_batch = np.array([
                reward_dict[f'agent_{i}'] for i in range(self.num_agents)
            ])
            episode_reward += reward_batch.mean()
            episode_length += 1
            total_steps += 1

            done = terminated_dict['__all__'] or truncated_dict['__all__']

            if render:
                self.env.render()

        # 收集统计信息
        info = info_dict.get('agent_0', {})

        result = {
            'reward': episode_reward,
            'length': episode_length,
            'tasks_completed': info.get('completed_tasks', 0),
            'collisions': info.get('collisions', 0),
            'direction_changes': np.mean(direction_changes),
            'task_times': episode_task_times  # ✨ 新增
        }

        if env_config.BIDIRECTIONAL:
            backward_usage = np.mean([b / max(total_steps, 1) for b in backward_steps]) * 100
            result['backward_usage'] = backward_usage

        return result

    def evaluate(self, num_episodes: int = 100, render: bool = False):
        """运行完整评估"""
        mode_name = "双向" if env_config.BIDIRECTIONAL else "单向"
        print("\n" + "=" * 60)
        print(f"🎯 开始评估：水平{mode_name} ({self.experiment_name})")
        print("=" * 60)
        print(f"评估轮数: {num_episodes}")
        print(f"AGV数量: {self.num_agents}")
        print(f"双向路由: {'启用' if env_config.BIDIRECTIONAL else '禁用'}")
        print("=" * 60 + "\n")

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            result = self.evaluate_episode(render=render)

            self.eval_results['episode_rewards'].append(result['reward'])
            self.eval_results['episode_lengths'].append(result['length'])
            self.eval_results['tasks_completed'].append(result['tasks_completed'])
            self.eval_results['collisions'].append(result['collisions'])
            self.eval_results['direction_changes'].append(result['direction_changes'])

            # ✨ 收集所有任务完成时间
            if result['task_times']:
                self.eval_results['task_completion_times'].extend(result['task_times'])

            if env_config.BIDIRECTIONAL:
                self.eval_results['backward_usage'].append(result['backward_usage'])

        # 打印统计
        self.print_statistics()

        # 可视化结果
        self.visualize_results()

        # 保存结果
        self.save_results()

    def print_statistics(self):
        """打印统计信息"""
        mode_name = "双向" if env_config.BIDIRECTIONAL else "单向"
        print("\n" + "=" * 60)
        print(f"📊 评估结果统计 - 水平{mode_name} ({self.experiment_name})")
        print("=" * 60)

        for key, values in self.eval_results.items():
            if values is None or len(values) == 0:
                continue

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

        # ✨ 特别强调任务完成时间
        task_times = self.eval_results['task_completion_times']
        if task_times:
            print(f"\n{'=' * 60}")
            print(f"⏱️  任务完成时间分析 ✨")
            print(f"{'=' * 60}")
            print(f"  总完成任务数: {len(task_times)}")
            print(f"  平均完成时间: {np.mean(task_times):.1f}秒")
            print(f"  标准差: {np.std(task_times):.1f}秒")
            print(f"  最短时间: {np.min(task_times):.1f}秒")
            print(f"  最长时间: {np.max(task_times):.1f}秒")
            print(f"  中位数: {np.median(task_times):.1f}秒")

            # 时间分布统计
            fast_tasks = sum(1 for t in task_times if t < 100)
            medium_tasks = sum(1 for t in task_times if 100 <= t < 200)
            slow_tasks = sum(1 for t in task_times if t >= 200)

            print(f"\n  时间分布:")
            print(f"    快速(<100s): {fast_tasks} ({fast_tasks / len(task_times) * 100:.1f}%)")
            print(f"    中等(100-200s): {medium_tasks} ({medium_tasks / len(task_times) * 100:.1f}%)")
            print(f"    慢速(>=200s): {slow_tasks} ({slow_tasks / len(task_times) * 100:.1f}%)")
            print(f"{'=' * 60}\n")

    def visualize_results(self):
        """可视化评估结果 - 包含任务时间图表"""
        mode_name = "双向" if env_config.BIDIRECTIONAL else "单向"

        # ✨ 调整为2x4布局（8个图）
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Evaluation Results - 水平{mode_name} ({self.experiment_name}) v2.2',
                     fontsize=16, fontweight='bold')

        # 前6个图保持不变...
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

        # 3. Episode Lengths
        ax = axes[0, 2]
        ax.plot(self.eval_results['episode_lengths'], alpha=0.6, color='orange')
        ax.axhline(np.mean(self.eval_results['episode_lengths']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length (steps)')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Collisions
        ax = axes[0, 3]
        ax.plot(self.eval_results['collisions'], alpha=0.6, color='red')
        ax.axhline(np.mean(self.eval_results['collisions']),
                   color='darkred', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Collisions')
        ax.set_title('Collision Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Direction Changes or Backward Usage
        ax = axes[1, 0]
        if env_config.BIDIRECTIONAL and self.eval_results['backward_usage']:
            ax.plot(self.eval_results['backward_usage'], alpha=0.6,
                    color='purple', marker='s')
            ax.axhline(np.mean(self.eval_results['backward_usage']),
                       color='r', linestyle='--', label='Mean')
            ax.set_ylabel('Backward Usage (%)')
            ax.set_title('Backward Usage Rate')
        else:
            ax.plot(self.eval_results['direction_changes'], alpha=0.6, color='purple')
            ax.axhline(np.mean(self.eval_results['direction_changes']),
                       color='r', linestyle='--', label='Mean')
            ax.set_ylabel('Direction Changes')
            ax.set_title('Direction Changes')
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Task Distribution
        ax = axes[1, 1]
        ax.hist(self.eval_results['tasks_completed'], bins=20,
                alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(self.eval_results['tasks_completed']),
                   color='r', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Tasks Completed')
        ax.set_ylabel('Frequency')
        ax.set_title('Task Completion Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ✨ 7. 任务完成时间分布（新增）
        ax = axes[1, 2]
        task_times = self.eval_results['task_completion_times']
        if task_times:
            ax.hist(task_times, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.mean(task_times), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(task_times):.1f}s')
            ax.axvline(np.median(task_times), color='orange', linestyle=':',
                       linewidth=2, label=f'Median: {np.median(task_times):.1f}s')
            ax.set_xlabel('Task Completion Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('⏱️  Task Time Distribution ✨')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Task Time Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        # ✨ 8. 任务完成时间箱线图（新增）
        ax = axes[1, 3]
        if task_times:
            bp = ax.boxplot([task_times], labels=['Task Time'],
                            patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('lightgreen')
            ax.set_ylabel('Task Completion Time (seconds)')
            ax.set_title('⏱️  Task Time Statistics ✨')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加统计信息文本
            stats_text = f"Mean: {np.mean(task_times):.1f}s\n"
            stats_text += f"Median: {np.median(task_times):.1f}s\n"
            stats_text += f"Min: {np.min(task_times):.1f}s\n"
            stats_text += f"Max: {np.max(task_times):.1f}s"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No Task Time Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        plt.tight_layout()

        save_path = os.path.join(train_config.LOG_DIR, f'evaluation_results_{self.experiment_name}_v2.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化结果已保存: {save_path}")

        plt.show()

    def save_results(self):
        """保存评估结果"""
        mode_name = "双向" if env_config.BIDIRECTIONAL else "单向"

        # 处理可能为None的值
        processed_results = {}
        for key, values in self.eval_results.items():
            if values is None or len(values) == 0:
                continue
            processed_results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        results = {
            'experiment': self.experiment_name,
            'description': f'水平布局 + {mode_name}',
            'statistics': processed_results,
            'raw_data': {
                key: [float(v) for v in values]
                for key, values in self.eval_results.items()
                if values is not None and len(values) > 0
            },
            'config': {
                'num_agents': self.num_agents,
                'bidirectional': env_config.BIDIRECTIONAL,
                'num_lanes': env_config.NUM_HORIZONTAL_LANES,
            }
        }

        save_path = os.path.join(train_config.LOG_DIR, f'evaluation_results_{self.experiment_name}_v2.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ 评估数据已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate Model with Task Time Tracking (v2.2)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment'
    )
    parser.add_argument(
        '--exp',
        type=str,
        default='h_bi',
        choices=['h_bi', 'h_uni'],
        help='Experiment name (h_bi or h_uni)'
    )

    args = parser.parse_args()

    # 运行评估
    evaluator = EvaluatorV2(args.checkpoint, experiment_name=args.exp)
    evaluator.evaluate(num_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()