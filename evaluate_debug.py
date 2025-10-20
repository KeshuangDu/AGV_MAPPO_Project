"""
评估程序 - 带详细调试信息
可以查看模型的行为和决策过程

运行方法：
    python evaluate_debug.py --checkpoint ./data/checkpoints_quick/mappo_final_100ep.pt --episodes 10 --verbose
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config import train_config
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class DebugEvaluator:
    """调试评估器类"""

    def __init__(self, checkpoint_path: str):
        """初始化评估器"""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and train_config.USE_CUDA
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        # 打印当前使用的配置
        print("\n" + "="*60)
        print("📋 当前配置")
        print("="*60)
        print(f"  - 任务管理器: {env_config.USE_TASK_MANAGER}")
        print(f"  - 奖励类型: {env_config.REWARD_TYPE}")
        print(f"  - 任务完成奖励: {env_config.REWARD_WEIGHTS['task_completion']}")
        print(f"  - 碰撞惩罚: {env_config.REWARD_WEIGHTS['collision']}")
        print(f"  - 到达阈值: {env_config.ARRIVAL_THRESHOLD}米")
        print("="*60 + "\n")

        # 初始化环境
        self.env = PortEnvironment(env_config)
        self.num_agents = env_config.NUM_AGVS

        # 计算观察维度
        self.obs_dim = 7 + 5*4 + 6 + env_config.NUM_HORIZONTAL_LANES

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
            print(f"❌ 错误：找不到检查点 {checkpoint_path}")
            print("\n可用的检查点：")
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                for cp in checkpoints:
                    print(f"  - {os.path.join(checkpoint_dir, cp)}")
            sys.exit(1)

        # 评估结果
        self.eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'tasks_completed': [],
            'collisions': [],
            'direction_changes': []
        }

    def flatten_observation(self, obs_dict: dict) -> np.ndarray:
        """展平观察"""
        obs_list = []
        obs_list.append(obs_dict['own_state'])
        obs_list.append(obs_dict['nearby_agvs'].flatten())
        obs_list.append(obs_dict['task_info'])
        obs_list.append(obs_dict['path_occupancy'])
        return np.concatenate(obs_list, axis=0)

    def evaluate_episode(self, render: bool = False, verbose: bool = False) -> dict:
        """评估单个episode，verbose=True时打印详细信息"""
        obs_dict, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # 记录AGV方向变化
        direction_changes = [0] * self.num_agents
        prev_directions = [agv.moving_forward for agv in self.env.agvs]

        # 记录任务完成情况
        initial_tasks = len(self.env.tasks)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎬 Episode 开始")
            print(f"{'='*60}")
            print(f"初始任务数: {initial_tasks}")
            print(f"AGV数量: {self.num_agents}")
            
            # 打印初始AGV状态
            print(f"\n📍 初始AGV位置:")
            for i, agv in enumerate(self.env.agvs):
                print(f"  AGV{i}: pos=({agv.position[0]:.1f}, {agv.position[1]:.1f})")
            
            # 打印初始任务
            print(f"\n📋 初始任务列表:")
            for task in self.env.tasks[:3]:  # 只打印前3个
                print(f"  Task{task.id}: {task.type}, "
                      f"pickup={task.pickup_location}, "
                      f"delivery={task.delivery_location}")
            if len(self.env.tasks) > 3:
                print(f"  ... (还有 {len(self.env.tasks)-3} 个任务)")

        step = 0
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

            # 统计方向变化
            for i, agv in enumerate(self.env.agvs):
                if agv.moving_forward != prev_directions[i]:
                    direction_changes[i] += 1
                prev_directions[i] = agv.moving_forward

            reward_batch = np.array([
                reward_dict[f'agent_{i}'] for i in range(self.num_agents)
            ])
            episode_reward += reward_batch.mean()
            episode_length += 1
            step += 1

            # 详细输出（每100步或任务完成时）
            if verbose and (step % 100 == 0 or len(self.env.completed_tasks) > 0):
                completed_now = len(self.env.completed_tasks)
                if step % 100 == 0 or completed_now > 0:
                    print(f"\n⏱️  Step {step}")
                    print(f"  完成任务: {completed_now}/{initial_tasks}")
                    print(f"  待完成任务: {len(self.env.tasks)}")
                    print(f"  当前奖励: {episode_reward:.2f}")
                    
                    # 打印每个AGV的状态
                    for i, agv in enumerate(self.env.agvs):
                        has_task = agv.current_task is not None
                        if has_task:
                            if not agv.has_container:
                                target = agv.current_task['pickup_location']
                                dist = np.linalg.norm(agv.position - target)
                                phase = "→pickup"
                            else:
                                target = agv.current_task['delivery_location']
                                dist = np.linalg.norm(agv.position - target)
                                phase = "→delivery"
                            
                            # 显示动作
                            action = env_actions[f'agent_{i}']
                            print(f"  AGV{i}: pos=({agv.position[0]:.1f},{agv.position[1]:.1f}), "
                                  f"{phase}, dist={dist:.1f}m, "
                                  f"lane={action['lane']}, dir={action['direction']}, "
                                  f"accel={action['motion'][0]:.2f}")
                        else:
                            print(f"  AGV{i}: pos=({agv.position[0]:.1f},{agv.position[1]:.1f}), idle")

            done = terminated_dict['__all__'] or truncated_dict['__all__']

            if render:
                self.env.render()

        # 收集统计信息
        info = info_dict.get('agent_0', {})
        completed_tasks = info.get('completed_tasks', 0)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🏁 Episode 结束")
            print(f"{'='*60}")
            print(f"运行步数: {episode_length}")
            print(f"完成任务数: {completed_tasks}")
            print(f"总奖励: {episode_reward:.2f}")
            print(f"碰撞次数: {info.get('collisions', 0)}")
            print(f"平均方向变化: {np.mean(direction_changes):.2f}")
            
            if completed_tasks == 0:
                print(f"\n⚠️  警告：没有完成任何任务！")
                print(f"   可能原因：")
                print(f"   1. 模型训练不足（需要更多训练轮数）")
                print(f"   2. 模型还没学会导航到目标")
                print(f"   3. 奖励设置可能需要调整")
            else:
                completion_rate = (completed_tasks / initial_tasks) * 100
                print(f"\n✅ 完成率: {completion_rate:.1f}%")
            print(f"{'='*60}\n")

        return {
            'reward': episode_reward,
            'length': episode_length,
            'tasks_completed': completed_tasks,
            'collisions': info.get('collisions', 0),
            'direction_changes': np.mean(direction_changes)
        }

    def evaluate(self, num_episodes: int = 100, render: bool = False, verbose: bool = False):
        """运行完整评估"""
        print("\n" + "="*60)
        print("🎯 开始评估")
        print("="*60)
        print(f"评估轮数: {num_episodes}")
        print(f"AGV数量: {self.num_agents}")
        print(f"详细输出: {verbose}")
        print("="*60 + "\n")

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # 前3个episode使用详细输出
            verbose_this_episode = verbose or (episode < 3)

            result = self.evaluate_episode(
                render=render,
                verbose=verbose_this_episode
            )

            self.eval_results['episode_rewards'].append(result['reward'])
            self.eval_results['episode_lengths'].append(result['length'])
            self.eval_results['tasks_completed'].append(result['tasks_completed'])
            self.eval_results['collisions'].append(result['collisions'])
            self.eval_results['direction_changes'].append(result['direction_changes'])

        # 计算统计量
        self.print_statistics()

        # 可视化结果
        self.visualize_results()

        # 保存结果
        self.save_results()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 评估结果统计")
        print("="*60)

        for key, values in self.eval_results.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)

                print(f"\n{key.replace('_', ' ').title()}:")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}")
                print(f"  Min:  {min_val:.2f}")
                print(f"  Max:  {max_val:.2f}")

        # 特别强调任务完成情况
        tasks_completed = self.eval_results['tasks_completed']
        if tasks_completed:
            total_tasks = sum(tasks_completed)
            print(f"\n{'='*60}")
            print(f"⭐ 关键指标 ⭐")
            print(f"{'='*60}")
            print(f"  总完成任务数: {total_tasks}")
            print(f"  平均每轮完成: {np.mean(tasks_completed):.2f}")

            if total_tasks == 0:
                print(f"\n❌ 警告：没有完成任何任务！")
                print(f"   这说明模型还没有学会任务完成策略")
                print(f"\n💡 建议：")
                print(f"   1. 训练更多轮次（至少1000轮）")
                print(f"   2. 检查奖励设置是否合理")
                print(f"   3. 考虑使用课程学习")
                print(f"   4. 增加密集奖励的引导性")
            else:
                completed_episodes = sum(1 for t in tasks_completed if t > 0)
                completion_rate = (completed_episodes / len(tasks_completed)) * 100
                print(f"  有任务完成的轮数: {completed_episodes}/{len(tasks_completed)}")
                print(f"  轮次完成率: {completion_rate:.1f}%")
                
                if np.mean(tasks_completed) < 1.0:
                    print(f"\n⚠️  任务完成数较低")
                    print(f"   建议继续训练以提高性能")
                else:
                    print(f"\n✅ 模型已经学会完成任务！")

        print("\n" + "="*60)

    def visualize_results(self):
        """可视化评估结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('MAPPO Evaluation Results - Debug Mode',
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

        # 2. Episode Lengths
        ax = axes[0, 1]
        ax.plot(self.eval_results['episode_lengths'], alpha=0.6, color='green')
        ax.axhline(np.mean(self.eval_results['episode_lengths']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length (steps)')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Tasks Completed  ✨ 重点关注
        ax = axes[0, 2]
        ax.plot(self.eval_results['tasks_completed'], alpha=0.6, color='orange', marker='o')
        ax.axhline(np.mean(self.eval_results['tasks_completed']),
                   color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Tasks')
        ax.set_title('Tasks Completed (KEY METRIC)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 如果全是0，添加警告文本
        if sum(self.eval_results['tasks_completed']) == 0:
            ax.text(0.5, 0.5, '⚠️ No Tasks Completed\nModel Needs More Training',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 4. Collisions
        ax = axes[1, 0]
        ax.plot(self.eval_results['collisions'], alpha=0.6, color='red')
        ax.axhline(np.mean(self.eval_results['collisions']),
                   color='darkred', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Collisions')
        ax.set_title('Collision Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Direction Changes
        ax = axes[1, 1]
        ax.plot(self.eval_results['direction_changes'], alpha=0.6, color='purple')
        ax.axhline(np.mean(self.eval_results['direction_changes']),
                   color='darkviolet', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Direction Changes')
        ax.set_title('Bidirectional Routing Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Task Completion Distribution
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

        save_path = os.path.join(train_config.LOG_DIR, 'evaluation_results_debug.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化结果已保存: {save_path}")

        plt.show()

    def save_results(self):
        """保存评估结果"""
        results = {
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
                'bidirectional': env_config.BIDIRECTIONAL,
                'num_lanes': env_config.NUM_HORIZONTAL_LANES,
                'use_task_manager': env_config.USE_TASK_MANAGER,
                'reward_type': env_config.REWARD_TYPE
            }
        }

        save_path = os.path.join(train_config.LOG_DIR, 'evaluation_results_debug.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ 评估数据已保存: {save_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate MAPPO Model - Debug Mode')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./data/checkpoints_quick/mappo_final_100ep.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for all episodes'
    )

    args = parser.parse_args()

    # 检查检查点是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误：找不到检查点 {args.checkpoint}")
        print("\n可用的检查点：")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            for cp in checkpoints:
                print(f"  - {os.path.join(checkpoint_dir, cp)}")
        return

    # 运行评估
    evaluator = DebugEvaluator(args.checkpoint)
    evaluator.evaluate(
        num_episodes=args.episodes,
        render=args.render,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
