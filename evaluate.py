"""
AGV MAPPO 评估程序
评估训练好的模型性能

使用方法：
    # 标准评估
    python evaluate.py --checkpoint ./data/checkpoints/mappo_final_standard.pt --episodes 100

    # 详细输出（调试模式）
    python evaluate.py --checkpoint ./data/checkpoints/mappo_episode_500.pt --episodes 10 --verbose

    # 保存结果
    python evaluate.py --checkpoint <path> --episodes 50 --save-results
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config import train_config
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Evaluator:
    """评估器类"""

    def __init__(self, checkpoint_path: str, verbose: bool = False):
        """
        初始化评估器

        Args:
            checkpoint_path: 模型检查点路径
            verbose: 是否打印详细信息
        """
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and train_config.USE_CUDA
            else 'cpu'
        )
        print(f"🖥️  Using device: {self.device}")

        if self.verbose:
            print("\n📋 Configuration:")
            print(f"   Task Manager: {env_config.USE_TASK_MANAGER}")
            print(f"   Reward Type: {env_config.REWARD_TYPE}")
            print(f"   Bidirectional: {env_config.BIDIRECTIONAL}")
            print(f"   Arrival Threshold: {env_config.ARRIVAL_THRESHOLD}m")

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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.actor_critic.eval()
            print(f"✅ Model loaded from: {checkpoint_path}")
            print(f"   Trained episodes: {checkpoint.get('episode', 'Unknown')}")
        else:
            raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    def evaluate(self, num_episodes: int = 100):
        """
        评估模型

        Args:
            num_episodes: 评估回合数
        """
        print(f"\n{'='*60}")
        print(f"🎯 Starting Evaluation: {num_episodes} episodes")
        print(f"{'='*60}\n")

        # 统计数据
        episode_rewards = []
        episode_lengths = []
        episode_tasks_completed = []
        episode_collisions = []

        all_distances = []  # 记录所有AGV到目标的距离

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs, info = self.env.reset()
            episode_reward = {f'agent_{i}': 0 for i in range(self.num_agents)}
            episode_length = 0
            done = False

            # 记录本episode的详细数据
            step_data = []

            while not done and episode_length < env_config.MAX_EPISODE_STEPS:
                # 选择动作
                actions = {}
                step_info = {'step': episode_length}

                for agent_id in range(self.num_agents):
                    agent_name = f'agent_{agent_id}'
                    obs_tensor = self._process_obs(obs[agent_name])

                    with torch.no_grad():
                        action, _, _ = self.mappo.select_action(obs_tensor)

                    actions[agent_name] = action

                    # 记录详细信息（verbose模式）
                    if self.verbose and episode_length % 100 == 0:
                        agv = self.env.agvs[agent_id]
                        step_info[f'agv{agent_id}'] = {
                            'position': agv.position,
                            'velocity': agv.velocity,
                            'task_status': agv.task_status,
                            'distance_to_target': obs[agent_name]['task_info'][4],
                            'accel': action['motion'][0],
                            'steering': action['motion'][1]
                        }
                        all_distances.append(obs[agent_name]['task_info'][4])

                # 环境步进
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                # 累计奖励
                for agent_id in range(self.num_agents):
                    agent_name = f'agent_{agent_id}'
                    episode_reward[agent_name] += rewards[agent_name]

                obs = next_obs
                episode_length += 1

                if self.verbose and episode_length % 100 == 0 and step_info:
                    step_data.append(step_info)

            # 记录统计
            avg_reward = np.mean([episode_reward[f'agent_{i}']
                                 for i in range(self.num_agents)])
            episode_rewards.append(avg_reward)
            episode_lengths.append(episode_length)
            episode_tasks_completed.append(len(self.env.completed_tasks))
            episode_collisions.append(self.env.episode_stats['collisions'])

            # 打印详细信息（verbose模式）
            if self.verbose:
                print(f"\n{'─'*60}")
                print(f"📊 Episode {episode + 1} Summary:")
                print(f"   Reward: {avg_reward:.2f}")
                print(f"   Length: {episode_length} steps")
                print(f"   Tasks Completed: {len(self.env.completed_tasks)}")
                print(f"   Collisions: {self.env.episode_stats['collisions']}")

                if step_data and len(step_data) > 0:
                    print(f"\n   Step Details (every 100 steps):")
                    for data in step_data[-3:]:  # 显示最后3个采样点
                        print(f"   Step {data['step']}:")
                        for agv_key, agv_info in data.items():
                            if agv_key != 'step':
                                print(f"      {agv_key}: pos={agv_info['position']}, "
                                      f"dist={agv_info['distance_to_target']:.1f}m, "
                                      f"accel={agv_info['accel']:.2f}")

        # 计算统计结果
        results = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'avg_tasks_completed': np.mean(episode_tasks_completed),
            'avg_collisions': np.mean(episode_collisions),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'total_tasks': np.sum(episode_tasks_completed),
        }

        if all_distances:
            results['avg_distance_to_target'] = np.mean(all_distances)
            results['min_distance_to_target'] = np.min(all_distances)

        # 打印结果
        self._print_results(results)

        return results, {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'tasks': episode_tasks_completed,
            'collisions': episode_collisions
        }

    def _process_obs(self, obs):
        """处理观察值"""
        obs_array = np.concatenate([
            obs['own_state'],
            obs['nearby_agvs'].flatten(),
            obs['task_info'],
            obs['path_occupancy']
        ])
        return torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)

    def _print_results(self, results):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"📈 Evaluation Results ({results['num_episodes']} episodes)")
        print(f"{'='*60}")
        print(f"  Average Reward:        {results['avg_reward']:>10.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Episode Length: {results['avg_length']:>10.0f} steps")
        print(f"  Average Tasks Completed: {results['avg_tasks_completed']:>10.2f}")
        print(f"  Total Tasks Completed:  {results['total_tasks']:>10.0f}")
        print(f"  Average Collisions:     {results['avg_collisions']:>10.2f}")
        print(f"  Reward Range:          [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")

        if 'avg_distance_to_target' in results:
            print(f"  Avg Distance to Target: {results['avg_distance_to_target']:>10.2f}m")
            print(f"  Min Distance to Target: {results['min_distance_to_target']:>10.2f}m")

        print(f"{'='*60}\n")

    def save_results(self, results, details, output_dir='./evaluation_results'):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON
        json_path = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"💾 Results saved to: {json_path}")

        # 绘制图表
        self._plot_results(details, output_dir, timestamp)

    def _plot_results(self, details, output_dir, timestamp):
        """绘制评估结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励分布
        axes[0, 0].hist(details['rewards'], bins=30, edgecolor='black')
        axes[0, 0].set_title('Episode Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')

        # Episode长度
        axes[0, 1].plot(details['lengths'])
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # 完成任务数
        axes[1, 0].plot(details['tasks'])
        axes[1, 0].set_title('Tasks Completed per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Tasks')

        # 碰撞次数
        axes[1, 1].plot(details['collisions'])
        axes[1, 1].set_title('Collisions per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Collisions')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Plots saved to: {plot_path}")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AGV MAPPO Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')

    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed step-by-step information')

    parser.add_argument('--save-results', action='store_true',
                       help='Save evaluation results to file')

    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        verbose=args.verbose
    )

    # 执行评估
    results, details = evaluator.evaluate(num_episodes=args.episodes)

    # 保存结果
    if args.save_results:
        evaluator.save_results(results, details)


if __name__ == '__main__':
    main()