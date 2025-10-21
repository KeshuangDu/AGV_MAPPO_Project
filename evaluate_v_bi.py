"""
垂直布局 + 双向路由 评估脚本
Vertical Layout + Bidirectional Routing Evaluation

运行方法：
    python evaluate_v_bi.py --checkpoint ./data/checkpoints_v_bi/mappo_episode_100.pt --episodes 50
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config_v_bi import train_config
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Evaluator:
    """评估器类"""

    def __init__(self, checkpoint_path: str, experiment_name: str = 'v_bi'):
        """初始化评估器"""

        self.experiment_name = experiment_name

        # ✨ 设置环境为垂直+双向模式
        env_config.LAYOUT_TYPE = 'vertical'
        env_config.BIDIRECTIONAL = True

        print(f"\n{'='*60}")
        print(f"📊 评估实验: 垂直布局 + 双向路由 (v_bi)")
        print(f"{'='*60}")
        print(f"⚙️  LAYOUT_TYPE = {env_config.LAYOUT_TYPE}")
        print(f"⚙️  BIDIRECTIONAL = {env_config.BIDIRECTIONAL}")
        print(f"{'='*60}\n")

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化环境
        self.env = PortEnvironment(env_config)
        self.num_agents = env_config.NUM_AGVS

        # 计算观察维度
        obs_dict, _ = self.env.reset()
        sample_obs = obs_dict['agent_0']
        self.obs_dim = (
            sample_obs['own_state'].shape[0] +
            sample_obs['nearby_agvs'].size +
            sample_obs['task_info'].shape[0] +
            sample_obs['path_occupancy'].shape[0]
        )

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

        # 评估结果（双向模式包含后退使用率）
        self.eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'tasks_completed': [],
            'collisions': [],
            'direction_changes': [],
            'backward_usage': []  # ✨ 双向模式特有
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
        backward_steps = [0] * self.num_agents
        total_steps = 0
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

            # ✅ 修复：转换为环境格式，使用motion
            env_actions = {}
            for i in range(self.num_agents):
                env_actions[f'agent_{i}'] = {
                    'lane': actions_tensor['lane'][i].cpu().item(),
                    'direction': actions_tensor['direction'][i].cpu().item(),
                    'motion': actions_tensor['motion'][i].cpu().numpy()
                }

            # 环境步进
            next_obs_dict, rewards_dict, terminated, truncated, info = self.env.step(env_actions)
            done = terminated['__all__'] or truncated['__all__']

            # 统计方向变化和后退使用
            for i in range(self.num_agents):
                curr_direction = self.env.agvs[i].moving_forward
                if curr_direction != prev_directions[i]:
                    direction_changes[i] += 1
                if not curr_direction:
                    backward_steps[i] += 1
                prev_directions[i] = curr_direction

            # 累计奖励
            rewards = np.array([rewards_dict[f'agent_{i}'] for i in range(self.num_agents)])
            episode_reward += rewards.mean()

            obs_dict = next_obs_dict
            episode_length += 1
            total_steps += 1

            if render:
                self.env.render()

        # 计算后退使用率
        backward_usage = [backward_steps[i] / total_steps if total_steps > 0 else 0
                          for i in range(self.num_agents)]

        return {
            'reward': episode_reward,
            'length': episode_length,
            'tasks_completed': info.get('tasks_completed', 0),
            'collisions': info.get('collisions', 0),
            'direction_changes': direction_changes,
            'backward_usage': backward_usage
        }

    def evaluate(self, num_episodes: int = 50, render: bool = False):
        """运行评估"""
        print(f"开始评估 - {num_episodes} episodes\n")

        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            result = self.evaluate_episode(render=render)

            self.eval_results['episode_rewards'].append(result['reward'])
            self.eval_results['episode_lengths'].append(result['length'])
            self.eval_results['tasks_completed'].append(result['tasks_completed'])
            self.eval_results['collisions'].append(result['collisions'])
            self.eval_results['direction_changes'].append(np.mean(result['direction_changes']))
            self.eval_results['backward_usage'].append(np.mean(result['backward_usage']))

        self.print_results()
        self.save_results()
        self.plot_results()

    def print_results(self):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"📊 评估结果统计")
        print(f"{'='*60}")
        print(f"平均奖励: {np.mean(self.eval_results['episode_rewards']):.2f}")
        print(f"平均步数: {np.mean(self.eval_results['episode_lengths']):.2f}")
        print(f"平均任务完成数: {np.mean(self.eval_results['tasks_completed']):.2f}")
        print(f"平均碰撞次数: {np.mean(self.eval_results['collisions']):.2f}")
        print(f"平均方向变化: {np.mean(self.eval_results['direction_changes']):.2f}次")
        print(f"✨ 平均后退使用率: {np.mean(self.eval_results['backward_usage'])*100:.2f}%")
        print(f"{'='*60}")

    def save_results(self):
        """保存评估结果"""
        results_dir = f'./data/eval_results_{self.experiment_name}'
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, 'eval_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'mean_reward': float(np.mean(self.eval_results['episode_rewards'])),
                'std_reward': float(np.std(self.eval_results['episode_rewards'])),
                'mean_length': float(np.mean(self.eval_results['episode_lengths'])),
                'mean_tasks': float(np.mean(self.eval_results['tasks_completed'])),
                'mean_collisions': float(np.mean(self.eval_results['collisions'])),
                'mean_direction_changes': float(np.mean(self.eval_results['direction_changes'])),
                'mean_backward_usage': float(np.mean(self.eval_results['backward_usage'])),
                'all_results': {k: [float(x) for x in v] for k, v in self.eval_results.items()}
            }, f, indent=2)

        print(f"\n📊 结果已保存到: {results_path}")

    def plot_results(self):
        """绘制评估结果"""
        results_dir = f'./data/eval_results_{self.experiment_name}'
        os.makedirs(results_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # 奖励分布
        axes[0].hist(self.eval_results['episode_rewards'], bins=20, edgecolor='black')
        axes[0].set_title('Episode Rewards Distribution')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')

        # 任务完成数
        axes[1].hist(self.eval_results['tasks_completed'], bins=20, edgecolor='black')
        axes[1].set_title('Tasks Completed Distribution')
        axes[1].set_xlabel('Tasks')
        axes[1].set_ylabel('Frequency')

        # 碰撞次数
        axes[2].hist(self.eval_results['collisions'], bins=20, edgecolor='black')
        axes[2].set_title('Collisions Distribution')
        axes[2].set_xlabel('Collisions')
        axes[2].set_ylabel('Frequency')

        # Episode长度
        axes[3].hist(self.eval_results['episode_lengths'], bins=20, edgecolor='black')
        axes[3].set_title('Episode Length Distribution')
        axes[3].set_xlabel('Steps')
        axes[3].set_ylabel('Frequency')

        # ✨ 后退使用率分布
        axes[4].hist(np.array(self.eval_results['backward_usage'])*100, bins=20,
                    edgecolor='black', color='purple', alpha=0.7)
        axes[4].set_title('Backward Usage Distribution')
        axes[4].set_xlabel('Backward Usage (%)')
        axes[4].set_ylabel('Frequency')

        # 方向变化
        axes[5].hist(self.eval_results['direction_changes'], bins=20, edgecolor='black')
        axes[5].set_title('Direction Changes Distribution')
        axes[5].set_xlabel('Changes')
        axes[5].set_ylabel('Frequency')

        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'eval_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存到: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估垂直布局+双向路由模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--episodes', type=int, default=50, help='评估轮数')
    parser.add_argument('--render', action='store_true', help='是否渲染')

    args = parser.parse_args()

    evaluator = Evaluator(checkpoint_path=args.checkpoint, experiment_name='v_bi')
    evaluator.evaluate(num_episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()