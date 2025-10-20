"""
AGV MAPPO 训练主程序
支持多种训练模式：quick(100轮)/medium(1000轮)/standard(5000轮)

使用方法：
    # 快速测试
    python train.py --mode quick

    # 中等规模训练
    python train.py --mode medium

    # 完整训练
    python train.py --mode standard

    # 自定义轮数
    python train.py --episodes 500

    # 从检查点恢复
    python train.py --mode medium --resume ./data/checkpoints/mappo_episode_500.pt
"""

import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime
import pickle
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config import train_config
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Trainer:
    """训练器类"""

    def __init__(self, mode='standard', custom_episodes=None, resume_path=None):
        """
        初始化训练器

        Args:
            mode: 训练模式 ('quick', 'medium', 'standard')
            custom_episodes: 自定义训练轮数
            resume_path: 恢复训练的检查点路径
        """
        self.mode = mode
        self.resume_path = resume_path

        # 根据模式调整配置
        self.config = self._get_config(mode, custom_episodes)

        # 设置随机种子
        self.set_seed(self.config['seed'])

        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['use_cuda']
            else 'cpu'
        )
        print(f"🖥️  Using device: {self.device}")
        print(f"🎯 Training mode: {mode}")
        print(f"📊 Total episodes: {self.config['num_episodes']}")

        # 创建目录
        self.create_directories()

        # 初始化环境
        self.env = PortEnvironment(env_config)
        self.num_agents = env_config.NUM_AGVS

        # 计算观察维度
        self.obs_dim = self.calculate_obs_dim()

        # 初始化模型
        self.actor_critic = ActorCritic(
            obs_dim=self.obs_dim,
            actor_hidden_dims=model_config.ACTOR_HIDDEN_DIMS,
            critic_hidden_dims=model_config.CRITIC_HIDDEN_DIMS,
            num_lanes=env_config.NUM_HORIZONTAL_LANES,
            num_directions=2,
            use_centralized_critic=model_config.USE_CENTRALIZED_CRITIC
        )

        # 初始化MAPPO算法
        self.mappo = MAPPO(
            actor_critic=self.actor_critic,
            num_agents=self.num_agents,
            lr_actor=self.config['actor_lr'],
            lr_critic=self.config['critic_lr'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_epsilon=self.config['clip_epsilon'],
            value_loss_coef=self.config['value_loss_coef'],
            entropy_coef=self.config['entropy_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            ppo_epochs=self.config['ppo_epochs'],
            device=self.device
        )

        # TensorBoard
        if self.config['use_tensorboard']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(self.config['log_dir'], f"{mode}_{timestamp}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"📈 TensorBoard logging to: {log_dir}")
        else:
            self.writer = None

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.tasks_completed = []

        # 恢复训练
        self.start_episode = 0
        if self.resume_path:
            self.load_checkpoint(self.resume_path)

    def _get_config(self, mode, custom_episodes):
        """根据模式获取配置"""
        config = {
            'seed': train_config.SEED,
            'use_cuda': train_config.USE_CUDA,
            'actor_lr': train_config.ACTOR_LR,
            'critic_lr': train_config.CRITIC_LR,
            'gamma': train_config.GAMMA,
            'gae_lambda': train_config.GAE_LAMBDA,
            'clip_epsilon': train_config.CLIP_EPSILON,
            'value_loss_coef': train_config.VALUE_LOSS_COEF,
            'entropy_coef': train_config.ENTROPY_COEF,
            'max_grad_norm': train_config.MAX_GRAD_NORM,
            'ppo_epochs': train_config.PPO_EPOCHS,
            'use_tensorboard': train_config.USE_TENSORBOARD,
        }

        # 根据模式设置参数
        if mode == 'quick':
            config.update({
                'num_episodes': 100,
                'save_interval': 20,
                'log_dir': './runs/quick',
                'checkpoint_dir': './data/checkpoints_quick'
            })
        elif mode == 'medium':
            config.update({
                'num_episodes': 1000,
                'save_interval': 100,
                'log_dir': './runs/medium',
                'checkpoint_dir': './data/checkpoints_medium'
            })
        else:  # standard
            config.update({
                'num_episodes': 5000,
                'save_interval': 500,
                'log_dir': './runs/standard',
                'checkpoint_dir': './data/checkpoints'
            })

        # 自定义轮数覆盖
        if custom_episodes:
            config['num_episodes'] = custom_episodes

        return config

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs('./data/logs', exist_ok=True)

    def calculate_obs_dim(self):
        """计算观察空间维度"""
        return 7 + 5 * 4 + 6 + env_config.NUM_HORIZONTAL_LANES

    def train(self):
        """训练主循环"""
        print("\n" + "="*60)
        print("🚀 Starting Training...")
        print("="*60 + "\n")

        for episode in range(self.start_episode, self.config['num_episodes']):
            obs, info = self.env.reset()
            episode_reward = {f'agent_{i}': 0 for i in range(self.num_agents)}
            episode_length = 0
            done = False

            # 收集轨迹
            trajectories = {f'agent_{i}': {
                'obs': [], 'actions': [], 'rewards': [],
                'values': [], 'log_probs': [], 'dones': []
            } for i in range(self.num_agents)}

            # Episode循环
            with tqdm(total=env_config.MAX_EPISODE_STEPS,
                     desc=f"Episode {episode+1}/{self.config['num_episodes']}",
                     leave=False) as pbar:

                while not done and episode_length < env_config.MAX_EPISODE_STEPS:
                    # 选择动作
                    actions = {}
                    for agent_id in range(self.num_agents):
                        agent_name = f'agent_{agent_id}'
                        obs_tensor = self._process_obs(obs[agent_name])

                        with torch.no_grad():
                            action, value, log_prob = self.mappo.select_action(obs_tensor)

                        actions[agent_name] = action

                        # 记录轨迹
                        trajectories[agent_name]['obs'].append(obs[agent_name])
                        trajectories[agent_name]['actions'].append(action)
                        trajectories[agent_name]['values'].append(value)
                        trajectories[agent_name]['log_probs'].append(log_prob)

                    # 环境步进
                    next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                    done = terminated or truncated

                    # 记录奖励和done
                    for agent_id in range(self.num_agents):
                        agent_name = f'agent_{agent_id}'
                        trajectories[agent_name]['rewards'].append(rewards[agent_name])
                        trajectories[agent_name]['dones'].append(done)
                        episode_reward[agent_name] += rewards[agent_name]

                    obs = next_obs
                    episode_length += 1
                    pbar.update(1)

            # 更新策略
            losses = self.mappo.update(trajectories)

            # 记录统计
            avg_reward = np.mean([episode_reward[f'agent_{i}']
                                 for i in range(self.num_agents)])
            self.episode_rewards.append(avg_reward)
            self.episode_lengths.append(episode_length)

            tasks_done = len(self.env.completed_tasks)
            self.tasks_completed.append(tasks_done)

            # 打印信息
            if (episode + 1) % 10 == 0:
                print(f"\n📊 Episode {episode+1}")
                print(f"   Reward: {avg_reward:.2f}")
                print(f"   Length: {episode_length}")
                print(f"   Tasks: {tasks_done}")
                print(f"   Actor Loss: {losses['actor_loss']:.4f}")
                print(f"   Critic Loss: {losses['critic_loss']:.4f}")

            # TensorBoard记录
            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', avg_reward, episode)
                self.writer.add_scalar('Train/EpisodeLength', episode_length, episode)
                self.writer.add_scalar('Train/TasksCompleted', tasks_done, episode)
                self.writer.add_scalar('Train/ActorLoss', losses['actor_loss'], episode)
                self.writer.add_scalar('Train/CriticLoss', losses['critic_loss'], episode)
                self.writer.add_scalar('Train/Entropy', losses['entropy'], episode)

            # 保存检查点
            if (episode + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(episode + 1)

        # 保存最终模型
        self.save_checkpoint(self.config['num_episodes'], final=True)

        print("\n" + "="*60)
        print("✅ Training Completed!")
        print("="*60)

        if self.writer:
            self.writer.close()

    def _process_obs(self, obs):
        """处理观察值"""
        obs_array = np.concatenate([
            obs['own_state'],
            obs['nearby_agvs'].flatten(),
            obs['task_info'],
            obs['path_occupancy']
        ])
        return torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)

    def save_checkpoint(self, episode, final=False):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.mappo.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'tasks_completed': self.tasks_completed,
        }

        if final:
            save_path = os.path.join(
                self.config['checkpoint_dir'],
                f'mappo_final_{self.mode}.pt'
            )
            print(f"\n💾 Saving final model to: {save_path}")
        else:
            save_path = os.path.join(
                self.config['checkpoint_dir'],
                f'mappo_episode_{episode}.pt'
            )
            print(f"💾 Checkpoint saved: Episode {episode}")

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path):
        """加载检查点"""
        print(f"📂 Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.mappo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_episode = checkpoint['episode']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.tasks_completed = checkpoint['tasks_completed']

        print(f"✅ Resumed from episode {self.start_episode}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AGV MAPPO Training')

    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'medium', 'standard'],
                       help='Training mode: quick(100), medium(1000), standard(5000)')

    parser.add_argument('--episodes', type=int, default=None,
                       help='Custom number of episodes (overrides mode)')

    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')

    args = parser.parse_args()

    # 创建训练器
    trainer = Trainer(
        mode=args.mode,
        custom_episodes=args.episodes,
        resume_path=args.resume
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()