"""
实验2：垂直布局 + 单向路由 训练脚本
Vertical Layout + Unidirectional Routing

运行方法：
    python train_v_uni.py

TensorBoard监控：
    tensorboard --logdir=./runs_v_uni

✨ 关键配置：
- LAYOUT_TYPE = 'vertical'  (垂直布局)
- BIDIRECTIONAL = False     (单向路由)
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config_v_uni import train_config
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Trainer:
    """训练器类"""

    def __init__(self):
        """初始化训练器"""

        # ✨ 设置环境为垂直+单向模式
        env_config.LAYOUT_TYPE = 'vertical'    # ✨ 垂直布局
        env_config.BIDIRECTIONAL = False       # ✨ 单向路由

        print(f"\n{'='*60}")
        print(f"🚀 实验2：垂直布局 + 单向路由 (v_uni)")
        print(f"{'='*60}")
        print(f"⚙️  LAYOUT_TYPE = {env_config.LAYOUT_TYPE}")
        print(f"⚙️  BIDIRECTIONAL = {env_config.BIDIRECTIONAL}")
        print(f"⚙️  QC在下边, YC在上边, AGV在垂直通道上移动")
        print(f"⚙️  AGV只能前进（单向）")
        print(f"{'='*60}\n")

        # 设置随机种子
        self.set_seed(train_config.SEED)

        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and train_config.USE_CUDA
            else 'cpu'
        )
        print(f"Using device: {self.device}")

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
            lr_actor=train_config.ACTOR_LR,
            lr_critic=train_config.CRITIC_LR,
            gamma=train_config.GAMMA,
            gae_lambda=train_config.GAE_LAMBDA,
            clip_epsilon=train_config.CLIP_EPSILON,
            value_loss_coef=train_config.VALUE_LOSS_COEF,
            entropy_coef=train_config.ENTROPY_COEF,
            max_grad_norm=train_config.MAX_GRAD_NORM,
            device=self.device
        )

        # TensorBoard
        self.writer = SummaryWriter(train_config.TENSORBOARD_DIR)

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.tasks_completed = []

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(train_config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(train_config.LOG_DIR, exist_ok=True)
        os.makedirs(train_config.TENSORBOARD_DIR, exist_ok=True)

    def calculate_obs_dim(self):
        """计算观察维度"""
        obs_dict, _ = self.env.reset()
        sample_obs = obs_dict['agent_0']

        obs_dim = (
            sample_obs['own_state'].shape[0] +
            sample_obs['nearby_agvs'].size +
            sample_obs['task_info'].shape[0] +
            sample_obs['path_occupancy'].shape[0]
        )
        return obs_dim

    def flatten_observation(self, obs_dict):
        """展平观察"""
        obs_list = []
        obs_list.append(obs_dict['own_state'])
        obs_list.append(obs_dict['nearby_agvs'].flatten())
        obs_list.append(obs_dict['task_info'])
        obs_list.append(obs_dict['path_occupancy'])
        return np.concatenate(obs_list, axis=0)

    def train_episode(self, episode):
        """训练一个episode"""
        obs_dict, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # 收集轨迹
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }

        while not done and episode_length < train_config.MAX_STEPS_PER_EPISODE:
            # 收集所有agent的观察
            obs_batch = []
            for i in range(self.num_agents):
                obs = self.flatten_observation(obs_dict[f'agent_{i}'])
                obs_batch.append(obs)

            obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)

            # 选择动作
            actions, log_probs, values = self.mappo.select_action(obs_tensor)

            # 转换为环境格式
            env_actions = {}
            for i in range(self.num_agents):
                env_actions[f'agent_{i}'] = {
                    'lane': actions['lane'][i].cpu().item(),
                    'direction': actions['direction'][i].cpu().item(),
                    'motion': actions['motion'][i].cpu().numpy()
                }

            # 环境步进
            next_obs_dict, rewards_dict, terminated, truncated, info = self.env.step(env_actions)

            # 正确获取全局终止状态
            done = terminated['__all__'] or truncated['__all__']

            # 收集奖励
            rewards = np.array([rewards_dict[f'agent_{i}'] for i in range(self.num_agents)])
            episode_reward += rewards.mean()

            # 存储轨迹
            trajectory['observations'].append(obs_batch)
            trajectory['actions'].append({
                'lane': actions['lane'].cpu().numpy(),
                'direction': actions['direction'].cpu().numpy(),
                'motion': actions['motion'].cpu().numpy()
            })
            trajectory['rewards'].append(rewards)
            trajectory['dones'].append(done)
            trajectory['values'].append(values.cpu().numpy())
            trajectory['log_probs'].append(log_probs.cpu().numpy())

            obs_dict = next_obs_dict
            episode_length += 1

        # 计算最后状态的价值（用于GAE计算）
        next_obs_batch = []
        for i in range(self.num_agents):
            obs = self.flatten_observation(next_obs_dict[f'agent_{i}'])
            next_obs_batch.append(obs)

        next_obs_tensor = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)

        with torch.no_grad():
            next_values = self.mappo.actor_critic.get_value(next_obs_tensor).cpu()

        # 将列表转换为张量
        observations = torch.FloatTensor(np.array(trajectory['observations'])).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(trajectory['rewards'])).to(self.device)
        values_tensor = torch.FloatTensor(np.array(trajectory['values'])).squeeze(-1).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(trajectory['dones'], dtype=np.float32)).to(self.device)
        log_probs_tensor = torch.FloatTensor(np.array(trajectory['log_probs'])).to(self.device)

        # 计算GAE和returns
        advantages, returns = self.mappo.compute_gae(
            rewards_tensor,
            values_tensor,
            dones_tensor,
            next_values.squeeze(-1).to(self.device)
        )

        # 构建完整的rollout buffer
        processed_trajectory = {
            'observations': observations.reshape(-1, self.obs_dim),
            'actions': {
                'lane': torch.LongTensor(np.array([a['lane'] for a in trajectory['actions']])).reshape(-1).to(self.device),
                'direction': torch.LongTensor(np.array([a['direction'] for a in trajectory['actions']])).reshape(-1).to(self.device),
                'motion': torch.FloatTensor(np.array([a['motion'] for a in trajectory['actions']])).reshape(-1, 2).to(self.device)
            },
            'log_probs': log_probs_tensor.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1)
        }

        # 训练更新
        loss_info = self.mappo.update(processed_trajectory)

        return episode_reward, episode_length, info, loss_info

    def train(self):
        """主训练循环"""
        print(f"\n开始训练 - {train_config.EXPERIMENT_NAME}")
        print(f"总轮数: {train_config.NUM_EPISODES}")
        print(f"保存间隔: {train_config.SAVE_INTERVAL}\n")

        for episode in tqdm(range(1, train_config.NUM_EPISODES + 1), desc="Training"):
            # 训练一个episode
            ep_reward, ep_length, info, loss_info = self.train_episode(episode)

            # 记录统计
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.tasks_completed.append(info.get('tasks_completed', 0))

            # TensorBoard记录
            self.writer.add_scalar('Train/Episode_Reward', ep_reward, episode)
            self.writer.add_scalar('Train/Episode_Length', ep_length, episode)
            self.writer.add_scalar('Train/Tasks_Completed', info.get('tasks_completed', 0), episode)
            self.writer.add_scalar('Train/Collisions', info.get('collisions', 0), episode)

            if loss_info:
                self.writer.add_scalar('Loss/Actor', loss_info.get('actor_loss', 0), episode)
                self.writer.add_scalar('Loss/Critic', loss_info.get('critic_loss', 0), episode)
                self.writer.add_scalar('Loss/Entropy', loss_info.get('entropy', 0), episode)

            # 定期保存
            if episode % train_config.SAVE_INTERVAL == 0:
                self.save_checkpoint(episode)

                # 打印进度
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_tasks = np.mean(self.tasks_completed[-100:])
                print(f"\nEpisode {episode}/{train_config.NUM_EPISODES}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Avg Tasks (last 100): {avg_tasks:.2f}")

        # 保存最终模型
        self.save_checkpoint('final')
        self.save_training_data()
        print("\n✅ 训练完成!")

    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint_path = os.path.join(
            train_config.CHECKPOINT_DIR,
            f'mappo_episode_{episode}.pt'
        )
        self.mappo.save(checkpoint_path)
        if train_config.VERBOSE:
            print(f"💾 检查点已保存: {checkpoint_path}")

    def save_training_data(self):
        """保存训练数据"""
        data = {
            'experiment': train_config.EXPERIMENT_NAME,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'tasks_completed': self.tasks_completed,
            'config': {
                'layout_type': env_config.LAYOUT_TYPE,
                'bidirectional': env_config.BIDIRECTIONAL,
                'num_agents': env_config.NUM_AGVS,
                'num_episodes': train_config.NUM_EPISODES
            }
        }

        # 保存为JSON
        json_path = os.path.join(train_config.LOG_DIR, 'training_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # 保存为pickle
        pkl_path = os.path.join(train_config.LOG_DIR, 'training_data.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"📊 训练数据已保存:")
        print(f"   - JSON: {json_path}")
        print(f"   - Pickle: {pkl_path}")


def main():
    """主函数"""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()