"""
训练主程序
执行MAPPO训练循环

✨ 修改：使用快速训练配置（100轮测试）
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from config.train_config_quick import train_config  # ✨ 改为快速训练配置
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Trainer:
    """训练器类"""

    def __init__(self):
        """初始化训练器"""
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
            ppo_epochs=train_config.PPO_EPOCHS,
            device=self.device
        )

        # TensorBoard
        if train_config.USE_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(train_config.TB_LOG_DIR, f"run_{timestamp}")
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
            print(f"启动TensorBoard: tensorboard --logdir={train_config.TB_LOG_DIR}")
        else:
            self.writer = None

        # 训练统计
        self.total_steps = 0
        self.episode_rewards = []
        self.training_data = []  # 存储随机生成的训练数据

    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def create_directories(self):
        """创建必要的目录"""
        os.makedirs(train_config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(train_config.LOG_DIR, exist_ok=True)
        os.makedirs(train_config.DATA_DIR, exist_ok=True)
        os.makedirs(train_config.TB_LOG_DIR, exist_ok=True)

    def calculate_obs_dim(self) -> int:
        """计算观察维度"""
        # own_state: 7 + nearby_agvs: 5*4 + task_info: 6 + path_occupancy: 3
        obs_dim = 7 + 5*4 + 6 + env_config.NUM_HORIZONTAL_LANES
        return obs_dim

    def flatten_observation(self, obs_dict: dict) -> np.ndarray:
        """
        将字典观察展平为向量

        Args:
            obs_dict: 观察字典

        Returns:
            展平的观察向量
        """
        obs_list = []
        obs_list.append(obs_dict['own_state'])
        obs_list.append(obs_dict['nearby_agvs'].flatten())
        obs_list.append(obs_dict['task_info'])
        obs_list.append(obs_dict['path_occupancy'])

        return np.concatenate(obs_list, axis=0)

    def collect_rollout(self) -> dict:
        """
        收集一个episode的经验

        Returns:
            经验缓冲区字典
        """
        # 初始化缓冲区
        observations = []
        actions = {'lane': [], 'direction': [], 'motion': []}
        log_probs = []
        rewards = []
        values = []
        dones = []

        # 重置环境
        obs_dict, info = self.env.reset()

        # 保存初始环境状态(用于后续分析)
        episode_data = {
            'initial_state': {
                'agv_positions': [agv.position.tolist() for agv in self.env.agvs],
                'tasks': [task.get_info() for task in self.env.tasks]
            },
            'transitions': []
        }

        episode_reward = 0

        for step in range(train_config.MAX_STEPS_PER_EPISODE):
            # 收集所有智能体的观察
            obs_batch = []
            for i in range(self.num_agents):
                obs = self.flatten_observation(obs_dict[f'agent_{i}'])
                obs_batch.append(obs)

            obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)

            # 选择动作
            actions_tensor, log_probs_tensor, values_tensor = self.mappo.select_action(
                obs_tensor
            )

            # 转换为环境格式
            env_actions = {}
            for i in range(self.num_agents):
                env_actions[f'agent_{i}'] = {
                    'lane': actions_tensor['lane'][i].cpu().item(),
                    'direction': actions_tensor['direction'][i].cpu().item(),
                    'motion': actions_tensor['motion'][i].cpu().numpy()
                }

            # 环境步进
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(env_actions)

            # 收集奖励
            reward_batch = np.array([
                reward_dict[f'agent_{i}'] for i in range(self.num_agents)
            ])

            # 检查是否结束
            done = terminated_dict['__all__'] or truncated_dict['__all__']
            done_batch = np.array([done] * self.num_agents)

            # 存储经验
            observations.append(obs_tensor.cpu())
            actions['lane'].append(actions_tensor['lane'].cpu())
            actions['direction'].append(actions_tensor['direction'].cpu())
            actions['motion'].append(actions_tensor['motion'].cpu())
            log_probs.append(log_probs_tensor.cpu())
            rewards.append(torch.FloatTensor(reward_batch))
            values.append(values_tensor.cpu())
            dones.append(torch.FloatTensor(done_batch))

            # 保存转换数据
            episode_data['transitions'].append({
                'step': step,
                'observations': obs_batch,
                'actions': {k: v.cpu().numpy().tolist() for k, v in actions_tensor.items()},
                'rewards': reward_batch.tolist(),
                'info': info_dict.get('agent_0', {})
            })

            episode_reward += reward_batch.mean()

            obs_dict = next_obs_dict

            if done:
                break

        # 计算最后的value
        next_obs_batch = []
        for i in range(self.num_agents):
            obs = self.flatten_observation(obs_dict[f'agent_{i}'])
            next_obs_batch.append(obs)
        next_obs_tensor = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)

        with torch.no_grad():
            next_values = self.mappo.actor_critic.get_value(next_obs_tensor).cpu()

        # 转换为tensor
        observations = torch.stack(observations)  # [num_steps, num_agents, obs_dim]
        log_probs = torch.stack(log_probs)  # [num_steps, num_agents]
        rewards = torch.stack(rewards)  # [num_steps, num_agents]
        values = torch.stack(values).squeeze(-1)  # [num_steps, num_agents]
        dones = torch.stack(dones)  # [num_steps, num_agents]

        # 计算GAE
        advantages, returns = self.mappo.compute_gae(
            rewards, values, dones, next_values.squeeze(-1)
        )

        # 构建rollout buffer
        rollout_buffer = {
            'observations': observations.reshape(-1, self.obs_dim),
            'actions': {
                'lane': torch.stack(actions['lane']).reshape(-1),
                'direction': torch.stack(actions['direction']).reshape(-1),
                'motion': torch.stack(actions['motion']).reshape(-1, 2)
            },
            'log_probs': log_probs.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1)
        }

        # 保存episode数据
        episode_data['episode_reward'] = episode_reward
        episode_data['episode_length'] = step + 1
        self.training_data.append(episode_data)

        return rollout_buffer, episode_reward

    def train(self):
        """主训练循环"""
        print("\n" + "="*50)
        print("Starting MAPPO Training - Quick Test (100 Episodes)")
        print("="*50)
        print(f"Environment: Horizontal Layout Port")
        print(f"Number of AGVs: {self.num_agents}")
        print(f"Bidirectional Routing: Enabled")
        print(f"Total Episodes: {train_config.NUM_EPISODES}")
        print(f"Max Steps per Episode: {train_config.MAX_STEPS_PER_EPISODE}")
        print(f"Reward Type: {env_config.REWARD_TYPE}")
        print("="*50 + "\n")

        for episode in tqdm(range(train_config.NUM_EPISODES), desc="Training"):
            # 收集经验
            rollout_buffer, episode_reward = self.collect_rollout()

            # 更新策略
            metrics = self.mappo.update(rollout_buffer)

            # 记录奖励
            self.episode_rewards.append(episode_reward)

            # 日志记录
            if (episode + 1) % train_config.LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-train_config.LOG_INTERVAL:])

                print(f"\nEpisode {episode + 1}/{train_config.NUM_EPISODES}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
                    self.writer.add_scalar('Train/AvgReward', avg_reward, episode)
                    self.writer.add_scalar('Train/ActorLoss', metrics['actor_loss'], episode)
                    self.writer.add_scalar('Train/ValueLoss', metrics['value_loss'], episode)
                    self.writer.add_scalar('Train/Entropy', metrics['entropy'], episode)

            # 保存检查点
            if (episode + 1) % train_config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    train_config.CHECKPOINT_DIR,
                    f"mappo_episode_{episode + 1}.pt"
                )
                self.mappo.save(checkpoint_path)

                # 保存训练数据
                data_path = os.path.join(
                    train_config.DATA_DIR,
                    f"training_data_episode_{episode + 1}.pkl"
                )
                with open(data_path, 'wb') as f:
                    pickle.dump(self.training_data[-train_config.SAVE_INTERVAL:], f)
                print(f"Training data saved to {data_path}")

            # 评估
            if (episode + 1) % train_config.EVAL_INTERVAL == 0:
                eval_reward = self.evaluate()
                print(f"  Evaluation Reward: {eval_reward:.2f}")

                if self.writer:
                    self.writer.add_scalar('Eval/Reward', eval_reward, episode)

        # 训练结束
        print("\n" + "="*50)
        print("Training Completed!")
        print("="*50)

        # 保存最终模型
        final_model_path = os.path.join(
            train_config.CHECKPOINT_DIR,
            "mappo_final_100ep.pt"
        )
        self.mappo.save(final_model_path)

        # 保存所有训练数据
        all_data_path = os.path.join(
            train_config.DATA_DIR,
            "all_training_data.pkl"
        )
        with open(all_data_path, 'wb') as f:
            pickle.dump(self.training_data, f)
        print(f"All training data saved to {all_data_path}")

        # 保存训练曲线
        self.save_training_curves()

        if self.writer:
            self.writer.close()

    def evaluate(self, num_episodes: int = None) -> float:
        """
        评估策略

        Args:
            num_episodes: 评估轮数

        Returns:
            平均奖励
        """
        if num_episodes is None:
            num_episodes = train_config.NUM_EVAL_EPISODES

        eval_rewards = []

        for _ in range(num_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = 0
            done = False

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
                obs_dict, reward_dict, terminated_dict, truncated_dict, _ = self.env.step(env_actions)

                reward_batch = np.array([
                    reward_dict[f'agent_{i}'] for i in range(self.num_agents)
                ])
                episode_reward += reward_batch.mean()

                done = terminated_dict['__all__'] or truncated_dict['__all__']

            eval_rewards.append(episode_reward)

        return np.mean(eval_rewards)

    def save_training_curves(self):
        """保存训练曲线数据"""
        import matplotlib.pyplot as plt

        # 绘制奖励曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')

        # 移动平均
        window = min(20, len(self.episode_rewards) // 2)  # 快速训练用小窗口
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(window)/window,
                mode='valid'
            )
            plt.plot(
                range(window-1, len(self.episode_rewards)),
                moving_avg,
                'r-',
                linewidth=2,
                label=f'Moving Average ({window})'
            )

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Curve - MAPPO Quick Test (100 Episodes)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        curve_path = os.path.join(train_config.LOG_DIR, 'training_curve.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {curve_path}")
        plt.close()

        # 保存数据
        def make_serializable(obj):
            """将配置对象转换为可序列化的字典"""
            if hasattr(obj, '__dict__'):
                return {
                    k: v for k, v in obj.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            return obj

        data = {
            'episode_rewards': self.episode_rewards,
            'config': {
                'env_config': make_serializable(env_config),
                'train_config': make_serializable(train_config),
                'model_config': make_serializable(model_config)
            }
        }

        json_path = os.path.join(train_config.LOG_DIR, 'training_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Training data saved to {json_path}")


def main():
    """主函数"""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()