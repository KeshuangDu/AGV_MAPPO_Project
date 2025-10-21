"""
实验2：水平布局 + 单向路由 训练脚本
Horizontal Layout + Unidirectional Routing

运行方法：
    python train_h_uni.py

TensorBoard监控：
    tensorboard --logdir=./runs_h_uni

✨ 关键差异：
- 使用 train_config_h_uni 配置
- 设置 BIDIRECTIONAL = False
- 其余代码与 train_h_bi.py 完全相同
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
from config.train_config_h_uni import train_config  # ✨ 使用水平单向配置
from config.model_config import model_config
from environment.port_env import PortEnvironment
from models.actor_critic import ActorCritic
from algorithm.mappo import MAPPO


class Trainer:
    """训练器类"""

    def __init__(self):
        """初始化训练器"""

        # ✨ 设置环境为单向模式
        env_config.BIDIRECTIONAL = False  # ✨✨ 关键！
        print(f"\n{'='*60}")
        print(f"🚀 实验2：水平布局 + 单向路由 (h_uni)")  # ✨ 差异3
        print(f"{'='*60}")
        print(f"⚠️  BIDIRECTIONAL = {env_config.BIDIRECTIONAL}")
        print(f"⚠️  AGV只能前进，不能后退！")  # ✨ 差异4
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
            ppo_epochs=train_config.PPO_EPOCHS,
            device=self.device
        )

        # TensorBoard
        if train_config.USE_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(train_config.TB_LOG_DIR, f"run_{timestamp}")
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs: {log_dir}")
            print(f"启动命令: tensorboard --logdir={train_config.TB_LOG_DIR}")
        else:
            self.writer = None

        # 训练统计
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_tasks_completed = []
        self.training_data = []

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
        obs_dim = 7 + 5 * 4 + 6 + env_config.NUM_HORIZONTAL_LANES
        return obs_dim

    def flatten_observation(self, obs_dict: dict) -> np.ndarray:
        """展平观察"""
        obs_list = []
        obs_list.append(obs_dict['own_state'])
        obs_list.append(obs_dict['nearby_agvs'].flatten())
        obs_list.append(obs_dict['task_info'])
        obs_list.append(obs_dict['path_occupancy'])
        return np.concatenate(obs_list, axis=0)

    def collect_rollout(self) -> dict:
        """收集一个episode的经验"""
        # 初始化缓冲区
        observations = []
        actions = {'lane': [], 'direction': [], 'motion': []}
        log_probs = []
        rewards = []
        values = []
        dones = []

        # 重置环境
        obs_dict, info = self.env.reset()

        episode_data = {
            'initial_state': {
                'agv_positions': [agv.position.tolist() for agv in self.env.agvs],
                'tasks': [task.get_info() for task in self.env.tasks]
            },
            'transitions': []
        }

        episode_reward = 0
        tasks_completed = 0

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

            episode_data['transitions'].append({
                'step': step,
                'observations': obs_batch,
                'actions': {k: v.cpu().numpy().tolist() for k, v in actions_tensor.items()},
                'rewards': reward_batch.tolist(),
                'info': info_dict.get('agent_0', {})
            })

            episode_reward += reward_batch.mean()
            tasks_completed = info_dict.get('agent_0', {}).get('completed_tasks', 0)

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
        observations = torch.stack(observations)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        values = torch.stack(values).squeeze(-1)
        dones = torch.stack(dones)

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

        episode_data['episode_reward'] = episode_reward
        episode_data['episode_length'] = step + 1
        episode_data['tasks_completed'] = tasks_completed
        self.training_data.append(episode_data)

        return rollout_buffer, episode_reward, tasks_completed

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print(f"🚀 开始训练：水平布局 + 双向路由")
        print("=" * 60)
        print(f"环境: 水平布局港口 + 双向路由")
        print(f"AGV数量: {self.num_agents}")
        print(f"训练轮数: {train_config.NUM_EPISODES}")
        print(f"每轮最大步数: {train_config.MAX_STEPS_PER_EPISODE}")
        print(f"奖励类型: {env_config.REWARD_TYPE}")
        print(f"任务管理器: {'启用' if env_config.USE_TASK_MANAGER else '禁用'}")
        print("=" * 60)
        print(f"💾 检查点目录: {train_config.CHECKPOINT_DIR}")
        print(f"📊 TensorBoard: tensorboard --logdir={train_config.TB_LOG_DIR}")
        print("=" * 60 + "\n")

        best_avg_tasks = 0

        for episode in tqdm(range(train_config.NUM_EPISODES), desc="Training"):
            # 收集经验
            rollout_buffer, episode_reward, tasks_completed = self.collect_rollout()

            # 更新策略
            metrics = self.mappo.update(rollout_buffer)

            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_tasks_completed.append(tasks_completed)

            # 日志记录
            if (episode + 1) % train_config.LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-train_config.LOG_INTERVAL:])
                avg_tasks = np.mean(self.episode_tasks_completed[-train_config.LOG_INTERVAL:])

                print(f"\n📈 Episode {episode + 1}/{train_config.NUM_EPISODES}")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  平均任务完成数: {avg_tasks:.2f}")
                print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
                    self.writer.add_scalar('Train/AvgReward', avg_reward, episode)
                    self.writer.add_scalar('Train/TasksCompleted', tasks_completed, episode)
                    self.writer.add_scalar('Train/AvgTasksCompleted', avg_tasks, episode)
                    self.writer.add_scalar('Train/ActorLoss', metrics['actor_loss'], episode)
                    self.writer.add_scalar('Train/ValueLoss', metrics['value_loss'], episode)
                    self.writer.add_scalar('Train/Entropy', metrics['entropy'], episode)

                # 如果任务完成数创新高，额外保存
                if avg_tasks > best_avg_tasks:
                    best_avg_tasks = avg_tasks
                    best_checkpoint_path = os.path.join(
                        train_config.CHECKPOINT_DIR,
                        f"mappo_best_tasks_{avg_tasks:.1f}.pt"
                    )
                    self.mappo.save(best_checkpoint_path)
                    print(f"  🌟 新纪录！保存最佳模型: {best_checkpoint_path}")

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

            # 评估
            if (episode + 1) % train_config.EVAL_INTERVAL == 0:
                eval_reward, eval_tasks = self.evaluate()
                print(f"  📊 评估 - 奖励: {eval_reward:.2f}, 任务: {eval_tasks:.2f}")

                if self.writer:
                    self.writer.add_scalar('Eval/Reward', eval_reward, episode)
                    self.writer.add_scalar('Eval/TasksCompleted', eval_tasks, episode)

        # 训练结束
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print(f"最佳平均任务完成数: {best_avg_tasks:.2f}")
        print("=" * 60)

        # 保存最终模型
        final_model_path = os.path.join(
            train_config.CHECKPOINT_DIR,
            f"mappo_final_{train_config.NUM_EPISODES}ep.pt"
        )
        self.mappo.save(final_model_path)

        # 保存所有训练数据
        all_data_path = os.path.join(
            train_config.DATA_DIR,
            "all_training_data.pkl"
        )
        with open(all_data_path, 'wb') as f:
            pickle.dump(self.training_data, f)

        # 保存训练曲线
        self.save_training_curves()

        if self.writer:
            self.writer.close()

    def evaluate(self, num_episodes: int = None) -> tuple:
        """评估策略"""
        if num_episodes is None:
            num_episodes = train_config.NUM_EVAL_EPISODES

        eval_rewards = []
        eval_tasks = []

        for _ in range(num_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                obs_batch = []
                for i in range(self.num_agents):
                    obs = self.flatten_observation(obs_dict[f'agent_{i}'])
                    obs_batch.append(obs)

                obs_tensor = torch.FloatTensor(np.array(obs_batch)).to(self.device)

                actions_tensor, _, _ = self.mappo.select_action(
                    obs_tensor, deterministic=True
                )

                env_actions = {}
                for i in range(self.num_agents):
                    env_actions[f'agent_{i}'] = {
                        'lane': actions_tensor['lane'][i].cpu().item(),
                        'direction': actions_tensor['direction'][i].cpu().item(),
                        'motion': actions_tensor['motion'][i].cpu().numpy()
                    }

                obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(env_actions)

                reward_batch = np.array([
                    reward_dict[f'agent_{i}'] for i in range(self.num_agents)
                ])
                episode_reward += reward_batch.mean()

                done = terminated_dict['__all__'] or truncated_dict['__all__']

            eval_rewards.append(episode_reward)
            eval_tasks.append(info_dict.get('agent_0', {}).get('completed_tasks', 0))

        return np.mean(eval_rewards), np.mean(eval_tasks)

    def save_training_curves(self):
        """保存训练曲线"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 奖励曲线
        ax = axes[0]
        ax.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        window = 50
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(
                range(window - 1, len(self.episode_rewards)),
                moving_avg,
                'r-',
                linewidth=2,
                label=f'Moving Average ({window})'
            )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Curve - Rewards (h_bi)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 任务完成数曲线
        ax = axes[1]
        ax.plot(self.episode_tasks_completed, alpha=0.6, color='green', label='Tasks Completed')
        if len(self.episode_tasks_completed) >= window:
            moving_avg_tasks = np.convolve(
                self.episode_tasks_completed,
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(
                range(window - 1, len(self.episode_tasks_completed)),
                moving_avg_tasks,
                'r-',
                linewidth=2,
                label=f'Moving Average ({window})'
            )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Tasks Completed')
        ax.set_title('Training Curve - Task Completion (h_bi)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        curve_path = os.path.join(train_config.LOG_DIR, 'training_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练曲线已保存: {curve_path}")
        plt.close()

        # 保存数据
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_tasks_completed': self.episode_tasks_completed,
            'config': {
                'experiment_name': train_config.EXPERIMENT_NAME,
                'experiment_desc': train_config.EXPERIMENT_DESC,
                'bidirectional': env_config.BIDIRECTIONAL,
                'num_episodes': train_config.NUM_EPISODES
            }
        }

        json_path = os.path.join(train_config.LOG_DIR, 'training_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 训练数据已保存: {json_path}")


def main():
    """主函数"""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()