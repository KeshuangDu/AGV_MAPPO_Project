"""
MAPPO (Multi-Agent Proximal Policy Optimization) 算法实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class MAPPO:
    """
    MAPPO算法类
    支持多智能体协同学习
    """

    def __init__(
            self,
            actor_critic: nn.Module,
            num_agents: int,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            value_loss_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            ppo_epochs: int = 10,
            device: str = 'cuda'
    ):
        """
        初始化MAPPO

        Args:
            actor_critic: Actor-Critic模型
            num_agents: 智能体数量
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数
            max_grad_norm: 梯度裁剪
            ppo_epochs: PPO更新轮数
            device: 设备
        """
        self.actor_critic = actor_critic.to(device)
        self.num_agents = num_agents

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=lr_actor
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.99
        )

    def compute_gae(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)

        Args:
            rewards: 奖励 [num_steps, num_agents]
            values: 状态价值 [num_steps, num_agents]
            dones: 终止标志 [num_steps, num_agents]
            next_values: 下一状态价值 [num_steps, num_agents]

        Returns:
            advantages: 优势 [num_steps, num_agents]
            returns: 回报 [num_steps, num_agents]
        """
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        last_gae = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values

        return advantages, returns

    def update(
            self,
            rollout_buffer: Dict
    ) -> Dict[str, float]:
        """
        更新策略

        Args:
            rollout_buffer: 经验缓冲区字典

        Returns:
            训练指标字典
        """
        # 从缓冲区提取数据
        obs = rollout_buffer['observations'].to(self.device)
        actions = {
            k: v.to(self.device)
            for k, v in rollout_buffer['actions'].items()
        }
        old_log_probs = rollout_buffer['log_probs'].to(self.device)
        advantages = rollout_buffer['advantages'].to(self.device)
        returns = rollout_buffer['returns'].to(self.device)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练指标
        metrics = defaultdict(list)

        # PPO更新
        for epoch in range(self.ppo_epochs):
            # 评估当前动作
            new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                obs, actions
            )

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = nn.MSELoss()(values.squeeze(-1), returns)

            # 熵损失
            entropy_loss = -entropy.mean()

            # 总损失
            total_loss = (
                    actor_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
            )

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()

            # 记录指标
            metrics['actor_loss'].append(actor_loss.item())
            metrics['value_loss'].append(value_loss.item())
            metrics['entropy'].append(-entropy_loss.item())
            metrics['total_loss'].append(total_loss.item())

        # 更新学习率
        self.scheduler.step()

        # 计算平均指标
        avg_metrics = {
            k: np.mean(v) for k, v in metrics.items()
        }

        return avg_metrics

    def select_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        选择动作

        Args:
            obs: 观察
            deterministic: 是否确定性

        Returns:
            actions: 动作字典
            action_log_probs: 动作对数概率
            values: 状态价值
        """
        with torch.no_grad():
            if deterministic:
                actions = self.actor_critic.actor.get_action(obs, deterministic=True)
                action_log_probs = None
                values = self.actor_critic.get_value(obs)
            else:
                actions, action_log_probs_dict = self.actor_critic.actor(obs)
                action_log_probs = action_log_probs_dict['total']
                values = self.actor_critic.get_value(obs)

        return actions, action_log_probs, values

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")