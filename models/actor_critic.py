"""
Actor-Critic神经网络
实现MAPPO的Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Tuple


class Actor(nn.Module):
    """
    Actor网络
    输出动作概率分布
    """

    def __init__(
            self,
            obs_dim: int,
            hidden_dims: list = [256, 256, 128],
            num_lanes: int = 3,
            num_directions: int = 2
    ):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim

        # 特征提取网络
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # 离散动作头：车道选择
        self.lane_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_lanes)
        )

        # 离散动作头：方向选择(双向路由关键)
        self.direction_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_directions)
        )

        # 连续动作头：运动控制
        self.motion_mean = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [acceleration, steering]
        )

        self.motion_log_std = nn.Parameter(torch.zeros(2))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        前向传播

        Args:
            obs: 观察 [batch_size, obs_dim]

        Returns:
            actions: 动作字典
            action_log_probs: 动作对数概率字典
        """
        # 特征提取
        features = self.feature_net(obs)

        # 车道选择(离散)
        lane_logits = self.lane_head(features)
        lane_dist = Categorical(logits=lane_logits)
        lane_action = lane_dist.sample()
        lane_log_prob = lane_dist.log_prob(lane_action)

        # 方向选择(离散，双向)
        direction_logits = self.direction_head(features)
        direction_dist = Categorical(logits=direction_logits)
        direction_action = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction_action)

        # 运动控制(连续)
        motion_mean = self.motion_mean(features)
        motion_std = self.motion_log_std.exp().expand_as(motion_mean)
        motion_dist = Normal(motion_mean, motion_std)
        motion_action = motion_dist.sample()
        motion_log_prob = motion_dist.log_prob(motion_action).sum(dim=-1)

        # 动作字典
        actions = {
            'lane': lane_action,
            'direction': direction_action,
            'motion': torch.tanh(motion_action)  # 限制到[-1, 1]
        }

        # 对数概率字典
        action_log_probs = {
            'lane': lane_log_prob,
            'direction': direction_log_prob,
            'motion': motion_log_prob,
            'total': lane_log_prob + direction_log_prob + motion_log_prob
        }

        return actions, action_log_probs

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            obs: 观察
            actions: 动作字典

        Returns:
            action_log_probs: 动作对数概率
            entropy: 熵
        """
        features = self.feature_net(obs)

        # 车道
        lane_logits = self.lane_head(features)
        lane_dist = Categorical(logits=lane_logits)
        lane_log_prob = lane_dist.log_prob(actions['lane'])
        lane_entropy = lane_dist.entropy()

        # 方向
        direction_logits = self.direction_head(features)
        direction_dist = Categorical(logits=direction_logits)
        direction_log_prob = direction_dist.log_prob(actions['direction'])
        direction_entropy = direction_dist.entropy()

        # 运动
        motion_mean = self.motion_mean(features)
        motion_std = self.motion_log_std.exp().expand_as(motion_mean)
        motion_dist = Normal(motion_mean, motion_std)

        # 注意：actions['motion']已经是tanh后的，需要inverse
        motion_action_raw = torch.atanh(torch.clamp(actions['motion'], -0.999, 0.999))
        motion_log_prob = motion_dist.log_prob(motion_action_raw).sum(dim=-1)
        motion_entropy = motion_dist.entropy().sum(dim=-1)

        # 总和
        total_log_prob = lane_log_prob + direction_log_prob + motion_log_prob
        total_entropy = lane_entropy + direction_entropy + motion_entropy

        return total_log_prob, total_entropy

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Dict:
        """
        获取动作(用于评估)

        Args:
            obs: 观察
            deterministic: 是否确定性动作

        Returns:
            动作字典
        """
        with torch.no_grad():
            features = self.feature_net(obs)

            # 车道
            lane_logits = self.lane_head(features)
            if deterministic:
                lane_action = lane_logits.argmax(dim=-1)
            else:
                lane_action = Categorical(logits=lane_logits).sample()

            # 方向
            direction_logits = self.direction_head(features)
            if deterministic:
                direction_action = direction_logits.argmax(dim=-1)
            else:
                direction_action = Categorical(logits=direction_logits).sample()

            # 运动
            motion_mean = self.motion_mean(features)
            if deterministic:
                motion_action = torch.tanh(motion_mean)
            else:
                motion_std = self.motion_log_std.exp().expand_as(motion_mean)
                motion_action = torch.tanh(Normal(motion_mean, motion_std).sample())

            return {
                'lane': lane_action,
                'direction': direction_action,
                'motion': motion_action
            }


class Critic(nn.Module):
    """
    Critic网络
    评估状态价值
    """

    def __init__(
            self,
            obs_dim: int,
            hidden_dims: list = [256, 256, 128],
            use_centralized: bool = True
    ):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.use_centralized = use_centralized

        # 价值网络
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.value_net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            obs: 观察 [batch_size, obs_dim]

        Returns:
            value: 状态价值 [batch_size, 1]
        """
        return self.value_net(obs)


class ActorCritic(nn.Module):
    """
    Actor-Critic模型
    组合Actor和Critic
    """

    def __init__(
            self,
            obs_dim: int,
            actor_hidden_dims: list = [256, 256, 128],
            critic_hidden_dims: list = [256, 256, 128],
            num_lanes: int = 3,
            num_directions: int = 2,
            use_centralized_critic: bool = True
    ):
        super(ActorCritic, self).__init__()

        self.actor = Actor(
            obs_dim=obs_dim,
            hidden_dims=actor_hidden_dims,
            num_lanes=num_lanes,
            num_directions=num_directions
        )

        # 如果使用中心化Critic，输入维度需要包含全局信息
        critic_obs_dim = obs_dim if not use_centralized_critic else obs_dim * 2

        self.critic = Critic(
            obs_dim=critic_obs_dim,
            hidden_dims=critic_hidden_dims,
            use_centralized=use_centralized_critic
        )

    def forward(self, obs: torch.Tensor):
        """前向传播"""
        actions, action_log_probs = self.actor(obs)
        value = self.critic(obs)
        return actions, action_log_probs, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        return self.critic(obs)

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作

        Returns:
            action_log_probs: 动作对数概率
            entropy: 熵
            value: 状态价值
        """
        action_log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic(obs)
        return action_log_probs, entropy, value