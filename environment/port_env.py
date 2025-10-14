"""
港口环境主类
实现水平布局的多AGV港口仿真环境
支持双向路由和多智能体交互
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import copy

from .agv import AGV
from .equipment import QuayCrane, YardCrane, Task


class PortEnvironment(gym.Env):
    """水平布局港口环境"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config):
        """
        初始化环境

        Args:
            config: 环境配置对象(EnvConfig)
        """
        super().__init__()

        self.config = config

        # 环境尺寸
        self.width = config.PORT_WIDTH
        self.height = config.PORT_HEIGHT

        # 时间
        self.dt = config.TIME_STEP
        self.current_time = 0.0
        self.max_steps = config.MAX_EPISODE_STEPS
        self.current_step = 0

        # 初始化设备
        self.num_agvs = config.NUM_AGVS
        self.num_qcs = config.NUM_QC
        self.num_ycs = config.NUM_YC

        self._init_equipment()

        # 任务管理
        self.tasks = []
        self.completed_tasks = []
        self.max_tasks = config.MAX_TASKS
        self.task_gen_rate = config.TASK_GENERATION_RATE

        # 定义动作空间和观察空间
        self._setup_spaces()

        # 奖励权重
        self.reward_weights = config.REWARD_WEIGHTS

        # 统计信息
        self.episode_stats = {
            'total_reward': 0.0,
            'collisions': 0,
            'tasks_completed': 0,
            'avg_task_time': 0.0
        }

        # 渲染相关
        self.render_mode = None
        self.screen = None

    def _init_equipment(self):
        """初始化港口设备"""
        # 初始化AGV
        self.agvs = []
        for i in range(self.num_agvs):
            # AGV初始位置均匀分布在中间区域
            init_x = np.random.uniform(200, 400)
            init_y = np.random.uniform(50, self.height - 50)
            init_pos = (init_x, init_y)

            agv = AGV(
                agv_id=i,
                init_position=init_pos,
                max_speed=self.config.AGV_MAX_SPEED,
                max_accel=self.config.AGV_MAX_ACCEL,
                length=self.config.AGV_LENGTH,
                width=self.config.AGV_WIDTH
            )
            self.agvs.append(agv)

        # 初始化岸桥(QC) - 位于左侧
        self.qcs = []
        for i in range(self.num_qcs):
            pos = self.config.QC_POSITIONS[i]
            qc = QuayCrane(
                crane_id=i,
                position=pos,
                operation_time_range=self.config.QC_OPERATION_TIME
            )
            self.qcs.append(qc)

        # 初始化场桥(YC) - 位于右侧
        self.ycs = []
        for i in range(self.num_ycs):
            pos = self.config.YC_POSITIONS[i]
            yc = YardCrane(
                crane_id=i,
                position=pos,
                operation_time_range=self.config.YC_OPERATION_TIME
            )
            self.ycs.append(yc)

    def _setup_spaces(self):
        """设置动作空间和观察空间"""
        # 每个AGV的观察空间
        self.single_obs_space = spaces.Dict({
            # 自身状态 (7维)
            'own_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,), dtype=np.float32
            ),
            # 附近AGV状态 (最多5个, 每个4维)
            'nearby_agvs': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(5, 4), dtype=np.float32
            ),
            # 任务信息 (6维)
            'task_info': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,), dtype=np.float32
            ),
            # 路径占用信息 (车道数量)
            'path_occupancy': spaces.Box(
                low=0, high=1,
                shape=(self.config.NUM_HORIZONTAL_LANES,),
                dtype=np.float32
            )
        })

        # 多智能体观察空间
        self.observation_space = spaces.Dict({
            f'agent_{i}': self.single_obs_space
            for i in range(self.num_agvs)
        })

        # 每个AGV的动作空间
        self.single_action_space = spaces.Dict({
            # 离散动作
            'lane': spaces.Discrete(self.config.NUM_HORIZONTAL_LANES),
            'direction': spaces.Discrete(2),  # 0: forward, 1: backward
            # 连续动作
            'motion': spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        })

        # 多智能体动作空间
        self.action_space = spaces.Dict({
            f'agent_{i}': self.single_action_space
            for i in range(self.num_agvs)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境

        Returns:
            observations: 初始观察
            info: 附加信息
        """
        super().reset(seed=seed)

        # 重置时间
        self.current_time = 0.0
        self.current_step = 0

        # 重置所有设备
        for agv in self.agvs:
            init_x = np.random.uniform(200, 400)
            init_y = np.random.uniform(50, self.height - 50)
            agv.reset((init_x, init_y))

        for qc in self.qcs:
            qc.reset()

        for yc in self.ycs:
            yc.reset()

        # 重置任务
        self.tasks = []
        self.completed_tasks = []
        Task.task_counter = 0

        # 生成初始任务
        self._generate_tasks(num_tasks=5)

        # 重置统计
        self.episode_stats = {
            'total_reward': 0.0,
            'collisions': 0,
            'tasks_completed': 0,
            'avg_task_time': 0.0
        }

        # 获取初始观察
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(self, actions: Dict):
        """
        执行一步动作

        Args:
            actions: 字典，key为agent_i，value为该智能体的动作

        Returns:
            observations: 新观察
            rewards: 奖励字典
            terminateds: 是否终止
            truncateds: 是否截断
            infos: 信息字典
        """
        # 1. 执行动作，更新AGV状态
        for i, agv in enumerate(self.agvs):
            action = actions.get(f'agent_{i}')
            if action is not None:
                self._execute_action(agv, action)

        # 2. 更新设备状态
        for qc in self.qcs:
            qc.update(self.dt)

        for yc in self.ycs:
            yc.update(self.dt)

        # 3. 检查碰撞
        collisions = self._check_collisions()

        # 4. 更新任务状态
        self._update_tasks()

        # 5. 生成新任务
        if np.random.random() < self.task_gen_rate:
            self._generate_tasks(num_tasks=1)

        # 6. 计算奖励
        rewards = self._compute_rewards(collisions)

        # 7. 更新时间和步数
        self.current_time += self.dt
        self.current_step += 1

        # 8. 检查是否结束
        terminated = self._check_done()
        truncated = self.current_step >= self.max_steps

        # 9. 获取观察和信息
        observations = self._get_observations()
        info = self._get_info()

        # 转换为多智能体格式
        terminateds = {f'agent_{i}': terminated for i in range(self.num_agvs)}
        terminateds['__all__'] = terminated

        truncateds = {f'agent_{i}': truncated for i in range(self.num_agvs)}
        truncateds['__all__'] = truncated

        infos = {f'agent_{i}': info for i in range(self.num_agvs)}

        return observations, rewards, terminateds, truncateds, infos

    def _execute_action(self, agv: AGV, action: Dict):
        """
        执行AGV动作

        Args:
            agv: AGV对象
            action: 动作字典
        """
        # 解析动作
        lane = action['lane']
        direction = action['direction']
        motion = action['motion']

        # 更新车道
        agv.current_lane = lane

        # 更新方向 (双向路由关键)
        target_forward = (direction == 0)
        if agv.moving_forward != target_forward:
            agv.switch_direction()

        # 更新运动状态
        acceleration = motion[0] * agv.max_accel
        steering = motion[1] * np.pi / 6  # 最大±30度

        agv.update_state(acceleration, steering, self.dt)

        # 边界约束
        agv.position[0] = np.clip(agv.position[0], 0, self.width)
        agv.position[1] = np.clip(agv.position[1], 0, self.height)

    def _check_collisions(self) -> List[Tuple[int, int]]:
        """
        检查AGV之间的碰撞

        Returns:
            碰撞的AGV对 [(i, j), ...]
        """
        collisions = []
        safe_dist = self.config.SAFE_DISTANCE

        for i in range(len(self.agvs)):
            for j in range(i + 1, len(self.agvs)):
                if self.agvs[i].is_collision(self.agvs[j], safe_dist):
                    collisions.append((i, j))
                    self.agvs[i].collision_flag = True
                    self.agvs[j].collision_flag = True
                    self.episode_stats['collisions'] += 1

        return collisions

    def _generate_tasks(self, num_tasks: int = 1):
        """
        生成新任务

        Args:
            num_tasks: 生成任务数量
        """
        for _ in range(num_tasks):
            if len(self.tasks) >= self.max_tasks:
                break

            # 随机选择任务类型
            task_type = np.random.choice(self.config.TASK_TYPES)

            # 随机选择QC和YC
            qc_id = np.random.randint(0, self.num_qcs)
            yc_id = np.random.randint(0, self.num_ycs)

            # 创建任务
            task = Task(task_type, qc_id, yc_id)

            # 设置pickup和delivery位置
            if task_type == 'import':
                task.pickup_location = self.qcs[qc_id].position.copy()
                task.delivery_location = self.ycs[yc_id].position.copy()
            else:
                task.pickup_location = self.ycs[yc_id].position.copy()
                task.delivery_location = self.qcs[qc_id].position.copy()

            self.tasks.append(task)

    def _update_tasks(self):
        """更新任务状态"""
        for task in self.tasks[:]:
            if task.status == 'completed':
                self.completed_tasks.append(task)
                self.tasks.remove(task)
                self.episode_stats['tasks_completed'] += 1

    def _compute_rewards(self, collisions: List) -> Dict:
        """
        计算奖励

        Args:
            collisions: 碰撞列表

        Returns:
            奖励字典
        """
        rewards = {}
        w = self.reward_weights

        for i, agv in enumerate(self.agvs):
            reward = 0.0

            # 时间惩罚
            reward += w['time_penalty']

            # 碰撞惩罚
            if agv.collision_flag:
                reward += w['collision']
                agv.collision_flag = False

            # 任务完成奖励
            if agv.current_task and agv.current_task['status'] == 'completed':
                reward += w['task_completion']

            # 双向路由奖励(如果有效利用双向)
            if hasattr(agv, 'direction_changes'):
                if agv.direction_changes > 0 and agv.direction_changes < 3:
                    reward += w['bidirectional_bonus']
                elif agv.direction_changes >= 3:
                    reward += w['direction_change']

            rewards[f'agent_{i}'] = reward
            self.episode_stats['total_reward'] += reward

        return rewards

    def _check_done(self) -> bool:
        """检查是否结束"""
        # 所有任务完成
        if len(self.tasks) == 0 and len(self.completed_tasks) > 0:
            return True

        # 发生严重碰撞
        if self.episode_stats['collisions'] > 10:
            return True

        return False

    def _get_observations(self) -> Dict:
        """获取所有智能体的观察"""
        observations = {}

        for i, agv in enumerate(self.agvs):
            obs = self._get_single_observation(agv)
            observations[f'agent_{i}'] = obs

        return observations

    def _get_single_observation(self, agv: AGV) -> Dict:
        """
        获取单个AGV的观察

        Args:
            agv: AGV对象

        Returns:
            观察字典
        """
        # 自身状态 [x, y, vx, vy, direction, has_container, moving_forward]
        own_state = np.array([
            agv.position[0] / self.width,
            agv.position[1] / self.height,
            agv.velocity / agv.max_speed,
            np.cos(agv.direction),
            np.sin(agv.direction),
            float(agv.has_container),
            float(agv.moving_forward)
        ], dtype=np.float32)

        # 附近AGV状态
        nearby_agvs = self._get_nearby_agvs(agv, max_count=5)

        # 任务信息
        task_info = self._get_task_info(agv)

        # 路径占用信息
        path_occupancy = self._get_path_occupancy()

        return {
            'own_state': own_state,
            'nearby_agvs': nearby_agvs,
            'task_info': task_info,
            'path_occupancy': path_occupancy
        }

    def _get_nearby_agvs(self, agv: AGV, max_count: int = 5) -> np.ndarray:
        """获取附近AGV信息"""
        nearby = np.zeros((max_count, 4), dtype=np.float32)

        other_agvs = [a for a in self.agvs if a.id != agv.id]
        distances = [agv.distance_to(a) for a in other_agvs]
        sorted_indices = np.argsort(distances)

        for i, idx in enumerate(sorted_indices[:max_count]):
            other = other_agvs[idx]
            rel_pos = other.position - agv.position
            nearby[i] = [
                rel_pos[0] / 100.0,
                rel_pos[1] / 100.0,
                other.velocity / other.max_speed,
                float(other.moving_forward)
            ]

        return nearby

    def _get_task_info(self, agv: AGV) -> np.ndarray:
        """获取任务信息"""
        task_info = np.zeros(6, dtype=np.float32)

        if agv.current_task:
            task = agv.current_task
            task_info[0] = 1.0  # 有任务
            task_info[1] = task['pickup_location'][0] / self.width
            task_info[2] = task['pickup_location'][1] / self.height
            task_info[3] = task['delivery_location'][0] / self.width
            task_info[4] = task['delivery_location'][1] / self.height
            task_info[5] = 1.0 if task['type'] == 'import' else 0.0

        return task_info

    def _get_path_occupancy(self) -> np.ndarray:
        """获取路径占用信息"""
        occupancy = np.zeros(self.config.NUM_HORIZONTAL_LANES, dtype=np.float32)

        for agv in self.agvs:
            lane = agv.current_lane
            if 0 <= lane < len(occupancy):
                occupancy[lane] += 1.0

        # 归一化
        occupancy = np.clip(occupancy / self.num_agvs, 0, 1)

        return occupancy

    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'current_time': self.current_time,
            'current_step': self.current_step,
            'num_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks),
            'collisions': self.episode_stats['collisions'],
            'total_reward': self.episode_stats['total_reward']
        }

    def render(self):
        """渲染环境(可选实现)"""
        if self.render_mode == 'human':
            # 可以用pygame实现可视化
            pass

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            import pygame
            pygame.quit()