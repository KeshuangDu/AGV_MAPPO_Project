"""
港口环境主类
实现水平布局的多AGV港口仿真环境
支持双向路由和多智能体交互

更新日期：2025.10.17
更新内容：放宽终止条件，避免过早结束训练
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import copy

from .agv import AGV
from .equipment import QuayCrane, YardCrane, Task

# 新增导入
from .task_manager import TaskManager, TaskManagerFactory
from .reward_shaper import RewardShaper, RewardShaperFactory


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

        # ========== 初始化任务管理器和奖励塑形器 ==========
        if config.USE_TASK_MANAGER:
            self.task_manager = TaskManagerFactory.create(
                'basic',  # 使用基础任务管理器
                config
            )
            self.reward_shaper = RewardShaperFactory.create(
                config.REWARD_TYPE,
                config
            )
            if config.VERBOSE:
                print(f"[PortEnv] Task Manager enabled: "
                      f"strategy={config.TASK_ASSIGNMENT_STRATEGY}")
                print(f"[PortEnv] Reward Shaper enabled: "
                      f"type={config.REWARD_TYPE}")
        else:
            self.task_manager = None
            self.reward_shaper = None
            if config.VERBOSE:
                print("[PortEnv] Using original task/reward logic")

    """
    port_env.py 中需要修改的部分（修正版）
    在原有文件中找到 _init_equipment() 方法，替换为以下代码
    """

    def _init_equipment(self):
        """
        初始化设备（AGV、QC、YC）
        支持水平和垂直两种布局
        """
        # 检查布局类型
        layout_type = getattr(self.config, 'LAYOUT_TYPE', 'horizontal')

        if layout_type == 'horizontal':
            self._init_horizontal_layout()
        elif layout_type == 'vertical':
            self._init_vertical_layout()
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

    def _init_horizontal_layout(self):
        """
        水平布局初始化（原有逻辑）
        - QC在左边（x=0附近）
        - YC在右边（x=640附近）
        - AGV在水平通道上移动
        """
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

        print(f"✅ 水平布局初始化完成:")
        print(f"   - QC数量: {len(self.qcs)} (左边)")
        print(f"   - YC数量: {len(self.ycs)} (右边)")
        print(f"   - AGV数量: {len(self.agvs)}")
        print(f"   - 水平通道数: {self.config.NUM_HORIZONTAL_LANES}")

    def _init_vertical_layout(self):
        """
        垂直布局初始化（新增）
        - QC在下边（y=0附近）
        - YC在上边（y=320附近）
        - AGV在垂直通道上移动
        """
        # 初始化AGV - 均匀分布在中间区域
        self.agvs = []
        for i in range(self.num_agvs):
            # AGV初始位置均匀分布在中间区域
            init_x = np.random.uniform(100, self.width - 100)
            init_y = np.random.uniform(100, self.height - 100)
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

        # 初始化岸桥(QC) - 位于底部
        # QC沿X轴均匀分布，Y坐标固定在底部
        self.qcs = []
        qc_x_positions = np.linspace(
            self.width * 0.2,
            self.width * 0.8,
            self.num_qcs
        )

        for i in range(self.num_qcs):
            pos = (qc_x_positions[i], 50.0)  # Y坐标固定在底部
            qc = QuayCrane(
                crane_id=i,
                position=pos,
                operation_time_range=self.config.QC_OPERATION_TIME
            )
            self.qcs.append(qc)

        # 初始化场桥(YC) - 位于顶部
        # YC沿X轴均匀分布，Y坐标固定在顶部
        self.ycs = []
        yc_x_positions = np.linspace(
            self.width * 0.2,
            self.width * 0.8,
            self.num_ycs
        )

        for i in range(self.num_ycs):
            pos = (yc_x_positions[i], self.height - 50.0)  # Y坐标固定在顶部
            yc = YardCrane(
                crane_id=i,
                position=pos,
                operation_time_range=self.config.YC_OPERATION_TIME
            )
            self.ycs.append(yc)

        print(f"✅ 垂直布局初始化完成:")
        print(f"   - QC数量: {len(self.qcs)} (底部)")
        print(f"   - YC数量: {len(self.ycs)} (顶部)")
        print(f"   - AGV数量: {len(self.agvs)}")
        print(f"   - 垂直通道数: {self.config.NUM_HORIZONTAL_LANES}")

    """
    使用说明：
    1. 在 port_env.py 中找到原有的 _init_equipment() 方法
    2. 替换为上面的新版本（包含布局类型判断）
    3. 添加 _init_horizontal_layout() 方法（将原有的初始化逻辑移入）
    4. 添加 _init_vertical_layout() 方法（新的垂直布局逻辑）

    关键修正：
    - ✅ 使用正确的参数名 crane_id（不是 qc_id）
    - ✅ 移除了不存在的 zone 参数
    - ✅ 使用与原代码相同的 AGV 初始化方式
    - ✅ 保持与原代码相同的参数顺序
    """

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

        # 重置任务管理器
        if self.task_manager:
            self.task_manager.reset()

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

        # ========== 使用任务管理器 ==========
        completed_this_step = []

        if self.task_manager:
            # 任务分配
            self.task_manager.assign_tasks(self.agvs, self.tasks)

            # 任务状态更新和完成检测
            completed_this_step = self.task_manager.update_task_status(
                self.agvs,
                self.tasks,
                self.current_time
            )

            # 移动已完成任务到completed_tasks
            for task in completed_this_step:
                if task in self.tasks:
                    self.tasks.remove(task)
                self.completed_tasks.append(task)
                self.episode_stats['tasks_completed'] += 1

        # 3. 检查碰撞
        collisions = self._check_collisions()

        # 4. 更新任务状态（如果不使用任务管理器）
        if not self.task_manager:
            self._update_tasks()

        # 5. 生成新任务
        if np.random.random() < self.task_gen_rate:
            self._generate_tasks(num_tasks=1)

        # ========== 使用奖励塑形器 ==========
        if self.reward_shaper:
            rewards = self.reward_shaper.compute_rewards(
                self.agvs,
                collisions,
                completed_this_step
            )
        else:
            # 使用原有奖励计算方法
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

        ✨ v2.1更新：添加BIDIRECTIONAL参数控制
        - BIDIRECTIONAL=True: 允许前进和后退
        - BIDIRECTIONAL=False: 强制只能前进
        """
        lane = action['lane']
        direction = action['direction']
        motion = action['motion']

        agv.current_lane = lane

        # ========== ✨ 核心修改：双向路由控制 ==========
        if self.config.BIDIRECTIONAL:
            # 双向模式：允许前进/后退切换
            target_forward = (direction == 0)
            if agv.moving_forward != target_forward:
                agv.switch_direction()

                if self.config.VERBOSE:
                    mode = "前进" if target_forward else "后退"
                    print(f"[PortEnv] AGV{agv.id} 切换方向 -> {mode}")
        else:
            # 单向模式：强制只能前进
            if not agv.moving_forward:
                # 如果当前是后退状态，强制切换为前进
                agv.moving_forward = True
                agv.velocity = abs(agv.velocity)  # 速度取绝对值

                if self.config.VERBOSE:
                    print(f"[PortEnv] AGV{agv.id} 单向模式，强制前进")

            # 如果网络尝试输出后退动作（direction=1），标记这个尝试
            if direction == 1:
                agv._attempted_backward = True
                if self.config.VERBOSE:
                    print(f"[PortEnv] AGV{agv.id} 尝试后退但被阻止（单向模式）")

        # ========== 以下保持不变 ==========
        acceleration = motion[0] * agv.max_accel
        steering = motion[1] * np.pi / 6

        agv.update_state(acceleration, steering, self.dt)

        # 边界限制
        agv.position[0] = np.clip(agv.position[0], 0, self.width)
        agv.position[1] = np.clip(agv.position[1], 0, self.height)

    def _check_collisions(self) -> List[Tuple[int, int]]:
        """检查AGV之间的碰撞"""
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
        """生成新任务"""
        for _ in range(num_tasks):
            if len(self.tasks) >= self.max_tasks:
                break

            task_type = np.random.choice(self.config.TASK_TYPES)
            qc_id = np.random.randint(0, self.num_qcs)
            yc_id = np.random.randint(0, self.num_ycs)

            task = Task(task_type, qc_id, yc_id)

            if task_type == 'import':
                task.pickup_location = self.qcs[qc_id].position.copy()
                task.delivery_location = self.ycs[yc_id].position.copy()
            else:
                task.pickup_location = self.ycs[yc_id].position.copy()
                task.delivery_location = self.qcs[qc_id].position.copy()

            self.tasks.append(task)

    def _update_tasks(self):
        """更新任务状态（原有方法，保留以兼容）"""
        for task in self.tasks[:]:
            if task.status == 'completed':
                self.completed_tasks.append(task)
                self.tasks.remove(task)
                self.episode_stats['tasks_completed'] += 1

    def _compute_rewards(self, collisions: List) -> Dict:
        """计算奖励（原有方法，保留以兼容）"""
        rewards = {}
        w = self.reward_weights

        for i, agv in enumerate(self.agvs):
            reward = 0.0

            reward += w['time_penalty']

            if agv.collision_flag:
                reward += w['collision']
                agv.collision_flag = False

            if agv.current_task and agv.current_task.get('status') == 'completed':
                reward += w['task_completion']

            if hasattr(agv, 'direction_changes'):
                if agv.direction_changes > 0 and agv.direction_changes < 3:
                    reward += w.get('bidirectional_bonus', 0)
                elif agv.direction_changes >= 3:
                    reward += w.get('direction_change', 0)

            rewards[f'agent_{i}'] = reward
            self.episode_stats['total_reward'] += reward

        return rewards

    def _check_done(self) -> bool:
        """
        检查是否结束
        
        ✨ 修改：放宽终止条件，避免过早结束
        """
        # 所有任务完成
        if len(self.tasks) == 0 and len(self.completed_tasks) > 0:
            return True

        # 碰撞次数过多（从30增加到100（改进版 v2））✨ 重要修改
        if self.episode_stats['collisions'] > 100:
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
        """获取单个AGV的观察"""
        own_state = np.array([
            agv.position[0] / self.width,
            agv.position[1] / self.height,
            agv.velocity / agv.max_speed,
            np.cos(agv.direction),
            np.sin(agv.direction),
            float(agv.has_container),
            float(agv.moving_forward)
        ], dtype=np.float32)

        nearby_agvs = self._get_nearby_agvs(agv, max_count=5)
        task_info = self._get_task_info(agv)
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
            task_info[0] = 1.0
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
            pass

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            import pygame
            pygame.quit()
