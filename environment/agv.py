"""
AGV实体类
定义AGV的状态、行为和属性

✨ v2.1更新：添加单向/双向模式支持
"""

import numpy as np
from typing import Tuple, Optional, Dict


class AGV:
    """自动导引车类"""

    def __init__(
            self,
            agv_id: int,
            init_position: Tuple[float, float],
            max_speed: float = 4.0,
            max_accel: float = 1.0,
            length: float = 6.0,
            width: float = 3.0
    ):
        """
        初始化AGV

        Args:
            agv_id: AGV编号
            init_position: 初始位置 (x, y)
            max_speed: 最大速度(米/秒)
            max_accel: 最大加速度(米/秒²)
            length: 长度(米)
            width: 宽度(米)
        """
        self.id = agv_id

        # 物理属性
        self.length = length
        self.width = width
        self.max_speed = max_speed
        self.max_accel = max_accel

        # 运动状态
        self.position = np.array(init_position, dtype=np.float32)
        self.velocity = 0.0
        self.direction = 0.0  # 方向角(弧度)

        # 双向运动状态
        self.moving_forward = True  # True: 前进, False: 后退

        # ✨ 新增：记录是否尝试后退（用于单向模式奖励）
        self._attempted_backward = False

        # 任务状态
        self.current_task = None
        self.has_container = False
        self.task_history = []

        # 路径状态
        self.current_lane = 0
        self.target_position = None

        # 历史轨迹(用于轨迹预测)
        self.trajectory_history = []
        self.max_history_len = 20

        # 碰撞检测
        self.collision_flag = False

    def update_state(
            self,
            acceleration: float,
            steering: float,
            dt: float = 0.5
    ):
        """
        更新AGV状态

        Args:
            acceleration: 加速度
            steering: 转向角
            dt: 时间步长
        """
        # 限制加速度
        acceleration = np.clip(acceleration, -self.max_accel, self.max_accel)

        # 更新速度
        if self.moving_forward:
            self.velocity += acceleration * dt
        else:
            self.velocity -= acceleration * dt

        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)

        # 更新方向
        self.direction += steering * dt
        self.direction = self.direction % (2 * np.pi)

        # 更新位置
        dx = self.velocity * np.cos(self.direction) * dt
        dy = self.velocity * np.sin(self.direction) * dt
        self.position += np.array([dx, dy])

        # 保存轨迹
        self.trajectory_history.append(self.position.copy())
        if len(self.trajectory_history) > self.max_history_len:
            self.trajectory_history.pop(0)

    def assign_task(self, task: Dict):
        """
        分配任务

        Args:
            task: 任务字典，包含pickup_location, delivery_location等
        """
        self.current_task = task
        self.target_position = task['pickup_location']
        self.task_history.append(task['id'])

    def complete_task(self):
        """完成当前任务"""
        if self.current_task is not None:
            self.current_task = None
            self.has_container = False
            self.target_position = None

    def switch_direction(self):
        """
        切换运动方向(双向路由关键)

        ✨ 新增：标记后退尝试（用于单向模式检测）
        """
        # 如果尝试从前进切换到后退，标记这个尝试
        if self.moving_forward:
            self._attempted_backward = True

        self.moving_forward = not self.moving_forward
        self.velocity = -self.velocity

    def get_state(self) -> Dict:
        """
        获取AGV状态

        Returns:
            状态字典
        """
        return {
            'id': self.id,
            'position': self.position.copy(),
            'velocity': self.velocity,
            'direction': self.direction,
            'moving_forward': self.moving_forward,
            'current_lane': self.current_lane,
            'has_container': self.has_container,
            'current_task': self.current_task,
            'trajectory_history': self.trajectory_history[-10:],  # 最近10步
            'collision_flag': self.collision_flag
        }

    def get_bounding_box(self) -> np.ndarray:
        """
        获取边界框(用于碰撞检测)

        Returns:
            边界框四个角点坐标
        """
        # 简化为矩形
        half_length = self.length / 2
        half_width = self.width / 2

        # 计算旋转后的四个角点
        cos_d = np.cos(self.direction)
        sin_d = np.sin(self.direction)

        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        rotation_matrix = np.array([
            [cos_d, -sin_d],
            [sin_d, cos_d]
        ])

        rotated_corners = corners @ rotation_matrix.T
        return rotated_corners + self.position

    def distance_to(self, other_agv: 'AGV') -> float:
        """
        计算与另一个AGV的距离

        Args:
            other_agv: 另一个AGV

        Returns:
            距离(米)
        """
        return np.linalg.norm(self.position - other_agv.position)

    def is_collision(self, other_agv: 'AGV', safe_distance: float = 10.0) -> bool:
        """
        检测是否与另一个AGV碰撞

        Args:
            other_agv: 另一个AGV
            safe_distance: 安全距离

        Returns:
            是否碰撞
        """
        distance = self.distance_to(other_agv)
        return distance < safe_distance

    def reset(self, init_position: Tuple[float, float]):
        """
        重置AGV状态

        Args:
            init_position: 初始位置
        """
        self.position = np.array(init_position, dtype=np.float32)
        self.velocity = 0.0
        self.direction = 0.0
        self.moving_forward = True
        self.current_task = None
        self.has_container = False
        self.task_history = []
        self.trajectory_history = []
        self.collision_flag = False
        self.target_position = None

        # ✨ 重置后退尝试标记
        self._attempted_backward = False

    def __repr__(self):
        return f"AGV_{self.id}(pos={self.position}, vel={self.velocity:.2f}, " \
               f"forward={self.moving_forward}, task={self.current_task is not None})"