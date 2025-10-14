"""
港口设备类
定义QC(岸桥)和YC(场桥)的行为
"""

import numpy as np
from typing import Tuple, Optional, Dict


class Equipment:
    """设备基类"""

    def __init__(
            self,
            equipment_id: int,
            position: Tuple[float, float],
            equipment_type: str,
            operation_time_range: Tuple[float, float] = (40, 200)
    ):
        """
        初始化设备

        Args:
            equipment_id: 设备编号
            position: 设备位置
            equipment_type: 设备类型 ('QC' or 'YC')
            operation_time_range: 操作时间范围(秒)
        """
        self.id = equipment_id
        self.position = np.array(position, dtype=np.float32)
        self.type = equipment_type
        self.operation_time_range = operation_time_range

        # 状态
        self.is_busy = False
        self.current_operation_time = 0.0
        self.remaining_time = 0.0

        # 任务队列
        self.task_queue = []
        self.completed_tasks = []

    def start_operation(self, task: Dict):
        """
        开始操作

        Args:
            task: 任务字典
        """
        if not self.is_busy:
            self.is_busy = True
            # 随机生成操作时间
            self.current_operation_time = np.random.uniform(
                *self.operation_time_range
            )
            self.remaining_time = self.current_operation_time
            self.task_queue.append(task)

    def update(self, dt: float):
        """
        更新设备状态

        Args:
            dt: 时间步长
        """
        if self.is_busy:
            self.remaining_time -= dt
            if self.remaining_time <= 0:
                self.complete_operation()

    def complete_operation(self):
        """完成当前操作"""
        if self.task_queue:
            completed_task = self.task_queue.pop(0)
            self.completed_tasks.append(completed_task)
        self.is_busy = False
        self.remaining_time = 0.0

    def get_state(self) -> Dict:
        """
        获取设备状态

        Returns:
            状态字典
        """
        return {
            'id': self.id,
            'type': self.type,
            'position': self.position.copy(),
            'is_busy': self.is_busy,
            'remaining_time': self.remaining_time,
            'queue_length': len(self.task_queue)
        }

    def reset(self):
        """重置设备状态"""
        self.is_busy = False
        self.remaining_time = 0.0
        self.task_queue = []
        self.completed_tasks = []


class QuayCrane(Equipment):
    """岸桥类(QC)"""

    def __init__(
            self,
            crane_id: int,
            position: Tuple[float, float],
            operation_time_range: Tuple[float, float] = (40, 200)
    ):
        super().__init__(crane_id, position, 'QC', operation_time_range)

        # 岸桥特有属性
        self.bay_number = crane_id  # 对应船舶的bay号

    def __repr__(self):
        return f"QC_{self.id}(pos={self.position}, busy={self.is_busy})"


class YardCrane(Equipment):
    """场桥类(YC)"""

    def __init__(
            self,
            crane_id: int,
            position: Tuple[float, float],
            operation_time_range: Tuple[float, float] = (40, 200)
    ):
        super().__init__(crane_id, position, 'YC', operation_time_range)

        # 场桥特有属性
        self.block_number = crane_id  # 对应堆场的block号

    def __repr__(self):
        return f"YC_{self.id}(pos={self.position}, busy={self.is_busy})"


class Task:
    """任务类"""

    task_counter = 0  # 任务计数器

    def __init__(
            self,
            task_type: str,
            qc_id: int,
            yc_id: int,
            priority: float = 1.0
    ):
        """
        初始化任务

        Args:
            task_type: 任务类型 ('import' or 'export')
            qc_id: 岸桥ID
            yc_id: 场桥ID
            priority: 优先级
        """
        self.id = Task.task_counter
        Task.task_counter += 1

        self.type = task_type
        self.qc_id = qc_id
        self.yc_id = yc_id
        self.priority = priority

        # 任务状态
        self.status = 'pending'  # pending, assigned, in_progress, completed
        self.assigned_agv = None
        self.start_time = None
        self.end_time = None

        # 位置信息(根据任务类型确定)
        if task_type == 'import':
            # 进口: QC -> YC
            self.pickup_location = None  # 将由QC位置确定
            self.delivery_location = None  # 将由YC位置确定
        else:
            # 出口: YC -> QC
            self.pickup_location = None
            self.delivery_location = None

    def assign_to_agv(self, agv_id: int):
        """分配给AGV"""
        self.assigned_agv = agv_id
        self.status = 'assigned'

    def start(self):
        """开始任务"""
        self.status = 'in_progress'
        self.start_time = 0  # 将由环境时间确定

    def complete(self):
        """完成任务"""
        self.status = 'completed'
        self.end_time = 0  # 将由环境时间确定

    def get_info(self) -> Dict:
        """获取任务信息"""
        return {
            'id': self.id,
            'type': self.type,
            'qc_id': self.qc_id,
            'yc_id': self.yc_id,
            'priority': self.priority,
            'status': self.status,
            'assigned_agv': self.assigned_agv,
            'pickup': self.pickup_location,
            'delivery': self.delivery_location
        }

    def __repr__(self):
        return f"Task_{self.id}({self.type}, QC{self.qc_id}->YC{self.yc_id}, {self.status})"