"""
任务管理器模块 - v2.2版本
负责任务分配、状态更新、完成检测
✨ 新增：任务完成时间追踪

更新日期：2025.10.20
更新内容：添加任务开始时间和完成时间记录
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .agv import AGV
from .equipment import Task


class TaskManager:
    """
    任务管理器基类 - v2.2

    功能：
    1. 任务分配：根据不同策略分配任务给AGV
    2. 状态更新：检测AGV是否到达pickup/delivery点
    3. 完成检测：判定任务完成并返回已完成任务列表
    4. ✨ 时间追踪：记录任务开始和完成时间
    """

    def __init__(self, config):
        """
        初始化任务管理器

        Args:
            config: 环境配置对象（EnvConfig）
        """
        self.config = config
        self.assignment_strategy = config.TASK_ASSIGNMENT_STRATEGY
        self.arrival_threshold = config.ARRIVAL_THRESHOLD
        self.verbose = config.VERBOSE

        # 统计信息
        self.total_assigned = 0
        self.total_completed = 0

    def assign_tasks(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """
        任务分配主函数（根据策略选择不同的分配方法）

        Args:
            agvs: AGV列表
            tasks: 待分配任务列表
        """
        if self.assignment_strategy == 'nearest':
            self._assign_nearest(agvs, tasks)
        elif self.assignment_strategy == 'sequential':
            self._assign_sequential(agvs, tasks)
        elif self.assignment_strategy == 'random':
            self._assign_random(agvs, tasks)
        elif self.assignment_strategy == 'priority':
            self._assign_priority(agvs, tasks)
        else:
            # 默认使用顺序分配
            self._assign_sequential(agvs, tasks)

    def _assign_nearest(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """最近距离分配策略"""
        for agv in agvs:
            if agv.current_task is None:
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                distances = [
                    np.linalg.norm(agv.position - t.pickup_location)
                    for t in available_tasks
                ]

                nearest_idx = np.argmin(distances)
                nearest_task = available_tasks[nearest_idx]

                self._assign_single_task(agv, nearest_task)

    def _assign_sequential(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """顺序分配策略"""
        for agv in agvs:
            if agv.current_task is None:
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                task = available_tasks[0]
                self._assign_single_task(agv, task)

    def _assign_random(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """随机分配策略"""
        for agv in agvs:
            if agv.current_task is None:
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                task = np.random.choice(available_tasks)
                self._assign_single_task(agv, task)

    def _assign_priority(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """优先级分配策略"""
        for agv in agvs:
            if agv.current_task is None:
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                sorted_tasks = sorted(
                    available_tasks,
                    key=lambda t: t.priority,
                    reverse=True
                )
                task = sorted_tasks[0]

                self._assign_single_task(agv, task)

    def _assign_single_task(self, agv: AGV, task: Task, current_time: float = None) -> None:
        """
        将单个任务分配给指定AGV

        ✨ v2.2：记录任务开始时间

        Args:
            agv: 目标AGV
            task: 待分配任务
            current_time: 当前时间（秒）
        """
        # ✨ 记录任务开始时间
        if current_time is not None:
            task.start_time = current_time

        # 构建任务字典
        task_dict = {
            'id': task.id,
            'type': task.type,
            'pickup_location': task.pickup_location.copy(),
            'delivery_location': task.delivery_location.copy(),
            'status': 'assigned',
            'priority': task.priority,
            'start_time': task.start_time  # ✨ 添加开始时间
        }

        # 分配给AGV
        agv.assign_task(task_dict)

        # 更新Task对象状态
        task.assign_to_agv(agv.id)
        task.start()

        # 统计
        self.total_assigned += 1

        if self.verbose:
            time_str = f" at t={current_time:.1f}s" if current_time is not None else ""
            print(f"[TaskManager] Assigned task {task.id} "
                  f"({task.type}) to AGV{agv.id}{time_str}")

    def update_task_status(
            self,
            agvs: List[AGV],
            tasks: List[Task],
            current_time: float = None  # ✨ 新增参数：当前时间
    ) -> List[Task]:
        """
        更新任务状态，检测pickup和delivery完成

        ✨ v2.2：记录任务完成时间

        Args:
            agvs: AGV列表
            tasks: 当前任务列表
            current_time: 当前时间（秒）

        Returns:
            completed_tasks: 本轮完成的任务列表
        """
        completed_tasks = []

        for agv in agvs:
            # 跳过没有任务的AGV
            if agv.current_task is None:
                continue

            task_dict = agv.current_task

            # === 阶段1：前往pickup点拾取货物 ===
            if not agv.has_container:
                dist_to_pickup = np.linalg.norm(
                    agv.position - task_dict['pickup_location']
                )

                # 到达pickup点
                if dist_to_pickup < self.arrival_threshold:
                    agv.has_container = True

                    if self.verbose:
                        print(f"[TaskManager] ✓ AGV{agv.id} picked up "
                              f"container from {task_dict['pickup_location']} "
                              f"(Task {task_dict['id']})")

            # === 阶段2：运送货物到delivery点 ===
            else:
                dist_to_delivery = np.linalg.norm(
                    agv.position - task_dict['delivery_location']
                )

                # 到达delivery点，完成任务！
                if dist_to_delivery < self.arrival_threshold:
                    # 在任务列表中找到对应的Task对象
                    for t in tasks:
                        if t.id == task_dict['id']:
                            # ✨ 记录任务完成时间
                            if current_time is not None:
                                t.end_time = current_time

                                # ✨ 计算任务完成用时
                                if t.start_time is not None:
                                    t.completion_time = t.end_time - t.start_time
                                else:
                                    t.completion_time = None

                            t.complete()
                            completed_tasks.append(t)

                            # 统计
                            self.total_completed += 1

                            if self.verbose:
                                time_info = ""
                                if hasattr(t, 'completion_time') and t.completion_time is not None:
                                    time_info = f" in {t.completion_time:.1f}s"

                                print(f"[TaskManager] ★ AGV{agv.id} "
                                      f"COMPLETED task {t.id}{time_info}!")
                            break

                    # 标记任务完成
                    agv.current_task['status'] = 'completed'

                    # 清空AGV的当前任务
                    agv.complete_task()

        return completed_tasks

    def get_statistics(self) -> Dict:
        """
        获取任务管理统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_assigned': self.total_assigned,
            'total_completed': self.total_completed,
            'completion_rate': (
                self.total_completed / self.total_assigned
                if self.total_assigned > 0 else 0.0
            )
        }

    def reset(self):
        """重置统计信息"""
        self.total_assigned = 0
        self.total_completed = 0


class TaskManagerFactory:
    """任务管理器工厂类"""

    @staticmethod
    def create(strategy: str, config) -> TaskManager:
        """根据策略创建任务管理器"""
        if strategy == 'basic':
            return TaskManager(config)
        elif strategy == 'rl':
            return TaskManager(config)
        elif strategy == 'hybrid':
            return TaskManager(config)
        else:
            return TaskManager(config)


# ===== 使用示例 =====
if __name__ == "__main__":
    print("TaskManager模块已加载 - v2.2版本")
    print("✨ 新增功能：任务完成时间追踪")