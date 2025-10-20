"""
任务管理器模块
负责任务分配、状态更新、完成检测
支持多种分配策略，便于消融实验

作者：AGV_MAPPO_Project
日期：2025.10.17
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .agv import AGV
from .equipment import Task


class TaskManager:
    """
    任务管理器基类

    功能：
    1. 任务分配：根据不同策略分配任务给AGV
    2. 状态更新：检测AGV是否到达pickup/delivery点
    3. 完成检测：判定任务完成并返回已完成任务列表
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
        """
        最近距离分配策略
        为每个空闲AGV分配距离最近的任务

        适用场景：传统调度算法对比baseline
        """
        for agv in agvs:
            if agv.current_task is None:
                # 筛选待分配任务
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                # 计算到每个任务pickup点的距离
                distances = [
                    np.linalg.norm(agv.position - t.pickup_location)
                    for t in available_tasks
                ]

                # 选择最近的任务
                nearest_idx = np.argmin(distances)
                nearest_task = available_tasks[nearest_idx]

                self._assign_single_task(agv, nearest_task)

    def _assign_sequential(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """
        顺序分配策略（简单baseline）
        按任务列表顺序依次分配给空闲AGV

        适用场景：RL训练初期，简单baseline
        """
        for agv in agvs:
            if agv.current_task is None:
                # 筛选待分配任务
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                # 选择第一个任务
                task = available_tasks[0]
                self._assign_single_task(agv, task)

    def _assign_random(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """
        随机分配策略
        随机选择任务分配给空闲AGV

        适用场景：消融实验，测试随机策略作为baseline
        """
        for agv in agvs:
            if agv.current_task is None:
                # 筛选待分配任务
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                # 随机选择
                task = np.random.choice(available_tasks)
                self._assign_single_task(agv, task)

    def _assign_priority(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """
        优先级分配策略
        优先分配高优先级任务

        适用场景：考虑任务紧急度的调度
        """
        for agv in agvs:
            if agv.current_task is None:
                # 筛选待分配任务
                available_tasks = [t for t in tasks if t.status == 'pending']
                if not available_tasks:
                    continue

                # 按优先级排序，选择最高优先级
                sorted_tasks = sorted(
                    available_tasks,
                    key=lambda t: t.priority,
                    reverse=True
                )
                task = sorted_tasks[0]

                self._assign_single_task(agv, task)

    def _assign_single_task(self, agv: AGV, task: Task) -> None:
        """
        将单个任务分配给指定AGV

        Args:
            agv: 目标AGV
            task: 待分配任务
        """
        # 构建任务字典
        task_dict = {
            'id': task.id,
            'type': task.type,
            'pickup_location': task.pickup_location.copy(),
            'delivery_location': task.delivery_location.copy(),
            'status': 'assigned',
            'priority': task.priority
        }

        # 分配给AGV
        agv.assign_task(task_dict)

        # 更新Task对象状态
        task.assign_to_agv(agv.id)
        task.start()

        # 统计
        self.total_assigned += 1

        if self.verbose:
            print(f"[TaskManager] Assigned task {task.id} "
                  f"({task.type}) to AGV{agv.id}")

    def update_task_status(
            self,
            agvs: List[AGV],
            tasks: List[Task]
    ) -> List[Task]:
        """
        更新任务状态，检测pickup和delivery完成

        Args:
            agvs: AGV列表
            tasks: 当前任务列表

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
                    # 标记任务完成
                    agv.current_task['status'] = 'completed'

                    # 在任务列表中找到对应的Task对象
                    for t in tasks:
                        if t.id == task_dict['id']:
                            t.complete()
                            completed_tasks.append(t)

                            # 统计
                            self.total_completed += 1

                            if self.verbose:
                                print(f"[TaskManager] ★ AGV{agv.id} "
                                      f"COMPLETED task {t.id} "
                                      f"at {task_dict['delivery_location']}!")
                            break

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
    """
    任务管理器工厂类
    便于扩展不同类型的任务管理器
    """

    @staticmethod
    def create(strategy: str, config) -> TaskManager:
        """
        根据策略创建任务管理器

        Args:
            strategy: 任务管理策略类型
                - 'basic': 基础任务管理器
                - 'rl': RL控制的任务管理器（未来扩展）
                - 'hybrid': 混合策略（未来扩展）
            config: 环境配置

        Returns:
            TaskManager实例
        """
        if strategy == 'basic':
            return TaskManager(config)
        elif strategy == 'rl':
            # 未来可以实现：由RL智能体决定任务分配
            # return RLTaskManager(config)
            return TaskManager(config)
        elif strategy == 'hybrid':
            # 未来可以实现：结合启发式规则和RL
            # return HybridTaskManager(config)
            return TaskManager(config)
        else:
            # 默认返回基础版本
            return TaskManager(config)


# ===== 使用示例 =====
if __name__ == "__main__":
    # 这是一个使用示例（实际使用时不需要运行此部分）
    print("TaskManager模块已加载")
    print("支持的分配策略：")
    print("  - 'sequential': 顺序分配")
    print("  - 'nearest': 最近距离分配")
    print("  - 'random': 随机分配")
    print("  - 'priority': 优先级分配")