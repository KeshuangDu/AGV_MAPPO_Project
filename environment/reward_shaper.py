"""
奖励塑形器模块 - 改进版 v2
基于调试评估结果优化

更新日期：2025.10.17
更新内容：
1. 增加加速奖励
2. 增强接近目标奖励
3. 减小负奖励影响
"""

import numpy as np
from typing import List, Dict, Tuple


class RewardShaper:
    """奖励塑形器 - 改进版"""

    def __init__(self, config):
        """初始化奖励塑形器"""
        self.config = config
        self.weights = config.REWARD_WEIGHTS
        self.reward_type = config.REWARD_TYPE
        self.verbose = config.VERBOSE

    def compute_rewards(
            self,
            agvs: List,
            collisions: List[Tuple[int, int]],
            completed_tasks: List
    ) -> Dict[str, float]:
        """计算奖励（主入口）"""
        if self.reward_type == 'sparse':
            return self._compute_sparse_rewards(
                agvs, collisions, completed_tasks
            )
        elif self.reward_type == 'dense':
            return self._compute_dense_rewards(
                agvs, collisions, completed_tasks
            )
        else:
            return self._compute_dense_rewards(
                agvs, collisions, completed_tasks
            )

    def _compute_sparse_rewards(
            self,
            agvs: List,
            collisions: List[Tuple[int, int]],
            completed_tasks: List
    ) -> Dict[str, float]:
        """稀疏奖励模式"""
        rewards = {}
        w = self.weights

        for i, agv in enumerate(agvs):
            reward = 0.0

            # 基础时间惩罚
            reward += w['time_penalty']

            # 碰撞惩罚
            if agv.collision_flag:
                reward += w['collision']
                agv.collision_flag = False

            # 任务完成大奖励
            if agv.current_task and agv.current_task.get('status') == 'completed':
                reward += w['task_completion']

                if self.verbose:
                    print(f"[RewardShaper] AGV{i} got task completion "
                          f"reward: +{w['task_completion']}")

            rewards[f'agent_{i}'] = reward

        return rewards

    def _compute_dense_rewards(
            self,
            agvs: List,
            collisions: List[Tuple[int, int]],
            completed_tasks: List
    ) -> Dict[str, float]:
        """
        密集奖励模式 - 改进版 v2 ✨✨
        
        基于调试发现的改进：
        1. 增加加速奖励（解决总是减速的问题）
        2. 增强接近目标奖励（AGV已能到31米，需要更强引导）
        3. 新增25米内奖励（填补50米和15米之间的空白）
        """
        rewards = {}
        w = self.weights

        for i, agv in enumerate(agvs):
            reward = 0.0

            # === 1. 基础时间惩罚 ===
            reward += w['time_penalty']

            # === 2. 任务相关奖励（密集奖励核心）===
            if agv.current_task is not None:
                task = agv.current_task

                # 确定当前目标位置
                if not agv.has_container:
                    target = task['pickup_location']
                    phase = 'pickup'
                else:
                    target = task['delivery_location']
                    phase = 'delivery'

                # 2.1 距离变化奖励（引导AGV朝目标移动）✨
                current_dist = np.linalg.norm(agv.position - target)

                # 初始化上一步距离
                if not hasattr(agv, 'prev_dist_to_target'):
                    agv.prev_dist_to_target = current_dist

                # 计算距离变化
                dist_delta = agv.prev_dist_to_target - current_dist

                # 只奖励靠近
                if 'distance_progress' in w:
                    reward += w['distance_progress'] * max(0, dist_delta)

                # 更新距离记录
                agv.prev_dist_to_target = current_dist

                # 2.2 接近目标的阶段性奖励（增强版）✨✨
                if current_dist < 50.0 and 'approach_bonus_50' in w:
                    if not hasattr(agv, '_approach_50_rewarded'):
                        reward += w['approach_bonus_50']
                        agv._approach_50_rewarded = True

                # ✨✨ 新增：25米内奖励（填补空白）
                if current_dist < 25.0 and 'reaching_near_target' in w:
                    if not hasattr(agv, '_approach_25_rewarded'):
                        reward += w['reaching_near_target']
                        agv._approach_25_rewarded = True
                        if self.verbose:
                            print(f"[RewardShaper] AGV{i} 到达25米内！ "
                                  f"+{w['reaching_near_target']}")

                if current_dist < 30.0 and 'approach_bonus_30' in w:
                    if not hasattr(agv, '_approach_30_rewarded'):
                        reward += w['approach_bonus_30']
                        agv._approach_30_rewarded = True

                # ✨✨ 增强：15米内大奖励
                if current_dist < 15.0 and 'approach_bonus_15' in w:
                    if not hasattr(agv, '_approach_15_rewarded'):
                        reward += w['approach_bonus_15']
                        agv._approach_15_rewarded = True
                        if self.verbose:
                            print(f"[RewardShaper] AGV{i} 进入15米内！ "
                                  f"+{w['approach_bonus_15']}")

                # 2.3 成功pickup的阶段奖励
                if agv.has_container and not hasattr(agv, '_pickup_rewarded'):
                    if 'pickup_success' in w:
                        reward += w['pickup_success']
                        agv._pickup_rewarded = True

                        # 重置接近奖励标记（为delivery阶段准备）
                        if hasattr(agv, '_approach_50_rewarded'):
                            delattr(agv, '_approach_50_rewarded')
                        if hasattr(agv, '_approach_25_rewarded'):
                            delattr(agv, '_approach_25_rewarded')
                        if hasattr(agv, '_approach_30_rewarded'):
                            delattr(agv, '_approach_30_rewarded')
                        if hasattr(agv, '_approach_15_rewarded'):
                            delattr(agv, '_approach_15_rewarded')

                        if self.verbose:
                            print(f"[RewardShaper] AGV{i} pickup成功！ "
                                  f"+{w['pickup_success']}")

                # 2.4 任务完成大奖励
                if task.get('status') == 'completed':
                    reward += w['task_completion']

                    # 重置所有标记
                    for attr in ['_pickup_rewarded', 'prev_dist_to_target',
                                 '_approach_50_rewarded', '_approach_25_rewarded',
                                 '_approach_30_rewarded', '_approach_15_rewarded']:
                        if hasattr(agv, attr):
                            delattr(agv, attr)

                    if self.verbose:
                        print(f"[RewardShaper] AGV{i} 完成任务！ "
                              f"+{w['task_completion']}")

            # === 3. 安全相关惩罚（减小影响）✨ ===
            if agv.collision_flag:
                reward += w['collision']  # 已减小到-10
                agv.collision_flag = False

            # === 4. 双向路由相关 ===
            if hasattr(agv, '_last_direction'):
                if agv.moving_forward != agv._last_direction:
                    if 'direction_change' in w:
                        reward += w['direction_change']
            agv._last_direction = agv.moving_forward

            # === 5. ✨✨ 新增：加速奖励（解决总是减速的问题）===
            if hasattr(agv, 'velocity') and agv.velocity > 0:
                # 如果AGV在移动（速度>0），给予小奖励
                if 'acceleration_bonus' in w:
                    reward += w['acceleration_bonus']
            
            # 记录速度用于下次检查
            if not hasattr(agv, '_prev_velocity'):
                agv._prev_velocity = 0
            
            # 如果速度增加（加速），额外奖励
            if agv.velocity > agv._prev_velocity and 'acceleration_bonus' in w:
                reward += w['acceleration_bonus'] * 0.5
            
            agv._prev_velocity = agv.velocity

            rewards[f'agent_{i}'] = reward

        return rewards

    def reset(self):
        """重置奖励塑形器状态"""
        pass


class RewardShaperFactory:
    """奖励塑形器工厂类"""

    @staticmethod
    def create(reward_type: str, config):
        """根据类型创建奖励塑形器"""
        return RewardShaper(config)


if __name__ == "__main__":
    print("RewardShaper模块已加载 - 改进版 v2")
    print("\n改进内容：")
    print("  1. ✅ 新增25米内奖励（填补50米和15米之间的空白）")
    print("  2. ✅ 增强15米内奖励（从5.0增加到15.0）")
    print("  3. ✅ 新增加速奖励（鼓励AGV积极移动）")
    print("  4. ✅ 减小碰撞惩罚（从-20减小到-10）")
