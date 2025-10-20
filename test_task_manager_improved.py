"""
改进版任务管理器测试脚本
添加简单的导航逻辑，让AGV能够到达目标位置
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.port_env import PortEnvironment
from config.env_config import env_config


def compute_navigation_action(agv, target_position):
    """
    计算导航到目标位置的动作

    Args:
        agv: AGV对象
        target_position: 目标位置 [x, y]

    Returns:
        action: 动作字典
    """
    # 计算目标方向
    delta = target_position - agv.position
    target_angle = np.arctan2(delta[1], delta[0])

    # 当前方向
    current_angle = agv.direction

    # 计算角度差（归一化到[-π, π]）
    angle_diff = target_angle - current_angle
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    # 距离
    distance = np.linalg.norm(delta)

    # 决策逻辑
    # 1. 如果角度差很大，先转向
    if abs(angle_diff) > np.pi / 6:  # 30度
        # 需要大转向
        steering = np.sign(angle_diff) * 1.0  # 最大转向
        acceleration = 0.3  # 慢速前进
    else:
        # 方向基本正确，加速前进
        steering = np.clip(angle_diff / (np.pi / 6), -1, 1)  # 微调方向

        # 根据距离调整速度
        if distance > 100:
            acceleration = 1.0  # 远距离全速
        elif distance > 50:
            acceleration = 0.7  # 中距离减速
        elif distance > 20:
            acceleration = 0.3  # 接近目标慢速
        else:
            acceleration = 0.1  # 非常接近，极慢速

    # 选择车道（简单策略：根据y坐标）
    if target_position[1] < 100:
        lane = 0
    elif target_position[1] < 200:
        lane = 1
    else:
        lane = 2

    # 方向（前进）
    direction = 0  # 0=前进，1=后退

    return {
        'lane': lane,
        'direction': direction,
        'motion': np.array([acceleration, steering])
    }


def main():
    print("=" * 60)
    print("任务管理器测试 - 改进版（带导航）")
    print("=" * 60)

    # 启用详细输出
    env_config.VERBOSE = True
    env_config.USE_TASK_MANAGER = True
    env_config.REWARD_TYPE = 'dense'

    print(f"\n配置：")
    print(f"  - 任务管理器: {env_config.USE_TASK_MANAGER}")
    print(f"  - 分配策略: {env_config.TASK_ASSIGNMENT_STRATEGY}")
    print(f"  - 奖励类型: {env_config.REWARD_TYPE}")
    print(f"  - 详细输出: {env_config.VERBOSE}")
    print(f"  - 到达阈值: {env_config.ARRIVAL_THRESHOLD}米")

    # 创建环境
    env = PortEnvironment(env_config)
    obs, info = env.reset()

    print(f"\n初始状态：")
    print(f"  - 任务数量: {len(env.tasks)}")
    print(f"  - AGV数量: {len(env.agvs)}")

    # 打印初始任务信息
    print(f"\n初始任务列表：")
    for task in env.tasks:
        print(f"  Task {task.id}: {task.type}, "
              f"QC{task.qc_id}->YC{task.yc_id}, "
              f"pickup={task.pickup_location}, "
              f"delivery={task.delivery_location}")

    print(f"\n开始运行（使用导航逻辑）...")
    print("=" * 60)

    total_reward = 0
    last_completed = 0

    for step in range(2000):  # 增加到2000步
        # 为每个AGV计算导航动作
        actions = {}
        for i, agv in enumerate(env.agvs):
            if agv.current_task is not None:
                # 确定当前目标
                if not agv.has_container:
                    target = agv.current_task['pickup_location']
                else:
                    target = agv.current_task['delivery_location']

                # 计算导航动作
                action = compute_navigation_action(agv, target)
            else:
                # 没有任务，原地待命
                action = {
                    'lane': i % 3,
                    'direction': 0,
                    'motion': np.array([0.0, 0.0])
                }

            actions[f'agent_{i}'] = action

        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)

        step_reward = sum(rewards.values())
        total_reward += step_reward

        # 检测任务完成
        if len(env.completed_tasks) > last_completed:
            print(f"\n🎉 [Step {step}] 新完成 "
                  f"{len(env.completed_tasks) - last_completed} 个任务!")
            last_completed = len(env.completed_tasks)

        # 每100步打印状态
        if step % 100 == 0:
            print(f"\n--- Step {step} ---")
            print(f"  已完成任务: {len(env.completed_tasks)}")
            print(f"  待完成任务: {len(env.tasks)}")
            print(f"  步奖励: {step_reward:.2f}")
            print(f"  总奖励: {total_reward:.2f}")

            # 打印每个AGV状态
            for i, agv in enumerate(env.agvs):
                has_task = agv.current_task is not None
                if has_task:
                    if not agv.has_container:
                        target = agv.current_task['pickup_location']
                        dist = np.linalg.norm(agv.position - target)
                        phase = "→pickup"
                    else:
                        target = agv.current_task['delivery_location']
                        dist = np.linalg.norm(agv.position - target)
                        phase = "→delivery"
                    print(f"  AGV{i}: pos={agv.position}, {phase}, dist={dist:.1f}m")
                else:
                    print(f"  AGV{i}: pos={agv.position}, idle")

        # 如果提前结束
        if terminated['__all__']:
            print(f"\n环境在第 {step} 步终止")
            break

    # 最终结果
    print("\n" + "=" * 60)
    print("测试结束 - 最终结果")
    print("=" * 60)
    print(f"✓ 运行步数: {step + 1}")
    print(f"✓ 完成任务数: {len(env.completed_tasks)}")
    print(f"✓ 剩余任务数: {len(env.tasks)}")
    print(f"✓ 总奖励: {total_reward:.2f}")
    print(f"✓ 平均步奖励: {total_reward / (step + 1):.3f}")
    print(f"✓ 碰撞次数: {env.episode_stats['collisions']}")

    # 判断测试是否成功
    if len(env.completed_tasks) > 0:
        print("\n✅ 测试成功！任务管理器工作正常！")
        completion_rate = len(env.completed_tasks) / (len(env.completed_tasks) + len(env.tasks)) * 100
        print(f"   完成率: {completion_rate:.1f}%")
    else:
        print("\n❌ 测试失败！没有完成任何任务！")
        print("   可能的问题：")
        print("   1. 导航逻辑需要调优")
        print("   2. 到达阈值可能需要调整")
        print("   3. AGV速度配置可能不合理")

    print("=" * 60)


if __name__ == "__main__":
    main()