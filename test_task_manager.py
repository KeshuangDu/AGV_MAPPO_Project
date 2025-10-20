"""
任务管理器测试脚本
用于验证任务分配和完成逻辑是否正常工作

使用方法：
    python test_task_manager.py

预期结果：
    - 应该看到任务分配信息
    - 应该看到AGV pickup货物
    - 应该看到任务完成信息
    - 最终tasks_completed > 0
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.port_env import PortEnvironment
from config.env_config import env_config


def main():
    print("=" * 60)
    print("任务管理器测试")
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
    print(f"  - QC位置: {env_config.QC_POSITIONS}")
    print(f"  - YC位置: {env_config.YC_POSITIONS}")

    # 打印初始任务信息
    print(f"\n初始任务列表：")
    for task in env.tasks:
        print(f"  Task {task.id}: {task.type}, "
              f"QC{task.qc_id}->YC{task.yc_id}, "
              f"pickup={task.pickup_location}, "
              f"delivery={task.delivery_location}")

    # 简单策略：全速前进
    actions = {
        f'agent_{i}': {
            'lane': i % 3,  # 分散到不同车道
            'direction': 0,  # 前进
            'motion': np.array([1.0, 0.0])  # 全速前进，不转向
        }
        for i in range(env.num_agvs)
    }

    print(f"\n开始运行...")
    print("=" * 60)

    # 运行最多1000步
    total_reward = 0
    last_completed = 0

    for step in range(1000):
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
                print(f"  AGV{i}: pos={agv.position}, "
                      f"has_task={has_task}, "
                      f"has_container={agv.has_container}")

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
        print(f"   完成率: {len(env.completed_tasks) / 5 * 100:.1f}%")
    else:
        print("\n❌ 测试失败！没有完成任何任务！")
        print("   请检查：")
        print("   1. AGV是否能够到达目标位置")
        print("   2. 到达阈值是否设置合理")
        print("   3. 任务分配是否正常")

    print("=" * 60)


if __name__ == "__main__":
    main()