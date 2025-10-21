"""
测试双向/单向路由的差异
验证BIDIRECTIONAL参数是否正确工作

运行方法：
    python test_bidirectional.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.env_config import env_config
from environment.port_env import PortEnvironment
import numpy as np


def test_mode(bidirectional: bool, mode_name: str):
    """测试指定模式"""
    print(f"\n{'=' * 60}")
    print(f"🧪 测试模式: {mode_name}")
    print(f"   BIDIRECTIONAL = {bidirectional}")
    print(f"{'=' * 60}")

    # 设置模式
    env_config.BIDIRECTIONAL = bidirectional
    env_config.VERBOSE = False

    # 创建环境
    env = PortEnvironment(env_config)
    obs, _ = env.reset()

    # 统计后退次数
    backward_attempts = 0  # 网络尝试后退的次数
    backward_successes = 0  # 实际后退成功的次数
    forward_count = 0  # 前进次数

    print(f"\n开始测试（运行100步）...")

    for step in range(100):
        # 构造动作（故意尝试后退）
        actions = {}
        for i in range(env_config.NUM_AGVS):
            # 随机选择前进或后退
            direction = 1 if np.random.random() < 0.5 else 0
            if direction == 1:
                backward_attempts += 1
            else:
                forward_count += 1

            actions[f'agent_{i}'] = {
                'lane': np.random.randint(0, 3),
                'direction': direction,  # 0=前进, 1=后退
                'motion': np.random.uniform(-0.5, 0.5, size=2)
            }

        # 执行
        obs, rewards, terminated, truncated, info = env.step(actions)

        # 统计实际后退次数（检查AGV状态）
        for agv in env.agvs:
            if not agv.moving_forward:
                backward_successes += 1

        if terminated['__all__']:
            break

    # 结果分析
    print(f"\n📊 测试结果：")
    print(f"{'=' * 60}")
    print(f"  尝试前进次数: {forward_count}")
    print(f"  尝试后退次数: {backward_attempts}")
    print(f"  实际后退成功次数: {backward_successes}")
    print(f"  后退成功率: {backward_successes / max(backward_attempts, 1) * 100:.1f}%")
    print(f"{'=' * 60}")

    # 验证结果
    if bidirectional:
        print(f"\n✅ 双向模式验证：")
        print(f"   期望：AGV应该能够后退")
        if backward_successes > 0:
            print(f"   结果：✅ 通过！AGV成功后退了{backward_successes}次")
            return True
        else:
            print(f"   结果：❌ 失败！AGV没有后退，但应该能后退")
            return False
    else:
        print(f"\n✅ 单向模式验证：")
        print(f"   期望：AGV不应该后退（即使尝试）")
        if backward_successes == 0:
            print(f"   结果：✅ 通过！AGV始终保持前进，没有后退")
            return True
        else:
            print(f"   结果：❌ 失败！AGV后退了{backward_successes}次，但不应该后退")
            return False


def main():
    print("\n" + "=" * 60)
    print("🎯 双向/单向路由验证测试")
    print("=" * 60)
    print("目的：验证BIDIRECTIONAL参数是否正确控制AGV行为")
    print("=" * 60)

    # 测试双向模式
    result_bi = test_mode(
        bidirectional=True,
        mode_name="双向路由（BIDIRECTIONAL=True）"
    )

    # 测试单向模式
    result_uni = test_mode(
        bidirectional=False,
        mode_name="单向路由（BIDIRECTIONAL=False）"
    )

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    print(f"  双向模式测试: {'✅ 通过' if result_bi else '❌ 失败'}")
    print(f"  单向模式测试: {'✅ 通过' if result_uni else '❌ 失败'}")
    print("=" * 60)

    if result_bi and result_uni:
        print("\n✅✅✅ 所有测试通过！BIDIRECTIONAL参数工作正常！")
        print("\n下一步：")
        print("  1. 训练水平双向: python train_h_bi.py")
        print("  2. 训练水平单向: python train_h_uni.py")
        print("  3. 对比两者的训练结果")
    else:
        print("\n❌ 测试失败！请检查以下文件的修改：")
        print("  1. environment/port_env.py - _execute_action方法")
        print("  2. environment/reward_shaper.py - 单向惩罚逻辑")
        print("  3. environment/agv.py - 后退尝试标记")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()