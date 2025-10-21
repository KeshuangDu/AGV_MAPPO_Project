"""
批量运行所有实验
依次运行：水平双向、水平单向

运行方法：
    python run_all_experiments.py

或单独运行某个实验：
    python run_all_experiments.py --exp h_bi   # 只运行水平双向
    python run_all_experiments.py --exp h_uni  # 只运行水平单向
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def run_experiment(exp_name: str, exp_desc: str, script_name: str):
    """运行单个实验"""
    print("\n" + "=" * 80)
    print(f"🚀 开始实验: {exp_name}")
    print(f"   描述: {exp_desc}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # 运行训练脚本
    try:
        subprocess.run(
            [sys.executable, script_name],
            check=True
        )

        print("\n" + "=" * 80)
        print(f"✅ 实验完成: {exp_name}")
        print("=" * 80 + "\n")
        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"❌ 实验失败: {exp_name}")
        print(f"   错误代码: {e.returncode}")
        print("=" * 80 + "\n")
        return False
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(f"⚠️  实验被用户中断: {exp_name}")
        print("=" * 80 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='运行AGV MAPPO对比实验')
    parser.add_argument(
        '--exp',
        type=str,
        choices=['all', 'h_bi', 'h_uni'],
        default='all',
        help='选择要运行的实验 (all/h_bi/h_uni)'
    )
    parser.add_argument(
        '--test-first',
        action='store_true',
        help='训练前先运行测试验证'
    )

    args = parser.parse_args()

    # 定义实验配置
    experiments = {
        'h_bi': {
            'name': '实验1 - 水平双向',
            'desc': '水平布局 + 双向路由（baseline）',
            'script': 'train_h_bi.py'
        },
        'h_uni': {
            'name': '实验2 - 水平单向',
            'desc': '水平布局 + 单向路由（对比实验）',
            'script': 'train_h_uni.py'
        }
    }

    # 欢迎信息
    print("\n" + "=" * 80)
    print("🎯 AGV MAPPO 对比实验批量运行工具")
    print("=" * 80)
    print("实验列表：")
    print("  1. h_bi  - 水平布局 + 双向路由")
    print("  2. h_uni - 水平布局 + 单向路由")
    print("=" * 80)

    # 先运行测试（如果需要）
    if args.test_first:
        print("\n🧪 运行验证测试...")
        try:
            subprocess.run(
                [sys.executable, 'test_bidirectional.py'],
                check=True
            )
            print("✅ 测试通过，开始实验")
        except subprocess.CalledProcessError:
            print("❌ 测试失败，请先修复问题")
            return
        except KeyboardInterrupt:
            print("⚠️  测试被中断")
            return

    # 确定要运行的实验
    if args.exp == 'all':
        exp_list = ['h_bi', 'h_uni']
    else:
        exp_list = [args.exp]

    print(f"\n将运行 {len(exp_list)} 个实验")

    # 运行实验
    results = {}
    start_time = datetime.now()

    for exp_id in exp_list:
        exp_config = experiments[exp_id]
        success = run_experiment(
            exp_config['name'],
            exp_config['desc'],
            exp_config['script']
        )
        results[exp_id] = success

    end_time = datetime.now()
    duration = end_time - start_time

    # 总结
    print("\n" + "=" * 80)
    print("📊 实验运行总结")
    print("=" * 80)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("\n实验结果：")

    for exp_id in exp_list:
        exp_config = experiments[exp_id]
        status = "✅ 成功" if results[exp_id] else "❌ 失败"
        print(f"  {exp_config['name']}: {status}")

    success_count = sum(results.values())
    print(f"\n成功: {success_count}/{len(exp_list)}")
    print("=" * 80)

    # 下一步提示
    if all(results.values()):
        print("\n✅ 所有实验完成！")
        print("\n下一步：对比分析")
        print("  1. 打开TensorBoard对比训练曲线：")
        print("     tensorboard --logdir_spec=h_bi:./runs_h_bi,h_uni:./runs_h_uni")
        print("\n  2. 运行评估脚本：")
        print("     python evaluate_h_bi.py")
        print("     python evaluate_h_uni.py")
        print("\n  3. 查看日志目录：")
        print("     ./data/logs_h_bi/")
        print("     ./data/logs_h_uni/")
    else:
        print("\n⚠️  部分实验失败，请检查错误信息")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()