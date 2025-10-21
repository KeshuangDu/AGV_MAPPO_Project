"""
对比分析脚本：水平双向 vs 水平单向
Compare h_bi (bidirectional) vs h_uni (unidirectional)

运行方法：
    python compare_results.py

功能：
1. 加载两个实验的评估结果
2. 生成对比图表
3. 输出详细的统计对比
4. 保存对比报告
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ResultComparator:
    """结果对比分析器"""

    def __init__(self):
        """初始化对比分析器"""
        self.results = {}

    def load_results(self, exp_name: str, result_path: str):
        """加载实验结果"""
        if not os.path.exists(result_path):
            print(f"⚠️  找不到结果文件: {result_path}")
            return False

        with open(result_path, 'r') as f:
            self.results[exp_name] = json.load(f)

        print(f"✅ 已加载: {exp_name} - {result_path}")
        return True

    def print_comparison_table(self):
        """打印对比表格"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能对比")
            return

        print("\n" + "=" * 80)
        print("📊 实验对比统计表")
        print("=" * 80)

        # 获取所有指标
        metrics = ['episode_rewards', 'tasks_completed', 'episode_lengths',
                   'collisions', 'direction_changes']

        # 创建对比表格
        data = []
        for metric in metrics:
            row = {'Metric': metric.replace('_', ' ').title()}

            for exp_name, result in self.results.items():
                if metric in result['statistics']:
                    stats = result['statistics'][metric]
                    row[f"{exp_name}_mean"] = f"{stats['mean']:.2f}"
                    row[f"{exp_name}_std"] = f"{stats['std']:.2f}"

            data.append(row)

        # 打印表格
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("=" * 80)

        # 计算改进百分比
        if 'h_bi' in self.results and 'h_uni' in self.results:
            print("\n📈 双向 vs 单向 改进分析")
            print("=" * 80)

            bi_tasks = self.results['h_bi']['statistics']['tasks_completed']['mean']
            uni_tasks = self.results['h_uni']['statistics']['tasks_completed']['mean']
            task_improvement = ((bi_tasks - uni_tasks) / uni_tasks * 100) if uni_tasks > 0 else 0

            bi_reward = self.results['h_bi']['statistics']['episode_rewards']['mean']
            uni_reward = self.results['h_uni']['statistics']['episode_rewards']['mean']
            reward_improvement = ((bi_reward - uni_reward) / abs(uni_reward) * 100) if uni_reward != 0 else 0

            bi_collisions = self.results['h_bi']['statistics']['collisions']['mean']
            uni_collisions = self.results['h_uni']['statistics']['collisions']['mean']
            collision_reduction = ((uni_collisions - bi_collisions) / uni_collisions * 100) if uni_collisions > 0 else 0

            print(f"任务完成数:")
            print(f"  双向: {bi_tasks:.2f}")
            print(f"  单向: {uni_tasks:.2f}")
            print(f"  改进: {task_improvement:+.1f}%")

            print(f"\n平均奖励:")
            print(f"  双向: {bi_reward:.2f}")
            print(f"  单向: {uni_reward:.2f}")
            print(f"  改进: {reward_improvement:+.1f}%")

            print(f"\n碰撞次数:")
            print(f"  双向: {bi_collisions:.2f}")
            print(f"  单向: {uni_collisions:.2f}")
            print(f"  减少: {collision_reduction:.1f}%")

            # 双向特有指标
            if 'backward_usage' in self.results['h_bi']['statistics']:
                backward = self.results['h_bi']['statistics']['backward_usage']['mean']
                print(f"\n✨ 双向路由特征:")
                print(f"  后退使用率: {backward:.1f}%")
                print(f"  说明: AGV利用后退功能提高灵活性")

            print("=" * 80)

    def plot_comparison(self):
        """绘制对比图表"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能对比")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Experimental Comparison: h_bi (Bidirectional) vs h_uni (Unidirectional)',
                     fontsize=16, fontweight='bold')

        metrics = [
            ('episode_rewards', 'Episode Rewards', 0, 0),
            ('tasks_completed', 'Tasks Completed', 0, 1),
            ('episode_lengths', 'Episode Lengths', 0, 2),
            ('collisions', 'Collisions', 1, 0),
            ('direction_changes', 'Direction Changes', 1, 1)
        ]

        colors = {'h_bi': 'blue', 'h_uni': 'orange'}
        labels = {'h_bi': 'Bidirectional', 'h_uni': 'Unidirectional'}

        # 绘制对比图
        for metric, title, row, col in metrics:
            ax = axes[row, col]

            for exp_name, result in self.results.items():
                if metric in result['raw_data']:
                    data = result['raw_data'][metric]
                    ax.plot(data, alpha=0.6, label=labels.get(exp_name, exp_name),
                            color=colors.get(exp_name, 'gray'))
                    ax.axhline(np.mean(data), color=colors.get(exp_name, 'gray'),
                               linestyle='--', alpha=0.8, linewidth=1.5)

            ax.set_xlabel('Episode')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 箱线图对比
        ax = axes[1, 2]
        if 'h_bi' in self.results and 'h_uni' in self.results:
            bi_tasks = self.results['h_bi']['raw_data']['tasks_completed']
            uni_tasks = self.results['h_uni']['raw_data']['tasks_completed']

            bp = ax.boxplot([bi_tasks, uni_tasks], labels=['Bidirectional', 'Unidirectional'],
                            patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightyellow')

            ax.set_ylabel('Tasks Completed')
            ax.set_title('Task Completion Distribution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        save_path = './data/comparison_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 对比图表已保存: {save_path}")

        plt.show()

    def generate_report(self):
        """生成对比报告"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能生成报告")
            return

        report = []
        report.append("# 实验对比报告：水平双向 vs 水平单向")
        report.append("\n## 实验配置")

        for exp_name, result in self.results.items():
            report.append(f"\n### {exp_name}")
            report.append(f"- 描述: {result.get('description', 'N/A')}")
            if 'config' in result:
                config = result['config']
                report.append(f"- 双向路由: {'启用' if config.get('bidirectional') else '禁用'}")
                report.append(f"- AGV数量: {config.get('num_agents', 'N/A')}")
                report.append(f"- 车道数量: {config.get('num_lanes', 'N/A')}")

        report.append("\n## 性能对比")
        report.append("\n| 指标 | 双向 (h_bi) | 单向 (h_uni) | 改进 |")
        report.append("|------|-------------|--------------|------|")

        if 'h_bi' in self.results and 'h_uni' in self.results:
            metrics = [
                ('episode_rewards', '平均奖励'),
                ('tasks_completed', '任务完成数'),
                ('episode_lengths', 'Episode长度'),
                ('collisions', '碰撞次数')
            ]

            for metric_key, metric_name in metrics:
                bi_val = self.results['h_bi']['statistics'][metric_key]['mean']
                uni_val = self.results['h_uni']['statistics'][metric_key]['mean']

                if metric_key == 'collisions':
                    # 碰撞次数：越少越好
                    improvement = ((uni_val - bi_val) / uni_val * 100) if uni_val > 0 else 0
                    improvement_str = f"{improvement:.1f}% 减少"
                else:
                    # 其他指标：越多越好
                    improvement = ((bi_val - uni_val) / abs(uni_val) * 100) if uni_val != 0 else 0
                    improvement_str = f"{improvement:+.1f}%"

                report.append(f"| {metric_name} | {bi_val:.2f} | {uni_val:.2f} | {improvement_str} |")

        report.append("\n## 结论")
        report.append("\n基于实验结果，双向路由相比单向路由的优势：")

        if 'h_bi' in self.results and 'h_uni' in self.results:
            bi_tasks = self.results['h_bi']['statistics']['tasks_completed']['mean']
            uni_tasks = self.results['h_uni']['statistics']['tasks_completed']['mean']

            if bi_tasks > uni_tasks:
                report.append("- ✅ 任务完成数更高，表明双向路由提高了调度效率")

            if 'backward_usage' in self.results['h_bi']['statistics']:
                backward = self.results['h_bi']['statistics']['backward_usage']['mean']
                report.append(f"- ✅ AGV利用后退功能（使用率{backward:.1f}%），提高灵活性")

            bi_collisions = self.results['h_bi']['statistics']['collisions']['mean']
            uni_collisions = self.results['h_uni']['statistics']['collisions']['mean']

            if bi_collisions < uni_collisions:
                report.append("- ✅ 碰撞次数更少，后退功能有助于避障")

        report_text = "\n".join(report)

        # 保存报告
        report_path = './data/comparison_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✅ 对比报告已保存: {report_path}")

        # 打印报告
        print("\n" + "=" * 80)
        print(report_text)
        print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("📊 实验结果对比分析工具")
    print("=" * 80)

    comparator = ResultComparator()

    # 加载实验结果
    results_loaded = 0

    # 尝试加载水平双向结果
    h_bi_path = './data/logs_h_bi/evaluation_results_h_bi.json'
    if comparator.load_results('h_bi', h_bi_path):
        results_loaded += 1

    # 尝试加载水平单向结果
    h_uni_path = './data/logs_h_uni/evaluation_results_h_uni.json'
    if comparator.load_results('h_uni', h_uni_path):
        results_loaded += 1

    if results_loaded < 2:
        print("\n❌ 错误：找不到足够的评估结果文件")
        print("\n请先运行评估脚本：")
        print("  python evaluate_h_bi.py")
        print("  python evaluate_h_uni.py")
        return

    print(f"\n✅ 成功加载 {results_loaded} 个实验结果")

    # 生成对比分析
    print("\n1️⃣  生成对比表格...")
    comparator.print_comparison_table()

    print("\n2️⃣  生成对比图表...")
    comparator.plot_comparison()

    print("\n3️⃣  生成对比报告...")
    comparator.generate_report()

    print("\n" + "=" * 80)
    print("✅ 对比分析完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print("  - ./data/comparison_results.png  (对比图表)")
    print("  - ./data/comparison_report.md    (对比报告)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()