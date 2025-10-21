"""
对比分析脚本 v2.2：水平双向 vs 水平单向
✨ 新增：任务完成时间对比

运行方法：
    python compare_results_v2.py

功能：
1. 加载两个实验的评估结果（包含任务时间）
2. 生成对比图表（包含时间对比）
3. 输出详细的统计对比（包含时间效率）
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


class ResultComparatorV2:
    """结果对比分析器 v2.2 - 包含任务时间"""

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
        """打印对比表格 - 包含任务时间"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能对比")
            return

        print("\n" + "=" * 80)
        print("📊 实验对比统计表 v2.2")
        print("=" * 80)

        # 获取所有指标（包含任务时间）
        metrics = ['episode_rewards', 'tasks_completed', 'task_completion_times',  # ✨ 新增
                   'episode_lengths', 'collisions', 'direction_changes']

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

            # 任务完成数
            bi_tasks = self.results['h_bi']['statistics']['tasks_completed']['mean']
            uni_tasks = self.results['h_uni']['statistics']['tasks_completed']['mean']
            task_improvement = ((bi_tasks - uni_tasks) / uni_tasks * 100) if uni_tasks > 0 else 0

            # 平均奖励
            bi_reward = self.results['h_bi']['statistics']['episode_rewards']['mean']
            uni_reward = self.results['h_uni']['statistics']['episode_rewards']['mean']
            reward_improvement = ((bi_reward - uni_reward) / abs(uni_reward) * 100) if uni_reward != 0 else 0

            # 碰撞次数
            bi_collisions = self.results['h_bi']['statistics']['collisions']['mean']
            uni_collisions = self.results['h_uni']['statistics']['collisions']['mean']
            collision_reduction = ((uni_collisions - bi_collisions) / uni_collisions * 100) if uni_collisions > 0 else 0

            # ✨ 任务完成时间（新增）
            if 'task_completion_times' in self.results['h_bi']['statistics'] and \
                    'task_completion_times' in self.results['h_uni']['statistics']:
                bi_time = self.results['h_bi']['statistics']['task_completion_times']['mean']
                uni_time = self.results['h_uni']['statistics']['task_completion_times']['mean']
                time_improvement = ((uni_time - bi_time) / uni_time * 100) if uni_time > 0 else 0
            else:
                bi_time = uni_time = time_improvement = None

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

            # ✨ 任务完成时间对比（新增）
            if bi_time is not None and uni_time is not None:
                print(f"\n⏱️  任务完成时间: ✨")
                print(f"  双向: {bi_time:.1f}秒")
                print(f"  单向: {uni_time:.1f}秒")
                if time_improvement > 0:
                    print(f"  改进: 双向快 {time_improvement:.1f}% ⚡")
                elif time_improvement < 0:
                    print(f"  结果: 单向快 {-time_improvement:.1f}%")
                else:
                    print(f"  结果: 两者相当")

            # 双向特有指标
            if 'backward_usage' in self.results['h_bi']['statistics']:
                backward = self.results['h_bi']['statistics']['backward_usage']['mean']
                print(f"\n✨ 双向路由特征:")
                print(f"  后退使用率: {backward:.1f}%")
                print(f"  说明: AGV利用后退功能提高灵活性")

            print("=" * 80)

    def plot_comparison(self):
        """绘制对比图表 - 包含任务时间对比"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能对比")
            return

        # ✨ 扩展为3x3布局（9个图）
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Experimental Comparison v2.2: h_bi vs h_uni (with Task Time)',
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

        # 绘制前5个对比图
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

        # 6. 任务完成箱线图
        ax = axes[1, 2]
        if 'h_bi' in self.results and 'h_uni' in self.results:
            bi_tasks = self.results['h_bi']['raw_data']['tasks_completed']
            uni_tasks = self.results['h_uni']['raw_data']['tasks_completed']

            bp = ax.boxplot([bi_tasks, uni_tasks],
                            labels=['Bidirectional', 'Unidirectional'],
                            patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightyellow')

            ax.set_ylabel('Tasks Completed')
            ax.set_title('Task Completion Distribution')
            ax.grid(True, alpha=0.3)

        # ✨ 7. 任务完成时间分布对比（新增）
        ax = axes[2, 0]
        if ('h_bi' in self.results and 'h_uni' in self.results and
                'task_completion_times' in self.results['h_bi']['raw_data'] and
                'task_completion_times' in self.results['h_uni']['raw_data']):

            bi_times = self.results['h_bi']['raw_data']['task_completion_times']
            uni_times = self.results['h_uni']['raw_data']['task_completion_times']

            if bi_times and uni_times:
                # 使用相同的bins范围
                all_times = bi_times + uni_times
                bins = np.linspace(min(all_times), max(all_times), 30)

                ax.hist(bi_times, bins=bins, alpha=0.5, color='blue',
                        label='Bidirectional', edgecolor='black')
                ax.hist(uni_times, bins=bins, alpha=0.5, color='orange',
                        label='Unidirectional', edgecolor='black')

                ax.axvline(np.mean(bi_times), color='blue', linestyle='--', linewidth=2)
                ax.axvline(np.mean(uni_times), color='orange', linestyle='--', linewidth=2)

                ax.set_xlabel('Task Completion Time (seconds)')
                ax.set_ylabel('Frequency')
                ax.set_title('⏱️  Task Time Distribution ✨')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # ✨ 8. 任务完成时间箱线图对比（新增）
        ax = axes[2, 1]
        if ('h_bi' in self.results and 'h_uni' in self.results and
                'task_completion_times' in self.results['h_bi']['raw_data'] and
                'task_completion_times' in self.results['h_uni']['raw_data']):

            bi_times = self.results['h_bi']['raw_data']['task_completion_times']
            uni_times = self.results['h_uni']['raw_data']['task_completion_times']

            if bi_times and uni_times:
                bp = ax.boxplot([bi_times, uni_times],
                                labels=['Bidirectional', 'Unidirectional'],
                                patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightyellow')

                # 添加均值标注
                means = [np.mean(bi_times), np.mean(uni_times)]
                ax.plot([1, 2], means, 'ro-', linewidth=2,
                        markersize=8, label='Mean')

                ax.set_ylabel('Task Completion Time (seconds)')
                ax.set_title('⏱️  Task Time Comparison ✨')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # 添加改进百分比文本
                improvement = ((means[1] - means[0]) / means[1] * 100)
                text = f"Bi faster: {improvement:.1f}%" if improvement > 0 else f"Uni faster: {-improvement:.1f}%"
                ax.text(0.5, 0.95, text, transform=ax.transAxes,
                        fontsize=10, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ✨ 9. 任务效率综合对比（新增）
        ax = axes[2, 2]
        if 'h_bi' in self.results and 'h_uni' in self.results:
            metrics_names = ['Tasks/Episode', 'Avg Time (s)', 'Collisions']

            bi_values = [
                self.results['h_bi']['statistics']['tasks_completed']['mean'],
                self.results['h_bi']['statistics'].get('task_completion_times', {}).get('mean', 0),
                self.results['h_bi']['statistics']['collisions']['mean']
            ]

            uni_values = [
                self.results['h_uni']['statistics']['tasks_completed']['mean'],
                self.results['h_uni']['statistics'].get('task_completion_times', {}).get('mean', 0),
                self.results['h_uni']['statistics']['collisions']['mean']
            ]

            x = np.arange(len(metrics_names))
            width = 0.35

            # 归一化显示（以单向为基准）
            normalized_bi = []
            normalized_uni = []
            for bi, uni in zip(bi_values, uni_values):
                if uni > 0:
                    normalized_bi.append(bi / uni * 100)
                    normalized_uni.append(100)
                else:
                    normalized_bi.append(0)
                    normalized_uni.append(0)

            ax.bar(x - width / 2, normalized_bi, width, label='Bidirectional',
                   color='lightblue', edgecolor='black')
            ax.bar(x + width / 2, normalized_uni, width, label='Unidirectional',
                   color='lightyellow', edgecolor='black')

            ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('Normalized Performance (%)')
            ax.set_title('⚡ Efficiency Comparison (Uni=100%)')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, rotation=15, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图表
        save_path = './data/comparison_results_v2.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 对比图表已保存: {save_path}")

        plt.show()

    def generate_report(self):
        """生成对比报告 - 包含任务时间"""
        if len(self.results) < 2:
            print("❌ 需要至少两个实验结果才能生成报告")
            return

        report = []
        report.append("# 实验对比报告 v2.2：水平双向 vs 水平单向")
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
                ('task_completion_times', '⏱️  任务完成时间(秒)'),  # ✨ 新增
                ('episode_lengths', 'Episode长度'),
                ('collisions', '碰撞次数')
            ]

            for metric_key, metric_name in metrics:
                if (metric_key in self.results['h_bi']['statistics'] and
                        metric_key in self.results['h_uni']['statistics']):

                    bi_val = self.results['h_bi']['statistics'][metric_key]['mean']
                    uni_val = self.results['h_uni']['statistics'][metric_key]['mean']

                    if metric_key in ['collisions', 'episode_lengths', 'task_completion_times']:
                        # 这些指标：越少越好
                        improvement = ((uni_val - bi_val) / uni_val * 100) if uni_val > 0 else 0
                        if improvement > 0:
                            improvement_str = f"{improvement:.1f}% 减少 ✅"
                        else:
                            improvement_str = f"{-improvement:.1f}% 增加"
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

            if 'task_completion_times' in self.results['h_bi']['statistics']:
                bi_time = self.results['h_bi']['statistics']['task_completion_times']['mean']
                uni_time = self.results['h_uni']['statistics']['task_completion_times']['mean']

                if bi_time < uni_time:
                    time_improv = (uni_time - bi_time) / uni_time * 100
                    report.append(f"- ✅ ⏱️  任务完成时间更短（快{time_improv:.1f}%），调度效率更高")

            if 'backward_usage' in self.results['h_bi']['statistics']:
                backward = self.results['h_bi']['statistics']['backward_usage']['mean']
                report.append(f"- ✅ AGV利用后退功能（使用率{backward:.1f}%），提高灵活性")

            bi_collisions = self.results['h_bi']['statistics']['collisions']['mean']
            uni_collisions = self.results['h_uni']['statistics']['collisions']['mean']

            if bi_collisions < uni_collisions:
                report.append("- ✅ 碰撞次数更少，后退功能有助于避障")

        report_text = "\n".join(report)

        # 保存报告
        report_path = './data/comparison_report_v2.md'
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
    print("📊 实验结果对比分析工具 v2.2 (with Task Time)")
    print("=" * 80)

    comparator = ResultComparatorV2()

    # 加载实验结果（优先加载v2版本）
    results_loaded = 0

    # 尝试加载水平双向结果（v2优先）
    h_bi_paths = [
        './data/logs_h_bi/evaluation_results_h_bi_v2.json',
        './data/logs_h_bi/evaluation_results_h_bi.json'
    ]
    for path in h_bi_paths:
        if comparator.load_results('h_bi', path):
            results_loaded += 1
            break

    # 尝试加载水平单向结果（v2优先）
    h_uni_paths = [
        './data/logs_h_uni/evaluation_results_h_uni_v2.json',
        './data/logs_h_uni/evaluation_results_h_uni.json'
    ]
    for path in h_uni_paths:
        if comparator.load_results('h_uni', path):
            results_loaded += 1
            break

    if results_loaded < 2:
        print("\n❌ 错误：找不到足够的评估结果文件")
        print("\n请先运行评估脚本：")
        print("  python evaluate_h_bi_v2.py --checkpoint xxx.pt --episodes 50 --exp h_bi")
        print("  python evaluate_h_uni_v2.py --checkpoint xxx.pt --episodes 50 --exp h_uni")
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
    print("  - ./data/comparison_results_v2.png  (对比图表)")
    print("  - ./data/comparison_report_v2.md    (对比报告)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()