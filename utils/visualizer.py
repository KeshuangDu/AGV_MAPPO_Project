"""
可视化工具
提供训练过程和结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict, Tuple
import os


class PortVisualizer:
    """港口环境可视化器"""

    def __init__(self, config):
        """
        初始化可视化器

        Args:
            config: 环境配置
        """
        self.config = config
        self.fig = None
        self.ax = None

        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

    def setup_figure(self, figsize: Tuple[int, int] = (14, 8)):
        """设置图形"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, self.config.PORT_WIDTH)
        self.ax.set_ylim(0, self.config.PORT_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('AGV Port Environment - Horizontal Layout with Bidirectional Routing')

        # 绘制水平通道
        self._draw_lanes()

        # 绘制QC和YC位置
        self._draw_equipment()

    def _draw_lanes(self):
        """绘制水平通道"""
        lane_positions = np.linspace(
            50,
            self.config.PORT_HEIGHT - 50,
            self.config.NUM_HORIZONTAL_LANES
        )

        for i, y in enumerate(lane_positions):
            # 双向车道
            self.ax.plot(
                [50, self.config.PORT_WIDTH - 50],
                [y, y],
                'k--',
                alpha=0.3,
                linewidth=1
            )

            # 标注车道编号
            self.ax.text(
                self.config.PORT_WIDTH / 2,
                y + 10,
                f'Lane {i}',
                ha='center',
                fontsize=9,
                color='gray'
            )

    def _draw_equipment(self):
        """绘制设备位置"""
        # 绘制QC (左侧，红色)
        for i, pos in enumerate(self.config.QC_POSITIONS):
            circle = plt.Circle(pos, 15, color='red', alpha=0.6)
            self.ax.add_patch(circle)
            self.ax.text(
                pos[0] - 25, pos[1],
                f'QC{i}',
                fontsize=10,
                fontweight='bold',
                color='darkred'
            )

        # 绘制YC (右侧，蓝色)
        for i, pos in enumerate(self.config.YC_POSITIONS):
            circle = plt.Circle(pos, 15, color='blue', alpha=0.6)
            self.ax.add_patch(circle)
            self.ax.text(
                pos[0] + 25, pos[1],
                f'YC{i}',
                fontsize=10,
                fontweight='bold',
                color='darkblue'
            )

    def draw_agvs(self, agvs: List, show_direction: bool = True):
        """
        绘制AGV

        Args:
            agvs: AGV列表
            show_direction: 是否显示方向箭头
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(agvs)))

        for i, agv in enumerate(agvs):
            pos = agv.position

            # AGV主体 (不同颜色)
            if agv.has_container:
                # 有货：实心
                marker = 's'
                alpha = 0.9
            else:
                # 空车：空心
                marker = 'o'
                alpha = 0.6

            self.ax.plot(
                pos[0], pos[1],
                marker=marker,
                markersize=12,
                color=colors[i],
                alpha=alpha,
                markeredgewidth=2,
                markeredgecolor='black'
            )

            # 显示方向箭头
            if show_direction:
                dx = 20 * np.cos(agv.direction)
                dy = 20 * np.sin(agv.direction)

                # 双向标识：前进用实线，后退用虚线
                linestyle = '-' if agv.moving_forward else '--'

                self.ax.arrow(
                    pos[0], pos[1], dx, dy,
                    head_width=8,
                    head_length=10,
                    fc=colors[i],
                    ec='black',
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.7
                )

            # AGV编号
            self.ax.text(
                pos[0], pos[1] + 20,
                f'AGV{i}',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.5)
            )

    def draw_tasks(self, tasks: List[Dict]):
        """
        绘制任务

        Args:
            tasks: 任务列表
        """
        for task in tasks:
            if task.get('status') == 'pending':
                pickup = task['pickup_location']
                delivery = task['delivery_location']

                # 用虚线连接pickup和delivery
                color = 'green' if task['type'] == 'import' else 'orange'

                self.ax.plot(
                    [pickup[0], delivery[0]],
                    [pickup[1], delivery[1]],
                    color=color,
                    linestyle=':',
                    alpha=0.4,
                    linewidth=1.5
                )

    def save_snapshot(self, filepath: str):
        """保存快照"""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Snapshot saved to {filepath}")

    def show(self):
        """显示图形"""
        plt.tight_layout()
        plt.show()

    def close(self):
        """关闭图形"""
        if self.fig:
            plt.close(self.fig)


class TrainingVisualizer:
    """训练过程可视化器"""

    @staticmethod
    def plot_training_metrics(
            metrics_dict: Dict[str, List[float]],
            save_path: str = None
    ):
        """
        绘制训练指标

        Args:
            metrics_dict: 指标字典
            save_path: 保存路径
        """
        num_metrics = len(metrics_dict)
        ncols = 2
        nrows = (num_metrics + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
        axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

        for idx, (name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            ax.plot(values, alpha=0.6)

            # 添加移动平均
            if len(values) >= 50:
                window = 50
                moving_avg = np.convolve(
                    values,
                    np.ones(window) / window,
                    mode='valid'
                )
                ax.plot(
                    range(window - 1, len(values)),
                    moving_avg,
                    'r-',
                    linewidth=2,
                    label=f'MA({window})'
                )

            ax.set_xlabel('Episode')
            ax.set_ylabel(name.replace('_', ' ').title())
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            if len(values) >= 50:
                ax.legend()

        # 隐藏多余的子图
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_comparison(
            data_dict: Dict[str, List[float]],
            title: str = "Comparison",
            ylabel: str = "Value",
            save_path: str = None
    ):
        """
        绘制对比图

        Args:
            data_dict: 数据字典 {label: values}
            title: 标题
            ylabel: y轴标签
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))

        for label, values in data_dict.items():
            plt.plot(values, label=label, alpha=0.7, linewidth=2)

        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_heatmap(
            data: np.ndarray,
            xlabel: str = "X",
            ylabel: str = "Y",
            title: str = "Heatmap",
            save_path: str = None
    ):
        """
        绘制热力图

        Args:
            data: 2D数组
            xlabel: x轴标签
            ylabel: y轴标签
            title: 标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            data,
            annot=False,
            cmap='YlOrRd',
            cbar=True,
            xticklabels=False,
            yticklabels=False
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        plt.show()


def demo_visualization():
    """演示可视化功能"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.env_config import env_config
    from environment.agv import AGV

    # 创建可视化器
    visualizer = PortVisualizer(env_config)
    visualizer.setup_figure()

    # 创建示例AGV
    agvs = []
    for i in range(5):
        x = np.random.uniform(200, 400)
        y = np.random.uniform(50, env_config.PORT_HEIGHT - 50)
        agv = AGV(i, (x, y))
        agv.direction = np.random.uniform(0, 2 * np.pi)
        agv.moving_forward = np.random.choice([True, False])
        agv.has_container = np.random.choice([True, False])
        agvs.append(agv)

    # 绘制AGV
    visualizer.draw_agvs(agvs, show_direction=True)

    # 示例任务
    tasks = [
        {
            'type': 'import',
            'status': 'pending',
            'pickup_location': env_config.QC_POSITIONS[0],
            'delivery_location': env_config.YC_POSITIONS[1]
        },
        {
            'type': 'export',
            'status': 'pending',
            'pickup_location': env_config.YC_POSITIONS[2],
            'delivery_location': env_config.QC_POSITIONS[1]
        }
    ]

    visualizer.draw_tasks(tasks)

    # 保存
    os.makedirs('./data/logs', exist_ok=True)
    visualizer.save_snapshot('./data/logs/demo_snapshot.png')

    visualizer.show()


if __name__ == "__main__":
    demo_visualization()