"""
可视化工具 - 增强版
✨ 新增功能：
1. 自动读取配置文件
2. 支持水平/垂直布局自动切换
3. 自适应AGV、QC、YC数量
4. 命令行参数支持
5. 更美观的可视化效果

运行方法：
    # 使用当前配置可视化
    python visualizer.py

    # 指定配置
    python visualizer.py --layout vertical --num-agvs 8 --num-qc 4 --num-yc 4

    # 从环境中可视化
    python visualizer.py --from-env
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PortVisualizer:
    """
    港口环境可视化器 - 增强版
    支持水平和垂直布局自动切换
    """

    def __init__(self, config=None):
        """
        初始化可视化器

        Args:
            config: 环境配置对象，如果为None则从env_config导入
        """
        if config is None:
            from config.env_config import env_config
            self.config = env_config
        else:
            self.config = config

        self.fig = None
        self.ax = None

        # 从配置读取参数
        self.layout_type = getattr(self.config, 'LAYOUT_TYPE', 'horizontal')
        self.num_agvs = self.config.NUM_AGVS
        self.num_qc = self.config.NUM_QC
        self.num_yc = self.config.NUM_YC
        self.num_lanes = self.config.NUM_HORIZONTAL_LANES
        self.bidirectional = getattr(self.config, 'BIDIRECTIONAL', True)

        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'

    def setup_figure(self, figsize: Tuple[int, int] = (16, 9)):
        """设置图形"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, self.config.PORT_WIDTH)
        self.ax.set_ylim(0, self.config.PORT_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')

        # 根据布局类型设置标题
        layout_name = "Horizontal" if self.layout_type == 'horizontal' else "Vertical"
        routing_type = "Bidirectional" if self.bidirectional else "Unidirectional"

        self.ax.set_title(
            f'AGV Port Environment - {layout_name} Layout with {routing_type} Routing\n'
            f'{self.num_agvs} AGVs | {self.num_qc} QCs | {self.num_yc} YCs | {self.num_lanes} Lanes',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # 根据布局类型绘制不同的元素
        if self.layout_type == 'horizontal':
            self._draw_horizontal_layout()
        else:
            self._draw_vertical_layout()

    def _draw_horizontal_layout(self):
        """绘制水平布局"""
        # 绘制水平通道
        lane_positions = np.linspace(
            50,
            self.config.PORT_HEIGHT - 50,
            self.num_lanes
        )

        for i, y in enumerate(lane_positions):
            # 车道线
            self.ax.plot(
                [50, self.config.PORT_WIDTH - 50],
                [y, y],
                'k--',
                alpha=0.3,
                linewidth=1.5
            )

            # 车道标签
            self.ax.text(
                self.config.PORT_WIDTH / 2,
                y + 15,
                f'Lane {i}',
                ha='center',
                fontsize=10,
                color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        # 绘制QC（左侧）和YC（右侧）
        self._draw_equipment_horizontal()

    def _draw_vertical_layout(self):
        """绘制垂直布局"""
        # 绘制垂直通道
        lane_positions = np.linspace(
            50,
            self.config.PORT_WIDTH - 50,
            self.num_lanes
        )

        for i, x in enumerate(lane_positions):
            # 车道线
            self.ax.plot(
                [x, x],
                [50, self.config.PORT_HEIGHT - 50],
                'k--',
                alpha=0.3,
                linewidth=1.5
            )

            # 车道标签
            self.ax.text(
                x,
                self.config.PORT_HEIGHT / 2,
                f'Lane {i}',
                ha='center',
                rotation=90,
                fontsize=10,
                color='gray',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        # 绘制QC（底部）和YC（顶部）
        self._draw_equipment_vertical()

    def _draw_equipment_horizontal(self):
        """绘制水平布局的设备（QC左，YC右）"""
        # 计算QC位置（左侧均匀分布）
        qc_y_positions = np.linspace(
            self.config.PORT_HEIGHT * 0.2,
            self.config.PORT_HEIGHT * 0.8,
            self.num_qc
        )

        for i, y in enumerate(qc_y_positions):
            pos = (50.0, y)

            # QC主体（红色大圆）
            circle = plt.Circle(pos, 20, color='red', alpha=0.7, zorder=10)
            self.ax.add_patch(circle)

            # QC边框
            circle_border = plt.Circle(pos, 20, color='darkred',
                                       fill=False, linewidth=2.5, zorder=11)
            self.ax.add_patch(circle_border)

            # QC标签
            self.ax.text(
                pos[0] - 35, pos[1],
                f'QC{i}',
                fontsize=11,
                fontweight='bold',
                color='darkred',
                ha='right',
                va='center',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor='darkred',
                          linewidth=2)
            )

        # 计算YC位置（右侧均匀分布）
        yc_y_positions = np.linspace(
            self.config.PORT_HEIGHT * 0.2,
            self.config.PORT_HEIGHT * 0.8,
            self.num_yc
        )

        for i, y in enumerate(yc_y_positions):
            pos = (self.config.PORT_WIDTH - 50.0, y)

            # YC主体（蓝色大圆）
            circle = plt.Circle(pos, 20, color='blue', alpha=0.7, zorder=10)
            self.ax.add_patch(circle)

            # YC边框
            circle_border = plt.Circle(pos, 20, color='darkblue',
                                       fill=False, linewidth=2.5, zorder=11)
            self.ax.add_patch(circle_border)

            # YC标签
            self.ax.text(
                pos[0] + 35, pos[1],
                f'YC{i}',
                fontsize=11,
                fontweight='bold',
                color='darkblue',
                ha='left',
                va='center',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor='darkblue',
                          linewidth=2)
            )

    def _draw_equipment_vertical(self):
        """绘制垂直布局的设备（QC下，YC上）"""
        # 计算QC位置（底部均匀分布）
        qc_x_positions = np.linspace(
            self.config.PORT_WIDTH * 0.2,
            self.config.PORT_WIDTH * 0.8,
            self.num_qc
        )

        for i, x in enumerate(qc_x_positions):
            pos = (x, 50.0)

            # QC主体（红色大圆）
            circle = plt.Circle(pos, 20, color='red', alpha=0.7, zorder=10)
            self.ax.add_patch(circle)

            # QC边框
            circle_border = plt.Circle(pos, 20, color='darkred',
                                       fill=False, linewidth=2.5, zorder=11)
            self.ax.add_patch(circle_border)

            # QC标签
            self.ax.text(
                pos[0], pos[1] - 35,
                f'QC{i}',
                fontsize=11,
                fontweight='bold',
                color='darkred',
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor='darkred',
                          linewidth=2)
            )

        # 计算YC位置（顶部均匀分布）
        yc_x_positions = np.linspace(
            self.config.PORT_WIDTH * 0.2,
            self.config.PORT_WIDTH * 0.8,
            self.num_yc
        )

        for i, x in enumerate(yc_x_positions):
            pos = (x, self.config.PORT_HEIGHT - 50.0)

            # YC主体（蓝色大圆）
            circle = plt.Circle(pos, 20, color='blue', alpha=0.7, zorder=10)
            self.ax.add_patch(circle)

            # YC边框
            circle_border = plt.Circle(pos, 20, color='darkblue',
                                       fill=False, linewidth=2.5, zorder=11)
            self.ax.add_patch(circle_border)

            # YC标签
            self.ax.text(
                pos[0], pos[1] + 35,
                f'YC{i}',
                fontsize=11,
                fontweight='bold',
                color='darkblue',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor='darkblue',
                          linewidth=2)
            )

    def draw_agvs(self, agvs: List, show_direction: bool = True, show_labels: bool = True):
        """
        绘制AGV

        Args:
            agvs: AGV列表或位置列表
            show_direction: 是否显示方向箭头
            show_labels: 是否显示AGV标签
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(agvs)))

        for i, agv in enumerate(agvs):
            # 兼容AGV对象和简单位置
            if hasattr(agv, 'position'):
                pos = agv.position
                has_container = getattr(agv, 'has_container', False)
                direction = getattr(agv, 'direction', 0)
                moving_forward = getattr(agv, 'moving_forward', True)
            else:
                pos = agv
                has_container = False
                direction = 0
                moving_forward = True

            # AGV主体
            if has_container:
                marker = 's'  # 方形 = 有货
                markersize = 15
                alpha = 0.9
                label_text = f'AGV{i}✓'
            else:
                marker = 'o'  # 圆形 = 空车
                markersize = 12
                alpha = 0.7
                label_text = f'AGV{i}'

            self.ax.plot(
                pos[0], pos[1],
                marker=marker,
                markersize=markersize,
                color=colors[i],
                alpha=alpha,
                markeredgewidth=2.5,
                markeredgecolor='black',
                zorder=20
            )

            # 显示方向箭头
            if show_direction and hasattr(agv, 'direction'):
                arrow_length = 25
                dx = arrow_length * np.cos(direction)
                dy = arrow_length * np.sin(direction)

                # 双向标识：前进实线，后退虚线
                linestyle = '-' if moving_forward else '--'
                arrow_color = colors[i] if moving_forward else 'gray'

                arrow = FancyArrowPatch(
                    (pos[0], pos[1]),
                    (pos[0] + dx, pos[1] + dy),
                    arrowstyle='-|>',
                    mutation_scale=20,
                    linewidth=2.5,
                    linestyle=linestyle,
                    color=arrow_color,
                    alpha=0.8,
                    zorder=19
                )
                self.ax.add_patch(arrow)

            # AGV标签
            if show_labels:
                self.ax.text(
                    pos[0], pos[1] - 20,
                    label_text,
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              alpha=0.9,
                              edgecolor=colors[i],
                              linewidth=1.5)
                )

    def draw_tasks(self, tasks: List[Dict], show_flow: bool = True):
        """
        绘制任务流

        Args:
            tasks: 任务列表
            show_flow: 是否显示任务流向箭头
        """
        for task in tasks:
            pickup = task.get('pickup_location')
            delivery = task.get('delivery_location')
            task_type = task.get('type', 'import')
            status = task.get('status', 'pending')

            if pickup is None or delivery is None:
                continue

            # 根据任务类型选择颜色
            if task_type == 'import':
                color = 'orange'
                label = 'Import'
            else:
                color = 'purple'
                label = 'Export'

            # 根据状态选择样式
            if status == 'completed':
                alpha = 0.3
                linestyle = ':'
            elif status == 'in_progress':
                alpha = 0.8
                linestyle = '-'
            else:
                alpha = 0.5
                linestyle = '--'

            if show_flow:
                # 绘制任务流向箭头
                arrow = FancyArrowPatch(
                    pickup, delivery,
                    arrowstyle='->',
                    mutation_scale=15,
                    linewidth=2,
                    linestyle=linestyle,
                    color=color,
                    alpha=alpha,
                    zorder=5
                )
                self.ax.add_patch(arrow)

    def add_legend(self):
        """添加图例"""
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Quay Crane (QC)'),
            Patch(facecolor='blue', alpha=0.7, label='Yard Crane (YC)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markersize=10,
                   markeredgecolor='black', markeredgewidth=2,
                   label='AGV (Empty)'),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='gray', markersize=10,
                   markeredgecolor='black', markeredgewidth=2,
                   label='AGV (Loaded)'),
        ]

        if self.bidirectional:
            legend_elements.extend([
                Line2D([0], [0], color='black', linewidth=2,
                       label='Forward Direction'),
                Line2D([0], [0], color='gray', linewidth=2,
                       linestyle='--', label='Backward Direction'),
            ])

        self.ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9,
            edgecolor='black'
        )

    def save_snapshot(self, save_path: str):
        """保存快照"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                         facecolor='white', edgecolor='none')
        print(f"📸 快照已保存: {save_path}")

    def show(self):
        """显示图形"""
        plt.tight_layout()
        plt.show()


def visualize_from_config(config=None, save_path: Optional[str] = None):
    """
    从配置文件创建可视化

    Args:
        config: 配置对象，如果为None则使用默认配置
        save_path: 保存路径，如果为None则不保存
    """
    from environment.agv import AGV

    # 创建可视化器
    visualizer = PortVisualizer(config)
    visualizer.setup_figure()

    # 创建示例AGV
    agvs = []
    for i in range(visualizer.num_agvs):
        x = np.random.uniform(150, visualizer.config.PORT_WIDTH - 150)
        y = np.random.uniform(100, visualizer.config.PORT_HEIGHT - 100)

        agv = AGV(i, (x, y))
        agv.direction = np.random.uniform(0, 2 * np.pi)
        agv.moving_forward = np.random.choice([True, False])
        agv.has_container = np.random.choice([True, False])
        agvs.append(agv)

    # 绘制AGV
    visualizer.draw_agvs(agvs, show_direction=True)

    # 添加图例
    visualizer.add_legend()

    # 保存或显示
    if save_path:
        visualizer.save_snapshot(save_path)

    visualizer.show()


def visualize_from_environment(env, save_path: Optional[str] = None):
    """
    从环境对象创建可视化

    Args:
        env: PortEnvironment对象
        save_path: 保存路径
    """
    visualizer = PortVisualizer(env.config)
    visualizer.setup_figure()

    # 绘制实际的AGV
    visualizer.draw_agvs(env.agvs, show_direction=True)

    # 绘制任务（如果有）
    if hasattr(env, 'tasks') and env.tasks:
        task_dicts = []
        for task in env.tasks:
            task_dicts.append({
                'type': task.type,
                'status': task.status,
                'pickup_location': task.pickup_location,
                'delivery_location': task.delivery_location
            })
        visualizer.draw_tasks(task_dicts)

    # 添加图例
    visualizer.add_legend()

    # 保存或显示
    if save_path:
        visualizer.save_snapshot(save_path)

    visualizer.show()


def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='港口环境可视化工具')
    parser.add_argument('--layout', type=str, choices=['horizontal', 'vertical'],
                        help='布局类型（默认从配置读取）')
    parser.add_argument('--num-agvs', type=int, help='AGV数量（默认从配置读取）')
    parser.add_argument('--num-qc', type=int, help='岸桥数量（默认从配置读取）')
    parser.add_argument('--num-yc', type=int, help='场桥数量（默认从配置读取）')
    parser.add_argument('--num-lanes', type=int, help='通道数量（默认从配置读取）')
    parser.add_argument('--bidirectional', action='store_true', help='启用双向路由')
    parser.add_argument('--unidirectional', action='store_true', help='禁用双向路由')
    parser.add_argument('--save', type=str, help='保存路径')
    parser.add_argument('--from-env', action='store_true', help='从实际环境创建可视化')

    args = parser.parse_args()

    # 导入配置
    from config.env_config import env_config

    # 根据命令行参数修改配置
    if args.layout:
        env_config.LAYOUT_TYPE = args.layout
    if args.num_agvs:
        env_config.NUM_AGVS = args.num_agvs
    if args.num_qc:
        env_config.NUM_QC = args.num_qc
    if args.num_yc:
        env_config.NUM_YC = args.num_yc
    if args.num_lanes:
        env_config.NUM_HORIZONTAL_LANES = args.num_lanes
    if args.bidirectional:
        env_config.BIDIRECTIONAL = True
    elif args.unidirectional:
        env_config.BIDIRECTIONAL = False

    print(f"\n{'=' * 60}")
    print(f"🎨 港口环境可视化")
    print(f"{'=' * 60}")
    print(f"布局类型: {env_config.LAYOUT_TYPE}")
    print(f"AGV数量: {env_config.NUM_AGVS}")
    print(f"岸桥数量: {env_config.NUM_QC}")
    print(f"场桥数量: {env_config.NUM_YC}")
    print(f"通道数量: {env_config.NUM_HORIZONTAL_LANES}")
    print(f"双向路由: {getattr(env_config, 'BIDIRECTIONAL', True)}")
    print(f"{'=' * 60}\n")

    if args.from_env:
        # 从实际环境创建可视化
        from environment.port_env import PortEnvironment
        env = PortEnvironment(env_config)
        env.reset()
        visualize_from_environment(env, args.save)
    else:
        # 从配置创建可视化
        visualize_from_config(env_config, args.save)


if __name__ == "__main__":
    main()