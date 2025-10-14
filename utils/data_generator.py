"""
数据生成器
生成和管理港口仿真数据
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple
from datetime import datetime
import os


class DataGenerator:
    """数据生成器类"""

    def __init__(self, config):
        """
        初始化数据生成器

        Args:
            config: 环境配置
        """
        self.config = config
        self.generated_scenarios = []

    def generate_random_scenario(self, num_tasks: int = 20) -> Dict:
        """
        生成随机港口场景

        Args:
            num_tasks: 任务数量

        Returns:
            场景字典
        """
        scenario = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_agvs': self.config.NUM_AGVS,
                'num_qcs': self.config.NUM_QC,
                'num_ycs': self.config.NUM_YC,
                'num_tasks': num_tasks
            },
            'initial_state': self._generate_initial_state(),
            'tasks': self._generate_tasks(num_tasks)
        }

        self.generated_scenarios.append(scenario)

        return scenario

    def _generate_initial_state(self) -> Dict:
        """生成初始状态"""
        # AGV初始位置
        agv_positions = []
        for i in range(self.config.NUM_AGVS):
            x = np.random.uniform(200, 400)
            y = np.random.uniform(50, self.config.PORT_HEIGHT - 50)
            agv_positions.append({
                'id': i,
                'position': [float(x), float(y)],
                'velocity': 0.0,
                'direction': float(np.random.uniform(0, 2 * np.pi)),
                'has_container': False
            })

        # QC状态
        qc_states = []
        for i in range(self.config.NUM_QC):
            qc_states.append({
                'id': i,
                'position': list(self.config.QC_POSITIONS[i]),
                'is_busy': False,
                'queue_length': 0
            })

        # YC状态
        yc_states = []
        for i in range(self.config.NUM_YC):
            yc_states.append({
                'id': i,
                'position': list(self.config.YC_POSITIONS[i]),
                'is_busy': False,
                'queue_length': 0
            })

        return {
            'agvs': agv_positions,
            'qcs': qc_states,
            'ycs': yc_states
        }

    def _generate_tasks(self, num_tasks: int) -> List[Dict]:
        """生成任务列表"""
        tasks = []

        for i in range(num_tasks):
            task_type = np.random.choice(self.config.TASK_TYPES)
            qc_id = np.random.randint(0, self.config.NUM_QC)
            yc_id = np.random.randint(0, self.config.NUM_YC)
            priority = float(np.random.uniform(0.5, 1.5))

            if task_type == 'import':
                pickup = list(self.config.QC_POSITIONS[qc_id])
                delivery = list(self.config.YC_POSITIONS[yc_id])
            else:
                pickup = list(self.config.YC_POSITIONS[yc_id])
                delivery = list(self.config.QC_POSITIONS[qc_id])

            task = {
                'id': i,
                'type': task_type,
                'qc_id': qc_id,
                'yc_id': yc_id,
                'priority': priority,
                'pickup_location': pickup,
                'delivery_location': delivery,
                'status': 'pending'
            }

            tasks.append(task)

        return tasks

    def generate_batch(self, num_scenarios: int = 100) -> List[Dict]:
        """
        批量生成场景

        Args:
            num_scenarios: 场景数量

        Returns:
            场景列表
        """
        scenarios = []

        print(f"Generating {num_scenarios} scenarios...")

        for i in range(num_scenarios):
            num_tasks = np.random.randint(10, 30)
            scenario = self.generate_random_scenario(num_tasks)
            scenarios.append(scenario)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_scenarios} scenarios")

        print("Generation complete!")

        return scenarios

    def save_scenarios(self, scenarios: List[Dict], filepath: str):
        """
        保存场景数据

        Args:
            scenarios: 场景列表
            filepath: 保存路径
        """
        # 确定文件格式
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(scenarios, f)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(scenarios, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")

        print(f"Scenarios saved to {filepath}")

    def load_scenarios(self, filepath: str) -> List[Dict]:
        """
        加载场景数据

        Args:
            filepath: 文件路径

        Returns:
            场景列表
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                scenarios = pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                scenarios = json.load(f)
        else:
            raise ValueError("Unsupported file format")

        print(f"Loaded {len(scenarios)} scenarios from {filepath}")

        return scenarios

    def export_to_csv(self, scenarios: List[Dict], output_dir: str):
        """
        导出为CSV格式

        Args:
            scenarios: 场景列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 导出AGV数据
        agv_data = []
        for scenario in scenarios:
            scenario_id = scenario['timestamp']
            for agv in scenario['initial_state']['agvs']:
                agv_data.append({
                    'scenario_id': scenario_id,
                    'agv_id': agv['id'],
                    'x': agv['position'][0],
                    'y': agv['position'][1],
                    'velocity': agv['velocity'],
                    'direction': agv['direction']
                })

        df_agv = pd.DataFrame(agv_data)
        df_agv.to_csv(os.path.join(output_dir, 'agv_data.csv'), index=False)

        # 导出任务数据
        task_data = []
        for scenario in scenarios:
            scenario_id = scenario['timestamp']
            for task in scenario['tasks']:
                task_data.append({
                    'scenario_id': scenario_id,
                    'task_id': task['id'],
                    'type': task['type'],
                    'qc_id': task['qc_id'],
                    'yc_id': task['yc_id'],
                    'priority': task['priority'],
                    'pickup_x': task['pickup_location'][0],
                    'pickup_y': task['pickup_location'][1],
                    'delivery_x': task['delivery_location'][0],
                    'delivery_y': task['delivery_location'][1]
                })

        df_task = pd.DataFrame(task_data)
        df_task.to_csv(os.path.join(output_dir, 'task_data.csv'), index=False)

        print(f"Data exported to {output_dir}")
        print(f"  - agv_data.csv: {len(df_agv)} records")
        print(f"  - task_data.csv: {len(df_task)} records")

    def get_statistics(self, scenarios: List[Dict]) -> Dict:
        """
        获取数据统计信息

        Args:
            scenarios: 场景列表

        Returns:
            统计信息字典
        """
        num_scenarios = len(scenarios)

        task_counts = [len(s['tasks']) for s in scenarios]
        import_counts = [
            sum(1 for t in s['tasks'] if t['type'] == 'import')
            for s in scenarios
        ]
        export_counts = [
            sum(1 for t in s['tasks'] if t['type'] == 'export')
            for s in scenarios
        ]

        stats = {
            'num_scenarios': num_scenarios,
            'tasks_per_scenario': {
                'mean': float(np.mean(task_counts)),
                'std': float(np.std(task_counts)),
                'min': int(np.min(task_counts)),
                'max': int(np.max(task_counts))
            },
            'import_tasks': {
                'mean': float(np.mean(import_counts)),
                'total': int(np.sum(import_counts))
            },
            'export_tasks': {
                'mean': float(np.mean(export_counts)),
                'total': int(np.sum(export_counts))
            },
            'agvs_per_scenario': self.config.NUM_AGVS,
            'qcs_per_scenario': self.config.NUM_QC,
            'ycs_per_scenario': self.config.NUM_YC
        }

        return stats

    def print_statistics(self, scenarios: List[Dict]):
        """打印统计信息"""
        stats = self.get_statistics(scenarios)

        print("\n" + "=" * 50)
        print("Data Statistics")
        print("=" * 50)
        print(f"Total Scenarios: {stats['num_scenarios']}")
        print(f"\nTasks per Scenario:")
        print(f"  Mean: {stats['tasks_per_scenario']['mean']:.2f}")
        print(f"  Std:  {stats['tasks_per_scenario']['std']:.2f}")
        print(f"  Range: [{stats['tasks_per_scenario']['min']}, "
              f"{stats['tasks_per_scenario']['max']}]")
        print(f"\nImport Tasks:")
        print(f"  Mean per Scenario: {stats['import_tasks']['mean']:.2f}")
        print(f"  Total: {stats['import_tasks']['total']}")
        print(f"\nExport Tasks:")
        print(f"  Mean per Scenario: {stats['export_tasks']['mean']:.2f}")
        print(f"  Total: {stats['export_tasks']['total']}")
        print("=" * 50 + "\n")


def main():
    """主函数 - 生成示例数据"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.env_config import env_config

    # 创建数据生成器
    generator = DataGenerator(env_config)

    # 生成场景
    scenarios = generator.generate_batch(num_scenarios=100)

    # 打印统计
    generator.print_statistics(scenarios)

    # 保存数据
    save_dir = './data/generated_data'
    os.makedirs(save_dir, exist_ok=True)

    generator.save_scenarios(scenarios, os.path.join(save_dir, 'scenarios.pkl'))
    generator.save_scenarios(scenarios, os.path.join(save_dir, 'scenarios.json'))
    generator.export_to_csv(scenarios, save_dir)

    print(f"\nAll data saved to {save_dir}")


if __name__ == "__main__":
    main()