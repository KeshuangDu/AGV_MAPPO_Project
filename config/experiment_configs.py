"""
实验配置管理模块
便于快速切换不同实验设置，支持消融实验和对比实验

使用方法：
    from config.experiment_configs import *
    env_config = apply_experiment_config(env_config, EXPERIMENT_1_MAPPO_DENSE)
"""

# ===== 实验1：MAPPO + 水平布局 + 密集奖励（推荐baseline）=====
EXPERIMENT_1_MAPPO_DENSE = {
    'name': 'MAPPO_Horizontal_Dense_Reward',
    'description': 'MAPPO with dense reward shaping (recommended)',

    # 任务管理
    'USE_TASK_MANAGER': True,
    'TASK_ASSIGNMENT_STRATEGY': 'sequential',
    'ARRIVAL_THRESHOLD': 20.0,

    # 奖励设置
    'REWARD_TYPE': 'dense',
    'USE_DENSE_REWARD': True,

    # 布局设置
    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,

    # 调试
    'VERBOSE': False,
}

# ===== 实验2：MAPPO + 稀疏奖励（消融实验）=====
EXPERIMENT_2_MAPPO_SPARSE = {
    'name': 'MAPPO_Horizontal_Sparse_Reward',
    'description': 'MAPPO with sparse reward (ablation study)',

    'USE_TASK_MANAGER': True,
    'TASK_ASSIGNMENT_STRATEGY': 'sequential',
    'ARRIVAL_THRESHOLD': 20.0,

    # 关键差异：稀疏奖励
    'REWARD_TYPE': 'sparse',
    'USE_DENSE_REWARD': False,

    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,
    'VERBOSE': False,
}

# ===== 实验3：传统调度算法 - 最近距离（对比baseline）=====
EXPERIMENT_3_TRADITIONAL_NEAREST = {
    'name': 'Traditional_Nearest_Distance',
    'description': 'Traditional scheduling with nearest distance assignment',

    'USE_TASK_MANAGER': True,
    # 关键差异：使用传统调度策略
    'TASK_ASSIGNMENT_STRATEGY': 'nearest',
    'ARRIVAL_THRESHOLD': 20.0,

    'REWARD_TYPE': 'dense',
    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,
    'VERBOSE': False,
}

# ===== 实验4：随机分配（最简单baseline）=====
EXPERIMENT_4_RANDOM_ASSIGNMENT = {
    'name': 'Random_Task_Assignment',
    'description': 'Random task assignment (baseline)',

    'USE_TASK_MANAGER': True,
    # 关键差异：随机分配
    'TASK_ASSIGNMENT_STRATEGY': 'random',
    'ARRIVAL_THRESHOLD': 20.0,

    'REWARD_TYPE': 'dense',
    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,
    'VERBOSE': False,
}

# ===== 实验5：单向路由（对比双向）=====
EXPERIMENT_5_UNIDIRECTIONAL = {
    'name': 'MAPPO_Unidirectional_Routing',
    'description': 'MAPPO without bidirectional routing (comparison)',

    'USE_TASK_MANAGER': True,
    'TASK_ASSIGNMENT_STRATEGY': 'sequential',
    'ARRIVAL_THRESHOLD': 20.0,

    'REWARD_TYPE': 'dense',
    # 关键差异：关闭双向路由
    'BIDIRECTIONAL': False,
    'NUM_HORIZONTAL_LANES': 3,
    'VERBOSE': False,
}

# ===== 实验6：增加通道数（消融实验）=====
EXPERIMENT_6_MORE_LANES = {
    'name': 'MAPPO_5_Lanes',
    'description': 'MAPPO with 5 horizontal lanes (ablation)',

    'USE_TASK_MANAGER': True,
    'TASK_ASSIGNMENT_STRATEGY': 'sequential',
    'ARRIVAL_THRESHOLD': 20.0,

    'REWARD_TYPE': 'dense',
    'BIDIRECTIONAL': True,
    # 关键差异：增加通道数
    'NUM_HORIZONTAL_LANES': 5,
    'VERBOSE': False,
}

# ===== 实验7：不使用任务管理器（原始方法对比）=====
EXPERIMENT_7_NO_TASK_MANAGER = {
    'name': 'Original_Without_Task_Manager',
    'description': 'Original implementation without task manager',

    # 关键差异：不使用任务管理器
    'USE_TASK_MANAGER': False,

    'REWARD_TYPE': 'dense',
    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,
    'VERBOSE': False,
}

# ===== 实验8：调试模式（详细输出）=====
EXPERIMENT_8_DEBUG_MODE = {
    'name': 'Debug_Mode_Verbose',
    'description': 'Debug mode with verbose output',

    'USE_TASK_MANAGER': True,
    'TASK_ASSIGNMENT_STRATEGY': 'sequential',
    'ARRIVAL_THRESHOLD': 20.0,

    'REWARD_TYPE': 'dense',
    'BIDIRECTIONAL': True,
    'NUM_HORIZONTAL_LANES': 3,
    # 关键差异：开启详细输出
    'VERBOSE': True,
}


def apply_experiment_config(env_config, experiment_config):
    """
    将实验配置应用到环境配置对象

    Args:
        env_config: 环境配置对象（EnvConfig实例）
        experiment_config: 实验配置字典

    Returns:
        env_config: 更新后的环境配置

    使用示例：
        from config.experiment_configs import *
        from config.env_config import env_config

        # 选择实验1
        env_config = apply_experiment_config(env_config, EXPERIMENT_1_MAPPO_DENSE)
    """
    print(f"\n应用实验配置: {experiment_config['name']}")
    print(f"描述: {experiment_config['description']}")
    print("配置变更：")

    for key, value in experiment_config.items():
        if key in ['name', 'description']:
            continue

        if hasattr(env_config, key):
            old_value = getattr(env_config, key)
            setattr(env_config, key, value)
            print(f"  - {key}: {old_value} -> {value}")
        else:
            print(f"  ⚠️ {key} 不存在于env_config中，跳过")

    print("")
    return env_config


def list_all_experiments():
    """列出所有可用的实验配置"""
    experiments = [
        EXPERIMENT_1_MAPPO_DENSE,
        EXPERIMENT_2_MAPPO_SPARSE,
        EXPERIMENT_3_TRADITIONAL_NEAREST,
        EXPERIMENT_4_RANDOM_ASSIGNMENT,
        EXPERIMENT_5_UNIDIRECTIONAL,
        EXPERIMENT_6_MORE_LANES,
        EXPERIMENT_7_NO_TASK_MANAGER,
        EXPERIMENT_8_DEBUG_MODE,
    ]

    print("\n可用的实验配置：")
    print("=" * 70)
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   {exp['description']}")
    print("=" * 70)


# ===== 使用示例 =====
if __name__ == "__main__":
    list_all_experiments()