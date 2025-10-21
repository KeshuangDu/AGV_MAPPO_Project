"""
训练配置 - 垂直布局 + 双向路由
Vertical Layout + Bidirectional Routing

实验标识：v_bi
"""

import os


class TrainConfig:
    """训练配置类"""

    # ========== 实验标识 ==========
    EXPERIMENT_NAME = 'v_bi'  # 垂直双向
    DESCRIPTION = 'Vertical Layout + Bidirectional Routing'

    # ========== 训练参数 ==========
    NUM_EPISODES = 1000  # 训练轮数
    MAX_STEPS_PER_EPISODE = 2000  # 每轮最大步数
    BATCH_SIZE = 256  # 批次大小
    BUFFER_SIZE = 10000  # 缓冲区大小

    # ========== PPO超参数 ==========
    PPO_EPOCHS = 10  # PPO更新轮数
    CLIP_EPSILON = 0.2  # PPO裁剪参数
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE参数

    # ========== 学习率 ==========
    ACTOR_LR = 3e-4  # Actor学习率
    CRITIC_LR = 3e-4  # Critic学习率
    LR_DECAY = True  # 是否使用学习率衰减
    LR_DECAY_RATE = 0.995  # 学习率衰减率

    # ========== 损失权重 ==========
    VALUE_LOSS_COEF = 0.5  # 价值损失系数
    ENTROPY_COEF = 0.01  # 熵正则化系数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪

    # ========== 保存设置 ==========
    SAVE_INTERVAL = 100  # 保存间隔（轮数）
    CHECKPOINT_DIR = './data/checkpoints_v_bi'  # 检查点目录
    LOG_DIR = './data/logs_v_bi'  # 日志目录
    TENSORBOARD_DIR = './runs_v_bi'  # TensorBoard目录

    # ========== 评估设置 ==========
    EVAL_INTERVAL = 50  # 评估间隔
    EVAL_EPISODES = 10  # 评估轮数

    # ========== 设备设置 ==========
    USE_CUDA = True  # 使用GPU
    SEED = 42  # 随机种子

    # ========== 其他 ==========
    VERBOSE = True  # 详细输出
    RENDER_TRAIN = False  # 训练时渲染


# 创建全局配置实例
train_config = TrainConfig()


# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    os.makedirs(train_config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(train_config.LOG_DIR, exist_ok=True)
    os.makedirs(train_config.TENSORBOARD_DIR, exist_ok=True)
    print(f"✅ 目录已创建:")
    print(f"   - 检查点: {train_config.CHECKPOINT_DIR}")
    print(f"   - 日志: {train_config.LOG_DIR}")
    print(f"   - TensorBoard: {train_config.TENSORBOARD_DIR}")


if __name__ == "__main__":
    create_directories()
    print(f"\n实验配置: {train_config.EXPERIMENT_NAME}")
    print(f"描述: {train_config.DESCRIPTION}")