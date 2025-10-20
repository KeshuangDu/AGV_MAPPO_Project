"""
训练配置文件
定义MAPPO算法相关的训练超参数
"""


class TrainConfig:
    """训练配置类"""

    # ===== 基础训练参数 =====
    NUM_EPISODES = 5000  # 训练总轮数
    MAX_STEPS_PER_EPISODE = 2000  # 每轮最大步数

    BATCH_SIZE = 256  # 批次大小
    BUFFER_SIZE = 10000  # 经验回放缓冲区大小

    # ===== PPO超参数 =====
    GAMMA = 0.99  # 折扣因子
    GAE_LAMBDA = 0.95  # GAE参数

    PPO_EPOCHS = 10  # PPO更新轮数
    CLIP_EPSILON = 0.2  # PPO裁剪参数

    VALUE_LOSS_COEF = 0.5  # 价值损失系数
    ENTROPY_COEF = 0.01  # 熵正则化系数
    MAX_GRAD_NORM = 0.5  # 梯度裁剪

    # ===== 学习率 =====
    ACTOR_LR = 3e-4  # Actor学习率
    CRITIC_LR = 3e-4  # Critic学习率

    LR_DECAY = True  # 是否衰减学习率
    LR_DECAY_RATE = 0.99  # 学习率衰减率

    # ===== 课程学习参数 =====
    USE_CURRICULUM = True  # 是否使用课程学习
    CURRICULUM_STAGES = [
        {'num_agvs': 1, 'episodes': 500},
        {'num_agvs': 2, 'episodes': 1000},
        {'num_agvs': 3, 'episodes': 1500},
        {'num_agvs': 5, 'episodes': 2000},
    ]

    # ===== 探索策略 =====
    EXPLORATION_NOISE = 0.1  # 探索噪声
    NOISE_DECAY = 0.995  # 噪声衰减
    MIN_NOISE = 0.01  # 最小噪声

    # ===== 保存和日志 =====
    SAVE_INTERVAL = 100  # 模型保存间隔(轮)
    LOG_INTERVAL = 10  # 日志记录间隔(轮)
    EVAL_INTERVAL = 50  # 评估间隔(轮)

    CHECKPOINT_DIR = './data/checkpoints'
    LOG_DIR = './data/logs'
    DATA_DIR = './data/generated_data'

    # ===== TensorBoard =====
    USE_TENSORBOARD = True  # 是否使用TensorBoard
    TB_LOG_DIR = './runs'  # TensorBoard日志目录

    # ===== 评估参数 =====
    NUM_EVAL_EPISODES = 10  # 评估轮数

    # ===== 随机种子 =====
    SEED = 42  # 随机种子

    # ===== GPU配置 =====
    USE_CUDA = True  # 是否使用GPU
    DEVICE = 'cuda'  # 设备选择

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }


# 创建全局配置实例
train_config = TrainConfig()