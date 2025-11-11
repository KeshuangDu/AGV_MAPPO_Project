"""
水平布局 + 双向路由 训练配置
Horizontal Layout + Bidirectional Routing

实验编号：实验1 (h_bi)
描述：baseline实验，使用v2改进的密集奖励
"""


class TrainConfig:
    """水平双向训练配置类"""

    # ===== 实验标识 =====
    EXPERIMENT_NAME = "h_bi"  # horizontal + bidirectional
    EXPERIMENT_DESC = "水平布局 + 双向路由（baseline）"

    # ===== 基础训练参数 =====
    NUM_EPISODES = 5000  # 训练1000轮
    MAX_STEPS_PER_EPISODE = 1000  # 每轮最大1000步

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
    USE_CURRICULUM = False  # 暂不使用课程学习

    # ===== 探索策略 =====
    EXPLORATION_NOISE = 0.1  # 探索噪声
    NOISE_DECAY = 0.995  # 噪声衰减
    MIN_NOISE = 0.01  # 最小噪声

    # ===== 保存和日志 =====
    SAVE_INTERVAL = 50  # 每50轮保存
    LOG_INTERVAL = 10  # 每10轮记录
    EVAL_INTERVAL = 50  # 每50轮评估

    # ✨ 实验专属目录
    CHECKPOINT_DIR = './data/checkpoints_h_bi'
    LOG_DIR = './data/logs_h_bi'
    DATA_DIR = './data/generated_data_h_bi'
    TB_LOG_DIR = './runs_h_bi'

    # ===== TensorBoard =====
    USE_TENSORBOARD = True

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


# ===== 打印配置信息 =====
if __name__ == "__main__":
    print("=" * 60)
    print("实验1：水平布局 + 双向路由 (h_bi)")
    print("=" * 60)
    print(f"训练轮数: {train_config.NUM_EPISODES}")
    print(f"检查点目录: {train_config.CHECKPOINT_DIR}")
    print(f"TensorBoard目录: {train_config.TB_LOG_DIR}")
    print("=" * 60)
    print("启动命令:")
    print(f"  训练: python train_h_bi.py")
    print(f"  监控: tensorboard --logdir={train_config.TB_LOG_DIR}")
    print(f"  评估: python evaluate_h_bi.py")
    print("=" * 60)