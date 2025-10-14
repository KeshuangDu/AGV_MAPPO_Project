"""
模型配置文件
定义神经网络架构相关参数
"""


class ModelConfig:
    """模型配置类"""

    # ===== 通用参数 =====
    HIDDEN_DIM = 256  # 隐藏层维度
    NUM_LAYERS = 2  # 网络层数
    ACTIVATION = 'relu'  # 激活函数

    # ===== GNN参数 =====
    USE_GNN = True  # 是否使用GNN
    GNN_TYPE = 'GAT'  # GNN类型: GAT, GCN, GraphSAGE

    GNN_HIDDEN_DIM = 128  # GNN隐藏层维度
    GNN_NUM_LAYERS = 2  # GNN层数
    GNN_NUM_HEADS = 4  # 注意力头数(GAT)
    GNN_DROPOUT = 0.1  # Dropout率

    # ===== Attention参数 =====
    USE_ATTENTION = True  # 是否使用注意力机制

    ATTN_HIDDEN_DIM = 256  # 注意力隐藏维度
    ATTN_NUM_HEADS = 8  # 多头注意力头数
    ATTN_DROPOUT = 0.1  # Dropout率

    # ===== Actor网络 =====
    ACTOR_HIDDEN_DIMS = [256, 256, 128]  # Actor隐藏层维度

    # 动作输出维度
    TASK_OUTPUT_DIM = 10  # 任务选择
    DIRECTION_OUTPUT_DIM = 2  # 方向选择(前进/后退)
    LANE_OUTPUT_DIM = 3  # 车道选择
    MOTION_OUTPUT_DIM = 2  # 运动控制(加速度, 转向)

    # ===== Critic网络 =====
    CRITIC_HIDDEN_DIMS = [256, 256, 128]  # Critic隐藏层维度

    USE_CENTRALIZED_CRITIC = False  # 是否使用中心化Critic

    # ===== 轨迹预测模块 =====
    USE_TRAJECTORY_PREDICTION = True  # 是否使用轨迹预测

    LSTM_HIDDEN_DIM = 128  # LSTM隐藏维度
    LSTM_NUM_LAYERS = 2  # LSTM层数
    PREDICTION_HORIZON = 10  # 预测时间步数

    # ===== 归一化 =====
    USE_LAYER_NORM = True  # 是否使用LayerNorm
    USE_BATCH_NORM = False  # 是否使用BatchNorm

    # ===== 初始化 =====
    WEIGHT_INIT = 'orthogonal'  # 权重初始化方法

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }


# 创建全局配置实例
model_config = ModelConfig()