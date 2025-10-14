"""
环境配置文件
定义港口布局、AGV参数等环境相关配置
"""


class EnvConfig:
    """环境配置类"""

    # ===== 港口布局参数 =====
    PORT_WIDTH = 640.0  # 港口宽度(米)
    PORT_HEIGHT = 320.0  # 港口高度(米)

    NUM_HORIZONTAL_LANES = 3  # 水平通道数量（简化版）
    LANE_WIDTH = 8.0  # 车道宽度(米)

    # QC和YC位置
    NUM_QC = 3  # 岸桥数量
    NUM_YC = 3  # 场桥数量
    QC_POSITIONS = [  # QC位置 (x, y)
        (50.0, 40.0),
        (50.0, 160.0),
        (50.0, 280.0)
    ]
    YC_POSITIONS = [  # YC位置 (x, y)
        (590.0, 40.0),
        (590.0, 160.0),
        (590.0, 280.0)
    ]

    # ===== AGV参数 =====
    NUM_AGVS = 5  # AGV数量
    AGV_MAX_SPEED = 4.0  # 最大速度(米/秒)
    AGV_MAX_ACCEL = 1.0  # 最大加速度(米/秒²)
    AGV_LENGTH = 6.0  # AGV长度(米)
    AGV_WIDTH = 3.0  # AGV宽度(米)

    # 双向路由参数
    BIDIRECTIONAL = True  # 是否启用双向路由
    SAFE_DISTANCE = 10.0  # 安全距离(米)

    # ===== 任务参数 =====
    MAX_TASKS = 20  # 最大任务数
    TASK_GENERATION_RATE = 0.3  # 任务生成率

    # 任务类型
    TASK_TYPES = ['import', 'export']  # 进口/出口

    # ===== 时间参数 =====
    TIME_STEP = 0.5  # 仿真时间步长(秒)
    MAX_EPISODE_STEPS = 2000  # 最大步数

    # QC/YC操作时间(秒) - 服从均匀分布
    QC_OPERATION_TIME = (40, 200)
    YC_OPERATION_TIME = (40, 200)

    # ===== 观察空间维度 =====
    OBS_DIM = 64  # 观察空间维度
    GLOBAL_STATE_DIM = 128  # 全局状态维度

    # ===== 动作空间 =====
    # 离散动作
    NUM_LANES = NUM_HORIZONTAL_LANES  # 车道选择
    NUM_DIRECTIONS = 2  # 方向选择：前进/后退

    # 连续动作
    CONTINUOUS_ACTION_DIM = 2  # [加速度, 转向角]

    # ===== 奖励参数 =====
    REWARD_WEIGHTS = {
        'task_completion': 10.0,  # 完成任务
        'time_penalty': -0.1,  # 时间惩罚
        'distance_penalty': -0.01,  # 距离惩罚
        'collision': -50.0,  # 碰撞
        'near_collision': -5.0,  # 险情
        'deadlock': -30.0,  # 死锁
        'cooperation': 2.0,  # 协作
        'blocking': -3.0,  # 阻塞
        'bidirectional_bonus': 5.0,  # 双向利用奖励
        'direction_change': -2.0,  # 频繁换向惩罚
    }

    # ===== 可视化参数 =====
    RENDER = False  # 是否渲染
    RENDER_FPS = 30  # 渲染帧率

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }


# 创建全局配置实例
env_config = EnvConfig()