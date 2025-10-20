"""
环境配置文件 - 改进版 v2
基于调试评估发现的问题优化

更新日期：2025.10.17
更新内容：
1. 放宽碰撞终止条件（30->100次）
2. 增强接近目标的奖励
3. 调整碰撞惩罚避免过度惩罚
4. 增加加速奖励
"""


class EnvConfig:
    """环境配置类 - 改进版"""

    # ===== 港口布局参数 =====
    PORT_WIDTH = 640.0  # 港口宽度(米)
    PORT_HEIGHT = 320.0  # 港口高度(米)

    NUM_HORIZONTAL_LANES = 3  # 水平通道数量
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

    # ===== 任务管理配置 =====
    USE_TASK_MANAGER = True  # 是否使用任务管理器
    TASK_ASSIGNMENT_STRATEGY = 'sequential'  # 任务分配策略
    
    ARRIVAL_THRESHOLD = 20.0  # 到达判定距离（米）
    VERBOSE = False  # 是否打印详细日志（调试用，训练时建议False）

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

    # ===== 奖励配置 ✨✨ 改进版 v2 =====
    REWARD_TYPE = 'dense'  # 奖励类型：'sparse' 或 'dense'
    USE_DENSE_REWARD = True

    REWARD_WEIGHTS = {
        # === 核心奖励（进一步增强）✨ ===
        'task_completion': 300.0,       # 任务完成奖励（从200增加）
        'time_penalty': -0.003,         # 时间惩罚（从-0.005减小）

        # === 安全惩罚（大幅减小，避免过度惩罚）✨ ===
        'collision': -10.0,             # 碰撞惩罚（从-20减小）
        'near_collision': -1.0,         # 险情惩罚（从-2减小）
        'deadlock': -10.0,              # 死锁惩罚（从-15减小）

        # === 协作奖励 ===
        'cooperation': 2.0,
        'blocking': -1.0,               # 阻塞惩罚（从-1.5减小）

        # === 双向路由 ===
        'bidirectional_bonus': 5.0,
        'direction_change': -0.1,       # 频繁换向惩罚（从-0.2减小）

        # === 密集奖励（大幅增强引导）✨✨ ===
        'distance_progress': 1.0,       # 朝目标前进奖励（从0.5增加）
        'approach_bonus_50': 2.0,       # 50米内奖励（从1.0增加）
        'approach_bonus_30': 5.0,       # 30米内奖励（从2.0增加）
        'approach_bonus_15': 15.0,      # 15米内奖励（从5.0增加）✨ 关键
        'pickup_success': 30.0,         # 成功pickup（从15.0增加）

        # === 新增：加速奖励（鼓励积极移动）✨✨ ===
        'acceleration_bonus': 0.5,      # 积极加速奖励（新增）
        'reaching_near_target': 10.0,   # 到达目标附近（25米内）奖励（新增）
    }

    # ===== 终止条件配置 ✨✨ 放宽 =====
    MAX_COLLISIONS = 100  # 最大碰撞次数（从30增加到100）

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


# ===== 使用说明 =====
if __name__ == "__main__":
    print("=" * 60)
    print("环境配置 - 改进版 v2")
    print("=" * 60)
    print("\n基于调试评估的改进：")
    print("1. ✅ 放宽碰撞终止：30次 -> 100次")
    print("2. ✅ 减小碰撞惩罚：-20 -> -10")
    print("3. ✅ 增强接近奖励：15米内从5.0增加到15.0")
    print("4. ✅ 新增加速奖励：鼓励AGV积极移动")
    print("5. ✅ 新增接近目标奖励：25米内给予奖励")
    print("\n关键改进原因：")
    print("  - 调试发现AGV能接近到31米，说明导航能力已初步形成")
    print("  - 碰撞次数过多（31-33次）导致过早终止")
    print("  - 加速度总是负数，需要鼓励积极加速")
    print("=" * 60)
