# 🚢 水平布局港口AGV双向调度 - MAPPO多智能体强化学习

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-DLMU-green.svg)](LICENSE)

> 基于多智能体近端策略优化（MAPPO）的自动化港口AGV双向路由调度系统

## 📋 项目特点

- ✅ **水平布局港口**：真实还原新型港口布局（如天津港C段），AGV在平行通道间高效运输
- ✅ **双向路由**：AGV支持前进/后退双向运动，无需掉头，节省时间和空间
- ✅ **多智能体学习**：基于MAPPO算法，支持多AGV协同决策和分布式控制
- ✅ **任务管理器**：创新的任务分配和完成检测机制
- ✅ **奖励塑形**：密集奖励引导，解决稀疏奖励问题
- ✅ **模块化设计**：清晰的代码结构，方便进行消融实验
- ✅ **完整流程**：从环境搭建、模型训练到结果评估的完整pipeline

---

## 🎯 核心创新点

相比原论文（Cao et al., 2023, C&IE）的改进：

| 维度 | 原论文（BDE算法） | 本项目（MAPPO算法） | 创新性 |
|------|-------------------|---------------------|--------|
| **布局** | 垂直布局 | 水平布局 | ⭐⭐⭐ 更符合新型港口 |
| **方法** | 差分进化算法 | 多智能体深度RL | ⭐⭐⭐⭐⭐ 自适应学习 |
| **路由** | 离线规划+动态调整 | 在线学习双向策略 | ⭐⭐⭐⭐ 实时决策 |
| **协作** | 集中式优化 | 分布式智能协作 | ⭐⭐⭐⭐⭐ 可扩展性强 |
| **任务管理** | 简单分配 | 智能任务管理器 | ⭐⭐⭐⭐ 创新模块 |

---

## 🏗️ 项目结构
```
AGV_MAPPO_Project/
├── algorithm/              # MAPPO算法
│   └── mappo.py           # PPO裁剪、GAE、梯度裁剪
│
├── config/                 # 配置文件
│   ├── env_config.py      # 环境配置（港口布局、AGV参数）
│   ├── model_config.py    # 模型配置（网络结构）
│   ├── train_config.py    # 训练配置（超参数）
│   └── experiment_configs.py  # 消融实验配置
│
├── environment/            # 环境模块
│   ├── port_env.py        # 港口环境主类
│   ├── agv.py             # AGV实体（双向运动）
│   ├── equipment.py       # QC/YC设备类
│   ├── task_manager.py    # 任务管理器⭐（创新）
│   └── reward_shaper.py   # 奖励塑形器⭐（创新）
│
├── models/                 # 神经网络模型
│   └── actor_critic.py    # Actor-Critic网络
│
├── utils/                  # 工具函数
│   ├── data_generator.py  # 数据生成器
│   └── visualizer.py      # 可视化工具
│
├── data/                   # 数据存储
│   ├── checkpoints/       # 模型检查点
│   └── logs/              # 日志文件
│
├── runs/                   # TensorBoard日志
│   ├── quick/             # 快速测试（100轮）
│   ├── medium/            # 中等规模（1000轮）
│   └── standard/          # 完整训练（5000轮）
│
├── train.py               # 训练主程序（统一入口）
├── evaluate.py            # 评估程序（统一入口）
├── test_environment.py    # 环境测试脚本
├── README.md              # 本文档
└── requirements.txt       # 依赖包
```

---

## 🚀 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone <repository-url>
cd AGV_MAPPO_Project

# 创建虚拟环境
conda create -n agv_mappo python=3.9
conda activate agv_mappo

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速测试（100轮，1-2小时）

验证代码是否正常运行：
```bash
# 快速训练100轮
python train.py --mode quick

# 启动TensorBoard监控
tensorboard --logdir=./runs/quick

# 快速评估
python evaluate.py \
    --checkpoint ./data/checkpoints_quick/mappo_final_quick.pt \
    --episodes 10 \
    --verbose
```

### 3. 中等规模训练（1000轮，6-10小时）

获得初步可用的模型：
```bash
# 中等规模训练
python train.py --mode medium

# 每200轮评估一次
python evaluate.py \
    --checkpoint ./data/checkpoints_medium/mappo_episode_200.pt \
    --episodes 50

# TensorBoard监控
tensorboard --logdir=./runs/medium
```

### 4. 完整训练（5000轮，24-48小时）

获得最佳性能模型：
```bash
# 完整训练
python train.py --mode standard

# 或从中断处恢复
python train.py --mode standard \
    --resume ./data/checkpoints/mappo_episode_1000.pt

# 最终评估
python evaluate.py \
    --checkpoint ./data/checkpoints/mappo_final_standard.pt \
    --episodes 500 \
    --save-results
```

### 5. 自定义训练
```bash
# 自定义轮数
python train.py --episodes 500

# 自定义轮数+模式（会覆盖模式的默认轮数）
python train.py --mode medium --episodes 1500
```

---

## 📊 核心功能模块

### 1. 环境模拟

**水平布局港口环境** (`environment/port_env.py`)
- 3条平行水平通道（可配置）
- 双向路由支持（AGV可前进/后退）
- QC/YC设备协调
- 进出口任务混合

**AGV实体** (`environment/agv.py`)
- 双向运动能力（无需掉头）
- 碰撞检测（chase/reverse/cross三种）
- 轨迹记录
- 任务状态管理

**任务管理器** (`environment/task_manager.py`) ⭐**创新点**
- 多种分配策略（sequential/nearest/random/priority）
- 自动到达检测（pickup/delivery）
- 任务完成统计
- 便于消融实验对比

**奖励塑形器** (`environment/reward_shaper.py`) ⭐**创新点**
- 密集奖励 vs 稀疏奖励
- 接近奖励（25米内+10，15米内+15）
- 加速奖励（鼓励积极移动）
- 解决"只学会减速"问题

### 2. MAPPO算法

**Actor-Critic网络** (`models/actor_critic.py`)
- 离散动作：车道选择、方向选择
- 连续动作：加速度、转向角
- 中心化Critic（可选）
- LayerNorm稳定训练

**MAPPO训练** (`algorithm/mappo.py`)
- PPO裁剪（clip_epsilon=0.2）
- GAE优势估计
- 梯度裁剪（max_grad_norm=0.5）
- 学习率调度

### 3. 可视化工具

**训练可视化**
- TensorBoard实时监控
- 奖励曲线、损失变化
- 任务完成数、碰撞次数
- 熵值趋势

**评估可视化** (`--save-results`)
- 奖励分布直方图
- Episode长度曲线
- 任务完成趋势
- 碰撞次数统计

---

## ⚙️ 配置说明

### 环境配置 (`config/env_config.py`)
```python
# 港口布局
NUM_HORIZONTAL_LANES = 3    # 水平通道数
PORT_WIDTH = 640.0          # 宽度(米)
PORT_HEIGHT = 320.0         # 高度(米)

# AGV参数
NUM_AGVS = 5                # AGV数量
AGV_MAX_SPEED = 4.0         # 最大速度(米/秒)
BIDIRECTIONAL = True        # 启用双向路由

# 任务管理
USE_TASK_MANAGER = True     # 使用任务管理器
TASK_ASSIGNMENT_STRATEGY = 'sequential'  # 分配策略
ARRIVAL_THRESHOLD = 20.0    # 到达判定距离(米)

# 奖励配置（改进版v2）
REWARD_TYPE = 'dense'       # 密集奖励
REWARD_WEIGHTS = {
    'task_completion': 300.0,     # 任务完成奖励↑
    'collision': -10.0,           # 碰撞惩罚↓（避免过度惩罚）
    'approach_25m': 10.0,         # 接近奖励（新增）
    'approach_15m': 15.0,         # 接近奖励（新增）
    'acceleration_bonus': 0.5,    # 加速奖励（新增）
    'bidirectional_bonus': 5.0,   # 双向路由奖励
}
```

### 训练配置 (`config/train_config.py`)
```python
# 训练参数（会被命令行参数覆盖）
NUM_EPISODES = 5000         # 训练轮数
BATCH_SIZE = 256            # 批次大小
PPO_EPOCHS = 10             # PPO更新轮数

# 学习率
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

# PPO超参数
GAMMA = 0.99                # 折扣因子
GAE_LAMBDA = 0.95           # GAE参数
CLIP_EPSILON = 0.2          # 裁剪参数
```

---

## 🔬 消融实验指南

### 实验配置 (`config/experiment_configs.py`)

项目提供8个预配置实验：
```python
EXPERIMENT_1_BASELINE          # 基线配置
EXPERIMENT_2_DENSE_REWARD      # 密集奖励
EXPERIMENT_3_NEAREST_ASSIGNMENT  # 最近任务分配
EXPERIMENT_4_RANDOM_ASSIGNMENT   # 随机任务分配
EXPERIMENT_5_UNIDIRECTIONAL      # 单向路由（对比）
EXPERIMENT_6_MORE_LANES          # 5条通道（对比3条）
EXPERIMENT_7_NO_TASK_MANAGER     # 不使用任务管理器
EXPERIMENT_8_DEBUG_MODE          # 调试模式
```

### 运行消融实验
```bash
# 修改 config/env_config.py 应用实验配置
# 例如：测试单向 vs 双向

# 实验1：双向路由（基线）
BIDIRECTIONAL = True
python train.py --mode medium

# 实验2：单向路由（对比）
BIDIRECTIONAL = False
python train.py --mode medium

# 对比结果
python evaluate.py --checkpoint <exp1_checkpoint> --episodes 100
python evaluate.py --checkpoint <exp2_checkpoint> --episodes 100
```

---

## 📈 预期实验结果

### 训练进度指标

| 训练阶段 | 轮数 | 任务完成数 | 平均奖励 | 平均距离 |
|----------|------|------------|----------|----------|
| **初期** | 100 | 0-1 | -100~0 | 25-30m |
| **中期** | 500 | 2-5 | -50~0 | 20-25m |
| **后期** | 1000 | 5-10 | 0-50 | 15-20m |
| **收敛** | 5000 | 10-20 | 50-150 | <15m |

### 改进效果对比

基于evaluate_debug.py的调试发现的问题及改进：

#### **问题诊断：**
1. ❌ 碰撞过多导致过早终止（31-33次碰撞→终止）
2. ❌ 模型只学会减速（accel总是负数）
3. ❌ 精准到达控制不足（最近31米，阈值20米）

#### **改进方案：**
| 改进项 | 改进前 | 改进后 | 效果 |
|--------|--------|--------|------|
| 碰撞终止条件 | 30次 | 100次 | 给予更多学习机会 ✅ |
| 碰撞惩罚 | -20.0 | -10.0 | 减少过度惩罚 ✅ |
| 任务完成奖励 | 200.0 | 300.0 | 增强完成动力 ✅ |
| 25米内奖励 | 无 | +10.0 | 填补引导空白 ✨ |
| 15米内奖励 | 5.0 | 15.0 | 强化精准控制 ✅ |
| 加速奖励 | 无 | +0.5 | 鼓励积极移动 ✨ |

#### **改进效果：**
- ✅ Episode长度增加：250步 → 400-600步（+60-140%）
- ✅ 平均距离减小：31米 → 25米 → 15米
- ✅ 任务完成数提升：0-2 → 5-10个
- ✅ 奖励转正：-100 → 0 → +50

---

## 🎯 成功标准

### 最小成功（100轮训练后）
- ✅ Episode长度 > 300步
- ✅ 平均距离 < 28米
- ✅ 奖励波动减小

### 基础成功（500轮训练后）
- ✅ 任务完成数 > 1
- ✅ 平均奖励 > -50
- ✅ 平均距离 < 25米

### 理想成功（1000轮训练后）
- ✅ 任务完成数 > 5
- ✅ 平均奖励 > 0
- ✅ 完成率 > 30%

### 论文级别（5000轮训练后）
- ✅ 任务完成率 > 50%
- ✅ 平均奖励 > 50
- ✅ 双向路由使用率 > 60%
- ✅ 碰撞率 < 10%

---

## 💡 常见问题

### Q1: 训练速度太慢？
**A:** 优化建议：
- 确保使用GPU：`torch.cuda.is_available()`
- 减少PPO更新轮数：`PPO_EPOCHS = 5`（默认10）
- 减少最大步数：`MAX_EPISODE_STEPS = 1000`（默认2000）
- 减少Batch Size：`BATCH_SIZE = 128`（默认256）

### Q2: 内存不足？
**A:** 
- 减少AGV数量：`NUM_AGVS = 3`（默认5）
- 减少Buffer大小（如果实现了buffer）
- 使用CPU训练（较慢但省内存）

### Q3: 模型不收敛？
**A:** 
- 延长训练时间（至少1000轮）
- 检查奖励设置是否合理
- 降低学习率：`ACTOR_LR = 1e-4`
- 增加Entropy系数：`ENTROPY_COEF = 0.02`

### Q4: 如何对比改进效果？
**A:** 
```bash
# 训练改进前模型
# （修改env_config.py使用旧奖励）
python train.py --mode quick

# 训练改进后模型
# （使用新奖励配置）
python train.py --mode quick

# 对比评估
python evaluate.py --checkpoint <old_model> --episodes 50
python evaluate.py --checkpoint <new_model> --episodes 50
```

### Q5: 中断后如何恢复？
**A:** 
```bash
python train.py --mode medium \
    --resume ./data/checkpoints_medium/mappo_episode_500.pt
```

---

## 📚 参考文献

1. **MAPPO算法**:  
   Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2021

2. **原论文（对比基线）**:  
   Cao et al., "AGV dispatching and bidirectional conflict-free routing problem in automated container terminal", C&IE 2023

3. **AGV调度综述**:  
   Yang et al., "An integrated scheduling method for AGV routing in automated container terminals", C&IE 2018

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发指南

#### 添加新的任务分配策略

在 `environment/task_manager.py` 中添加：
```python
def _assign_your_strategy(self, agvs, tasks):
    """你的分配策略"""
    # 实现逻辑
    pass
```

#### 添加新的奖励项

在 `environment/reward_shaper.py` 中添加：
```python
def compute_reward(self, ...):
    # 新奖励项
    if condition:
        reward += self.config.REWARD_WEIGHTS['new_reward']
```

在 `config/env_config.py` 中添加权重：
```python
REWARD_WEIGHTS = {
    'new_reward': 1.0,
}
```

---

## 📄 许可证

本项目采用 DLMU 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

如有问题，请提交Issue或联系开发者。

---

**祝实验顺利！🎉**

---

## 📌 更新日志

### v2.0 (2025.10.17) - 重大改进
- ✨ 新增任务管理器模块
- ✨ 新增奖励塑形器模块
- 🔧 统一训练和评估脚本
- 📝 合并所有README文档
- 🎯 放宽碰撞终止条件（30→100次）
- 📈 改进奖励函数（密集奖励+接近奖励+加速奖励）

### v1.0 (2025.10.10) - 初始版本
- ✅ 基础环境实现
- ✅ MAPPO算法实现
- ✅ 双向路由支持