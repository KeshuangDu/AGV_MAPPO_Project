# 🚢 水平布局港口AGV双向调度 - MAPPO多智能体强化学习

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-DLMU-green.svg)](LICENSE)

基于多智能体近端策略优化（MAPPO）的自动化港口AGV双向路由调度系统。支持水平布局港口、双向路由和多AGV协同决策。

## 📋 项目特点

- ✅ **水平布局港口**：真实还原新型港口布局，AGV在平行通道间高效运输
- ✅ **双向路由**：AGV支持前进/后退双向运动，无需掉头，节省时间和空间
- ✅ **多智能体学习**：基于MAPPO算法，支持多AGV协同决策
- ✅ **模块化设计**：清晰的代码结构，方便替换布局和进行消融实验
- ✅ **完整训练流程**：包含训练、评估、可视化全套工具
- ✅ **数据保存**：自动保存随机生成的训练数据，支持复现和分析

## 🏗️ 项目结构

```
AGV_MAPPO_Project/
├── config/                  # 配置文件
│   ├── env_config.py       # 环境配置（港口布局、AGV参数等）
│   ├── train_config.py     # 训练配置（超参数、保存路径等）
│   └── model_config.py     # 模型配置（网络结构等）
├── environment/            # 环境模块
│   ├── port_env.py        # 港口环境主类
│   ├── agv.py             # AGV实体类
│   └── equipment.py       # QC/YC设备类
├── models/                 # 神经网络模型
│   ├── actor_critic.py    # Actor-Critic网络
│   ├── gnn.py             # 图神经网络（可选）
│   └── attention.py       # 注意力模块（可选）
├── algorithm/              # MAPPO算法
│   ├── mappo.py           # MAPPO主算法
│   └── buffer.py          # 经验回放缓冲区
├── utils/                  # 工具函数
│   ├── data_generator.py  # 数据生成器
│   └── visualizer.py      # 可视化工具
├── data/                   # 数据存储
│   ├── checkpoints/       # 模型检查点
│   ├── logs/              # 日志文件
│   └── generated_data/    # 随机生成的训练数据
├── train.py               # 训练主程序
├── evaluate.py            # 评估程序
└── requirements.txt       # 依赖包
```

## 🚀 快速开始

### 1. 环境安装

首先克隆项目并创建Anaconda虚拟环境：

```bash
# 创建虚拟环境
conda create -n agv_mappo python=3.9
conda activate agv_mappo

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置参数

在 `config/` 目录下修改配置文件：

- `env_config.py`: 调整港口布局、AGV数量、任务参数等
- `train_config.py`: 修改训练轮数、学习率、保存间隔等
- `model_config.py`: 自定义网络结构

### 3. 开始训练

```bash
# 基础训练（默认参数）
python train.py

# 训练完成后，模型和数据将保存在 data/ 目录下
```

训练过程中会自动：
- 保存模型检查点（每100轮）
- 记录TensorBoard日志
- 保存随机生成的训练数据
- 绘制训练曲线

### 4. 查看训练过程

启动TensorBoard：

```bash
tensorboard --logdir=runs
```

在浏览器中打开 `http://localhost:6006` 查看训练指标。

### 5. 评估模型

```bash
# 评估最终模型
python evaluate.py --checkpoint ./data/checkpoints/500mappo_final.pt --episodes 100

# 评估特定检查点
python evaluate.py --checkpoint ./data/checkpoints/mappo_episode_500.pt --episodes 50
```

评估结果包括：
- 奖励分布
- 任务完成率
- 碰撞次数
- 双向路由使用率
- 详细统计图表

### 6. 生成数据

```bash
# 批量生成港口场景数据
cd utils
python data_generator.py
```

生成的数据将保存为：
- `scenarios.pkl`: Python pickle格式
- `scenarios.json`: JSON格式
- `agv_data.csv`: AGV数据CSV
- `task_data.csv`: 任务数据CSV

## 📊 主要功能模块

### 1. 环境模拟

**水平布局港口环境** (`environment/port_env.py`)
- 3条平行水平通道
- 双向路由支持
- QC/YC设备协调
- 进出口任务混合

**AGV实体** (`environment/agv.py`)
- 双向运动能力（前进/后退）
- 碰撞检测
- 轨迹记录
- 任务状态管理

### 2. MAPPO算法

**Actor-Critic网络** (`models/actor_critic.py`)
- 离散动作：车道选择、方向选择
- 连续动作：加速度、转向角
- 中心化Critic（可选）
- LayerNorm稳定训练

**MAPPO训练** (`algorithm/mappo.py`)
- PPO裁剪
- GAE优势估计
- 梯度裁剪
- 学习率调度

### 3. 可视化工具

**环境可视化** (`utils/visualizer.py`)
- 实时港口状态展示
- AGV运动轨迹
- 双向路由标识
- 任务流向可视化

**训练可视化**
- TensorBoard实时监控
- 奖励曲线
- 损失变化
- 熵值趋势

## ⚙️ 配置说明

### 环境配置 (env_config.py)

```python
# 港口布局
NUM_HORIZONTAL_LANES = 3    # 水平通道数
PORT_WIDTH = 640.0          # 宽度(米)
PORT_HEIGHT = 320.0         # 高度(米)

# AGV参数
NUM_AGVS = 5                # AGV数量
AGV_MAX_SPEED = 4.0         # 最大速度(米/秒)
BIDIRECTIONAL = True        # 启用双向路由

# 奖励权重
REWARD_WEIGHTS = {
    'task_completion': 10.0,
    'collision': -50.0,
    'bidirectional_bonus': 5.0,  # 双向路由奖励
    'direction_change': -2.0,    # 频繁换向惩罚
}
```

### 训练配置 (train_config.py)

```python
# 训练参数
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

### 模型配置 (model_config.py)

```python
# 网络结构
ACTOR_HIDDEN_DIMS = [256, 256, 128]
CRITIC_HIDDEN_DIMS = [256, 256, 128]

# 高级模块（可选）
USE_GNN = True              # 图神经网络
USE_ATTENTION = True        # 注意力机制
```

## 📈 实验结果

训练5000轮后的典型结果：

| 指标 | 数值 |
|-----|------|
| 平均奖励 | 150-200 |
| 任务完成率 | 85-95% |
| 碰撞率 | < 5% |
| 双向路由使用率 | 60-80% |
| 平均Episode长度 | 800-1200步 |

## 🔬 消融实验指南

### 1. 对比单向 vs 双向路由

修改 `config/env_config.py`:

```python
# 单向模式
BIDIRECTIONAL = False

# 双向模式
BIDIRECTIONAL = True
```

分别训练和评估，对比结果。

### 2. 测试不同AGV数量

```python
NUM_AGVS = 3   # 少量AGV
NUM_AGVS = 5   # 中等
NUM_AGVS = 10  # 大量AGV
```

### 3. 修改通道数量

```python
NUM_HORIZONTAL_LANES = 2  # 拥挤
NUM_HORIZONTAL_LANES = 3  # 平衡
NUM_HORIZONTAL_LANES = 5  # 宽松
```

### 4. 替换为垂直布局

在 `environment/port_env.py` 中修改布局逻辑，改为垂直通道布局。

## 🛠️ 开发指南

### 添加新的布局

1. 在 `config/env_config.py` 中添加新布局配置
2. 修改 `environment/port_env.py` 中的 `_init_equipment()` 方法
3. 更新 `_setup_spaces()` 适配新观察空间

### 添加新的奖励项

在 `environment/port_env.py` 的 `_compute_rewards()` 方法中添加：

```python
# 新奖励项
if condition:
    reward += self.reward_weights['new_reward']
```

在 `config/env_config.py` 中添加权重：

```python
REWARD_WEIGHTS = {
    'new_reward': 1.0,
}
```

### 扩展网络架构

在 `models/` 目录下添加新模块：

```python
# models/new_module.py
class NewModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络结构
```

在 `actor_critic.py` 中引入使用。

## 📚 参考文献

1. **MAPPO算法**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2021
2. **AGV调度**: Yang et al., "An integrated scheduling method for AGV routing in automated container terminals", C&IE 2018
3. **双向路由**: Cao et al., "AGV dispatching and bidirectional conflict-free routing problem in automated container terminal", C&IE 2023

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 DLMU 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题，请提交Issue或联系开发者。

---

**祝您实验顺利！🎉**