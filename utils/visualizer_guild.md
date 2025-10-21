# 🎨 港口环境可视化工具使用指南

## ✨ 新功能

增强版 `visualizer.py` 支持：

1. ✅ **自动读取配置** - 从 `env_config.py` 自动读取所有参数
2. ✅ **双布局支持** - 自动识别水平/垂直布局
3. ✅ **动态适配** - 自动适配 AGV、QC、YC 数量变化
4. ✅ **命令行参数** - 灵活的命令行控制
5. ✅ **更美观** - 改进的视觉效果和标注

---

## 🚀 快速开始

### 方法1：使用当前配置

```bash
# 直接运行，使用 env_config.py 中的配置
python visualizer.py
```

### 方法2：指定参数

```bash
# 垂直布局，8个AGV，4个QC，4个YC
python visualizer.py --layout vertical --num-agvs 8 --num-qc 4 --num-yc 4

# 水平布局，3个AGV，单向路由
python visualizer.py --layout horizontal --num-agvs 3 --unidirectional

# 保存图片
python visualizer.py --save ./output/port_layout.png
```

### 方法3：从实际环境可视化

```bash
# 从运行中的环境创建可视化（包含实际AGV位置和任务）
python visualizer.py --from-env --save ./output/current_state.png
```

---

## 📋 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--layout` | 布局类型 | `--layout vertical` |
| `--num-agvs` | AGV数量 | `--num-agvs 10` |
| `--num-qc` | 岸桥数量 | `--num-qc 5` |
| `--num-yc` | 场桥数量 | `--num-yc 5` |
| `--num-lanes` | 通道数量 | `--num-lanes 5` |
| `--bidirectional` | 启用双向路由 | `--bidirectional` |
| `--unidirectional` | 禁用双向路由 | `--unidirectional` |
| `--save` | 保存路径 | `--save output.png` |
| `--from-env` | 从实际环境创建 | `--from-env` |

---

## 💡 使用场景

### 场景1：查看当前配置的布局

当你修改了 `env_config.py` 中的参数后，想立即查看效果：

```bash
python visualizer.py
```

这会自动读取你的配置并显示布局。

### 场景2：对比不同配置

生成不同配置的对比图：

```bash
# 水平布局 - 3通道
python visualizer.py --layout horizontal --num-lanes 3 --save h_3lanes.png

# 水平布局 - 5通道
python visualizer.py --layout horizontal --num-lanes 5 --save h_5lanes.png

# 垂直布局 - 3通道
python visualizer.py --layout vertical --num-lanes 3 --save v_3lanes.png

# 垂直布局 - 5通道
python visualizer.py --layout vertical --num-lanes 5 --save v_5lanes.png
```

### 场景3：可视化训练过程

在训练脚本中调用可视化：

```python
from visualizer import visualize_from_environment

# 在训练循环中
if episode % 100 == 0:
    visualize_from_environment(
        env, 
        save_path=f'./output/episode_{episode}.png'
    )
```

### 场景4：生成论文图表

生成高质量的论文插图：

```bash
# 水平布局展示图
python visualizer.py --layout horizontal --num-agvs 5 \
    --num-qc 3 --num-yc 3 --bidirectional \
    --save paper_horizontal_layout.png

# 垂直布局展示图
python visualizer.py --layout vertical --num-agvs 5 \
    --num-qc 3 --num-yc 3 --bidirectional \
    --save paper_vertical_layout.png
```

---

## 🎨 可视化元素说明

### 颜色含义

- 🔴 **红色圆圈** - 岸桥（QC，Quay Crane）
- 🔵 **蓝色圆圈** - 场桥（YC，Yard Crane）
- 🌈 **彩色标记** - AGV
  - 圆形（○）- 空车
  - 方形（□）- 载货

### 方向箭头

- **实线箭头** - AGV前进方向
- **虚线箭头** - AGV后退方向（仅双向模式）

### 通道标识

- **灰色虚线** - 车道中心线
- **灰色标签** - 车道编号

---

## 🔧 在代码中使用

### 示例1：基础可视化

```python
from visualizer import PortVisualizer
from config.env_config import env_config

# 创建可视化器
visualizer = PortVisualizer(env_config)
visualizer.setup_figure()

# 添加图例
visualizer.add_legend()

# 保存
visualizer.save_snapshot('./output/port_layout.png')

# 显示
visualizer.show()
```

### 示例2：添加AGV

```python
from visualizer import PortVisualizer
from environment.agv import AGV

visualizer = PortVisualizer()
visualizer.setup_figure()

# 创建AGV
agvs = []
for i in range(5):
    agv = AGV(i, (200 + i*100, 160))
    agv.direction = 0
    agv.has_container = (i % 2 == 0)
    agvs.append(agv)

# 绘制AGV
visualizer.draw_agvs(agvs, show_direction=True)
visualizer.add_legend()
visualizer.show()
```

### 示例3：从环境可视化

```python
from visualizer import visualize_from_environment
from environment.port_env import PortEnvironment
from config.env_config import env_config

# 创建环境
env = PortEnvironment(env_config)
env.reset()

# 运行几步
for _ in range(10):
    actions = {...}  # 你的动作
    env.step(actions)

# 可视化当前状态
visualize_from_environment(env, save_path='current_state.png')
```

---

## 📊 批量生成图表

创建一个批量生成脚本 `generate_layouts.py`：

```python
#!/usr/bin/env python
import subprocess
import os

configs = [
    # (layout, num_agvs, num_qc, num_yc, num_lanes, bidirectional, name)
    ('horizontal', 5, 3, 3, 3, True, 'h_bi_3lanes'),
    ('horizontal', 5, 3, 3, 5, True, 'h_bi_5lanes'),
    ('vertical', 5, 3, 3, 3, True, 'v_bi_3lanes'),
    ('vertical', 5, 3, 3, 5, True, 'v_bi_5lanes'),
    ('horizontal', 5, 3, 3, 3, False, 'h_uni_3lanes'),
    ('vertical', 5, 3, 3, 3, False, 'v_uni_3lanes'),
]

os.makedirs('./output/layouts', exist_ok=True)

for config in configs:
    layout, agvs, qc, yc, lanes, bi, name = config
    
    cmd = [
        'python', 'visualizer.py',
        '--layout', layout,
        '--num-agvs', str(agvs),
        '--num-qc', str(qc),
        '--num-yc', str(yc),
        '--num-lanes', str(lanes),
        '--bidirectional' if bi else '--unidirectional',
        '--save', f'./output/layouts/{name}.png'
    ]
    
    print(f"生成: {name}")
    subprocess.run(cmd)

print("\n✅ 所有图表生成完成！")
```

运行：
```bash
python generate_layouts.py
```

---

## 🎯 与配置文件联动

### 自动适配流程

1. **修改配置文件** `config/env_config.py`
```python
# 修改这些参数
LAYOUT_TYPE = 'vertical'  # 改为垂直
NUM_AGVS = 10             # 改为10个AGV
NUM_QC = 5                # 改为5个岸桥
NUM_YC = 5                # 改为5个场桥
NUM_HORIZONTAL_LANES = 5  # 改为5条通道
BIDIRECTIONAL = True      # 启用双向
```

2. **运行可视化**
```bash
python visualizer.py
```

3. **查看效果** - 可视化会自动显示新配置！

### 验证配置变化

修改配置后，运行可视化确认：

```bash
# 保存一张图片记录
python visualizer.py --save config_verification.png

# 在浏览器中查看
open config_verification.png  # Mac
# 或
xdg-open config_verification.png  # Linux
```

---

## 📐 布局对比示例

### 水平布局（Horizontal）

```
QC0 ────────── Lane 0 ────────── YC0
  │                                │
QC1 ────────── Lane 1 ────────── YC1
  │                                │
QC2 ────────── Lane 2 ────────── YC2
```

特点：
- QC在左，YC在右
- AGV在水平通道上移动
- 适合传统港口布局

### 垂直布局（Vertical）

```
        YC0    YC1    YC2
         │      │      │
    Lane 0  Lane 1  Lane 2
         │      │      │
        QC0    QC1    QC2
```

特点：
- QC在下，YC在上
- AGV在垂直通道上移动
- 适合新型自动化港口

---

## 🐛 故障排除

### 问题1：图形不显示

```bash
# 确保安装了matplotlib
pip install matplotlib seaborn

# 在远程服务器上，保存而不是显示
python visualizer.py --save output.png
```

### 问题2：配置未生效

```bash
# 检查是否使用了正确的配置文件
python -c "from config.env_config import env_config; print(env_config.LAYOUT_TYPE)"

# 确认配置文件路径
python visualizer.py --layout vertical  # 强制指定
```

### 问题3：AGV位置重叠

这是正常的随机初始化，在实际训练中AGV会自动分散。

---

## 💻 完整示例

创建 `demo_visualizer.py`：

```python
"""
可视化演示脚本
展示如何使用增强版visualizer
"""

from visualizer import PortVisualizer, visualize_from_config
from config.env_config import env_config
import os

# 设置输出目录
output_dir = './output/visualizer_demo'
os.makedirs(output_dir, exist_ok=True)

print("🎨 可视化演示开始...\n")

# Demo 1: 使用当前配置
print("1️⃣ 使用当前配置生成可视化")
visualize_from_config(
    env_config, 
    save_path=f'{output_dir}/demo_1_current_config.png'
)

# Demo 2: 修改配置测试
print("2️⃣ 测试不同AGV数量")
for num_agvs in [3, 5, 8, 10]:
    temp_config = env_config
    temp_config.NUM_AGVS = num_agvs
    visualize_from_config(
        temp_config,
        save_path=f'{output_dir}/demo_2_agvs_{num_agvs}.png'
    )

# Demo 3: 测试两种布局
print("3️⃣ 测试两种布局")
for layout in ['horizontal', 'vertical']:
    temp_config = env_config
    temp_config.LAYOUT_TYPE = layout
    visualize_from_config(
        temp_config,
        save_path=f'{output_dir}/demo_3_layout_{layout}.png'
    )

print(f"\n✅ 所有演示完成！查看 {output_dir} 目录")
```

运行：
```bash
python demo_visualizer.py
```

---

## 📚 参考资料

- 配置文件：`config/env_config.py`
- 环境实现：`environment/port_env.py`
- AGV类：`environment/agv.py`

---

**现在你的可视化工具已经完全联动配置文件了！** 🎉

修改配置 → 运行 `python visualizer.py` → 自动看到新布局！