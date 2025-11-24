# DeepCFR with Feature Transform 使用指南

## 📋 概述

本项目提供了两种方式在 DeepCFR 中添加手动特征（起手牌强度、位置优势等）：

1. **简单版本（推荐）**：直接拼接7维手动特征到原始信息状态
2. **复杂版本**：特征转换层 + 可学习特征 + 降维

## 🚀 快速开始

### 方法1: 使用训练脚本（最简单，推荐）

训练脚本已经集成了特征转换功能，默认使用**简单版本**：

```bash
conda activate open_spiel
python train_deep_cfr_texas.py --num_players 6 --num_iterations 100
```

**默认配置**：
- ✅ 使用简单版本（`--use_simple_feature`，默认启用）
- ✅ 直接拼接7维手动特征
- ✅ 信息状态(281维) + 手动特征(7维) = 288维 -> MLP

### 方法2: 在代码中直接使用

#### 简单版本（推荐）

```python
from deep_cfr_simple_feature import DeepCFRSimpleFeature

solver = DeepCFRSimpleFeature(
    game,
    policy_network_layers=(256, 256),
    advantage_network_layers=(128, 128),
    num_iterations=100,
    num_traversals=20,
    learning_rate=1e-4,
    memory_capacity=int(1e6),
    device=device,
)
```

**流程**：信息状态(281维) + 手动特征(7维) = 288维 -> MLP

#### 复杂版本

```python
from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform

solver = DeepCFRWithFeatureTransform(
    game,
    policy_network_layers=(256, 256),
    advantage_network_layers=(128, 128),
    transformed_size=150,  # 转换后的特征大小
    use_hybrid_transform=True,  # 使用混合特征转换
    num_iterations=100,
    num_traversals=20,
    learning_rate=1e-4,
    memory_capacity=int(1e6),
    device=device,
)
```

**流程**：信息状态(281维) + 手动特征(7维) + 可学习特征(64维) = 352维 -> 降维到150维 -> MLP

## 📊 两种版本对比

| 特性 | 简单版本（推荐） | 复杂版本 |
|------|----------------|---------|
| **实现复杂度** | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| **输入维度** | 288维（281+7） | 352维（281+7+64） |
| **输出维度** | 直接到MLP | 降维到150维再MLP |
| **可学习特征** | ❌ 无 | ✅ 64维 |
| **特征归一化** | ❌ 无（依赖BatchNorm） | ✅ BatchNorm + LayerNorm |
| **计算效率** | ⭐⭐⭐ 高 | ⭐⭐ 中等 |
| **推荐场景** | 快速开始、简单需求 | 需要更多特征学习 |

## 📝 命令行参数

### 训练脚本参数

```bash
# 使用简单版本（默认，推荐）
python train_deep_cfr_texas.py --num_players 6 --num_iterations 100

# 使用复杂版本
python train_deep_cfr_texas.py --num_players 6 --no_simple_feature

# 不使用特征转换（标准DeepCFR）
python train_deep_cfr_texas.py --num_players 6 --no_feature_transform

# 完整参数示例
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 100 \
    --num_traversals 20 \
    --policy_layers 256 256 \
    --advantage_layers 128 128 \
    --learning_rate 1e-4 \
    --use_simple_feature \  # 使用简单版本（默认）
    --save_history
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_simple_feature` | 使用简单版本（直接拼接7维特征） | ✅ 启用 |
| `--no_simple_feature` | 不使用简单版本（使用复杂版本） | - |
| `--use_feature_transform` | 使用特征转换 | ✅ 启用 |
| `--no_feature_transform` | 不使用特征转换（标准DeepCFR） | - |
| `--transformed_size` | 复杂版本的转换后特征大小 | 150 |
| `--use_hybrid_transform` | 复杂版本使用混合特征转换 | ✅ 启用 |

## 🔍 手动特征说明（7维）

两种版本都使用相同的手动特征：

### 1. 位置优势特征（4维）

- **位置优势值**（0.0-1.0）：BTN=1.0, CO=0.6, MP=0.3, UTG=0.0
- **是否早期位置**（0或1）：UTG, MP
- **是否后期位置**（0或1）：CO, BTN
- **是否盲注位置**（0或1）：SB, BB

### 2. 起手牌强度特征（1维）

- **Preflop手牌强度**（0.0-1.0）：基于标准排名表
- 例如：AA=1.0, KK=0.95, AKs=0.88, 72o=0.0

### 3. 下注统计特征（2维）

- **归一化最大下注**：单次最大下注 / 20000
- **归一化总下注**：累计总下注 / 20000

## 💡 使用示例

### 示例1: 简单版本（推荐）

```python
from deep_cfr_simple_feature import DeepCFRSimpleFeature
import pyspiel

# 创建游戏
game_config = {
    "numPlayers": 6,
    "numBoardCards": "0 3 1 1",
    "numRanks": 13,
    "numSuits": 4,
    "firstPlayer": "2",
    "stack": "20000 20000 20000 20000 20000 20000",
    "blind": "50 100 0 0 0 0",
    "numHoleCards": 2,
    "numRounds": 4,
    "betting": "nolimit",
    "maxRaises": "3",
}
game = pyspiel.load_game("universal_poker", game_config)

# 创建简单版本的 DeepCFR Solver
solver = DeepCFRSimpleFeature(
    game,
    policy_network_layers=(256, 256),
    advantage_network_layers=(128, 128),
    num_iterations=100,
    num_traversals=20,
    learning_rate=1e-4,
)

# 训练（与标准 DeepCFR 相同）
for iteration in range(100):
    for player in range(game.num_players()):
        for _ in range(20):
            solver._traverse_game_tree(solver._root_node, player)
        solver._learn_advantage_network(player)
    solver._learn_strategy_network()
```

### 示例2: 复杂版本

```python
from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform

solver = DeepCFRWithFeatureTransform(
    game,
    policy_network_layers=(256, 256),
    advantage_network_layers=(128, 128),
    transformed_size=150,  # 特征维度从352降到150
    use_hybrid_transform=True,  # 使用混合特征转换
    num_iterations=100,
    num_traversals=20,
    learning_rate=1e-4,
)
```

### 示例3: 在训练脚本中使用

训练脚本已经集成，直接运行即可：

```bash
# 使用简单版本（默认）
conda activate open_spiel
python train_deep_cfr_texas.py --num_players 6 --num_iterations 100

# 使用复杂版本
python train_deep_cfr_texas.py --num_players 6 --no_simple_feature

# 使用标准版本（无特征）
python train_deep_cfr_texas.py --num_players 6 --no_feature_transform
```

## ⚙️ 训练建议

### 简单版本推荐配置

```python
solver = DeepCFRSimpleFeature(
    game,
    policy_network_layers=(256, 256),  # 可以适当减小，如(128, 128)
    advantage_network_layers=(128, 128),
    num_iterations=100,
    num_traversals=20,
    learning_rate=1e-4,
    memory_capacity=int(1e6),
)
```

**优点**：
- ✅ 实现简单，易于理解
- ✅ 计算效率高
- ✅ 直接利用7维手动特征
- ✅ 保持原始架构

### 复杂版本推荐配置

```python
solver = DeepCFRWithFeatureTransform(
    game,
    policy_network_layers=(256, 256),
    advantage_network_layers=(128, 128),
    transformed_size=150,  # 推荐值
    use_hybrid_transform=True,  # 推荐启用
    num_iterations=100,
    num_traversals=20,
    learning_rate=5e-5,  # 可以稍微调低
    memory_capacity=int(1e6),
)
```

**优点**：
- ✅ 有可学习特征（64维）
- ✅ 有特征归一化
- ✅ 先降维再处理，可能更高效

## 🧪 测试

### 测试简单版本

```bash
conda activate open_spiel
python deep_cfr_simple_feature.py
```

### 测试复杂版本

```bash
conda activate open_spiel
python test_deep_cfr_feature_transform.py
```

## 📊 与标准 DeepCFR 的对比

| 特性 | 标准 DeepCFR | 简单版本 | 复杂版本 |
|------|-------------|---------|---------|
| **输入维度** | 281维 | 288维（281+7） | 352维（281+7+64） |
| **输出维度** | 直接到MLP | 直接到MLP | 降维到150维再MLP |
| **领域知识** | ❌ 无 | ✅ 7维手动特征 | ✅ 7维手动特征 + 64维可学习特征 |
| **特征归一化** | ❌ 无 | ❌ 无（依赖BatchNorm） | ✅ BatchNorm + LayerNorm |
| **实现复杂度** | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **计算效率** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **推荐场景** | 基准测试 | **快速开始（推荐）** | 需要更多特征学习 |

## ⚠️ 注意事项

1. **模型保存/加载**：
   - 保存和加载方式与标准 DeepCFR 相同
   - 使用 `torch.save()` 和 `torch.load()`

2. **设备**：
   - 自动检测 GPU，如果没有则使用 CPU
   - 可以手动指定：`device=torch.device("cuda:0")`

3. **单样本推理**：
   - 简单版本：无需特殊处理
   - 复杂版本：自动处理 BatchNorm 问题（使用 LayerNorm 替代）

4. **评估**：
   - 与标准 DeepCFR 完全兼容
   - 可以使用相同的评估脚本

## 📚 相关文件

- `deep_cfr_simple_feature.py`: 简单版本实现（推荐）
- `deep_cfr_with_feature_transform.py`: 复杂版本实现
- `train_deep_cfr_texas.py`: 训练脚本（已集成两种版本）
- `test_deep_cfr_feature_transform.py`: 复杂版本测试脚本
- `ENHANCED_FEATURES_SUMMARY.md`: 特征详细说明
- `FEATURE_TRANSFORM_USAGE.md`: 特征转换使用指南
- `SIMPLE_FEATURE_APPROACH.md`: 简单版本说明

## 🎯 总结

**推荐使用简单版本**：

1. **导入**：`from deep_cfr_simple_feature import DeepCFRSimpleFeature`
2. **创建**：直接创建 `DeepCFRSimpleFeature`，无需额外参数
3. **训练**：其他代码无需修改，直接运行

或者直接使用训练脚本：

```bash
conda activate open_spiel
python train_deep_cfr_texas.py --num_players 6 --num_iterations 100
```

就这么简单！🎉
