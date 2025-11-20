# 德州扑克训练和推理指南

## 📋 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [DeepCFR 训练](#deepcfr-训练)
- [DeepCFR 推理](#deepcfr-推理)
- [MCCFR 训练](#mccfr-训练)
- [MCCFR 测试](#mccfr-测试)
- [注意事项](#注意事项)
- [常见问题](#常见问题)

## 概述

本项目提供了使用 OpenSpiel 进行德州扑克策略训练的完整解决方案，支持两种算法：

1. **DeepCFR** - 基于深度神经网络的 CFR 算法
   - 适合大规模训练
   - 支持 GPU 加速
   - 模型可保存和加载

2. **MCCFR** - 蒙特卡洛 CFR 算法
   - Tabular 策略
   - 适合小规模快速训练
   - 策略可保存和加载

## 环境要求

### 必需环境
- Python 3.8+
- PyTorch (支持 CUDA)
- OpenSpiel
- NumPy

### Conda 环境
```bash
conda activate open_spiel
```

### 验证环境
```bash
python -c "import torch; import pyspiel; print('环境正常')"
```

## 快速开始

### DeepCFR 快速训练（2人场）
```bash
# 10 次迭代，跳过 NashConv，每 5 次迭代评估
python train_deep_cfr_texas.py \
    --num_players 2 \
    --num_iterations 10 \
    --skip_nashconv \
    --eval_interval 5
```

### DeepCFR 快速训练（6人场）
```bash
# 6人场快速测试
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 10 \
    --num_traversals 10 \
    --skip_nashconv \
    --eval_interval 5 \
    --save_prefix deepcfr_texas_6player
```

### DeepCFR 快速推理
```bash
# 测试 10 局游戏
python inference_simple.py --num_games 10
```

### MCCFR 快速训练
```bash
# 2人场，100 次迭代
python train_texas_holdem_mccfr.py --num_players 2 --iterations 100

# 6人场，100 次迭代
python train_texas_holdem_mccfr.py --num_players 6 --iterations 100
```

## DeepCFR 训练

### 基本使用

```bash
python train_deep_cfr_texas.py \
    --num_iterations 100 \
    --num_traversals 20 \
    --skip_nashconv \
    --eval_interval 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_players` | 2 | 玩家数量 |
| `--num_iterations` | 10 | 迭代次数 |
| `--num_traversals` | 20 | 每次迭代的遍历次数 |
| `--policy_layers` | 64 64 | 策略网络层大小 |
| `--advantage_layers` | 32 32 | 优势网络层大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--memory_capacity` | 1000000 | 内存容量 |
| `--save_prefix` | deepcfr_texas | 保存文件前缀 |
| `--save_dir` | models | 模型保存目录（默认保存到 models/ 子文件夹） |
| `--use_gpu` | True | 使用 GPU |
| `--skip_nashconv` | False | 跳过 NashConv 计算（推荐） |
| `--eval_interval` | 10 | 每 N 次迭代进行一次评估 |
| `--eval_with_games` | False | 评估时包含测试对局 |
| `--save_history` | True | 保存训练历史到JSON文件 |

### 推荐配置

#### 2人场 - 小规模训练（测试）
```bash
python train_deep_cfr_texas.py \
    --num_players 2 \
    --num_iterations 10 \
    --num_traversals 10 \
    --skip_nashconv \
    --eval_interval 5
```

#### 2人场 - 中等规模训练
```bash
python train_deep_cfr_texas.py \
    --num_players 2 \
    --num_iterations 100 \
    --num_traversals 20 \
    --skip_nashconv \
    --eval_interval 10 \
    --eval_with_games
```

#### 2人场 - 大规模训练
```bash
python train_deep_cfr_texas.py \
    --num_players 2 \
    --num_iterations 500 \
    --num_traversals 50 \
    --policy_layers 256 256 128 \
    --advantage_layers 128 128 64 \
    --memory_capacity 10000000 \
    --skip_nashconv \
    --eval_interval 20
```

#### 6人场 - 快速测试
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 10 \
    --num_traversals 10 \
    --skip_nashconv \
    --eval_interval 5 \
    --save_prefix deepcfr_texas_6player
```

#### 6人场 - 中等规模训练（推荐）
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 100 \
    --num_traversals 20 \
    --policy_layers 128 128 \
    --advantage_layers 64 64 \
    --memory_capacity 2000000 \
    --skip_nashconv \
    --eval_interval 10 \
    --eval_with_games \
    --save_prefix deepcfr_texas_6player
```

#### 6人场 - 大规模训练
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 500 \
    --num_traversals 50 \
    --policy_layers 256 256 128 \
    --advantage_layers 128 128 64 \
    --memory_capacity 10000000 \
    --skip_nashconv \
    --eval_interval 20 \
    --eval_with_games \
    --save_prefix deepcfr_texas_6player_large
```

### 训练输出

训练过程中会显示：
- 每次迭代的耗时
- 损失值（每 10 次迭代）
- GPU 内存使用（如果使用 GPU）
- 评估指标（如果启用）

示例输出：
```
迭代 10/100... 完成 (耗时: 2.50秒) | 玩家0损失: 1234.567 | GPU内存: 2.34GB

  评估训练效果（迭代 10）... 完成
  策略熵: 1.2345 | 策略缓冲区: 1,000,000 | 优势样本: 2,000,000
```

### 模型保存

训练完成后会自动保存到 `models/{save_prefix}/` 目录（如果目录已存在则添加时间戳）：
- `{save_prefix}_policy_network.pt` - 策略网络
- `{save_prefix}_advantage_player_0.pt` - 玩家 0 优势网络
- `{save_prefix}_advantage_player_1.pt` - 玩家 1 优势网络
- ... (每个玩家一个优势网络，6人场会有6个优势网络)
- `{save_prefix}_training_history.json` - 训练历史记录
- `config.json` - 训练配置信息

**示例**（6人场）：
```
models/
└── deepcfr_texas_6player/
    ├── deepcfr_texas_6player_policy_network.pt
    ├── deepcfr_texas_6player_advantage_player_0.pt
    ├── deepcfr_texas_6player_advantage_player_1.pt
    ├── deepcfr_texas_6player_advantage_player_2.pt
    ├── deepcfr_texas_6player_advantage_player_3.pt
    ├── deepcfr_texas_6player_advantage_player_4.pt
    ├── deepcfr_texas_6player_advantage_player_5.pt
    ├── deepcfr_texas_6player_training_history.json
    └── config.json
```

## DeepCFR 推理

### 基本使用

```bash
python inference_simple.py --num_games 10
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_prefix` | deepcfr_texas | 模型文件前缀 |
| `--num_games` | 10 | 测试游戏数量 |
| `--policy_layers` | 64 64 | 策略网络层大小（必须与训练时一致） |
| `--use_gpu` | True | 使用 GPU |

### 使用示例

```bash
# 基本测试
python inference_simple.py --num_games 10

# 大规模测试
python inference_simple.py --num_games 100

# 自定义模型
python inference_simple.py \
    --model_prefix my_model \
    --num_games 20 \
    --policy_layers 256 256
```

### 输出示例

```
======================================================================
DeepCFR 简化推理
======================================================================

使用设备: cuda

[0/2] 创建游戏...
  ✓ 游戏创建成功: universal_poker
  信息状态大小: 169
  动作数量: 4

[1/2] 加载模型...
  加载策略网络: deepcfr_texas_policy_network.pt
  ✓ 策略网络加载成功

[2/2] 测试模型 (10 局游戏)...
  进行第 5/10 局...
  进行第 10/10 局...

  测试结果:
    玩家 0:
      平均收益: 80.0000
      胜率: 60.0%
      总收益: 800.00
    玩家 1:
      平均收益: -80.0000
      胜率: 20.0%
      总收益: -800.00

======================================================================
✓ 推理完成
======================================================================
```

## MCCFR 训练

### 基本使用

```bash
python train_texas_holdem_mccfr.py \
    --num_players 2 \
    --iterations 1000
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_players` | 2 | 玩家数量 |
| `--iterations` | 1000 | 迭代次数 |
| `--save_path` | mccfr_strategy.pkl | 保存路径 |

### 使用示例

```bash
# 快速测试
python train_texas_holdem_mccfr.py --num_players 2 --iterations 100

# 正式训练
python train_texas_holdem_mccfr.py --num_players 2 --iterations 10000
```

## MCCFR 测试

### 加载和测试策略

```bash
python load_and_test_strategy.py
```

### 功能

- 加载保存的策略文件
- 测试策略性能（计算 NashConv）
- 交互式测试

## 注意事项

### ⚠️ 重要提示

1. **游戏配置必须一致**
   - 训练和推理时使用的游戏配置必须完全一致
   - 否则信息状态张量大小不匹配，无法加载模型
   - 当前配置：`blind=100 100, stack=2000 2000, firstPlayer=2 1 1 1`

2. **网络结构必须一致**
   - 推理时 `--policy_layers` 和 `--advantage_layers` 必须与训练时一致
   - 默认值：`policy_layers=64 64, advantage_layers=32 32`

3. **跳过 NashConv 计算（推荐）**
   - 对于大规模训练，强烈建议使用 `--skip_nashconv`
   - NashConv 计算会消耗大量 CPU 和内存
   - 训练过程不受影响，只是无法获得 NashConv 评估值
   - 可以使用训练过程中的评估指标（损失值、策略熵等）

4. **GPU 使用**
   - 默认启用 GPU（如果可用）
   - 如果 GPU 不可用，会自动使用 CPU
   - 可以通过 `--use_gpu` 控制

5. **模型文件**
   - 确保模型文件存在：`{prefix}_policy_network.pt`
   - 推理时需要与训练时相同的网络结构

### 训练建议

1. **从小规模开始**
   - 先用少量迭代测试（10-20 次）
   - 确认训练正常后再进行大规模训练

2. **监控训练过程**
   - 观察损失值趋势
   - 使用 `--eval_interval` 定期评估
   - 使用 `--eval_with_games` 进行详细评估

3. **资源管理**
   - 大规模训练时使用 `--skip_nashconv`
   - 根据 GPU 内存调整 `--memory_capacity`
   - 根据系统资源调整 `--num_traversals`

### 推理建议

1. **测试游戏数量**
   - 快速测试：5-10 局
   - 详细测试：50-100 局
   - 大规模测试：1000+ 局

2. **结果解读**
   - 胜率 > 50%：模型优于随机策略
   - 平均收益 > 0：模型策略有效
   - 胜率越高，模型质量越好

## 常见问题

### Q1: 训练时 CPU 和内存被占满怎么办？

**A**: 使用 `--skip_nashconv` 跳过 NashConv 计算。NashConv 计算会消耗大量资源，但训练过程不受影响。

### Q2: 模型加载失败，提示大小不匹配？

**A**: 检查两点：
1. 游戏配置是否与训练时一致
2. 网络结构参数（`--policy_layers`, `--advantage_layers`）是否与训练时一致

### Q3: 训练时没有输出？

**A**: 训练脚本会自动显示进度。如果看不到输出：
- 检查是否使用了 `--skip_nashconv`
- 检查 `--eval_interval` 设置（默认每 10 次迭代显示）

### Q4: GPU 没有被使用？

**A**: 检查：
1. CUDA 是否可用：`python -c "import torch; print(torch.cuda.is_available())"`
2. 是否使用了 `--use_gpu` 参数
3. GPU 内存是否足够

### Q5: 训练损失值增加是正常的吗？

**A**: 是的，在 DeepCFR 的早期迭代中，损失值增加是正常现象：
- 游戏树探索加深
- 优势值范围扩大
- 网络还在学习

关注损失值的趋势，而不是绝对值。

### Q6: 如何判断训练效果？

**A**: 可以通过以下指标：
1. **损失值趋势**：应该稳定或下降
2. **策略熵**：应该在合理范围内
3. **缓冲区大小**：应该持续增长
4. **测试对局**：胜率应该 > 50%

### Q7: 训练需要多长时间？

**A**: 取决于：
- 迭代次数和遍历次数
- 网络大小
- 是否使用 GPU

示例：
- 10 次迭代，20 次遍历：约 1-5 分钟
- 100 次迭代，20 次遍历：约 10-30 分钟
- 500 次迭代，50 次遍历：约 1-3 小时

## 文件结构

```
.
├── train_deep_cfr_texas.py      # DeepCFR 训练脚本
├── inference_simple.py           # DeepCFR 推理脚本（推荐）
├── evaluate_model.py            # 完整模型评估脚本
├── compare_models.py             # 模型对比评估脚本
├── training_evaluator.py         # 训练评估模块
├── nash_conv_gpu.py              # NashConv GPU 加速
├── train_texas_holdem_mccfr.py  # MCCFR 训练脚本
├── load_and_test_strategy.py    # MCCFR 策略加载和测试
├── models/                       # 模型保存目录（自动创建）
│   ├── deepcfr_texas_6player/   # 6人场模型目录
│   │   ├── *.pt                 # 模型文件
│   │   ├── *_training_history.json
│   │   └── config.json
│   └── ...
└── *.pkl                         # MCCFR 策略文件
```

## 相关文档

详细文档请参考：
- `TRAIN_6PLAYER_EXAMPLE.md` - **6人场训练完整指南**（推荐查看）
- `TRAINING_EVALUATION_GUIDE.md` - 训练评估详细指南（评估指标、使用方法、指标解读）
- `NUM_TRAVERSALS_EXPLANATION.md` - num_traversals 参数详解
- `SKIP_NASHCONV_EXPLANATION.md` - NashConv 详细说明（为什么可以跳过、影响分析）
- `INFERENCE_GUIDE.md` - 推理详细指南（推理脚本使用、测试结果解读）
- `NASHCONV_RESOURCE_FIX.md` - NashConv 资源问题修复（资源限制、故障排除）

## 技术支持

如果遇到问题：
1. 检查本文档的"常见问题"部分
2. 查看相关详细文档
3. 检查环境配置是否正确
4. 确认文件路径和参数设置

## 更新日志

### 最新版本
- ✅ DeepCFR GPU 支持
- ✅ **6人场训练支持**（自动配置行动顺序）
- ✅ **模型保存到子文件夹**（避免覆盖）
- ✅ **训练历史记录**（自动保存JSON）
- ✅ 训练过程评估功能
- ✅ 简化推理脚本
- ✅ 完整模型评估和对比脚本
- ✅ NashConv 资源限制
- ✅ 完整的文档和示例

