# 6人场无限注德州扑克 DeepCFR 训练指南

## 快速开始

### 基本训练命令

```bash
# 6人场，10次迭代（快速测试）
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 10 \
    --num_traversals 10 \
    --skip_nashconv \
    --eval_interval 5 \
    --save_prefix deepcfr_texas_6player
```

**注意**：模型会自动保存到 `models/deepcfr_texas_6player/` 目录，避免覆盖之前的模型。

### 推荐配置

#### 小规模训练（测试）
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 20 \
    --num_traversals 10 \
    --skip_nashconv \
    --eval_interval 5 \
    --save_prefix deepcfr_texas_6player_test
```

#### 中等规模训练
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

#### 大规模训练
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

## 6人场配置说明

### 游戏配置
- **玩家数量**: 6
- **盲注结构**: 所有玩家都是 100（小盲大盲相同，实际游戏中会区分）
- **初始筹码**: 每个玩家 2000
- **位置顺序**:
  - Player 0: 小盲 (SB)
  - Player 1: 大盲 (BB)
  - Player 2: 枪口位 (UTG)
  - Player 3: 中位 (MP)
  - Player 4: 劫持位 (HJ)
  - Player 5: 按钮位 (BTN)

### 行动顺序
- **Preflop (Round 0)**: 第一个行动的是 UTG (Player 2)
- **Flop/Turn/River (Rounds 1-3)**: 第一个行动的是小盲 (Player 0)

## 训练注意事项

### 1. 资源需求
6人场的游戏状态空间比2人场大得多，需要：
- **更多内存**: 建议至少 16GB RAM
- **更大缓冲区**: `--memory_capacity` 建议设置为 2,000,000 或更大
- **更多训练时间**: 每次迭代耗时会更长

### 2. 网络结构
6人场建议使用更大的网络：
- **策略网络**: `128 128` 或 `256 256 128`
- **优势网络**: `64 64` 或 `128 128 64`

### 3. 训练参数
- **迭代次数**: 建议至少 100 次迭代
- **遍历次数**: 建议 20-50 次
- **评估间隔**: 建议每 10-20 次迭代评估一次

### 4. 跳过 NashConv
强烈建议使用 `--skip_nashconv`，因为6人场的 NashConv 计算会非常耗时。

## 训练输出

训练过程中会显示：
- 每次迭代的耗时
- 每个玩家的优势损失
- GPU 内存使用（如果使用 GPU）
- 评估指标（策略熵、缓冲区大小等）

示例输出：
```
迭代 10/100... 完成 (耗时: 15.50秒) | 玩家0损失: 1234.567 | 玩家1损失: 2345.678 | ... | GPU内存: 4.34GB

  评估训练效果（迭代 10）... 完成
  策略熵: 1.2345 | 策略缓冲区: 500,000 | 优势样本: 3,000,000
```

## 模型保存

训练完成后会自动保存到 `models/{save_prefix}/` 目录（如果目录已存在则添加时间戳避免覆盖）：

**保存的文件**：
- `{save_prefix}_policy_network.pt` - 策略网络
- `{save_prefix}_advantage_player_0.pt` - 玩家 0 优势网络
- `{save_prefix}_advantage_player_1.pt` - 玩家 1 优势网络
- `{save_prefix}_advantage_player_2.pt` - 玩家 2 优势网络
- `{save_prefix}_advantage_player_3.pt` - 玩家 3 优势网络
- `{save_prefix}_advantage_player_4.pt` - 玩家 4 优势网络
- `{save_prefix}_advantage_player_5.pt` - 玩家 5 优势网络
- `{save_prefix}_training_history.json` - 训练历史记录
- `config.json` - 训练配置信息（包含所有训练参数）

**目录结构示例**：
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

**自定义保存目录**：
```bash
# 使用自定义保存目录
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 10 \
    --save_prefix deepcfr_texas_6player \
    --save_dir my_models  # 保存到 my_models/ 目录
```

## 推理和评估

### 快速推理测试
```bash
# 注意: 推理脚本需要修改 num_players 参数
# 或者使用 evaluate_model.py
python evaluate_model.py \
    --model_prefix deepcfr_texas_6player \
    --num_games 100 \
    --policy_layers 128 128
```

### 完整评估
```bash
python evaluate_model.py \
    --model_prefix deepcfr_texas_6player \
    --opponents random call fold \
    --num_games 200 \
    --policy_layers 128 128 \
    --output evaluation_6player.json
```

## 常见问题

### Q1: 训练速度很慢？
**A**: 6人场的状态空间很大，训练速度会比2人场慢很多。建议：
- 使用 GPU 加速
- 减少 `num_traversals`（但会影响训练质量）
- 使用 `--skip_nashconv` 跳过 NashConv 计算

### Q2: 内存不足？
**A**: 6人场需要更多内存，建议：
- 减少 `--memory_capacity`
- 使用更小的网络结构
- 减少 `--num_traversals`

### Q3: 如何判断训练效果？
**A**: 观察以下指标：
- 损失值趋势（应该稳定或下降）
- 策略熵（应该在合理范围内）
- 缓冲区大小（应该持续增长）
- 测试对局胜率（应该 > 1/6，即优于随机）

### Q4: 训练需要多长时间？
**A**: 取决于配置：
- 10 次迭代，20 次遍历：约 5-15 分钟
- 100 次迭代，20 次遍历：约 1-3 小时
- 500 次迭代，50 次遍历：约 5-15 小时

## 与2人场的区别

| 项目 | 2人场 | 6人场 |
|------|-------|-------|
| 状态空间 | 较小 | 大得多 |
| 训练时间 | 较快 | 慢很多 |
| 内存需求 | 较低 | 较高 |
| 网络大小 | 64 64 | 建议 128 128 或更大 |
| 缓冲区大小 | 1,000,000 | 建议 2,000,000+ |
| 迭代次数 | 50-100 | 建议 100-500 |

## 下一步

训练完成后，可以：
1. 使用 `evaluate_model.py` 评估模型性能
2. 使用 `compare_models.py` 对比不同训练阶段的模型
3. 分析训练历史文件，查看训练曲线
4. 进行实际对局测试

