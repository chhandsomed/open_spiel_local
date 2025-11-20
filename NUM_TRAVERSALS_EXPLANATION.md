# num_traversals 参数详解

## 什么是 num_traversals？

`num_traversals` 是 DeepCFR 算法中的一个重要超参数，表示**每次迭代中，对每个玩家进行游戏树遍历的次数**。

## 在训练过程中的作用

### 训练循环结构

```
对于每次迭代 (iteration):
    对于每个玩家 (player):
        进行 num_traversals 次游戏树遍历
        更新该玩家的优势网络
    更新策略网络
```

### 具体流程

在代码中，训练循环如下：

```python
for iteration in range(num_iterations):
    for player in range(game.num_players()):
        # 对当前玩家进行 num_traversals 次遍历
        for _ in range(num_traversals):
            deep_cfr_solver._traverse_game_tree(deep_cfr_solver._root_node, player)
        
        # 更新该玩家的优势网络
        deep_cfr_solver._learn_advantage_network(player)
    
    # 更新策略网络
    deep_cfr_solver._learn_strategy_network()
```

## 游戏树遍历 (Traversal) 是什么？

游戏树遍历是指：
1. 从游戏初始状态开始
2. 根据当前策略选择动作
3. 沿着游戏树向下探索
4. 收集信息状态和优势值
5. 将这些信息存储到缓冲区中

每次遍历都会：
- 探索游戏树的不同路径
- 收集更多的训练样本
- 更新后悔值（regret）的估计

## num_traversals 的影响

### 1. 训练样本数量

- **更大的 num_traversals**：
  - ✅ 每次迭代收集更多训练样本
  - ✅ 更充分地探索游戏树
  - ✅ 优势网络和策略网络有更多数据学习
  - ❌ 训练时间更长

- **更小的 num_traversals**：
  - ✅ 训练速度更快
  - ❌ 每次迭代样本较少
  - ❌ 可能探索不充分

### 2. 训练质量

- **num_traversals 太小**：
  - 可能导致训练不稳定
  - 策略收敛较慢
  - 最终策略质量可能较差

- **num_traversals 太大**：
  - 训练时间显著增加
  - 收益递减（超过一定值后，增加效果不明显）

### 3. 内存使用

- 每次遍历都会向缓冲区添加样本
- `num_traversals` 越大，缓冲区增长越快
- 需要确保 `memory_capacity` 足够大

## 推荐设置

### 2人场德州扑克

| 场景 | num_traversals | 说明 |
|------|----------------|------|
| 快速测试 | 10-20 | 验证代码和配置 |
| 中等训练 | 20-30 | 平衡速度和质量 |
| 高质量训练 | 50-100 | 追求更好的策略 |

### 6人场德州扑克

| 场景 | num_traversals | 说明 |
|------|----------------|------|
| 快速测试 | 10-15 | 验证配置 |
| 中等训练 | 20-30 | 推荐设置 |
| 高质量训练 | 50+ | 需要更多时间 |

## 实际训练示例

### 示例 1：快速测试
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 10 \
    --num_traversals 10 \  # 较少遍历，快速测试
    --skip_nashconv
```
- 总遍历次数：10 迭代 × 6 玩家 × 10 遍历 = 600 次遍历
- 训练时间：约 5-10 分钟

### 示例 2：中等训练
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 100 \
    --num_traversals 20 \  # 平衡设置
    --skip_nashconv
```
- 总遍历次数：100 迭代 × 6 玩家 × 20 遍历 = 12,000 次遍历
- 训练时间：约 1-3 小时

### 示例 3：高质量训练
```bash
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 500 \
    --num_traversals 50 \  # 更多遍历，更好质量
    --skip_nashconv
```
- 总遍历次数：500 迭代 × 6 玩家 × 50 遍历 = 150,000 次遍历
- 训练时间：约 5-15 小时

## 如何选择合适的 num_traversals？

### 1. 根据游戏复杂度
- **简单游戏**（如 Kuhn Poker）：10-20 足够
- **中等游戏**（如 2人德州）：20-50
- **复杂游戏**（如 6人德州）：30-100

### 2. 根据训练时间预算
- **时间有限**：10-20
- **中等时间**：20-30
- **充足时间**：50+

### 3. 根据观察训练效果
- 如果损失值波动很大 → 可能需要增加 `num_traversals`
- 如果训练很慢但效果不好 → 可能需要增加 `num_iterations` 而不是 `num_traversals`
- 如果缓冲区增长很慢 → 可能需要增加 `num_traversals`

## num_traversals vs num_iterations

| 参数 | 作用 | 影响 |
|------|------|------|
| `num_iterations` | 训练的总迭代次数 | 决定训练的总轮数，影响最终收敛 |
| `num_traversals` | 每次迭代的遍历次数 | 决定每次迭代的探索深度，影响样本质量 |

**建议**：
- 如果训练时间有限，优先增加 `num_iterations` 而不是 `num_traversals`
- 如果发现训练不稳定，可以增加 `num_traversals`
- 两者需要平衡，不能只关注一个

## 总结

`num_traversals` 控制每次迭代中每个玩家的游戏树遍历次数：
- **作用**：决定每次迭代收集多少训练样本
- **影响**：训练速度、训练质量、内存使用
- **建议**：根据游戏复杂度、时间预算和训练效果调整
- **默认值**：通常 20-30 是一个好的起点

对于 6 人场德州扑克，建议从 `num_traversals=20` 开始，根据训练效果和时间预算调整。


