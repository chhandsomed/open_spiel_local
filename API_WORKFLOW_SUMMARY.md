# API工作流程总结

## 你的理解完全正确！✅

API的工作流程就是：

1. **根据传过来的筹码和游戏配置，新建一个游戏实例**
2. **根据手牌和公共牌的多少，发牌**
3. **按照传过来的历史动作，跑到当前的状态**
4. **然后给出动作推荐**

## 详细流程

### 步骤1：创建游戏实例

**根据配置创建游戏**：
```python
# api_server.py:1173-1178
if blinds is not None and stacks is not None:
    game = create_game_with_config(
        num_players, 
        blinds,      # 盲注列表
        stacks,      # 筹码列表
        betting_abstraction, 
        dealer_pos   # Dealer位置
    )
```

**创建的游戏实例包含**：
- 玩家数量
- 盲注配置
- 筹码配置
- 行动顺序（firstPlayer，基于dealer_pos）
- 下注抽象（fchpa等）

### 步骤2：创建初始状态并发牌

**创建初始状态**：
```python
# api_server.py:303
state = game.new_initial_state()
```

**发手牌**：
```python
# api_server.py:337-355
# 根据hole_cards发当前玩家的手牌
# 其他玩家的手牌随机分配
while state.is_chance_node() and hole_card_idx < len(all_hole_cards):
    state.apply_action(target_card)  # 发指定的手牌
```

**发公共牌**：
```python
# api_server.py:357-374
# 根据board_cards发公共牌
while state.is_chance_node() and board_card_idx < len(board_indices):
    state.apply_action(target_card)  # 发指定的公共牌
```

**关键点**：
- 手牌数量：根据`hole_cards`的长度（通常是2张）
- 公共牌数量：根据`board_cards`的长度（0/3/4/5张）
- OpenSpiel根据已发牌的数量自动判断是发手牌还是公共牌

### 步骤3：应用历史动作

**按照历史动作顺序应用**：
```python
# api_server.py:385-408
for action in action_history:
    # 如果遇到chance节点（需要发Turn/River），先发完
    while state.is_chance_node():
        state.apply_action(random.choice(legal_actions))
    
    # 应用玩家动作
    state.apply_action(action)
    # OpenSpiel内部自动更新：
    # - 下注状态
    # - 当前玩家
    # - 筹码
    # - 游戏阶段等
```

**历史动作的作用**：
- 重建游戏状态到当前时刻
- 包括所有玩家的下注动作
- 包括弃牌、跟注、加注等动作
- 不包括发牌动作（chance节点）

### 步骤4：获取动作推荐

**调用模型推理**：
```python
# api_server.py:1232-1236
recommended_action, action_probs, legal_actions = get_recommended_action(
    state,           # 重建后的游戏状态
    MODEL,           # 训练好的模型
    DEVICE,          # 设备（cpu/cuda）
    dealer_pos=dealer_pos  # Dealer位置（用于位置编码映射）
)
```

**模型推理过程**：
1. 提取信息状态：`state.information_state_tensor(player_id)`
2. 位置编码映射（如果需要）：将位置编码映射到训练时的配置
3. 模型推理：输入信息状态，输出动作概率
4. 选择推荐动作：概率最大的动作

## 完整流程图

```
请求到达
    ↓
步骤1：创建游戏实例
    ├─ 根据blinds、stacks、dealer_pos创建游戏
    └─ 配置firstPlayer（行动顺序）
    ↓
步骤2：创建初始状态并发牌
    ├─ state = game.new_initial_state()
    ├─ 发手牌（根据hole_cards）
    │   └─ hole_cards_dealt_ = 12 (6人*2张)
    └─ 发公共牌（根据board_cards）
        └─ board_cards_dealt_ = 0/3/4/5
    ↓
步骤3：应用历史动作
    ├─ for action in action_history:
    │   ├─ 处理chance节点（发Turn/River）
    │   └─ state.apply_action(action)
    └─ 状态更新到当前时刻
    ↓
步骤4：验证状态
    ├─ 检查是否terminal
    ├─ 检查是否chance节点
    └─ 检查当前玩家是否匹配
    ↓
步骤5：获取动作推荐
    ├─ 提取信息状态
    ├─ 位置编码映射（如果需要）
    ├─ 模型推理
    └─ 返回推荐动作和概率
    ↓
返回响应
```

## 实际示例

### 请求示例

```json
{
  "player_id": 2,
  "dealer_pos": 5,
  "hole_cards": ["As", "Kh"],
  "board_cards": ["2d", "3c", "4h"],
  "action_history": [1, 1, 1, 1],
  "blinds": [50, 100, 0, 0, 0, 0],
  "stacks": [2000, 2000, 2000, 2000, 2000, 2000]
}
```

### 处理流程

**步骤1：创建游戏实例**
```python
game = create_game_with_config(
    num_players=6,
    blinds=[50, 100, 0, 0, 0, 0],
    stacks=[2000, 2000, 2000, 2000, 2000, 2000],
    betting_abstraction='fchpa',
    dealer_pos=5
)
# 配置：firstPlayer = 2 (UTG)
```

**步骤2：创建初始状态并发牌**
```python
state = game.new_initial_state()
# 发12张手牌（6人*2张）
# hole_cards_dealt_ = 12
# 发3张公共牌
# board_cards_dealt_ = 3
```

**步骤3：应用历史动作**
```python
for action in [1, 1, 1, 1]:
    state.apply_action(action)  # Call动作
# 状态更新：
# - 4个玩家都Call了
# - 当前玩家 = 某个玩家
# - 下注状态已更新
```

**步骤4：获取动作推荐**
```python
info_state = state.information_state_tensor(2)
recommended_action = model.predict(info_state)
# 返回：推荐动作和概率
```

## 关键点总结

### ✅ 你的理解完全正确

1. **新建游戏实例**：根据配置（blinds、stacks、dealer_pos）
2. **发牌**：根据手牌和公共牌的数量
3. **应用历史动作**：按照顺序应用，重建到当前状态
4. **动作推荐**：基于重建的状态进行AI推理

### 补充说明

1. **游戏实例**：每次请求都创建新的（如果传了blinds/stacks）
2. **发牌顺序**：OpenSpiel根据状态变量自动判断
3. **历史动作**：只包含玩家动作，不包含发牌动作
4. **状态重建**：完全通过动作序列重建，确保状态正确

## 代码位置

### 主要函数

1. **`create_game_with_config()`** (api_server.py:213)
   - 创建游戏实例

2. **`build_state_from_cards()`** (api_server.py:277)
   - 创建初始状态
   - 发手牌和公共牌
   - 应用历史动作

3. **`get_recommended_action()`** (api_server.py:811)
   - 获取动作推荐

4. **`recommend_action()`** (api_server.py:1072)
   - API端点处理函数

## 总结

你的理解完全正确！API的工作流程就是：

1. ✅ **新建游戏实例**（根据配置）
2. ✅ **发牌**（根据手牌和公共牌数量）
3. ✅ **应用历史动作**（重建到当前状态）
4. ✅ **动作推荐**（基于重建的状态）

这就是整个API的核心工作流程！

