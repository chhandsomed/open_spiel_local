# DeepCFR 算法原理与德州扑克训练详解

## 📚 目录

1. [DeepCFR 算法概述](#deepcfr-算法概述)
2. [核心原理](#核心原理)
3. [网络架构](#网络架构)
4. [训练流程](#训练流程)
5. [代码实现详解](#代码实现详解)
6. [在德州扑克中的应用](#在德州扑克中的应用)

---

## DeepCFR 算法概述

**DeepCFR (Deep Counterfactual Regret Minimization)** 是 CFR (Counterfactual Regret Minimization) 算法的深度学习版本，用于求解大规模不完全信息博弈的纳什均衡。

### 为什么需要 DeepCFR？

传统 CFR 算法需要：
- ❌ 存储所有信息集（information sets）的策略
- ❌ 对于德州扑克这样的游戏，信息集数量巨大（10^18+）
- ❌ 内存和计算资源需求巨大

DeepCFR 通过神经网络：
- ✅ 用神经网络近似策略和优势值
- ✅ 只需要存储训练样本，不需要存储所有信息集
- ✅ 可以处理大规模游戏

### 核心思想

DeepCFR 将 CFR 的两个核心组件用神经网络替代：
1. **优势网络 (Advantage Networks)**：近似每个玩家的后悔值（regret）
2. **策略网络 (Policy Network)**：近似平均策略（average strategy）

---

## 核心原理

### 1. CFR 基础回顾

CFR 算法的核心是**后悔值匹配 (Regret Matching)**：

```
后悔值 R^T(a) = Σ_t (u(a) - u(σ^t))
策略 σ^{T+1}(a) = R^T_+(a) / Σ_b R^T_+(b)
```

其中：
- `R^T_+(a) = max(0, R^T(a))` 是正后悔值
- `u(a)` 是选择动作 a 的期望收益
- `u(σ^t)` 是当前策略的期望收益

### 2. DeepCFR 的改进

DeepCFR 使用神经网络来近似：

1. **优势网络**：`A^θ(s, a) ≈ R^T(s, a)` （后悔值）
2. **策略网络**：`π^φ(s, a) ≈ σ^T(s, a)` （平均策略）

### 3. 训练过程

```
对于每次迭代 t:
    对于每个玩家 p:
        进行多次游戏树遍历，收集样本
        用优势网络计算后悔值
        更新优势网络
    更新策略网络（使用平均策略）
```

---

## 网络架构

### 1. 优势网络 (Advantage Networks)

**作用**：为每个玩家学习后悔值，用于策略选择

```python
# 每个玩家一个独立的优势网络
self._advantage_networks = [
    MLP(embedding_size, [128, 128], num_actions) 
    for _ in range(num_players)
]
```

**输入**：信息状态向量 (information state tensor)
- 德州扑克中：包含手牌、公共牌、下注历史等信息
- 大小：266 维（6人场）

**输出**：每个动作的优势值（后悔值）
- 4个动作：Fold, Call/Check, Bet/Raise, All-in

### 2. 策略网络 (Policy Network)

**作用**：学习平均策略，用于最终决策

```python
# 所有玩家共享一个策略网络
self._policy_network = MLP(
    embedding_size, 
    [256, 256, 128], 
    num_actions
)
```

**输入**：信息状态向量
**输出**：每个动作的概率分布

### 3. 网络结构 (MLP)

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        # 输入层 -> 隐藏层1 -> 隐藏层2 -> ... -> 输出层
        self._layers = [
            SonnetLinear(input_size, hidden_sizes[0]),
            SonnetLinear(hidden_sizes[0], hidden_sizes[1]),
            ...
            SonnetLinear(hidden_sizes[-1], output_size)
        ]
```

**激活函数**：ReLU（除了输出层）

---

## 训练流程

### 整体训练循环

```python
for iteration in range(num_iterations):  # 例如：500次迭代
    for player in range(num_players):     # 对每个玩家
        # 1. 游戏树遍历（收集样本）
        for _ in range(num_traversals):   # 例如：50次遍历
            traverse_game_tree(root, player)
        
        # 2. 更新优势网络
        learn_advantage_network(player)
    
    # 3. 更新策略网络
    learn_strategy_network()
```

### 详细步骤

#### 步骤 1: 游戏树遍历 (`_traverse_game_tree`)

这是 DeepCFR 的核心，用于收集训练样本：

```python
def _traverse_game_tree(self, state, player):
    if state.is_terminal():
        return state.returns()[player]  # 游戏结束，返回收益
    
    elif state.is_chance_node():
        # 机会节点（发牌）：随机选择
        action = sample_from_chance_outcomes()
        return traverse_game_tree(state.child(action), player)
    
    elif state.current_player() == player:
        # 当前玩家：计算后悔值
        _, strategy = sample_action_from_advantage(state, player)
        
        # 递归计算每个动作的期望收益
        for action in legal_actions:
            expected_payoff[action] = traverse_game_tree(
                state.child(action), player
            )
        
        # 计算当前策略的期望收益（CFV）
        cfv = Σ strategy[a] * expected_payoff[a]
        
        # 计算后悔值（优势值）
        for action in legal_actions:
            regret[action] = expected_payoff[action] - cfv
        
        # 存储到优势缓冲区
        advantage_memories[player].add(
            info_state, iteration, regret, action
        )
        
        return cfv
    
    else:
        # 其他玩家：使用策略采样动作
        _, strategy = sample_action_from_advantage(state, other_player)
        action = sample_from_strategy(strategy)
        
        # 存储到策略缓冲区
        strategy_memories.add(info_state, iteration, strategy)
        
        return traverse_game_tree(state.child(action), player)
```

**关键点**：
- 对当前玩家：计算所有动作的期望收益，得到后悔值
- 对其他玩家：使用策略采样动作，记录策略分布

#### 步骤 2: 后悔值匹配 (`_sample_action_from_advantage`)

将优势值转换为策略：

```python
def _sample_action_from_advantage(self, state, player):
    # 1. 用优势网络获取优势值
    info_state = state.information_state_tensor(player)
    raw_advantages = advantage_networks[player](info_state)
    
    # 2. 只保留正优势值（正后悔值）
    advantages = [max(0, a) for a in raw_advantages]
    
    # 3. 归一化得到策略（后悔值匹配）
    cumulative_regret = sum(advantages[action] for action in legal_actions)
    if cumulative_regret > 0:
        strategy[action] = advantages[action] / cumulative_regret
    else:
        # 如果所有后悔值都是负的，均匀分布
        strategy[action] = 1.0 / len(legal_actions)
    
    return advantages, strategy
```

#### 步骤 3: 更新优势网络 (`_learn_advantage_network`)

```python
def _learn_advantage_network(self, player):
    # 1. 从缓冲区采样
    samples = advantage_memories[player].sample(batch_size)
    
    # 2. 准备数据
    info_states = [s.info_state for s in samples]
    advantages = [s.advantage for s in samples]
    iterations = [sqrt(s.iteration) for s in samples]  # 加权
    
    # 3. 前向传播
    outputs = advantage_networks[player](info_states)
    
    # 4. 计算损失（加权 MSE）
    loss = MSE(iterations * outputs, iterations * advantages)
    
    # 5. 反向传播
    loss.backward()
    optimizer.step()
    
    return loss
```

**关键点**：
- 使用 `sqrt(iteration)` 加权，早期迭代权重较小
- 这是 DeepCFR 论文中的技巧，帮助网络学习

#### 步骤 4: 更新策略网络 (`_learn_strategy_network`)

```python
def _learn_strategy_network(self):
    # 1. 从策略缓冲区采样
    samples = strategy_memories.sample(batch_size)
    
    # 2. 准备数据
    info_states = [s.info_state for s in samples]
    action_probs = [s.strategy_action_probs for s in samples]
    iterations = [sqrt(s.iteration) for s in samples]
    
    # 3. 前向传播
    logits = policy_network(info_states)
    outputs = softmax(logits)
    
    # 4. 计算损失（加权 MSE）
    loss = MSE(iterations * outputs, iterations * action_probs)
    
    # 5. 反向传播
    loss.backward()
    optimizer.step()
    
    return loss
```

---

## 代码实现详解

### 1. 初始化 (`__init__`)

```python
def __init__(self, game, policy_network_layers, advantage_network_layers, ...):
    # 游戏信息
    self._game = game
    self._root_node = game.new_initial_state()
    self._embedding_size = len(root_node.information_state_tensor(0))
    self._num_actions = game.num_distinct_actions()
    
    # 网络初始化
    self._policy_network = MLP(embedding_size, policy_layers, num_actions)
    self._advantage_networks = [
        MLP(embedding_size, advantage_layers, num_actions)
        for _ in range(num_players)
    ]
    
    # 缓冲区（Reservoir Sampling）
    self._strategy_memories = ReservoirBuffer(memory_capacity)
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(num_players)
    ]
    
    # 优化器
    self._optimizer_policy = Adam(policy_network.parameters(), lr=learning_rate)
    self._optimizer_advantages = [
        Adam(advantage_networks[p].parameters(), lr=learning_rate)
        for p in range(num_players)
    ]
```

### 2. 缓冲区 (Reservoir Buffer)

使用**水库采样 (Reservoir Sampling)** 来均匀采样：

```python
class ReservoirBuffer:
    def add(self, element):
        if len(self._data) < capacity:
            self._data.append(element)
        else:
            # 随机替换
            idx = random.randint(0, self._add_calls)
            if idx < capacity:
                self._data[idx] = element
        self._add_calls += 1
```

**优点**：
- 内存固定（不会无限增长）
- 保证均匀采样（每个样本被保留的概率相等）

### 3. 训练脚本 (`train_deep_cfr_texas.py`)

```python
def train_deep_cfr(...):
    # 1. 创建游戏
    game = pyspiel.load_game("universal_poker(...)")
    
    # 2. 创建求解器
    solver = DeepCFRSolver(
        game,
        policy_network_layers=(256, 256, 128),
        advantage_network_layers=(128, 128, 64),
        num_iterations=500,
        num_traversals=50,
        ...
    )
    
    # 3. 训练循环
    for iteration in range(num_iterations):
        for player in range(num_players):
            # 遍历游戏树
            for _ in range(num_traversals):
                solver._traverse_game_tree(root, player)
            
            # 更新优势网络
            loss = solver._learn_advantage_network(player)
        
        # 更新策略网络
        policy_loss = solver._learn_strategy_network()
```

---

## 在德州扑克中的应用

### 1. 游戏配置

```python
game_string = (
    f"universal_poker("
    f"betting=nolimit,"
    f"numPlayers=6,"
    f"numRounds=4,"           # Preflop, Flop, Turn, River
    f"numBoardCards=0 3 1 1," # 每轮公共牌数
    f"numHoleCards=2,"        # 每人2张手牌
    f"stack=2000 2000 ...,"   # 初始筹码
    f"blind=100 100 ..."      # 盲注
    f")"
)
```

### 2. 信息状态 (Information State)

德州扑克的信息状态包含：
- **玩家位置**：6个玩家，用6维 one-hot 编码
- **手牌**：52张牌，用52维向量（1表示有这张牌）
- **公共牌**：52维向量（1表示公共牌中有这张牌）
- **下注历史**：动作序列编码
- **投入金额**：每个玩家的投入

**总大小**：266维（6人场）

### 3. 动作空间

4个动作：
- `0`: Fold（弃牌）
- `1`: Call/Check（跟注/过牌）
- `2`: Bet/Raise（下注/加注）
- `3`: All-in（全押）

### 4. 训练参数（6人场大规模训练）

```python
num_iterations = 500      # 迭代次数
num_traversals = 50       # 每次迭代的遍历次数
policy_layers = (256, 256, 128)      # 策略网络
advantage_layers = (128, 128, 64)    # 优势网络
learning_rate = 0.001
memory_capacity = 10,000,000
```

**计算量**：
- 每次迭代：6个玩家 × 50次遍历 = 300次游戏树遍历
- 总遍历次数：500 × 300 = 150,000次
- 每次遍历可能探索数百到数千个节点

### 5. 训练过程示例

```
迭代 1/500...
  遍历游戏树（玩家0，50次）...
  遍历游戏树（玩家1，50次）...
  ...
  更新优势网络（玩家0）... 损失: 3.66M
  更新优势网络（玩家1）... 损失: 2.14M
  ...
  更新策略网络... 损失: 37.13

迭代 20/500...
  策略熵: 0.0000
  策略缓冲区: 85,722
  优势样本: 10,917
  测试对局: 玩家0平均收益=67.39, 胜率=16.0%

...

迭代 500/500...
  策略缓冲区: 2,258,501
  优势样本: 281,087
  最终损失: 1,144.36M (玩家0)
```

### 6. 为什么损失值会增长？

这是**正常的**！原因：
1. 随着训练深入，探索的游戏树更深
2. 优势值的范围扩大（后悔值可能很大）
3. 网络需要学习更大的数值范围

**关键指标**：
- ✅ 缓冲区持续增长（说明在探索）
- ✅ 所有玩家都在训练（损失都在增长）
- ⚠️ 策略熵为0（可能策略过于确定）

---

## 关键概念总结

### 1. 后悔值 (Regret)

```
后悔值 = 选择动作a的收益 - 当前策略的期望收益
```

如果后悔值为正，说明这个动作比当前策略好。

### 2. 后悔值匹配 (Regret Matching)

```
策略概率 = 正后悔值 / 所有正后悔值的和
```

只考虑正后悔值，负后悔值设为0。

### 3. 平均策略 (Average Strategy)

```
平均策略 = (1/T) * Σ_t 策略^t
```

所有迭代的策略的平均值，收敛到纳什均衡。

### 4. 信息集 (Information Set)

在德州扑克中，信息集是玩家能看到的所有信息：
- 自己的手牌
- 公共牌
- 下注历史
- 其他玩家的投入

**关键**：相同信息集的状态，玩家应该采用相同策略。

---

## 训练技巧

### 1. 优势网络重新初始化

```python
if reinitialize_advantage_networks:
    advantage_networks[player].reset()
```

**原因**：每轮重新学习，避免过拟合早期样本。

### 2. 迭代加权

```python
weight = sqrt(iteration)
loss = MSE(weight * prediction, weight * target)
```

**原因**：早期迭代的样本质量较低，权重较小。

### 3. 水库采样

使用固定大小的缓冲区，保证均匀采样，避免内存爆炸。

### 4. 外部采样 (External Sampling)

在遍历时，对其他玩家使用策略采样，而不是遍历所有动作，大大减少计算量。

---

## 评估指标

### 1. 策略熵

衡量策略的随机性：
```
熵 = -Σ p(a) * log(p(a))
```

- 熵=0：策略完全确定（总是选择同一个动作）
- 熵大：策略随机性强

### 2. 缓冲区大小

- 策略缓冲区：已探索的信息集数量
- 优势缓冲区：已收集的优势样本数量

### 3. 测试对局

与随机策略对局，统计：
- 平均收益
- 胜率

---

## 参考资料

- DeepCFR 论文：https://arxiv.org/abs/1811.00164
- OpenSpiel 文档：https://github.com/deepmind/open_spiel
- CFR 算法：https://en.wikipedia.org/wiki/Counterfactual_regret_minimization

---

## 总结

DeepCFR 通过神经网络近似 CFR 算法，使得可以处理像德州扑克这样的大规模不完全信息博弈：

1. **优势网络**：学习每个玩家的后悔值，用于策略选择
2. **策略网络**：学习平均策略，用于最终决策
3. **游戏树遍历**：收集训练样本
4. **迭代训练**：逐步改进策略，收敛到纳什均衡

在德州扑克中，DeepCFR 可以学习到接近最优的策略，即使游戏状态空间巨大（10^18+ 信息集）。


