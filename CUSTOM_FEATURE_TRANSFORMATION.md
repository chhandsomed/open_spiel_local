# 信息状态自定义特征转换层详解

## 📋 概述

虽然信息状态的格式是固定的（在 C++ 中定义），但我们可以在**网络输入层之前添加转换层**，将原始信息状态转换为自定义特征表示。

## 🎯 核心思想

### 当前架构

```
原始信息状态 (266维)
    ↓
MLP 网络 (256-256-128)
    ↓
输出 (4个动作的概率)
```

### 添加转换层后的架构

```
原始信息状态 (266维)
    ↓
特征转换层 (自定义)
    ↓
转换后的特征 (例如: 200维)
    ↓
MLP 网络 (256-256-128)
    ↓
输出 (4个动作的概率)
```

## 🔧 实现方法

### 方法1: 在 MLP 内部添加预处理层（推荐）

创建一个新的网络类，包含转换层和原始 MLP：

```python
class MLPWithFeatureTransform(nn.Module):
    """带特征转换层的 MLP"""
    
    def __init__(self, 
                 raw_input_size,      # 原始信息状态大小 (266)
                 transformed_size,    # 转换后的特征大小 (例如: 200)
                 hidden_sizes,        # MLP 隐藏层大小
                 output_size,         # 输出大小 (4)
                 transform_type="linear"):
        super().__init__()
        
        # 特征转换层
        if transform_type == "linear":
            # 线性转换：可以学习如何组合原始特征
            self.feature_transform = nn.Sequential(
                nn.Linear(raw_input_size, transformed_size),
                nn.ReLU(),
                nn.Linear(transformed_size, transformed_size),
                nn.ReLU()
            )
        elif transform_type == "custom":
            # 自定义转换：手动提取特征
            self.feature_transform = CustomFeatureExtractor(
                raw_input_size, transformed_size
            )
        else:
            # 恒等映射（不转换）
            self.feature_transform = nn.Identity()
            transformed_size = raw_input_size
        
        # 原始 MLP（使用转换后的特征大小作为输入）
        self.mlp = MLP(transformed_size, hidden_sizes, output_size)
    
    def forward(self, x):
        # 先转换特征
        transformed = self.feature_transform(x)
        # 再通过 MLP
        return self.mlp(transformed)
```

### 方法2: 自定义特征提取器

手动解析信息状态并提取有用特征：

```python
class CustomFeatureExtractor(nn.Module):
    """自定义特征提取器
    
    从原始信息状态中提取更有意义的特征：
    - 手牌强度（对子、同花、顺子等）
    - 位置信息（UTG, MP, CO, BTN, SB, BB）
    - 下注历史统计
    - 底池大小归一化
    - 等等
    """
    
    def __init__(self, raw_input_size, output_size):
        super().__init__()
        self.raw_input_size = raw_input_size
        self.output_size = output_size
        
        # 可以添加一些可学习的特征提取层
        self.learned_features = nn.Sequential(
            nn.Linear(raw_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        # 方法A: 完全可学习的转换
        return self.learned_features(x)
        
        # 方法B: 手动特征工程 + 可学习转换
        # manual_features = self.extract_manual_features(x)
        # learned_features = self.learned_features(x)
        # return torch.cat([manual_features, learned_features], dim=-1)
    
    def extract_manual_features(self, x):
        """手动提取特征（示例）"""
        # 解析信息状态
        num_players = 6
        num_cards = 52
        
        # 1. 玩家位置特征
        player_pos = x[:, 0:num_players]  # 6维
        
        # 2. 手牌特征
        hole_cards = x[:, num_players:num_players+num_cards]  # 52维
        hole_card_count = hole_cards.sum(dim=1, keepdim=True)  # 手牌数量
        
        # 3. 公共牌特征
        board_cards = x[:, num_players+num_cards:num_players+2*num_cards]  # 52维
        board_card_count = board_cards.sum(dim=1, keepdim=True)  # 公共牌数量
        
        # 4. 动作序列特征（简化）
        action_seq_start = num_players + 2 * num_cards
        # 可以提取：动作数量、下注总额等
        
        # 组合特征
        features = torch.cat([
            player_pos,
            hole_card_count,
            board_card_count,
            # ... 更多特征
        ], dim=1)
        
        return features
```

### 方法3: 在调用网络前手动转换（最简单）

不修改网络结构，在调用前转换：

```python
def transform_info_state(raw_info_state, num_players=6):
    """将原始信息状态转换为自定义特征"""
    info_state = np.array(raw_info_state)
    features = []
    
    # 1. 玩家位置（保持原样）
    features.extend(info_state[0:num_players])
    
    # 2. 手牌特征（提取手牌数量、手牌强度等）
    hole_cards = info_state[num_players:num_players+52]
    hole_card_count = np.sum(hole_cards)
    features.append(hole_card_count)
    # 可以添加更多手牌特征...
    
    # 3. 公共牌特征
    board_cards = info_state[num_players+52:num_players+104]
    board_card_count = np.sum(board_cards)
    features.append(board_card_count)
    
    # 4. 动作序列统计
    action_seq = info_state[num_players+104:]
    # 提取统计特征：动作数量、平均下注等
    
    return np.array(features)

# 在网络调用前使用
info_state = state.information_state_tensor(player)
transformed = transform_info_state(info_state)
state_tensor = torch.FloatTensor(transformed).to(device)
output = network(state_tensor)
```

## 📝 具体实现步骤

### 步骤1: 创建带转换层的网络类

```python
# 在 deep_cfr.py 中添加

class MLPWithFeatureTransform(nn.Module):
    def __init__(self, raw_input_size, transformed_size, 
                 hidden_sizes, output_size):
        super().__init__()
        # 转换层
        self.transform = nn.Sequential(
            nn.Linear(raw_input_size, transformed_size),
            nn.ReLU(),
            nn.Dropout(0.1)  # 可选：防止过拟合
        )
        # MLP
        self.mlp = MLP(transformed_size, hidden_sizes, output_size)
    
    def forward(self, x):
        x = self.transform(x)
        return self.mlp(x)
    
    def reset(self):
        # 重置 MLP，但保留转换层
        self.mlp.reset()
```

### 步骤2: 修改 DeepCFRSolver 初始化

```python
# 在 DeepCFRSolver.__init__ 中

# 原始代码：
self._policy_network = MLP(self._embedding_size, ...)

# 修改为：
transformed_size = 200  # 转换后的特征大小
self._policy_network = MLPWithFeatureTransform(
    raw_input_size=self._embedding_size,  # 266
    transformed_size=transformed_size,    # 200
    hidden_sizes=list(policy_network_layers),
    output_size=self._num_actions
)
```

### 步骤3: 优势网络同样修改

```python
self._advantage_networks = [
    MLPWithFeatureTransform(
        raw_input_size=self._embedding_size,
        transformed_size=transformed_size,
        hidden_sizes=list(advantage_network_layers),
        output_size=self._num_actions
    ) for _ in range(self._num_players)
]
```

## 🎨 自定义特征示例

### 示例1: 提取手牌强度特征

```python
def extract_hand_strength_features(hole_cards_bits, board_cards_bits):
    """提取手牌强度相关特征"""
    features = []
    
    # 手牌数量
    features.append(np.sum(hole_cards_bits))
    
    # 公共牌数量
    features.append(np.sum(board_cards_bits))
    
    # 手牌是否为对子（需要解析牌面值）
    # ... 实现对子检测逻辑
    
    # 手牌是否为同花（需要解析花色）
    # ... 实现同花检测逻辑
    
    return np.array(features)
```

### 示例2: 提取位置特征

```python
def extract_position_features(player_pos_one_hot):
    """提取位置相关特征"""
    player_idx = np.argmax(player_pos_one_hot)
    
    # 位置编码（UTG=0, MP=1, CO=2, BTN=3, SB=4, BB=5）
    position_features = np.zeros(6)
    position_features[player_idx] = 1.0
    
    # 位置数值（用于距离计算）
    position_value = player_idx / 5.0  # 归一化到 [0, 1]
    
    return np.concatenate([position_features, [position_value]])
```

### 示例3: 提取下注历史特征

```python
def extract_betting_features(action_seq_bits, action_sizings):
    """提取下注历史特征"""
    features = []
    
    # 动作数量
    num_actions = np.count_nonzero(action_sizings > 0)
    features.append(num_actions)
    
    # 总下注金额
    total_bet = np.sum(action_sizings)
    features.append(total_bet)
    
    # 平均下注金额
    if num_actions > 0:
        avg_bet = total_bet / num_actions
    else:
        avg_bet = 0
    features.append(avg_bet)
    
    # 最大下注金额
    max_bet = np.max(action_sizings) if len(action_sizings) > 0 else 0
    features.append(max_bet)
    
    return np.array(features)
```

## 🔄 完整转换流程

### 方案A: 端到端可学习转换

```python
class EndToEndFeatureTransform(nn.Module):
    """端到端可学习的特征转换"""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.transform(x)
```

**优点**：
- ✅ 完全可学习，网络自动找到最佳特征
- ✅ 实现简单
- ✅ 不需要手动特征工程

**缺点**：
- ❌ 需要更多参数
- ❌ 可能学习到不直观的特征
- ❌ 训练时间更长

### 方案B: 手动特征工程 + 可学习转换

```python
class HybridFeatureTransform(nn.Module):
    """混合特征转换：手动特征 + 可学习转换"""
    
    def __init__(self, raw_input_size, manual_feature_size, 
                 transformed_size):
        super().__init__()
        self.manual_extractor = ManualFeatureExtractor()
        self.learned_transform = nn.Sequential(
            nn.Linear(raw_input_size, 128),
            nn.ReLU()
        )
        # 合并手动特征和可学习特征
        self.final_transform = nn.Sequential(
            nn.Linear(manual_feature_size + 128, transformed_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        manual_features = self.manual_extractor(x)
        learned_features = self.learned_transform(x)
        combined = torch.cat([manual_features, learned_features], dim=1)
        return self.final_transform(combined)
```

**优点**：
- ✅ 结合领域知识和可学习特征
- ✅ 特征更可解释
- ✅ 可能性能更好

**缺点**：
- ❌ 需要手动设计特征
- ❌ 实现更复杂

## 📊 使用场景

### 场景1: 减少特征维度

原始信息状态：266维 → 转换后：150维

```python
# 可以降低网络复杂度，加快训练
transformed_size = 150
self._policy_network = MLPWithFeatureTransform(
    raw_input_size=266,
    transformed_size=150,
    hidden_sizes=[128, 64],  # 可以减小
    output_size=4
)
```

### 场景2: 增加领域知识

添加扑克相关的先验知识：

```python
# 例如：手牌强度、位置优势等
features = [
    hand_strength,      # 手牌强度 (0-1)
    position_advantage, # 位置优势
    pot_odds,          # 底池赔率
    stack_ratio,       # 筹码比例
    # ...
]
```

### 场景3: 特征归一化

```python
class NormalizedFeatureTransform(nn.Module):
    """特征归一化转换"""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        # 学习每个特征的均值和标准差
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
        self.transform = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # 归一化
        x_normalized = (x - self.mean) / (self.std + 1e-8)
        return self.transform(x_normalized)
```

## ⚠️ 注意事项

### 1. 保持可微性

转换层必须是可微分的，这样才能进行反向传播：

```python
# ✅ 正确：使用 PyTorch 操作
x = F.relu(self.linear(x))

# ❌ 错误：使用 NumPy 操作（不可微）
x = np.sum(x, axis=1)  # 这会断开梯度
```

### 2. 设备一致性

确保转换层和网络在同一设备上：

```python
self.feature_transform = self.feature_transform.to(device)
self.mlp = self.mlp.to(device)
```

### 3. 训练稳定性

转换层可能影响训练稳定性，建议：
- 使用 Batch Normalization
- 添加 Dropout
- 使用较小的学习率

### 4. 特征维度匹配

确保转换后的特征维度与 MLP 输入维度匹配：

```python
# 转换层输出大小
transformed_size = 200

# MLP 输入大小必须匹配
self.mlp = MLP(transformed_size, hidden_sizes, output_size)
```

## 🔗 集成到现有代码

### 修改点1: 网络定义

```python
# 在 deep_cfr.py 中修改 MLP 类或创建新类
class MLPWithTransform(MLP):
    def __init__(self, raw_size, transformed_size, hidden_sizes, output_size):
        # 添加转换层
        self.transform = nn.Linear(raw_size, transformed_size)
        # 调用父类，但使用转换后的尺寸
        super().__init__(transformed_size, hidden_sizes, output_size)
    
    def forward(self, x):
        x = F.relu(self.transform(x))
        return super().forward(x)
```

### 修改点2: DeepCFRSolver 初始化

```python
# 在 __init__ 中
self._embedding_size = len(state.information_state_tensor(0))  # 266
self._transformed_size = 200  # 自定义

# 修改网络创建
self._policy_network = MLPWithTransform(
    self._embedding_size,
    self._transformed_size,
    list(policy_network_layers),
    self._num_actions
)
```

### 修改点3: 优化器

转换层的参数也需要优化：

```python
# 优化器会自动包含转换层的参数
self._optimizer_policy = torch.optim.Adam(
    self._policy_network.parameters(),  # 包括转换层
    lr=learning_rate
)
```

## 📈 效果评估

添加转换层后，可以：

1. **监控特征分布**：
   ```python
   with torch.no_grad():
       transformed = transform_layer(raw_features)
       print(f"转换后特征统计: mean={transformed.mean()}, std={transformed.std()}")
   ```

2. **可视化特征**：
   ```python
   # 使用 t-SNE 或 PCA 可视化转换后的特征
   ```

3. **对比性能**：
   - 有转换层 vs 无转换层
   - 不同转换层设计的效果

## 🎯 总结

添加转换层的核心步骤：

1. **创建转换层类**：继承 `nn.Module`，实现 `forward` 方法
2. **修改网络结构**：在 MLP 前添加转换层
3. **调整输入维度**：MLP 的输入维度改为转换后的维度
4. **训练**：转换层参数会随网络一起训练

**关键优势**：
- ✅ 不修改 C++ 代码
- ✅ 可以添加领域知识
- ✅ 可以降维或增维
- ✅ 端到端可训练

**适用场景**：
- 需要添加先验知识
- 需要降维加速训练
- 需要特征归一化
- 实验不同的特征表示

