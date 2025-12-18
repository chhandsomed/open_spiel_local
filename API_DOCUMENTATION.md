# API 接口文档

## 1. 健康检查

```
GET /api/v1/health
```

**响应**:
```json
{
  "success": true,
  "message": "API server is running",
  "model_loaded": true,
  "game_loaded": true
}
```

---

## 2. 获取推荐动作

```
POST /api/v1/recommend_action
```

### 请求示例

**示例1: Preflop（翻牌前）**
```json
{
  "player_id": 2,  // 玩家ID (0-5)，固定座位索引。注意：dealer_pos=5时，UTG是Player 2
  "hole_cards": [51, 38],  // 手牌：As(51), Kh(38)
  "board_cards": [],  // 公共牌：Preflop阶段为空
  "action_history": [],  // 历史动作：Preflop阶段为空
  "action_sizings": [],  // 下注金额：Preflop阶段为空（bet to格式，累计总额）
  "blinds": [50, 100, 0, 0, 0, 0],  // 盲注：[SB, BB, 0, 0, 0, 0]
  "stacks": [2000, 2000, 2000, 2000, 2000, 2000],  // 当前剩余筹码（6人场）
  "dealer_pos": 5  // Dealer位置：5（SB=Player 0, BB=Player 1, UTG=Player 2）
}
```

**示例2: Preflop（5人场）**
```json
{
  "player_id": 2,  // 玩家ID：2（固定座位索引）。注意：5人场dealer_pos=4时，UTG是Player 2
  "hole_cards": [51, 38],  // 手牌：As(51), Kh(38)
  "board_cards": [],  // 公共牌：Preflop阶段为空
  "action_history": [],  // 历史动作：Preflop阶段为空
  "action_sizings": [],  // 下注金额：Preflop阶段为空（bet to格式，累计总额）
  "blinds": [100, 200, 0, 0, 0],  // 盲注：[SB, BB, 0, 0, 0]（5人场）
  "stacks": [50000, 50000, 50000, 50000, 50000],  // 当前剩余筹码（5人场）
  "dealer_pos": 4  // Dealer位置：4（SB=Player 0, BB=Player 1, UTG=Player 2）
}
```

**示例3: Flop（5人场，翻牌后，有历史动作）**
```json
{
  "player_id": 1,
  "hole_cards": [12, 25],
  "board_cards": [0, 13, 26],
  "action_history": [0, 0, 1, 2],
  "action_sizings": [0, 0, 0, 200],
  "blinds": [100, 200, 0, 0, 0],
  "stacks": [49800, 49800, 50000, 50000, 49800],
  "dealer_pos": 4
}
```

**示例4: action_sizings详细示例（展示所有动作类型）**
```json
{
  "player_id": 2,
  "hole_cards": [51, 38],
  "board_cards": [],
  "action_history": [0, 1, 2, 3, 4],
  "action_sizings": [0, 0, 300, 5000, 150],
  "blinds": [50, 100, 0, 0, 0, 0],
  "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
  "dealer_pos": 5
}
```
**action_sizings说明**:
- `action_history[0] = 0` (Fold) -> `action_sizings[0] = 0` (弃牌为0)
- `action_history[1] = 1` (Call/Check) -> `action_sizings[1] = 0` (Call/Check固定为0)
- `action_history[2] = 2` (Pot) -> `action_sizings[2] = 300` (加注到池，累计下注300)
- `action_history[3] = 3` (All-in) -> `action_sizings[3] = 5000` (全押，累计下注5000)
- `action_history[4] = 4` (Half-Pot) -> `action_sizings[4] = 150` (半池加注，累计下注150)

**action_sizings计算公式**:
```
对于每个动作i:
  player_id = 执行动作i的玩家ID
  action_sizings[i] = player_contributions[player_id] (动作i执行后)
```

**特殊情况**:
- **Fold (动作0)**: `action_sizings[i] = 0`（固定值）
- **Call/Check (动作1)**: `action_sizings[i] = 0`（OpenSpiel设计，固定值）
- **Pot/All-in/Half-Pot (动作2/3/4)**: `action_sizings[i] = player_contributions[player_id]`（动作i执行后）

**计算流程**:
```
1. 初始化: player_contributions = [0, 0, 0, ...] (所有玩家贡献为0)

2. 对于每个动作i:
   a. 获取执行动作i的玩家player_id
   b. 执行动作i
   c. OpenSpiel更新player_contributions[player_id]
   d. 如果动作i是Pot/All-in/Half-Pot:
        action_sizings[i] = player_contributions[player_id]
      否则（Fold/Call/Check）:
        action_sizings[i] = 0
```

**Pot和Half-Pot金额计算公式**:
```
Pot动作金额 = maxSpent + pot_after_call
Half-Pot动作金额 = maxSpent + 0.5 * pot_after_call

其中:
  pot_size = sum(player_contributions)  // 所有玩家累计贡献总和
  amount_to_call = maxSpent - player_contributions[current_player]  // 当前玩家需要跟注的金额
  pot_after_call = amount_to_call + pot_size  // 如果当前玩家跟注后的池子大小
  maxSpent = 当前轮次的最大下注金额（即需要跟注到的金额）
```

**计算示例（详细步骤）**:
```
初始状态: player_contributions = [0, 0, 0, 0, 0, 0]
          blinds = [50, 100, 0, 0, 0, 0]
          maxSpent = 100 (BB需要跟注到100)

动作0: Player 2执行Fold
  - player_contributions不变: [0, 0, 0, 0, 0, 0]
  - action_sizings[0] = 0 (Fold固定为0)

动作1: Player 3执行Call/Check，跟注到100
  - player_contributions更新: [50, 100, 0, 100, 0, 0]
    (SB已投50，BB已投100，Player 3跟注到100)
  - maxSpent = 100
  - action_sizings[1] = 0 (Call/Check固定为0)

动作2: Player 4执行Pot，加注到池
  - 动作前状态:
    * player_contributions = [50, 100, 0, 100, 0, 0]
    * pot_size = 50 + 100 + 0 + 100 + 0 + 0 = 250
    * amount_to_call = 100 - 0 = 100 (Player 4需要跟注100)
    * pot_after_call = 100 + 250 = 350
    * maxSpent = 100
  - Pot金额计算:
    * Pot金额 = maxSpent + pot_after_call = 100 + 350 = 450
    * 但这是"加注到"的金额，Player 4之前贡献为0，所以实际加注450
  - 动作后:
    * player_contributions更新: [50, 100, 0, 100, 450, 0]
    * action_sizings[2] = player_contributions[4] = 450
  - ⚠️ 注意：示例中的300是简化值，实际计算可能不同

动作3: Player 0执行All-in，全押5000
  - player_contributions更新: [5000, 100, 0, 100, 450, 0]
  - action_sizings[3] = player_contributions[0] = 5000

动作4: Player 1执行Half-Pot，半池加注
  - 动作前状态:
    * player_contributions = [5000, 100, 0, 100, 450, 0]
    * pot_size = 5000 + 100 + 0 + 100 + 450 + 0 = 5650
    * amount_to_call = 450 - 100 = 350 (Player 1需要跟注350到450)
    * pot_after_call = 350 + 5650 = 6000
    * maxSpent = 450
  - Half-Pot金额计算:
    * Half-Pot金额 = maxSpent + 0.5 * pot_after_call = 450 + 0.5 * 6000 = 450 + 3000 = 3450
    * Player 1之前贡献为100，所以实际加注 = 3450 - 100 = 3350
  - 动作后:
    * player_contributions更新: [5000, 3450, 0, 100, 450, 0]
    * action_sizings[4] = player_contributions[1] = 3450
  - ⚠️ 注意：示例中的150是简化值，实际计算可能不同
```

**关键点**:
- **Pot动作**: 加注到 `maxSpent + pot_after_call`（全池加注）
- **Half-Pot动作**: 加注到 `maxSpent + 0.5 * pot_after_call`（半池加注）
- **action_sizings**: 存储的是动作后玩家的累计贡献（bet to格式），不是增量
- 示例中的300和150是简化值，实际值由OpenSpiel根据当前游戏状态动态计算

**实际获取方式**:
- UI从OpenSpiel的`information_state_tensor`中提取`action_sizings`
- API也可以从`state.to_struct().player_contributions`获取玩家贡献
- `action_sizings[i]`就是动作i执行后，执行该动作的玩家的累计贡献值

**示例5: Flop（6人场，翻牌后，有历史动作）**
```json
{
  "player_id": 0,  // 玩家ID：0（固定座位索引），当前应该行动的玩家
  "hole_cards": [12, 25],  // 手牌：2♣(12), 2♥(25)
  "board_cards": [0, 13, 26],  // 公共牌：2♦(0), 2♣(13), 2♥(26)（翻牌3张）
  "action_history": [0, 0, 1, 2],  // 历史动作：[弃牌, 弃牌, 跟注, 加注到池]
  // 说明：Preflop阶段，Player 2弃牌，Player 3弃牌，Player 4跟注，Player 5加注到池
  // Flop阶段，现在轮到Player 0（SB）行动
  "action_sizings": [0, 0, 0, 200],  // 下注金额：[0, 0, 0, 200]（bet to格式，累计总额）
  // ⚠️ 重要：Call/Check(动作1)的action_sizings必须传0（OpenSpiel设计）
  // 实际下注金额从stacks变化计算：stacks从[2000,2000,2000]变为[1950,2000,2000]表示玩家0下注了50
  "blinds": [50, 100, 0, 0, 0, 0],  // 盲注：[SB, BB, 0, 0, 0, 0]
  "stacks": [1950, 1900, 2000, 2000, 2000, 2000],  // 当前剩余筹码（6人场）
  "dealer_pos": 5  // Dealer位置：5（SB=Player 0, BB=Player 1, UTG=Player 2）
}
```

**字段值说明**:
- `board_cards`: `[0, 13, 26]` 
  - `0` = 2♦ (Two of Diamonds)
  - `13` = 2♣ (Two of Clubs)
  - `26` = 2♥ (Two of Hearts)
  - 表示翻牌阶段的3张公共牌

- `action_history`: `[0, 0, 1, 2]`
  - `0` = Fold（弃牌）- 第1个玩家弃牌
  - `0` = Fold（弃牌）- 第2个玩家弃牌
  - `1` = Call/Check（跟注/过牌）- 第3个玩家跟注
  - `2` = Pot（加注到池）- 第4个玩家加注到池
  - 按时间顺序排列，只包含玩家动作

- `action_sizings`: `[0, 0, 0, 200]`（bet to格式，累计总额）
  - `0` - 对应action_history[0]，弃牌下注金额为0
  - `0` - 对应action_history[1]，弃牌下注金额为0
  - `0` - 对应action_history[2]，**跟注/过牌动作的action_sizings必须传0（OpenSpiel设计）**
    - ⚠️ **必须传0**：Call/Check的action_sizings固定为0，这是OpenSpiel的设计，必须遵守
    - ⚠️ **实际下注金额**：Call/Check的实际下注金额从`stacks`变化计算
      - 示例：如果stacks从`[2000, 2000, 2000]`变为`[1950, 2000, 2000]`，说明玩家0实际下注了50
      - 原因：Call/Check的下注金额由当前需要跟注的金额决定，不是固定值
  - `200` - 对应action_history[3]，加注到池后累计下注总额为200
  - 与action_history一一对应，表示每次动作后的累计下注总额（Call/Check固定为0）

**action_sizings传值规则（重要）**:
- **动作0 (Fold/弃牌)**: `action_sizings = 0`
- **动作1 (Call/Check/跟注/过牌)**: `action_sizings = 0`（OpenSpiel设计，必须传0）
- **动作2 (Pot/加注到池)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家累计下注了300，则传`300`
- **动作3 (All-in/全押)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家全押5000，则传`5000`
- **动作4 (Half-Pot/半池加注)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家累计下注了150，则传`150`

**完整示例**:
```json
{
  "action_history": [0, 1, 2, 3, 4],
  "action_sizings": [0, 0, 300, 5000, 150]
}
```
说明：
- `[0, 0]` - 前两个动作是Fold和Call/Check，都传0
- `[300]` - Pot动作，累计下注300
- `[5000]` - All-in动作，累计下注5000
- `[150]` - Half-Pot动作，累计下注150

**action_sizings传值规则（重要）**:
- **动作0 (Fold/弃牌)**: `action_sizings = 0`
- **动作1 (Call/Check/跟注/过牌)**: `action_sizings = 0`（OpenSpiel设计，必须传0）
- **动作2 (Pot/加注到池)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家累计下注了300，则传`300`
- **动作3 (All-in/全押)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家全押5000，则传`5000`
- **动作4 (Half-Pot/半池加注)**: `action_sizings = 累计下注总额`（bet to格式）
  - 示例：如果玩家累计下注了150，则传`150`

**完整示例**:
```json
{
  "action_history": [0, 1, 2, 3, 4],
  "action_sizings": [0, 0, 300, 5000, 150]
}
```
说明：
- `[0, 0]` - 前两个动作是Fold和Call/Check，都传0
- `[300]` - Pot动作，累计下注300
- `[5000]` - All-in动作，累计下注5000
- `[150]` - Half-Pot动作，累计下注150
  
**action_sizings处理说明**:
- **归一化**: 传入**归一化前的原始金额**（如200），API会在推理时自动归一化（除以max_stack）
- **格式**: **"bet to"格式（累计总额）**，API也支持增量格式（自动识别和转换）
  - **"bet to"格式（推荐）**: 本次动作后的累计下注总额（如：`[0, 0, 0, 200]`表示第4个动作后累计下注200）
  - **增量格式（兼容）**: 本次动作新增的下注金额（如：`[0, 0, 0, 200]`表示第4个动作新增200），API会自动转换为"bet to"格式
- **实际使用**: UI从OpenSpiel的`information_state_tensor`中提取，使用的是"bet to"格式
- **处理**: API会从OpenSpiel的`information_state_tensor`中提取action_sizings用于验证，但不影响状态构建
- **⚠️ Call/Check特殊说明（重要）**: 
  - **必须传0**：Call/Check动作（action_id=1）的action_sizings**必须传0**，这是OpenSpiel的设计，必须遵守
  - **实际下注金额**：Call/Check的实际下注金额需要从`stacks`的变化计算
    - 示例：如果stacks从`[2000, 2000, 2000]`变为`[1950, 2000, 2000]`，说明玩家0实际下注了50
    - 原因：Call/Check的下注金额由当前需要跟注的金额决定，不是固定值，所以OpenSpiel将其设为0
  - **API处理**：API不依赖action_sizings来构建状态（只使用action_history），action_sizings仅用于验证

**字段值说明**:
- `board_cards`: `[0, 13, 26]` = 2♦, 2♣, 2♥（翻牌3张）
- `action_history`: `[0, 0, 1, 2]` = [弃牌, 弃牌, 跟注, 加注到池]
- `action_sizings`: `[0, 0, 0, 200]` = [0, 0, 0, 200]（对应action_history的下注金额）

### 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `player_id` | integer | ✅ | 玩家ID (0到num_players-1)，固定座位索引 |
| `hole_cards` | array | ✅ | 手牌，2张。格式：`[51, 38]` 或 `["As", "Kh"]` |
| `board_cards` | array | ✅ | 公共牌。Preflop: `[]`, Flop: 3张, Turn: 4张, River: 5张 |
| `action_history` | array | ✅ | 历史动作ID列表（只包含玩家动作） |
| `action_sizings` | array | ❌ | 每次动作的下注金额，与action_history长度相同<br>**格式**: 归一化前的原始金额（API会自动归一化）<br>**类型**: **"bet to"格式（累计总额）**，API也支持增量格式（自动识别）<br>**示例**: `[0, 0, 0, 200]`（bet to格式，表示累计下注200） |
| `blinds` | array | ❌ | 盲注列表，长度=num_players。如`[50,100,0,0,0,0]` |
| `stacks` | array | ❌ | 当前剩余筹码，长度=num_players |
| `dealer_pos` | integer | 条件 | 如果传了blinds/stacks则必需。Dealer位置(0-5) |

### 卡牌格式

**数字格式（推荐）**: `0-51`
- `0-12`: 方块 (2♦-A♦)
- `13-25`: 梅花 (2♣-A♣)
- `26-38`: 红桃 (2♥-A♥)
- `39-51`: 黑桃 (2♠-A♠)

**传统格式**: `"RankSuit"`，如`"As"`(A♠), `"Kh"`(K♥)

### 响应

```json
{
  "success": true,
  "data": {
    "recommended_action": 1,
    "action_probabilities": {"0": 0.05, "1": 0.75, "2": 0.15, "3": 0.05},
    "legal_actions": [0, 1, 2, 3],
    "current_player": 0
  },
  "error": null
}
```

**字段说明**:
- `recommended_action`: 推荐动作ID（argmax策略）
- `action_probabilities`: 所有合法动作的概率分布
- `legal_actions`: 当前合法的动作ID列表

---

## 3. 动态替换模型

```
POST /api/v1/reload_model
```

### 请求示例

```json
{
  "model_dir": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",
  "num_players": 5,
  "device": "cpu"
}
```

### 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model_dir` | string | ✅ | 模型目录路径 |
| `num_players` | integer | ❌ | 场次(5或6)，不指定则从config.json读取 |
| `device` | string | ❌ | 设备类型("cpu"或"cuda") |

### 响应

```json
{
  "success": true,
  "message": "Model reloaded from models/xxx",
  "model_dir": "models/xxx",
  "device": "cpu",
  "num_players": 5,
  "loaded_models": {
    "5": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",
    "6": "models/deepcfr_stable_run/checkpoints/iter_32000"
  }
}
```

---

## 4. 获取动作映射表

```
GET /api/v1/action_mapping
```

### 响应

```json
{
  "success": true,
  "betting_abstraction": "fchpa",
  "action_mapping": {
    "0": "Fold",
    "1": "Call/Check",
    "2": "Pot (Raise to Pot)",
    "3": "All-in",
    "4": "Half-Pot (Raise to Half Pot)"
  }
}
```

---

## 5. 动作映射表

### fchpa抽象（5个动作）

| ID | 动作 | 说明 |
|----|------|------|
| 0 | Fold | 弃牌 |
| 1 | Call/Check | 跟注/过牌 |
| 2 | Pot | 加注到池 |
| 3 | All-in | 全押 |
| 4 | Half-Pot | 半池加注 |

### fcpa抽象（4个动作）

| ID | 动作 | 说明 |
|----|------|------|
| 0 | Fold | 弃牌 |
| 1 | Call/Check | 跟注/过牌 |
| 2 | Pot | 加注到池 |
| 3 | All-in | 全押 |

---

## 6. 使用示例

### Python示例

```python
import requests

BASE_URL = "http://localhost:8826/api/v1"

# 获取推荐动作（6人场）
response = requests.post(f"{BASE_URL}/recommend_action", json={
    "player_id": 2,  # 玩家ID (0-5)，dealer_pos=5时UTG是Player 2
    "hole_cards": [51, 38],  # 手牌：As(51), Kh(38)
    "board_cards": [],  # 公共牌：Preflop为空
    "action_history": [],  # 历史动作：Preflop为空
    "action_sizings": [],  # 下注金额：Preflop为空（bet to格式）
    "blinds": [50, 100, 0, 0, 0, 0],  # 盲注：[SB, BB, 0, 0, 0, 0]（6人场）
    "stacks": [2000, 2000, 2000, 2000, 2000, 2000],  # 当前剩余筹码（6人场）
    "dealer_pos": 5  # Dealer位置：5（SB=Player 0, BB=Player 1, UTG=Player 2）
})
result = response.json()

# 获取推荐动作（5人场）
response = requests.post(f"{BASE_URL}/recommend_action", json={
    "player_id": 2,  # 玩家ID (0-4)，dealer_pos=4时UTG是Player 2
    "hole_cards": [51, 38],  # 手牌：As(51), Kh(38)
    "board_cards": [],  # 公共牌：Preflop为空
    "action_history": [],  # 历史动作：Preflop为空
    "action_sizings": [],  # 下注金额：Preflop为空（bet to格式）
    "blinds": [100, 200, 0, 0, 0],  # 盲注：[SB, BB, 0, 0, 0]（5人场）
    "stacks": [50000, 50000, 50000, 50000, 50000],  # 当前剩余筹码（5人场）
    "dealer_pos": 4  # Dealer位置：4（SB=Player 0, BB=Player 1, UTG=Player 2）
})
result = response.json()

# 替换模型
response = requests.post(f"{BASE_URL}/reload_model", json={
    "model_dir": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",  # 模型目录路径
    "num_players": 5,  # 场次：5人场（可选）
    "device": "cpu"  # 设备：cpu或cuda（可选）
})
```

### curl示例

```bash
# 1. 获取推荐动作（6人场）
# 注意：JSON不支持注释，以下注释仅用于说明，实际使用时需要去掉
curl -X POST http://localhost:8826/api/v1/recommend_action \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": 2,
    "hole_cards": [51, 38],
    "board_cards": [],
    "action_history": [],
    "action_sizings": [],
    "blinds": [50, 100, 0, 0, 0, 0],
    "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
    "dealer_pos": 5
  }'
# 参数说明：
#   player_id: 2 - 玩家ID (0-5)，dealer_pos=5时UTG是Player 2
#   hole_cards: [51, 38] - 手牌：As(51), Kh(38)
#   board_cards: [] - 公共牌：Preflop为空
#   action_history: [] - 历史动作：Preflop为空
#   action_sizings: [] - 下注金额：Preflop为空（bet to格式）
#   blinds: [50, 100, 0, 0, 0, 0] - 盲注：[SB, BB, 0, 0, 0, 0]（6人场）
#   stacks: [2000, 2000, 2000, 2000, 2000, 2000] - 当前剩余筹码（6人场）
#   dealer_pos: 5 - Dealer位置：5（SB=Player 0, BB=Player 1, UTG=Player 2）

# 1-2. 获取推荐动作（5人场）
curl -X POST http://localhost:8826/api/v1/recommend_action \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": 2,
    "hole_cards": [51, 38],
    "board_cards": [],
    "action_history": [],
    "action_sizings": [],
    "blinds": [100, 200, 0, 0, 0],
    "stacks": [50000, 50000, 50000, 50000, 50000],
    "dealer_pos": 4
  }'
# 参数说明：
#   player_id: 2 - 玩家ID (0-4)，dealer_pos=4时UTG是Player 2
#   hole_cards: [51, 38] - 手牌：As(51), Kh(38)
#   board_cards: [] - 公共牌：Preflop为空
#   action_history: [] - 历史动作：Preflop为空
#   action_sizings: [] - 下注金额：Preflop为空（bet to格式）
#   blinds: [100, 200, 0, 0, 0] - 盲注：[SB, BB, 0, 0, 0]（5人场）
#   stacks: [50000, 50000, 50000, 50000, 50000] - 当前剩余筹码（5人场）
#   dealer_pos: 4 - Dealer位置：4（SB=Player 0, BB=Player 1, UTG=Player 2）

# 2. 替换模型
# 注意：JSON不支持注释，以下注释仅用于说明，实际使用时需要去掉
curl -X POST http://localhost:8826/api/v1/reload_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",
    "num_players": 5,
    "device": "cpu"
  }'
# 参数说明：
#   model_dir: 模型目录路径
#   num_players: 5 - 场次：5人场（可选，不指定则从config.json读取）
#   device: cpu - 设备：cpu或cuda（可选，默认使用当前设备）
```

---

## 7. 重要说明

1. **player_id**: 固定座位索引，不会因Dealer轮转而改变
   - ⚠️ **重要**：`player_id`必须是当前应该行动的玩家（`state.current_player()`）
   - Preflop阶段：UTG = `(dealer_pos + 3) % num_players`
   - 示例：6人场，`dealer_pos=5`时，UTG是Player 2，不是Player 0
2. **action_history**: 只包含玩家动作，不包含发牌动作
3. **stacks**: 当前剩余筹码，不是初始筹码
4. **场次选择**: API根据`blinds`/`stacks`长度自动选择模型（5人场或6人场）
5. **位置角色**: `(player_id - dealer_pos) % num_players` 计算角色（SB/BB/UTG等）
6. **action_sizings处理**:
   - **归一化**: 传入归一化前的原始金额（如200），API会在推理时自动归一化（除以max_stack）
   - **格式**: **"bet to"格式（累计总额）**，API也支持增量格式（自动识别和转换）
   - **实际使用**: UI从OpenSpiel提取，使用的是"bet to"格式
   - **处理**: API会从OpenSpiel提取action_sizings用于验证，但不影响状态构建（状态通过action_history构建）
