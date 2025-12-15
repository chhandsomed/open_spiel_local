# API 使用文档

## 概述

API服务器为后端提供德州扑克推荐动作接口。支持六人场，只推理一个位置的出牌动作。

**关键特性**：
- 后端只需传当前玩家的手牌和公共牌
- 其他玩家的手牌由系统随机分配（不影响推理结果）
- 支持自定义盲注和筹码配置
- 支持历史动作重建游戏状态

## 安装依赖

```bash
pip install flask requests
```

## 启动服务器

```bash
# 使用CPU
python api_server.py --model_dir models/deepcfr_stable_run --host 0.0.0.0 --port 5000 --device cpu

# 使用GPU（如果可用）
python api_server.py --model_dir models/deepcfr_stable_run --host 0.0.0.0 --port 5000 --device cuda
```

**参数说明**：
- `--model_dir`: 模型目录路径（必须包含`config.json`和模型文件）
- `--host`: 绑定的主机地址（默认：0.0.0.0）
- `--port`: 绑定的端口（默认：5000）
- `--device`: 使用的设备，`cpu`或`cuda`（默认：cpu）

## API 端点

### 1. 健康检查

**GET** `/api/v1/health`

检查服务器和模型是否已加载。

**响应示例**：
```json
{
  "success": true,
  "message": "API server is running",
  "model_loaded": true,
  "game_loaded": true
}
```

### 2. 获取推荐动作

**POST** `/api/v1/recommend_action`

获取当前状态下的推荐动作。

**请求格式**：
```json
{
  "player_id": 0,
  "hole_cards": ["As", "Kh"],
  "board_cards": ["2d", "3c", "4h"],
  "action_history": [1, 1, 1, 1],
  "seed": 12345
}
```

**字段说明**：
- `player_id` (必需): 当前需要推理的玩家ID，0-5
  - **这是OpenSpiel内部的固定座位索引，不会因为Dealer轮转而改变**
  - Player 0 永远是座位0，Player 1 永远是座位1，以此类推
  - 但Player 0在不同局中可能扮演不同角色（Dealer/SB/BB/UTG等），这取决于`dealer_pos`
  - 示例：
    - 如果`dealer_pos=5`，则Player 0是SB，Player 1是BB，Player 2是UTG
    - 如果`dealer_pos=0`，则Player 0是Dealer，Player 1是SB，Player 2是BB
- `hole_cards` (必需): 当前玩家的手牌列表，必须是2张牌
  - **支持两种格式**：
    1. **数字格式（推荐）**：`[0, 12]` - 0-51的整数，数字已包含花色信息
       - 花色顺序：方块(Diamond)[0-12] -> 梅花(Clubs)[13-25] -> 红桃(Hearts)[26-38] -> 黑桃(Spade)[39-51]
       - 每个花色内：2~JQKA 对应 0~12（rank）
    2. **传统格式（兼容）**：`["As", "Kh"]` - Rank + Suit 字符串
       - Rank: `2-9`, `T`(10), `X`(10), `J`, `Q`, `K`, `A`
       - Suit: `s`(spades), `h`(hearts), `d`(diamonds), `c`(clubs)
- `board_cards` (必需): 公共牌列表
  - Preflop: `[]`
  - Flop: `[13, 26, 39]` 或 `["2d", "3c", "4h"]` (3张)
  - Turn: `[13, 26, 39, 0]` 或 `["2d", "3c", "4h", "5s"]` (4张)
  - River: `[13, 26, 39, 0, 1]` 或 `["2d", "3c", "4h", "5s", "6h"]` (5张)
  - 格式与`hole_cards`相同，支持数字格式或传统格式
- `action_history` (必需): 历史动作列表（**只包含玩家动作，不包含发牌动作**）
  - 动作ID：`0`=Fold, `1`=Call/Check, `2`=Pot, `3`=All-in, `4`=Half-Pot (如果使用fchpa)
  - 动作顺序：按游戏进行的时间顺序
  - **注意**：不需要传发牌动作（chance actions），系统会自动处理发牌
- `action_sizings` (可选): 每次动作的下注金额列表，与`action_history`一一对应
  - 格式：`[0, 0, 100, 0]` - 每个动作对应的下注金额
  - 长度必须与`action_history`相同
  - Fold和Check/Call的下注金额为0
  - Raise/Bet的下注金额为实际下注数量
  - **注意**：如果不传，系统会根据动作ID自动计算下注金额（FCHPA模式下）
- `blinds` (可选): 盲注列表，如 `[50, 100, 0, 0, 0, 0]`
  - 如果不传，则使用模型默认配置
  - 如果传了，必须与`stacks`一起传，且长度必须等于玩家数量
- `stacks` (可选): 筹码列表，如 `[2000, 2000, 2000, 2000, 2000, 2000]`
  - 如果不传，则使用模型默认配置
  - 如果传了，必须与`blinds`一起传，且长度必须等于玩家数量
  - **注意**：传入的应该是当前剩余筹码，不是初始筹码
- `seed` (可选): 随机种子，用于分配其他玩家的手牌（确保可复现）

**响应格式**：
```json
{
  "success": true,
  "data": {
    "recommended_action": 1,
    "action_probabilities": {
      "0": 0.05,
      "1": 0.75,
      "2": 0.15,
      "3": 0.05
    },
    "legal_actions": [0, 1, 2, 3],
    "current_player": 0
  },
  "error": null
}
```

**响应字段说明**：
- `recommended_action`: 推荐的动作ID（概率最大的动作）
- `action_probabilities`: 所有合法动作的概率分布
- `legal_actions`: 当前状态下的合法动作列表
- `current_player`: 当前玩家ID

**错误响应**：
```json
{
  "success": false,
  "data": null,
  "error": "错误信息"
}
```

### 3. 重新加载模型

**POST** `/api/v1/reload_model`

动态切换模型（支持运行时更新模型）。

**请求格式**：
```json
{
  "model_dir": "models/deepcfr_stable_run",
  "device": "cpu"
}
```

**字段说明**：
- `model_dir` (必需): 模型目录路径
- `device` (可选): 设备类型，`cpu`或`cuda`（默认使用当前设备）

**响应示例**：
```json
{
  "success": true,
  "message": "Model reloaded from models/deepcfr_stable_run",
  "model_dir": "models/deepcfr_stable_run",
  "device": "cpu"
}
```

### 4. 获取动作映射表

**GET** `/api/v1/action_mapping`

获取动作ID到动作名称的映射。

**响应示例**：
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

## 使用示例

### Python示例

```python
import requests
import json

base_url = "http://localhost:5000/api/v1"

# 1. 健康检查
response = requests.get(f"{base_url}/health")
print(response.json())

# 2. 获取推荐动作（Preflop阶段，使用自定义盲注和筹码）
request_data = {
    "player_id": 0,
    "hole_cards": ["As", "Kh"],
    "board_cards": [],
    "action_history": [],
    "blinds": [50, 100, 0, 0, 0, 0],
    "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
    "seed": 12345
}

response = requests.post(f"{base_url}/recommend_action", json=request_data)
result = response.json()

if result["success"]:
    print(f"推荐动作: {result['data']['recommended_action']}")
    print(f"动作概率: {result['data']['action_probabilities']}")
else:
    print(f"错误: {result['error']}")
```

### cURL示例

```bash
# 健康检查
curl http://localhost:5000/api/v1/health

# 获取推荐动作
curl -X POST http://localhost:5000/api/v1/recommend_action \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": 0,
    "hole_cards": ["As", "Kh"],
    "board_cards": [],
    "action_history": [],
    "seed": 12345
  }'
```

## 测试脚本

使用提供的测试脚本：

```bash
python test_api.py http://localhost:5000/api/v1
```

## 注意事项

1. **牌面格式**：必须使用标准格式（如`"As"`, `"Kh"`），大小写不敏感
2. **手牌数量**：当前玩家的手牌必须是2张
3. **公共牌数量**：
   - Preflop: 0张
   - Flop: 3张
   - Turn: 4张
   - River: 5张
4. **历史动作**：**只包含玩家动作，不包含发牌动作（chance actions）**
   - 发牌动作由系统自动处理，后端不需要传
   - 只需要传玩家做出的决策动作（fold/call/raise等）
5. **盲注和筹码**：
   - 如果请求中提供了`blinds`和`stacks`，则使用请求中的值
   - 如果不提供，则使用模型默认配置（从`config.json`读取）
   - 如果提供了其中一个，必须同时提供另一个，且长度必须等于玩家数量
6. **其他玩家手牌**：由系统随机分配，不影响当前玩家的推理结果
7. **随机种子**：如果提供`seed`，可以确保其他玩家手牌的分配是可复现的
8. **模型切换**：支持通过`/api/v1/reload_model`接口动态切换模型，无需重启服务器

## 动作ID说明

根据`betting_abstraction`配置，动作ID可能不同：

### fchpa (推荐)
- `0`: Fold
- `1`: Call/Check
- `2`: Pot (Raise to Pot)
- `3`: All-in
- `4`: Half-Pot (Raise to Half Pot)

### fcpa
- `0`: Fold
- `1`: Call/Check
- `2`: Pot (Raise to Pot)
- `3`: All-in

### fc
- `0`: Fold
- `1`: Call/Check

## 故障排查

1. **模型未加载**：检查`--model_dir`路径是否正确，是否包含`config.json`和模型文件
2. **非法动作**：检查`action_history`中的动作是否在当前状态下合法
3. **牌面冲突**：确保`hole_cards`和`board_cards`中没有重复的牌
4. **状态不匹配**：确保`action_history`与当前游戏状态一致

## 技术细节

### 状态构建流程

1. 创建游戏状态：
   - 如果请求中提供了`blinds`和`stacks`，使用它们创建游戏实例
   - 否则使用模型默认配置（从`config.json`读取）
2. 在chance节点选择指定的手牌和公共牌：
   - 当前玩家的手牌：使用请求中指定的手牌
   - 公共牌：使用请求中指定的公共牌
   - 其他玩家的手牌：从剩余牌中随机分配
3. 应用历史动作重建游戏状态：
   - 历史动作只包含玩家动作（fold/call/raise等）
   - 不包含发牌动作（chance actions），系统会自动处理
4. 获取当前玩家的信息状态
5. 使用模型推理，返回推荐动作

### 发牌动作说明

**发牌动作（chance actions）**是OpenSpiel在chance节点时选择发哪张牌的动作。在实际推理时：
- **不需要**后端传发牌动作
- 系统会根据请求中的`hole_cards`和`board_cards`自动处理发牌
- 后端只需要传玩家做出的决策动作（`action_history`）

### 信息状态

OpenSpiel的信息状态只包含当前玩家可见的信息：
- 当前玩家的手牌（52 bits）
- 公共牌（52 bits）
- 动作序列
- 动作对应的筹码大小

**不包含其他玩家的手牌**，因此其他玩家手牌的分配不影响推理结果。
