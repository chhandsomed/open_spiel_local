# OpenSpiel 德州扑克 DeepCFR 完整使用指南

本文档详细介绍了 OpenSpiel 德州扑克 DeepCFR 环境的安装、训练、推理、评测以及交互式对战流程，并提供了针对 6 人无限注德州扑克（No-Limit Texas Hold'em）的参数调优建议。

## 0. 安装与环境准备 (Installation)

### 系统依赖
项目提供了一个脚本来自动安装所需的 C++ 依赖库（如 `abseil-cpp`, `dds` 等）和系统工具。
```bash
./install.sh
```

### Python 环境
建议使用 Conda 管理环境 (Python 3.9 - 3.12)：
```bash
conda create -n open_spiel python=3.12 -y
conda activate open_spiel
pip install -r requirements.txt
```

### 编译 OpenSpiel
安装 Python 绑定 (`pyspiel`)：
```bash
pip install .
```

---

## 1. 训练 (Training)

使用 `train_deep_cfr_texas.py` 脚本来训练模型。

### 推荐命令 (高性能版)
针对 RTX 4090 等高性能显卡，建议使用更大的网络和缓冲区以获得更强的策略。

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train_deep_cfr_texas.py \
    --num_players 6 \
    --betting_abstraction fchpa \
    --policy_layers 128 128 \
    --advantage_layers 128 128 \
    --memory_capacity 4000000 \
    --num_iterations 2000 \
    --num_traversals 40 \
    --learning_rate 0.0005 \
    --eval_interval 50 \
    --eval_with_games \
    --save_prefix deepcfr_texas_6p_fchpa_large \
    > train_log_large.txt 2>&1 &
```

### 关键参数说明

| 参数 | 默认值 | 推荐值 (6人局) | 说明 |
| :--- | :--- | :--- | :--- |
| `--betting_abstraction` | `fcpa` | **`fchpa`** | 下注抽象。`fchpa` 包含半池加注(Half-pot)，策略更灵活。 |
| `--policy_layers` | `64 64` | **`128 128`** | 策略网络层数。6人局状态复杂，增加层数可提升拟合能力。 |
| `--advantage_layers` | `32 32` | **`128 128`** | 优势网络层数。用于估计后悔值，建议与策略网络相当或略小。 |
| `--memory_capacity` | `1e6` | **`4e6` (400万)** | 经验回放缓冲区。越大越好，防止模型遗忘早期策略。 |
| `--num_iterations` | `100` | **`2000`+** | 总迭代次数。DeepCFR 收敛较慢，需要较多迭代。 |
| `--num_traversals` | `20` | **`40`** | 每次迭代采样的轨迹数。增加此值可减少方差，使训练更稳定。 |
| `--learning_rate` | `1e-3` | **`5e-4`** | 学习率。网络变大后，适当降低学习率有助于稳定收敛。 |

### 附录：动作映射表

| 模式 | 代码 | 动作 ID 及含义 | 动作数量 |
| :--- | :--- | :--- | :--- |
| **默认模式** | `fcpa` | 0:Fold, 1:Call/Check, 2:Pot, 3:All-in | 4 |
| **增强模式** | `fchpa` | 0:Fold, 1:Call/Check, 2:Pot, 3:All-in, **4:Half-Pot** | 5 |
| **测试模式** | `fc` | 0:Fold, 1:Call/Check | 2 |

---

## 2. 推理与自对弈 (Inference / Self-Play)

使用 `inference_simple.py` 让模型自己打自己，快速评估模型在各个位置的平均收益和胜率。

```bash
# 运行 1000 局自对弈
python inference_simple.py \
    --model_prefix models/deepcfr_texas_6p_fchpa_large/deepcfr_texas_6p_fchpa_large \
    --num_games 1000 \
    --use_gpu
```

**结果解读**:
*   **平均收益**: 长期来看，所有位置的平均收益之和应接近 0。
*   **位置优势**: 正常情况下，后位（Button, CO）收益应高于前位（SB, BB, UTG）。
*   **胜率**: 通常在 15% - 25% 之间。

---

## 3. 模型对比评测 (Head-to-Head Evaluation)

使用 `evaluate_models_head_to_head.py` 让两个不同的模型进行对战（例如：新模型 vs 旧模型）。

```bash
python evaluate_models_head_to_head.py \
    --model_a models/deepcfr_texas_6p_fchpa_large \
    --model_b models/deepcfr_texas_6p_fchpa_baseline \
    --num_games 2000 \
    --use_gpu
```

**注意**: 两个模型必须具有**相同的游戏配置**（玩家数、下注抽象必须一致）。脚本会自动进行两轮测试（交换座位），以消除位置优势带来的偏差。

---

## 4. 训练日志分析 (Log Analysis)

使用 `analyze_training.py` 分析训练过程中的指标变化，或对比两次训练的效果。

### 单模型分析
```bash
python analyze_training.py models/deepcfr_texas_6p_fchpa_large/deepcfr_texas_6p_fchpa_large_training_history.json
```

### 双模型对比
```bash
python analyze_training.py \
    models/new_model/history.json \
    --compare models/old_model/history.json
```

---

## 5. 交互式对战 (Interactive Play)

使用 `play_interactive.py` 亲自与训练好的模型对战。

```bash
# 作为玩家 0 (SB) 与模型对战
python play_interactive.py \
    --model_dir models/deepcfr_texas_6p_fchpa_large \
    --num_players 6 \
    --human_player 0
```

**功能**:
*   自动检测 `config.json` 中的 `betting_abstraction`，加载正确的动作空间。
*   支持 `fchpa` 模式下的 5 种动作：Fold, Check/Call, Pot, All-in, Half-pot。

---

## 6. MCCFR (Monte Carlo CFR)

除了 DeepCFR，本项目也支持传统的 MCCFR 算法（基于表格）。这适合小规模测试或理论研究。

### 训练
```bash
# 2人场，1000 次迭代
python train_texas_holdem_mccfr.py --num_players 2 --iterations 1000
```

### 测试
```bash
python load_and_test_strategy.py
```

---

## 7. 常见问题与调优建议

### Q: 模型训练很久但策略依然很随机（策略熵高）？
*   **原因**: 样本利用率低或网络欠拟合。
*   **解决**: 
    1. 增加 `policy_layers` (如 128x128)。
    2. 增加 `num_traversals` (如 40 或 60)。
    3. 稍微降低学习率。

### Q: 模型只会弃牌或只会 All-in？
*   **原因**: 早期探索不够，陷入局部极值；或者优势网络梯度爆炸。
*   **解决**:
    1. 增大 `memory_capacity`，确保覆盖更多历史策略。
    2. 检查 `advantage_layers` 是否过大或过小。
    3. 重新开始训练，DeepCFR 对初始化有一定敏感性。

### Q: 显存不足 (OOM)？
*   **解决**:
    1. 减小 `memory_capacity` (如 200万)。
    2. 减小 `batch_size` (在代码中默认较大，可修改 `DeepCFR` 构造函数参数)。
    3. 减小网络层数 (如回到 64x64)。

### Q: 6人局模型表现不如 2人局？
*   **原因**: 6人局复杂度是指数级增长的。
*   **解决**: 需要指数级增加的训练资源。2000 次迭代对于 6 人局可能只是起步，可能需要 5000+ 次迭代才能达到较强水平。

## 8. 文件结构

```
.
├── train_deep_cfr_texas.py      # DeepCFR 训练主脚本
├── inference_simple.py          # 快速推理/自对弈脚本
├── evaluate_models_head_to_head.py # 模型对战评测脚本
├── play_interactive.py          # 人机交互对战脚本
├── analyze_training.py          # 训练日志分析与对比脚本
├── deep_cfr_simple_feature.py   # 策略网络特征提取模块
├── models/                      # 模型保存目录
│   └── deepcfr_texas_.../       # 每次训练的独立目录
│       ├── config.json          # 训练配置
│       ├── *_policy_network.pt  # 策略网络权重
│       └── *_history.json       # 训练日志
└── train_texas_holdem_mccfr.py  # MCCFR 训练脚本
```
