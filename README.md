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

### 推荐命令 (单 GPU 版)
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
    --skip_nashconv \
    --save_prefix deepcfr_texas_6p_fchpa_large \
    > train_log_large.log 2>&1 &
```

### 推荐命令 (多 GPU 版 + Checkpoint)
支持多 GPU 并行训练和中间 checkpoint 保存，防止长时间训练中断丢失进度。

```bash
# 使用 4 张 GPU 并行训练，每 200 次迭代保存一次 checkpoint
nohup python train_deep_cfr_texas.py \
    --num_players 6 \
    --betting_abstraction fchpa \
    --policy_layers 128 128 \
    --advantage_layers 128 128 \
    --memory_capacity 4000000 \
    --num_iterations 20 \
    --num_traversals 40 \
    --learning_rate 0.0005 \
    --eval_interval 10 \
    --eval_with_games \
    --save_prefix deepcfr_texas_6p_multi_gpu \
    --skip_nashconv \
    --multi_gpu \
    --gpu_ids 0 1 2 3 \
    --checkpoint_interval 10 \
    > train_log_multi_gpu.log 2>&1 &
```

**Checkpoint 说明**:
- Checkpoint 保存在 `models/<save_prefix>/checkpoints/` 目录下
- 文件命名格式: `*_iter{N}.pt`（如 `deepcfr_texas_policy_network_iter200.pt`）
- 训练被中断（Ctrl+C）时会自动保存当前进度
- 最终模型保存在主目录，不带 `_iter` 后缀

### 推荐命令 (多进程并行版 - 真正的并行化) ⭐推荐
使用多个 CPU 进程并行遍历游戏树，充分利用多核 CPU，显著提升训练速度。

```bash
# 多进程并行 + 多 GPU（4张卡，16个Worker，推荐配置）
nohup python deep_cfr_parallel.py \
    --num_players 6 \
    --num_iterations 1000 \
    --num_traversals 500 \
    --num_workers 16 \
    --batch_size 4096 \
    --sync_interval 50 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --checkpoint_interval 50 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_network_layers 256 256 256 \
    --advantage_network_layers 256 256 256 \
    --memory_capacity 2000000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_6p \
    > train_parallel.log 2>&1 &
```

```bash
# 快速测试命令（验证多GPU是否正常工作）
python deep_cfr_parallel.py \
    --num_players 6 \
    --num_iterations 10 \
    --num_traversals 100 \
    --num_workers 4 \
    --batch_size 1024 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --skip_nashconv
```

**多进程并行说明**:
- 多个 Worker 进程并行遍历游戏树（CPU 密集型）
- 主进程在 GPU 上训练神经网络，支持多 GPU DataParallel
- N 个 Worker 可以获得接近 N 倍的遍历速度
- 适合多核 CPU 服务器，比纯 DataParallel 更高效
- 支持 `--skip_nashconv` 跳过 NashConv 计算（6人局强烈建议）
- 支持 `--checkpoint_interval` 保存中间 checkpoint
- 训练中断时自动保存当前进度

**参数建议**:
- `--num_workers`: 建议设为 CPU 核心数的一半到全部（如 8-16）
- `--batch_size`: 多 GPU 时建议 4096+，充分利用显存
- `--sync_interval`: 每 N 次遍历同步网络参数，建议 50-100
- `--gpu_ids`: 指定多张 GPU，如 `0 1 2 3` 使用 4 张卡

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
| `--multi_gpu` | `False` | - | 启用多 GPU 并行训练 (DataParallel)。 |
| `--gpu_ids` | `None` | `0 1 2 3` | 指定使用的 GPU ID 列表。不指定则使用所有可用 GPU。 |
| `--checkpoint_interval` | `0` | **`200`** | Checkpoint 保存间隔。0 表示不保存中间 checkpoint。 |

**多进程并行版参数** (`deep_cfr_parallel.py`):

| 参数 | 默认值 | 推荐值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--num_workers` | `4` | **`8-16`** | Worker 进程数量。建议设为 CPU 核心数。 |
| `--batch_size` | `2048` | **`4096`** | 训练批量大小。多 GPU 时越大利用率越高。 |
| `--sync_interval` | `100` | **`50`** | 每 N 次遍历同步网络参数到 Worker。 |
| `--skip_nashconv` | `False` | **`True`** | 跳过 NashConv 计算。6人局强烈建议开启。 |
| `--use_gpu` | `True` | `True` | 使用 GPU 训练网络。 |
| `--gpu_ids` | `None` | **`0 1 2 3`** | 指定多张 GPU，启用 DataParallel 并行训练。 |
| `--policy_network_layers` | `128 128` | **`256 256 256`** | 策略网络结构。多 GPU 可用更大网络。 |
| `--advantage_network_layers` | `128 128` | **`256 256 256`** | 优势网络结构。多 GPU 可用更大网络。 |

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

### 游戏流程
1.  **启动**: 脚本自动检测模型配置，加载环境。
2.  **状态**: 显示当前轮次（Preflop/Flop/Turn/River）、公共牌、底池、你的手牌。
3.  **行动**: 输入数字选择动作（弃牌/跟注/加注）。
4.  **结束**: 结算收益，显示所有玩家手牌。

### 故障排除
*   **"RuntimeError: size mismatch"**: 检查模型是否使用了不同的 `bettingAbstraction` 或特征配置。确保 `config.json` 存在且正确。
*   **"模型加载失败"**: 检查模型文件（.pt）是否存在于目录中。

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
    4. 使用多 GPU 分摊显存压力 (`--multi_gpu --gpu_ids 0 1 2 3`)。

### Q: 6人局模型表现不如 2人局？
*   **原因**: 6人局复杂度是指数级增长的。
*   **解决**: 需要指数级增加的训练资源。2000 次迭代对于 6 人局可能只是起步，可能需要 5000+ 次迭代才能达到较强水平。

### Q: 多 GPU 训练效果如何？
*   **说明**: 多 GPU 使用 PyTorch 的 `DataParallel` 实现，主要加速网络的前向/反向传播阶段。
*   **注意**: DeepCFR 的游戏树遍历（`_traverse_game_tree`）仍在 CPU 上进行，因此多 GPU 主要加速 `_learn_advantage_network()` 和 `_learn_strategy_network()` 阶段。
*   **建议**: 增大 `memory_capacity` 以积累更多样本，使训练批次更大，多 GPU 效果更明显。

### Q: 如何真正利用多核 CPU 加速训练？
*   **解决**: 使用 `deep_cfr_parallel.py` 多进程并行版本。
*   **原理**: 多个 Worker 进程并行遍历游戏树（CPU 密集型），主进程在 GPU 上训练网络。
*   **效果**: N 个 Worker 可以获得接近 N 倍的遍历速度，显著提升整体训练效率。
*   **命令**: `python deep_cfr_parallel.py --num_workers 8 --num_players 6 ...`

---

## 8. 进阶主题：特征转换 (Feature Engineering)

本项目默认使用**简单特征版本**（Simple Feature），即将 7 维手动特征（如手牌强度、位置优势）直接拼接到原始信息状态张量中。

### 两种模式
1.  **简单版本 (推荐)**: `info_state (281) + manual_features (7) -> MLP`。计算快，效果好。
2.  **复杂版本**: `info_state + features -> Transform Layer -> MLP`。包含可学习特征，适合更深的研究。

### 代码调用
```python
from deep_cfr_simple_feature import DeepCFRSimpleFeature
# 自动启用特征拼接
solver = DeepCFRSimpleFeature(game, policy_network_layers=(128, 128), ...)
```

---

## 9. 进阶主题：损失与评估 (Loss & Evaluation)

### 损失计算
DeepCFR 的损失函数包含 `sqrt(iteration)` 加权项。因此，**随着迭代次数增加，损失值自然会增长**。
*   **不要**仅凭损失值绝对值判断训练是否恶化。
*   应关注损失值的相对趋势。

### 评估指标
判断训练效果的最佳指标：
1.  **策略熵 (Policy Entropy)**: 应逐渐降低，表示策略在收敛。
2.  **缓冲区大小 (Buffer Size)**: 应持续增长，表示探索了更多状态。
3.  **测试对局 (Test Games)**: 胜率应稳定在 50% 以上（对随机策略）或与其他模型对战胜率提升。
4.  **NashConv**: 最准确但计算极其耗时，大规模训练时建议跳过 (`--skip_nashconv`)。

---

## 10. 文件结构

```
.
├── train_deep_cfr_texas.py      # DeepCFR 训练主脚本 (支持多 GPU)
├── deep_cfr_parallel.py         # 多进程并行 DeepCFR 训练脚本 (推荐)
├── inference_simple.py          # 快速推理/自对弈脚本
├── evaluate_models_head_to_head.py # 模型对战评测脚本
├── play_interactive.py          # 人机交互对战脚本
├── analyze_training.py          # 训练日志分析与对比脚本
├── deep_cfr_simple_feature.py   # 策略网络特征提取模块 (支持多 GPU)
├── deep_cfr_with_feature_transform.py # 复杂特征转换模块 (支持多 GPU)
├── models/                      # 模型保存目录
│   └── deepcfr_texas_.../       # 每次训练的独立目录
│       ├── config.json          # 训练配置 (含 multi_gpu, gpu_ids)
│       ├── *_policy_network.pt  # 策略网络权重 (用于推理)
│       ├── *_advantage_player_*.pt # 优势网络权重 (仅用于训练)
│       └── *_history.json       # 训练日志
└── train_texas_holdem_mccfr.py  # MCCFR 训练脚本

## 11. 附录：DeepCFR 网络结构说明

DeepCFR 包含两种类型的神经网络，它们作用不同：

### 1. 优势网络 (Advantage Network)
- **数量**: 每个玩家 1 个 (6人局有 6 个)
- **作用**: 预测每个动作的**后悔值 (Regret)**。它指导算法在训练过程中如何改进策略。
- **使用场景**: **仅训练阶段**。推理时不需要。

### 2. 策略网络 (Policy Network)
- **数量**: 所有玩家共用 1 个
- **作用**: 拟合所有迭代产生的**平均策略**。根据 DeepCFR 理论，平均策略会收敛到纳什均衡。
- **使用场景**: **推理、对战阶段**。这是最终产出的模型文件。
```
