# Texas Hold'em DeepCFR Solver (基于 OpenSpiel)

这是一个针对德州扑克（No-Limit Texas Hold'em）优化的 Deep CFR 求解器。项目基于 DeepMind 的 OpenSpiel 框架，添加了多进程并行训练、自定义特征工程、实时评估和交互式对战功能。

## 📋 目录

1. [安装与环境准备](#0-安装与环境准备-installation)
2. [项目更新与架构演进](#1-项目更新与架构演进-architecture--updates)
3. [核心功能与优化](#2-核心功能与优化-core-features)
4. [训练](#3-训练-training)
5. [推理与自对弈](#4-推理与自对弈-inference--self-play)
6. [模型对比评测](#5-模型对比评测-head-to-head-evaluation)
7. [交互式对战](#6-交互式对战-interactive-play)
8. [训练日志分析](#7-训练日志分析-log-analysis)
9. [文件结构](#8-文件结构)

---

## 0. 安装与环境准备 (Installation)

### 系统依赖
```bash
./install.sh
```

### Python 环境
建议使用 Conda (Python 3.9 - 3.12):
```bash
conda create -n open_spiel python=3.11
conda activate open_spiel
pip install -r requirements.txt
```

### 编译 OpenSpiel
```bash
pip install .
```

---

## 1. 项目更新与架构演进 (Architecture & Updates)

### 🚀 2025-12-08 最新架构升级

本项目已经历了多次核心迭代，解决了原始 DeepCFR 算法在 6 人局大规模场景下的多个瓶颈。

#### **1. 并行化架构重构 (Parallel DeepCFR)**
*   **问题**: 原生 DeepCFR 仅支持简单的 GPU 数据并行，CPU 游戏树遍历（采样）成为严重瓶颈。
*   **解决方案**: 实现了 **Master-Worker 架构 (`deep_cfr_parallel.py`)**。
    *   **Worker (CPU)**: N 个 Worker 进程并行进行 Monte Carlo 树遍历，生产样本。
    *   **Master (GPU)**: 主进程专注于从共享缓冲区采样并训练神经网络。
    *   **健壮性升级**: 
        *   新增 **Worker 存活监控**：主进程实时监测 Worker 状态，一旦发现 Worker 异常退出（如 OOM），立即抛出异常停止训练，防止主进程死锁空转。
        *   异常堆栈捕获：Worker 进程增加全局异常捕获，确保错误日志不丢失。
*   **效果**: 训练吞吐量提升 **7.8x** (16核 CPU)，彻底解耦计算密集型与 IO 密集型任务。

#### **2. 特征工程增强 (Feature Engineering)**
*   **问题**: 原始 InfoState 过于稀疏，且对大额筹码（如 20000）不敏感（数值未归一化）。
*   **解决方案**: 
    *   **Simple Feature 模式**: 在原始 InfoState 后拼接 **7 维专家特征**（位置优势、EHS 手牌强度、下注统计）。
    *   **自动特征归一化**: 自动读取游戏配置的 `stack`，将所有金额类特征（包括原始输入中的 `sizings`）归一化到 `[0, 1]`，解决了模型对大筹码数值脱敏的问题。

#### **3. 6人局专项适配**
*   **网络扩容**: 策略网络从 `64x64` 升级为 **`256x3`** 或 **`1024x4`**，以拟合 6 人局复杂的博弈逻辑。
*   **动作空间**: 采用 **`fchpa`** (Fold, Call, Half-Pot, Pot, All-in) 抽象，引入半池下注。

---

## 2. 核心功能与优化 (Core Features)

*   **多进程并行训练**: 真正的 CPU 多核利用。
*   **多 GPU 加速**: 支持 PyTorch `DataParallel`，单机多卡训练。
*   **增量式 Checkpoint**: 训练过程中无卡顿保存模型，支持从任意 Checkpoint 完美恢复训练 (`--resume`)。
*   **实时评估**: 训练中定期进行“策略熵”监控和“随机对战测试”，即使跳过 NashConv 也能掌握训练趋势。
*   **交互式对战**: 提供人类 vs AI 的实战接口，支持实时显示 AI 思考概率。

---

## 3. 训练 (Training)

### 推荐命令 (单 GPU 版)
针对 RTX 4090 等高性能显卡，建议使用更大的网络和缓冲区以获得更强的策略。

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train_deep_cfr_texas.py \
    --num_players 6 \
    --betting_abstraction fchpa \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 4000000 \
    --num_iterations 2000 \
    --num_traversals 100 \
    --learning_rate 0.001 \
    --batch_size 4096 \
    --eval_interval 100 \
    --checkpoint_interval 100 \
    --skip_nashconv \
    --save_prefix deepcfr_texas_6p_single \
    > train_single_gpu.log 2>&1 &
```

### 推荐命令 (多 GPU 版 + Checkpoint)
支持多 GPU 并行训练和中间 checkpoint 保存，防止长时间训练中断丢失进度。

```bash
# 使用 4 张 GPU 并行训练，每 100 次迭代保存一次 checkpoint
nohup python train_deep_cfr_texas.py \
    --num_players 6 \
    --betting_abstraction fchpa \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 4000000 \
    --num_iterations 2000 \
    --num_traversals 100 \
    --learning_rate 0.001 \
    --batch_size 4096 \
    --eval_interval 100 \
    --skip_nashconv \
    --multi_gpu \
    --gpu_ids 0 1 2 3 \
    --checkpoint_interval 100 \
    --save_prefix deepcfr_texas_6p_multi_gpu \
    > train_multi_gpu.log 2>&1 &
```

**Checkpoint 说明**:
- Checkpoint 保存在 `models/<save_prefix>/checkpoints/` 目录下
- 文件命名格式: `*_iter{N}.pt`（如 `deepcfr_texas_policy_network_iter200.pt`）
- 训练被中断（Ctrl+C）时会自动保存当前进度
- 最终模型保存在主目录，不带 `_iter` 后缀

### 推荐命令 (多进程并行版 - 真正的并行化) ⭐推荐
使用多个 CPU 进程并行遍历游戏树，充分利用多核 CPU，显著提升训练速度。

#### 针对 4张 4090 显卡的高性能配置 (推荐)
```bash
nohup python deep_cfr_parallel.py \
    --num_players 6 \
    --num_iterations 2000 \
    --num_traversals 500 \
    --num_workers 16 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 50 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 10 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 2000000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_6p > train_parallel.log 2>&1 &
```

#### 通用配置 (单卡/少核)
```bash
nohup python deep_cfr_parallel.py \
    --num_players 6 \
    --num_iterations 20000 \
    --num_workers 8 \
    --num_traversals 500 \
    --batch_size 4096 \
    --memory_capacity 2000000 \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --use_gpu \
    --gpu_ids 0 \
    --eval_interval 100 \
    --checkpoint_interval 100 \
    --skip_nashconv \
    --save_prefix test_parallel
```

#### 自定义盲注和筹码配置示例
```bash
# 5人场，自定义盲注和筹码
nohup python deep_cfr_parallel.py \
    --num_players 5 \
    --blinds "100 200 0 0 0" \
    --stack_size 50000 \
    --num_iterations 20000 \
    --num_traversals 1600 \
    --num_workers 16 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 50 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 2000000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_5p_custom \
    > train_parallel_5p.log 2>&1 &

#### 续训脚本（从 checkpoint 恢复训练）
```bash
# 从之前的训练目录恢复训练
# 会自动加载最新的 checkpoint 和配置（玩家数、网络结构、盲注、筹码等）
# 可以覆盖训练超参数（如 batch_size, learning_rate, num_iterations）
nohup python deep_cfr_parallel.py \
    --resume models/deepcfr_parallel_5p_custom \
    --num_iterations 20000 \
    --num_workers 16 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 50 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --memory_capacity 2000000 \
    > train_parallel_5p_resume.log 2>&1 &
```

**续训说明**:
- `--resume` 会自动从 `config.json` 加载：玩家数、网络结构、遍历次数、盲注、筹码、下注抽象等
- 可以覆盖的训练超参数：`--num_iterations`, `--batch_size`, `--learning_rate`, `--memory_capacity` 等
- 会自动找到最新的 checkpoint 并从中继续训练
- 建议使用不同的日志文件（如 `train_parallel_5p_resume.log`）以便区分

# 2人场高额桌配置
nohup python deep_cfr_parallel.py \
    --num_players 2 \
    --blinds "200 100" \
    --stack_size 10000 \
   --num_iterations 2000 \
    --num_traversals 500 \
    --num_workers 16 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 50 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 2000000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_2p_high_stakes \
    > train_parallel_2p.log 2>&1 &
```

**多进程并行说明**:
- 多个 Worker 进程并行遍历游戏树（CPU 密集型）
- 主进程在 GPU 上训练神经网络，支持多 GPU DataParallel
- N 个 Worker 可以获得接近 N 倍的遍历速度
- 适合多核 CPU 服务器，比纯 DataParallel 更高效
- 支持 `--skip_nashconv` 跳过 NashConv 计算（6人局强烈建议）
- 支持 `--checkpoint_interval` 保存中间 checkpoint
- 支持 `--resume` 从 checkpoint 恢复训练
- 训练中断时自动保存当前进度

```bash
nohup python deep_cfr_parallel.py \
    --resume models/deepcfr_stable_run \
    --memory_capacity 2000000  \
    --num_iterations 30000 \
    --num_workers 16 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --checkpoint_interval 50 \
    --eval_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv > train_parallel_resume_v8.log 2>&1 &
```

**参数建议**:
- `--num_workers`: 建议设为 CPU 核心数的一半到全部（如 8-16）
- `--batch_size`: 多 GPU 时建议 4096+，充分利用显存
- `--gpu_ids`: 指定多张 GPU，如 `0 1 2 3` 使用 4 张卡
- `--blinds`: 如果不指定，会根据玩家数量自动生成：
  - 2人场：`"100 50"` (BB=100, SB=50)
  - 多人场：`"50 100 0 0 0 0"` (SB=50, BB=100, 其他=0)
- `--stack_size`: 如果不指定，默认每个玩家 2000 筹码
- `--resume`: 指定要恢复的模型目录，自动加载最新 checkpoint 和关键参数（玩家数、网络结构、遍历次数、盲注、筹码等）
- `--num_test_games`: 评估时的测试对局数量。6人局建议 50-100，如果对局失败率较高可适当增加

**盲注和筹码配置说明**:
- `--blinds` 和 `--stack_size` 参数会在训练时保存到 `config.json` 中
- 恢复训练时，如果命令行未指定这些参数，会自动从 `config.json` 加载
- 如果命令行显式指定了这些参数，会优先使用命令行参数（允许覆盖配置）

### 关键参数说明

| 参数 | 默认值 | 推荐值 (6人局) | 说明 |
| :--- | :--- | :--- | :--- |
| `--betting_abstraction` | `fcpa` | **`fchpa`** | 下注抽象。`fchpa` 包含半池加注(Half-pot)，策略更灵活。 |
| `--policy_layers` | `64 64` | **`256 256 256`** | 策略网络结构。6人局状态复杂，建议3层256节点。 |
| `--advantage_layers` | `32 32` | **`256 256 256`** | 优势网络结构。用于估计后悔值，建议与策略网络相同。 |
| `--memory_capacity` | `1e6` | **`4e6` (400万)** | 经验回放缓冲区。越大越好，防止模型遗忘早期策略。 |
| `--num_iterations` | `100` | **`2000`+** | 总迭代次数。DeepCFR 收敛较慢，需要较多迭代。 |
| `--num_traversals` | `20` | **`100`** | 每次迭代采样的轨迹数。增加此值可减少方差，使训练更稳定。 |
| `--learning_rate` | `1e-3` | **`1e-3`** | 学习率。 |
| `--batch_size` | `2048` | **`4096`** | 训练批量大小。多 GPU 时越大利用率越高。 |
| `--multi_gpu` | `False` | `True` | 启用多 GPU 并行训练 (DataParallel)。 |
| `--gpu_ids` | `None` | `0 1 2 3` | 指定使用的 GPU ID 列表。不指定则使用所有可用 GPU。 |
| `--checkpoint_interval` | `0` | **`100`** | Checkpoint 保存间隔。0 表示不保存中间 checkpoint。 |
| `--skip_nashconv` | `False` | **`True`** | 跳过 NashConv 计算。6人局强烈建议开启。 |

**多进程并行版参数** (`deep_cfr_parallel.py`):

| 参数 | 默认值 | 推荐值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--num_players` | `2` | **`6`** | 玩家数量。支持 2-10 人。 |
| `--num_workers` | `4` | **`16`** | Worker 进程数量。建议设为 CPU 核心数。 |
| `--num_traversals` | `100` | **`500`** | 每次迭代遍历次数。多 Worker 时可设更大值。 |
| `--batch_size` | `2048` | **`4096`** | 训练批量大小。多 GPU 时越大利用率越高。 |
| `--policy_layers` | `128 128` | **`256 256 256`** | 策略网络结构。多 GPU 可用更大网络。 |
| `--advantage_layers` | `128 128` | **`256 256 256`** | 优势网络结构。多 GPU 可用更大网络。 |
| `--memory_capacity` | `1e6` | **`2e6`** | 经验回放缓冲区大小。 |
| `--learning_rate` | `1e-3` | **`1e-3`** | 学习率。 |
| `--blinds` | `None` | - | 盲注配置。格式：`"小盲 大盲"` (2人场) 或 `"50 100 0 0 0 0"` (多人场完整配置)。不指定时根据玩家数量自动生成。 |
| `--stack_size` | `None` | **`2000`** | 每个玩家的初始筹码。不指定时默认 2000。 |
| `--use_gpu` | `False` | **`True`** | 使用 GPU 训练网络。 |
| `--gpu_ids` | `None` | **`0 1 2 3`** | 指定多张 GPU，启用 DataParallel 并行训练。 |
| `--eval_interval` | `10` | **`100`** | 评估间隔。每 N 次迭代评估一次策略质量。 |
| `--eval_with_games` | `False` | `True` | 评估时运行测试对局。 |
| `--num_test_games` | `50` | **`50-100`** | 评估时的测试对局数量。6人局可能因复杂度导致部分对局失败，可适当增加此值。 |
| `--checkpoint_interval` | `0` | **`50`** | Checkpoint 保存间隔。 |
| `--skip_nashconv` | `False` | **`True`** | 跳过 NashConv 计算。6人局强烈建议开启。 |
| `--resume` | `None` | - | 从指定目录恢复训练。自动从 config.json 加载关键参数（玩家数、网络结构、遍历次数、盲注、筹码等）。 |

**性能对比** (6人德扑, 5次迭代, 50次遍历):

| 版本 | 时间 | 加速比 |
| :--- | :--- | :--- |
| `train_deep_cfr_texas.py` (多GPU版) | 65.8 秒 | 1x |
| `deep_cfr_parallel.py` (16 Workers) | 8.48 秒 | **7.8x** |

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
# 推荐方式：只传模型目录（自动从 config.json 读取配置）
python inference_simple.py \
    --model_dir models/deepcfr_parallel_6p \
    --num_games 1000 \
    --use_gpu

# 支持 checkpoint 目录（自动选择最新的 checkpoint）
python inference_simple.py \
    --model_dir models/deepcfr_parallel_6p/checkpoints/iter_1750 \
    --num_games 1000 \
    --use_gpu
```

**结果解读**:
*   **平均收益**: 长期来看，所有位置的平均收益之和应接近 0。
*   **位置优势**: 正常情况下，后位（Button, CO）收益应高于前位（SB, BB, UTG）。
*   **胜率**: 通常在 15% - 25% 之间。

---

## 5. 模型对比评测 (Head-to-Head Evaluation)

使用 `evaluate_models_head_to_head.py` 让两个不同的模型进行对战（例如：新模型 vs 旧模型）。

```bash
# 对比两个不同的模型目录
python evaluate_models_head_to_head.py \
    --model_a models/deepcfr_texas_6p_fchpa_large \
    --model_b models/deepcfr_texas_6p_fchpa_baseline \
    --num_games 2000 \
    --use_gpu

# 支持 checkpoint 目录（对比不同迭代的模型）
python evaluate_models_head_to_head.py \
    --model_a models/deepcfr_parallel_6p/checkpoints/iter_1750 \
    --model_b models/deepcfr_parallel_6p/checkpoints/iter_1600 \
    --num_games 1000 \
    --use_gpu
```

**注意**: 两个模型必须具有**相同的游戏配置**（玩家数、下注抽象必须一致）。脚本会自动进行两轮测试（交换座位），以消除位置优势带来的偏差。

### 批量评估所有 Checkpoint

使用 `evaluate_all_checkpoints.py` 自动评估所有 checkpoint，找出最佳模型：

```bash
# 评估所有 checkpoint，每个测试 500 局
python evaluate_all_checkpoints.py \
    --model_dir models/deepcfr_parallel_6p \
    --num_games 500 \
    --use_gpu \
    --top_k 10

# 保存结果到文件
python evaluate_all_checkpoints.py \
    --model_dir models/deepcfr_parallel_6p \
    --num_games 500 \
    --use_gpu \
    --output checkpoint_evaluation.json
```

**输出说明**:
- 按玩家0平均收益排序，显示前 K 个最佳模型
- 显示每个 checkpoint 的迭代号、平均收益、胜率、收益方差等指标
- 收益方差越小，说明策略越平衡（所有位置表现相近）

---

## 6. 交互式对战 (Interactive Play)

使用 `play_interactive.py` 亲自与训练好的模型对战。

```bash
# 作为玩家 0 (SB) 与模型对战（交互模式，一局一问是否继续）
python play_interactive.py \
    --model_dir models/deepcfr_stable_run/checkpoints/iter_10900 \
    --num_players 6 \
    --human_player 0

# 自动自对弈模式：人类座位也由模型控制，连续打 10 局并输出详细日志
python play_interactive.py \
    --model_dir models/deepcfr_parallel_6p/checkpoints/iter_16550 \
    --num_players 6 \
    --human_player 0 \
    --auto_play \
    --num_games 10 \
    > play_interactive_16550_10games.log
```

### 游戏流程
1.  **启动**: 脚本自动检测模型配置，加载环境。
2.  **状态**: 显示当前轮次（Preflop/Flop/Turn/River）、公共牌、底池、你的手牌。
3.  **行动**:
    - 交互模式：输入数字选择动作（弃牌/跟注/加注）。
    - 自动模式（`--auto_play`）：人类位置也由模型决策，并打印该状态下各动作的概率分布。
4.  **结束**: 结算收益，显示所有玩家手牌。

---

## 7. 训练日志分析 (Log Analysis)

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

### 关键指标解读
1.  **策略熵 (Policy Entropy)**: 应逐渐降低，表示策略在收敛。
2.  **缓冲区大小 (Buffer Size)**: 应持续增长，表示探索了更多状态。
3.  **测试对局 (Test Games)**: 胜率应稳定在 50% 以上（对随机策略）或与其他模型对战胜率提升。

---

## 10. 文件结构

```
.
├── train_deep_cfr_texas.py      # DeepCFR 训练主脚本 (支持多 GPU)
├── deep_cfr_parallel.py         # 多进程并行 DeepCFR 训练脚本 (推荐)
├── inference_simple.py          # 快速推理/自对弈脚本 (支持 checkpoint)
├── evaluate_models_head_to_head.py # 模型对战评测脚本 (支持 checkpoint)
├── evaluate_all_checkpoints.py  # 批量评估所有 checkpoint，找出最佳模型
├── play_interactive.py          # 人机交互对战脚本 (支持 checkpoint)
├── analyze_training.py          # 训练日志分析与对比脚本
├── deep_cfr_simple_feature.py   # 策略网络特征提取模块 (支持多 GPU)
├── deep_cfr_with_feature_transform.py # 复杂特征转换模块 (支持多 GPU)
├── models/                      # 模型保存目录
│   └── deepcfr_texas_.../       # 每次训练的独立目录
│       ├── config.json          # 训练配置 (含 multi_gpu, gpu_ids)
│       ├── *_policy_network.pt  # 策略网络权重 (用于推理)
│       ├── checkpoints/         # Checkpoint 目录
│       │   └── iter_N/          # 迭代 N 的 checkpoint
│       │       ├── *_policy_network_iterN.pt
│       │       └── *_advantage_player_*_iterN.pt
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
