# Texas Hold'em DeepCFR Solver (基于 OpenSpiel)

这是一个针对德州扑克（No-Limit Texas Hold'em）优化的 Deep CFR 求解器。项目基于 DeepMind 的 OpenSpiel 框架，添加了多进程并行训练、自定义特征工程、实时评估和交互式对战功能。

## 📋 目录

1. [安装与环境准备](#0-安装与环境准备-installation)
2. [项目更新与架构演进](#1-项目更新与架构演进-architecture--updates)
3. [核心功能与优化](#2-核心功能与优化-core-features)
4. [训练](#3-训练-training)
5. [推理与自对弈](#4-推理与自对弈-inference--self-play)
6. [模型对比评测](#5-模型对比评测-head-to-head-evaluation)
7. [API 接口服务](#6-api-接口服务-api-server)
8. [交互式对战](#7-交互式对战-interactive-play)
9. [训练日志分析](#8-训练日志分析-log-analysis)
10. [文件结构](#9-文件结构)

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
*   **实时评估**: 训练中定期进行"策略熵"监控和"随机对战测试"，即使跳过 NashConv 也能掌握训练趋势。
*   **TensorBoard 可视化**: 自动记录训练损失、样本数量、评估指标等，支持实时查看训练曲线。
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
    --eval_interval 100 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 4000000 \
    --queue_maxsize 30000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_5p_custom_v2 \
    > train_parallel_5p_v2_6.log 2>&1 &


# 6人场
nohup python deep_cfr_parallel.py \
    --num_players 6 \
    --blinds "100 200 0 0 0 0" \
    --stack_size 50000 \
    --num_iterations 20000 \
    --num_traversals 1600 \
    --num_workers 16 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 100 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --policy_layers 256 256 256 \
    --advantage_layers 256 256 256 \
    --memory_capacity 4000000 \
    --queue_maxsize 30000 \
    --betting_abstraction fchpa \
    --save_prefix deepcfr_parallel_6p_custom_v3 \
    > train_parallel_6p_v3.log 2>&1 &

#### 续训脚本（从 checkpoint 恢复训练）
```bash
# 从之前的训练目录恢复训练
# 会自动加载最新的 checkpoint 和配置（玩家数、网络结构、盲注、筹码等）
# 可以覆盖训练超参数（如 batch_size, learning_rate, num_iterations）
现在的api_server.py 是不是不支持我最新的只有1个自己添加的特征的模型，能不能兼容一下。然后有个接口可以看目前线上

nohup python deep_cfr_parallel.py \
    --resume models/deepcfr_parallel_5p_custom_v2_20251219_111930 \
    --num_iterations 20000 \
    --num_workers 16 \
    --num_traversals 1600 \
    --batch_size 4096 \
    --use_gpu \
    --gpu_ids 0 1 2 3 \
    --eval_interval 100 \
    --checkpoint_interval 100 \
    --eval_with_games \
    --num_test_games 100 \
    --skip_nashconv \
    --learning_rate 0.001 \
    --memory_capacity 4000000 \
    --queue_maxsize 30000 \
    > train_parallel_5p_v2_resume_2.log 2>&1 &
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

### TensorBoard 可视化监控

训练过程中会自动记录训练指标到 TensorBoard，方便实时查看训练曲线和监控训练进度。

#### 安装 TensorBoard

```bash
pip install tensorboard
```

#### 启动训练（自动记录）

训练时会自动在模型目录下创建 `tensorboard_logs/` 目录并记录日志：

```bash
python deep_cfr_parallel.py \
    --num_players 5 \
    --num_iterations 20000 \
    --save_prefix deepcfr_parallel_5p_custom \
    ...
```

训练开始时会显示：
```
✓ TensorBoard日志目录: models/deepcfr_parallel_5p_custom/tensorboard_logs
  查看命令: tensorboard --logdir models/deepcfr_parallel_5p_custom/tensorboard_logs
```

#### 查看训练曲线

在另一个终端启动 TensorBoard：

```bash
# 查看单个模型的训练日志
tensorboard --logdir models/deepcfr_parallel_5p_custom/tensorboard_logs

# 或者查看多个模型的对比（推荐）
tensorboard --logdir models/
```

然后在浏览器打开 `http://localhost:6006` 即可查看：

**记录的指标包括**：

1. **损失曲线** (`Loss/`):
   - `Advantage_Player_0`, `Advantage_Player_1`, ... - 每个玩家的优势网络损失
   - `Policy` - 策略网络损失

2. **训练指标** (`Metrics/`):
   - `Total_Advantage_Samples` - 总优势样本数量
   - `Strategy_Buffer_Size` - 策略缓冲区大小
   - `Policy_Entropy` - 策略熵（策略的随机性）

3. **评估结果** (`Evaluation/`) - 如果启用了 `--eval_with_games`:
   - `Avg_Return` - 平均回报（vs 随机策略）
   - `Win_Rate` - 胜率（vs 随机策略）

**使用技巧**：
- 使用对数刻度查看损失曲线（TensorBoard 默认支持）
- 可以同时加载多个训练日志进行对比
- 损失增长是正常的（因为使用了 `sqrt(iteration)` 加权）
- 重要的是观察损失的趋势：应该逐渐稳定或缓慢增长
- 如果损失突然大幅增长，可能是训练不稳定，需要调整学习率或网络结构

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

## 6. API 接口服务 (API Server)

使用 `api_server.py` 提供 RESTful API 接口，供前后端调用获取推荐动作。

### 安装 API 依赖

```bash
pip install -r requirements_api.txt
```

### 启动 API 服务器

#### 单模型模式（向后兼容）

```bash
# 使用 CPU
python api_server.py --model_dir models/deepcfr_parallel_6p --host 0.0.0.0 --port 5000 --device cpu

# 使用 GPU（如果可用）
python api_server.py --model_dir models/deepcfr_parallel_6p --host 0.0.0.0 --port 5000 --device cuda
```

#### 多模型模式（推荐，支持5人场和6人场）

```bash
# 同时加载5人场和6人场模型
nohup python api_server.py \
  --model_5p models/deepcfr_parallel_5p_custom/checkpoints/iter_4100 \
  --model_6p models/deepcfr_6p_multi_20260116_171819/checkpoints/iter_6600 \
  --host 0.0.0.0 \
  --port 8826 \
  --device cpu > api_server_multi_model_8826.log 2>&1 &
```

**多模型模式说明**：
- API服务器会根据请求中的`blinds`/`stacks`长度自动选择对应场次的模型
- 5人场：`blinds`长度为5，`stacks`长度为5
- 6人场：`blinds`长度为6，`stacks`长度为6
- 支持同时加载多个场次的模型，无需重启即可切换

### API 端点

1. **健康检查**: `GET /api/v1/health`
2. **获取推荐动作**: `POST /api/v1/recommend_action`
3. **获取动作映射表**: `GET /api/v1/action_mapping`
4. **动态替换模型**: `POST /api/v1/reload_model` ⭐新增

### 请求格式

```json
{
  "player_id": 0,
  "hole_cards": [0, 12],
  "board_cards": [13, 26, 39],
  "action_history": [1, 1, 1, 1],
  "action_sizings": [0, 0, 0, 0],
  "blinds": [50, 100, 0, 0, 0, 0],
  "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
  "dealer_pos": 5
}
```

**关键字段说明**：
- `player_id` (必需): 当前需要推理的玩家ID（0-5）
  - **这是OpenSpiel内部的固定座位索引，不会因为Dealer轮转而改变**
  - Player 0 永远是座位0，Player 1 永远是座位1，以此类推
  - 但Player 0在不同局中可能扮演不同角色（Dealer/SB/BB/UTG等），这取决于`dealer_pos`
- `hole_cards` (必需): 当前玩家手牌，支持两种格式：
  - **数字格式（推荐）**：`[0, 12]` - 0-51的整数，数字已包含花色信息
    - 花色顺序：方块(Diamond)[0-12] -> 梅花(Clubs)[13-25] -> 红桃(Hearts)[26-38] -> 黑桃(Spade)[39-51]
  - **传统格式（兼容）**：`["As", "Kh"]` - Rank + Suit 字符串
- `board_cards` (必需): 公共牌，格式同上
- `action_history` (必需): 历史动作列表（**只包含玩家动作，不包含发牌动作**）
- `action_sizings` (可选): 每次动作的下注金额，与`action_history`一一对应
- `blinds` (可选): 盲注列表，如 `[50, 100, 0, 0, 0, 0]`
  - 如果传了，必须与`stacks`和`dealer_pos`一起传
- `stacks` (可选): 当前剩余筹码列表（不是初始筹码）
  - 如果传了，必须与`blinds`和`dealer_pos`一起传
- `dealer_pos` (必需，如果传了blinds和stacks): Dealer位置（0-5）
  - 用于确定当前局中每个座位的角色（Dealer/SB/BB/UTG等）
  - 如果不传且提供了blinds/stacks，API会返回错误

### 响应格式

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

### 动作映射

**fchpa抽象**（5个动作）：
- `0`: Fold（弃牌）
- `1`: Call/Check（跟注/过牌）
- `2`: Pot（加注到当前底池大小）
- `3`: All-in（全押）
- `4`: Half-Pot（加注到当前底池的一半）

### 使用示例

#### 获取推荐动作

**Python调用示例**：
```python
import requests

url = "http://localhost:5000/api/v1/recommend_action"
data = {
    "player_id": 0,
    "hole_cards": ["As", "Kh"],  # 或使用数字格式 [51, 38]
    "board_cards": ["2d", "3c", "4h"],  # 或使用数字格式 [0, 13, 26]
    "action_history": [1, 1, 2],
    "action_sizings": [0, 0, 100],
    "blinds": [50, 100, 0, 0, 0, 0],
    "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
    "dealer_pos": 5
}

response = requests.post(url, json=data)
result = response.json()
```

#### 动态替换模型 ⭐新增

**替换指定场次的模型**：

```bash
# 替换5人场模型（明确指定num_players=5）
curl -X POST http://localhost:8826/api/v1/reload_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "models/deepcfr_parallel_5p_custom_v3/checkpoints/iter_2900",
    "num_players": 5
  }'

# 替换6人场模型（明确指定num_players=6）
curl -X POST http://localhost:8826/api/v1/reload_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "models/deepcfr_6p_multi_20260116_171819/checkpoints/iter_114200",
    "num_players": 6
  }'



  # 替换6人场模型（明确指定num_players=6）
curl -X POST http://localhost:8828/api/v1/reload_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "models/deepcfr_6p_multi_v5.3_20260119_183722/checkpoints/iter_317600",
    "num_players": 6
  }'
```

**自动检测场次（从config.json读取）**：

```bash
# 不指定num_players，自动从config.json读取
curl -X POST http://localhost:8826/api/v1/reload_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "models/some_model"
  }'
```

**Python调用示例**：

```python
import requests

# 替换5人场模型
url = "http://localhost:8826/api/v1/reload_model"
data = {
    "model_dir": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",
    "num_players": 5,  # 可选：明确指定场次
    "device": "cpu"     # 可选：默认使用当前设备
}

response = requests.post(url, json=data)
result = response.json()
print(result)
# {
#   "success": true,
#   "message": "Model reloaded from models/xxx",
#   "model_dir": "models/xxx",
#   "device": "cpu",
#   "num_players": 5,
#   "loaded_models": {
#     "5": "models/deepcfr_parallel_5p_custom/checkpoints/iter_5000",
#     "6": "models/deepcfr_stable_run/checkpoints/iter_32000"
#   }
# }
```

**请求参数说明**：
- `model_dir` (必需): 模型目录路径
- `num_players` (可选): 明确指定场次（5或6）。如果不指定，从`config.json`自动检测
- `device` (可选): 设备类型（`cpu`或`cuda`），默认使用当前设备

**响应说明**：
- `success`: 是否成功
- `num_players`: 实际加载的场次
- `loaded_models`: 当前所有已加载的模型列表（key为场次，value为模型路径）

### 并发请求

**重要**：为了确保并发安全，**建议总是传递`blinds`和`stacks`参数**。这样每次请求都会创建新的游戏实例，完全隔离。

**测试并发**：
```bash
python test_concurrent_api.py
```

### 工作流程

API的工作流程：
1. **创建游戏实例**：根据`blinds`、`stacks`、`dealer_pos`创建游戏
2. **发牌**：根据`hole_cards`和`board_cards`的数量自动发牌
3. **应用历史动作**：按照`action_history`顺序应用，重建到当前状态
4. **动作推荐**：基于重建的状态进行AI推理

**详细工作流程**：请参考 [API_WORKFLOW_SUMMARY.md](API_WORKFLOW_SUMMARY.md)

### 关键说明

- `action_history` **只包含玩家动作，不包含发牌动作**（系统会自动处理发牌）
- 动作必须按游戏进行的时间顺序排列
- 支持数字格式（0-51）和传统格式（"As", "Kh"）的卡牌输入
- `action_sizings` 可选，如果不传则系统会根据动作ID自动计算下注金额
- `stacks` 传入的是**当前剩余筹码**，不是初始筹码
- `dealer_pos` 必需（如果传了blinds和stacks），用于正确计算firstPlayer和行动顺序
- `player_id` 是固定的座位索引（0-5），不会因为Dealer轮转而改变
- **位置角色计算**：`(player_id - dealer_pos) % num_players` 确定角色（BTN/SB/BB/UTG/MP/CO）
- **位置编码映射**：
  - Solver模型：位置编码映射已禁用，`dealer_pos`参数会被忽略
  - Standard Network：如果提供`dealer_pos`，会进行位置编码映射以匹配训练配置

---

## 7. 交互式对战 (Interactive Play)

### 7.1 基于API的Gradio界面

使用 `play_gradio_api.py` 提供基于API服务器的Web界面，所有位置的AI动作都通过API服务器获取：

```bash
# 1. 先启动API服务器（在另一个终端，推荐使用多模型模式）
python api_server.py \
  --model_5p models/deepcfr_parallel_5p_custom/checkpoints/iter_4100 \
  --model_6p models/deepcfr_stable_run/checkpoints/iter_31000 \
  --host 0.0.0.0 \
  --port 8826 \
  --device cpu

# 2. 启动Gradio界面
python play_gradio_api.py
```

**特性**：
- 所有位置的AI动作都通过API服务器获取
- 支持动态盲注和筹码配置
- 实时显示每个位置的API请求和响应（包括Dealer位置）
- 支持模型动态切换（通过API服务器的reload_model端点）
- 自动传递位置信息（`dealer_pos`）给API服务器
- **场次切换功能** ⭐新增：支持在UI中切换5人场和6人场

**场次切换**：
- UI界面提供场次选择Radio控件（5人场/6人场）
- 切换场次时自动更新游戏配置、筹码和盲注
- 5人场配置：`stacks=[50000]*5`, `dealer_pos=4`, `blinds=[100,200]`
- 6人场配置：`stacks=[2000]*6`, `dealer_pos=5`, `blinds=[50,100]`
- 切换后提示用户点击"开始新游戏"

**配置**：
- API服务器地址：默认 `http://localhost:8826/api/v1`（可在代码中修改 `API_BASE_URL`）
- Gradio端口：默认 `8823`（可在 `demo.launch()` 中修改）

**位置信息说明**：
- `player_id`：OpenSpiel内部的固定座位索引（0-5），不会因为Dealer轮转而改变
- `dealer_pos`：每局游戏的Dealer位置，用于确定每个座位的角色（Dealer/SB/BB/UTG等）
- 位置信息会自动从`TOURNAMENT_STATE`获取并传递给API服务器

### 7.2 传统交互式对战

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

## 8. 训练日志分析 (Log Analysis)

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

## 9. 文件结构

```
.
├── train_deep_cfr_texas.py      # DeepCFR 训练主脚本 (支持多 GPU)
├── deep_cfr_parallel.py         # 多进程并行 DeepCFR 训练脚本 (推荐)
├── inference_simple.py          # 快速推理/自对弈脚本 (支持 checkpoint)
├── evaluate_models_head_to_head.py # 模型对战评测脚本 (支持 checkpoint)
├── evaluate_all_checkpoints.py  # 批量评估所有 checkpoint，找出最佳模型
├── play_interactive.py          # 人机交互对战脚本 (支持 checkpoint)
├── api_server.py                # API 服务器 (提供 RESTful 接口)
├── test_api.py                  # API 测试脚本
├── API_USAGE.md                 # API 使用文档
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

## 10. 附录：DeepCFR 网络结构说明

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
