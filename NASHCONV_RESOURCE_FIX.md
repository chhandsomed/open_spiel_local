# NashConv 资源占用问题修复

> **提示**：这是详细的技术文档。快速开始请参考 [README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md)

## 问题描述

计算 NashConv 时，CPU 和内存都被占满，系统无法正常运行。

## 原因分析

1. **`policy.tabular_policy_from_callable` 会遍历所有信息集**
   - 对于德州扑克这样的大游戏，状态空间巨大（~10^18）
   - 会创建巨大的策略表格，消耗大量内存
   - 遍历过程消耗大量 CPU

2. **游戏树遍历**
   - Best Response 计算需要遍历整个游戏树
   - 对于大规模游戏，这是指数级复杂度

3. **没有资源限制**
   - 默认使用所有 CPU 核心
   - 没有内存限制
   - 没有超时机制

## 解决方案

### 1. 使用资源限制版本（推荐）

已更新 `nash_conv_gpu.py`，添加了资源限制功能：

```python
from nash_conv_gpu import nash_conv_lightweight

conv = nash_conv_lightweight(
    game,
    deep_cfr_solver,
    max_cpu_threads=2,    # 限制 CPU 线程数
    max_memory_gb=8,      # 限制内存使用（8GB）
    timeout_seconds=600,  # 10 分钟超时
    verbose=True,
    device=device
)
```

**默认限制**：
- CPU 线程数：2（避免占满所有核心）
- 内存：8GB（避免内存溢出）
- 超时：10 分钟（避免无限运行）

### 2. 训练时跳过 NashConv（最推荐）

对于大规模训练，**强烈建议跳过 NashConv 计算**：

```bash
python train_deep_cfr_texas.py --skip_nashconv
```

**原因**：
- 训练时不需要每次都计算 NashConv
- 可以训练完成后单独计算
- 避免训练过程被中断

### 3. 自定义资源限制

如果需要自定义资源限制：

```python
from nash_conv_gpu import nash_conv_gpu

conv = nash_conv_gpu(
    game,
    deep_cfr_solver,
    max_cpu_threads=4,      # 自定义 CPU 线程数
    max_memory_gb=16,      # 自定义内存限制
    timeout_seconds=1800,   # 30 分钟超时
    verbose=True,
    device=device
)
```

## 使用方法

### 方法 1: 训练时跳过（推荐）

```bash
# 训练时跳过 NashConv
python train_deep_cfr_texas.py --num_iterations 100 --skip_nashconv
```

### 方法 2: 使用资源限制版本（已集成）

训练脚本已自动使用资源限制版本：

```bash
# 自动使用资源限制（CPU: 2线程, 内存: 8GB, 超时: 10分钟）
python train_deep_cfr_texas.py --num_iterations 100
```

### 方法 3: 单独计算 NashConv

训练完成后，单独计算 NashConv：

```python
import pyspiel
from open_spiel.python import policy
from nash_conv_gpu import nash_conv_lightweight

# 加载游戏和模型
game = pyspiel.load_game("universal_poker(...)")
# ... 加载 deep_cfr_solver ...

# 使用资源限制计算 NashConv
conv = nash_conv_lightweight(
    game,
    deep_cfr_solver,
    max_cpu_threads=2,
    max_memory_gb=8,
    timeout_seconds=600,
    verbose=True
)
```

## 资源限制说明

### CPU 线程数限制

- **默认**: 2 线程
- **原因**: 避免占满所有 CPU 核心，保持系统响应
- **调整**: 根据系统核心数调整（建议不超过核心数的 50%）

### 内存限制

- **默认**: 8GB
- **原因**: 避免内存溢出，导致系统卡死
- **调整**: 根据系统内存调整（建议不超过系统内存的 50%）

### 超时限制

- **默认**: 10 分钟（600 秒）
- **原因**: 避免无限运行，占用资源
- **调整**: 根据游戏规模调整（大规模游戏可能需要更长时间）

## 故障排除

### 问题 1: 仍然内存不足

**解决方案**：
1. 进一步降低 `max_memory_gb`（例如 4GB）
2. 使用 `--skip_nashconv` 跳过计算
3. 使用更小的游戏配置进行测试

### 问题 2: 超时

**解决方案**：
1. 增加 `timeout_seconds`（例如 1800 秒）
2. 使用 `--skip_nashconv` 跳过计算
3. 使用采样方法（如果实现）

### 问题 3: CPU 仍然占满

**解决方案**：
1. 进一步降低 `max_cpu_threads`（例如 1）
2. 使用 `taskset` 限制 CPU 使用：
   ```bash
   taskset -c 0-1 python train_deep_cfr_texas.py
   ```

## 最佳实践

1. **训练时跳过 NashConv**
   ```bash
   python train_deep_cfr_texas.py --skip_nashconv
   ```

2. **定期单独计算**
   - 每 100 次迭代计算一次
   - 或训练完成后计算

3. **使用资源限制**
   - 自动使用资源限制版本
   - 根据系统资源调整限制

4. **监控资源使用**
   ```bash
   # 监控 CPU 和内存
   top
   # 或
   htop
   ```

## 技术细节

### 资源限制实现

1. **CPU 线程数限制**
   ```python
   os.environ['OMP_NUM_THREADS'] = str(max_cpu_threads)
   os.environ['MKL_NUM_THREADS'] = str(max_cpu_threads)
   os.environ['NUMEXPR_NUM_THREADS'] = str(max_cpu_threads)
   ```

2. **内存限制**
   ```python
   resource.setrlimit(
       resource.RLIMIT_AS,
       (max_bytes, max_bytes)
   )
   ```

3. **超时机制**
   ```python
   signal.signal(signal.SIGALRM, _timeout_handler)
   signal.alarm(int(timeout_seconds))
   ```

## 总结

- ✅ **已添加资源限制**：CPU 线程数、内存、超时
- ✅ **自动使用资源限制版本**：训练脚本已更新
- ✅ **推荐跳过 NashConv**：训练时使用 `--skip_nashconv`
- ⚠️ **大规模游戏**：仍然可能消耗大量资源，建议跳过

**建议**：对于大规模训练，始终使用 `--skip_nashconv` 跳过 NashConv 计算。

