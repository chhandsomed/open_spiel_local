# OpenSpiel 德州扑克训练项目

## 📖 快速开始

### 环境设置
```bash
conda activate open_spiel
```

### DeepCFR 训练（推荐）
```bash
# 6人场训练（默认使用简单特征版本）
python train_deep_cfr_texas.py \
    --num_players 6 \
    --num_iterations 100 \
    --skip_nashconv \
    --eval_interval 10
```

### 交互式游戏
```bash
python play_interactive.py --model_dir models/deepcfr_texas_*
```

## 📚 完整文档

- **[README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md)** - 完整使用指南（训练、推理、参数说明）
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 特征转换使用指南（简单版本 vs 复杂版本）
- **[LOSS_AND_EVALUATION.md](LOSS_AND_EVALUATION.md)** - 损失计算与评估方法详解
- **[INTERACTIVE_PLAY_GUIDE.md](INTERACTIVE_PLAY_GUIDE.md)** - 交互式游戏使用指南

## 🎯 核心功能

1. **DeepCFR 训练** - 支持2-6人场，GPU加速，自动特征提取
2. **特征转换** - 起手牌强度、位置优势等7维手动特征
3. **训练评估** - 策略熵、缓冲区大小、测试对局等轻量级指标
4. **交互式游戏** - 与训练好的模型对局

## ⚠️ 重要提示

- 训练时使用 `--skip_nashconv` 避免资源问题
- 游戏配置必须与训练时一致
- 网络结构参数必须与训练时一致

详细说明请参考 [README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md)
