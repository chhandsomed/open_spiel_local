# OpenSpiel 德州扑克训练项目

## 📖 主要文档

**👉 [README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md) - 完整使用指南（推荐从这里开始）**

这是项目的完整使用指南，包含：
- 快速开始
- DeepCFR 和 MCCFR 训练方法
- 推理使用方法
- 注意事项和常见问题

## 🚀 快速开始

### DeepCFR 训练
```bash
python train_deep_cfr_texas.py --num_iterations 100 --skip_nashconv
```

### DeepCFR 推理
```bash
python inference_simple.py --num_games 10
```

### MCCFR 训练
```bash
python train_texas_holdem_mccfr.py --num_players 2 --iterations 1000
```

## 📁 核心文件

### DeepCFR
- `train_deep_cfr_texas.py` - 训练脚本
- `inference_simple.py` - 推理脚本（推荐）
- `training_evaluator.py` - 训练评估模块
- `nash_conv_gpu.py` - NashConv GPU 加速

### MCCFR
- `train_texas_holdem_mccfr.py` - 训练脚本
- `load_and_test_strategy.py` - 策略加载和测试

## 📚 详细文档

- [README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md) - **主文档（完整指南）**
- [TRAINING_EVALUATION_GUIDE.md](TRAINING_EVALUATION_GUIDE.md) - 训练评估详细指南
- [SKIP_NASHCONV_EXPLANATION.md](SKIP_NASHCONV_EXPLANATION.md) - NashConv 说明
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - 推理详细指南
- [NASHCONV_RESOURCE_FIX.md](NASHCONV_RESOURCE_FIX.md) - NashConv 资源修复
- [TEXAS_HOLDEM_GUIDE.md](TEXAS_HOLDEM_GUIDE.md) - 德州扑克技术指南

## ⚠️ 重要提示

1. **训练时使用 `--skip_nashconv`**：避免资源问题
2. **游戏配置必须一致**：训练和推理时配置要相同
3. **网络结构必须一致**：推理时参数要与训练时相同

详细说明请参考 [README_TEXAS_HOLDEM.md](README_TEXAS_HOLDEM.md)
