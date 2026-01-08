#!/bin/bash
# 6人场DeepCFR训练脚本（多GPU版本）
# 使用4张GPU，最大化训练速度

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用6张GPU

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate open_spiel

# 训练参数配置
NUM_PLAYERS=6
NUM_WORKERS=8               # Worker数量（平衡样本产生速度和内存，8个Worker适合当前配置）
NUM_ITERATIONS=20000        # 迭代次数（DeepCFR收敛较慢，需要较多迭代）
NUM_TRAVERSALS=2000          # 每次迭代遍历次数（适配batch_size=4096，增加样本产生速度，保证新样本充足）
BATCH_SIZE=4096              # 训练批量大小（多GPU时4096利用率高，适配其他参数）
MEMORY_CAPACITY=500000      # 优势网络经验回放缓冲区容量（50万，每个玩家，适配batch_size=4096，建议至少100x batch_size，总内存约105GB）
STRATEGY_MEMORY_CAPACITY=1600000  # 策略网络经验回放缓冲区容量（160万，所有玩家共享，适配策略样本产生速度16K/迭代，建议100x新增样本）
QUEUE_MAXSIZE=50000         # 队列最大大小（平衡内存和性能，总内存约12GB）
NEW_SAMPLE_RATIO=0.5        # 新样本占比（分层加权采样，50%新样本+50%重要性加权老样本）
LEARNING_RATE=0.001          # 学习率
POLICY_LAYERS="256 256 256"  # 策略网络结构（3层256节点，6人局状态复杂）
ADVANTAGE_LAYERS="256 256 256"  # 优势网络结构（与策略网络相同）
ADVANTAGE_TRAIN_STEPS=1      # 优势网络训练步骤数（每次迭代训练1次，大幅减少以加快迭代速度）
POLICY_TRAIN_STEPS=2         # 策略网络训练步骤数（每次迭代训练2次，减少以加快迭代速度）
EVAL_INTERVAL=50            # 每N次迭代进行一次评估
CHECKPOINT_INTERVAL=100       # Checkpoint保存间隔
NUM_TEST_GAMES=200            # 评估时的测试对局数量（推荐200局，平衡速度和准确性）
MAX_MEMORY_GB=4              # Worker内存限制（每个Worker最多4GB，防止OOM）
betting_abstraction="fchpa"
blinds="100 200 0 0 0 0"
stack_size=50000
resume_model_dir="models/deepcfr_6p_multi_20260107_180305"

# 续训时不需要设置新的save_prefix，代码会自动从config.json读取
# 如果需要设置新的save_prefix，可以取消下面的注释
# SAVE_PREFIX="deepcfr_6p_multi_$(date +%Y%m%d_%H%M%S)"

# 运行训练（后台运行，输出到日志文件）
nohup python deep_cfr_parallel.py \
    --num_players $NUM_PLAYERS \
    --stack_size $stack_size \
    --blinds "$blinds" \
    --resume $resume_model_dir \
    --num_workers $NUM_WORKERS \
    --num_iterations $NUM_ITERATIONS \
    --num_traversals $NUM_TRAVERSALS \
    --batch_size $BATCH_SIZE \
    --memory_capacity $MEMORY_CAPACITY \
    --strategy_memory_capacity $STRATEGY_MEMORY_CAPACITY \
    --queue_maxsize $QUEUE_MAXSIZE \
    --new_sample_ratio $NEW_SAMPLE_RATIO \
    --learning_rate $LEARNING_RATE \
    --policy_layers $POLICY_LAYERS \
    --advantage_layers $ADVANTAGE_LAYERS \
    --eval_interval $EVAL_INTERVAL \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --num_test_games $NUM_TEST_GAMES \
    --use_gpu \
    --gpu_ids 0 1 2 3 4 5 \
    --skip_nashconv \
    --eval_with_games \
    --betting_abstraction $betting_abstraction \
    > train_6p_resume_$(basename $resume_model_dir).log 2>&1 &

# 获取进程ID
PID=$!
LOG_FILE="train_6p_resume_$(basename $resume_model_dir).log"
echo "训练已启动，进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "续训目录: $resume_model_dir"
echo ""
echo "查看训练进度: tail -f $LOG_FILE"
echo "停止训练: kill $PID"

