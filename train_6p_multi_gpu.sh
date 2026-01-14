#!/bin/bash
# 6人场DeepCFR训练脚本（多GPU版本）
# 使用6张GPU，最大化训练速度
# 
# 训练策略优化：
# - 策略网络只在checkpoint时训练（减少99%训练时间）
# - 评估只在checkpoint时进行（减少99%评估时间）
# - Checkpoint时策略网络训练20步，充分学习
# - Checkpoint时评估1000局，提高准确性

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用6张GPU

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate open_spiel

# 训练参数配置（针对 256核CPU + 503GB内存 + 8×RTX4090 优化）
NUM_PLAYERS=6
NUM_WORKERS=20               # Worker数量（增加到20，提高队列使用率，避免主进程sleep）
NUM_ITERATIONS=20000        # 迭代次数（DeepCFR收敛较慢，需要较多迭代）
NUM_TRAVERSALS=4096          # 每次迭代遍历次数（适配batch_size=4096，增加样本产生速度，保证新样本充足）
BATCH_SIZE=4096              # 训练批量大小（多GPU时4096利用率高，适配其他参数）
MEMORY_CAPACITY=360000      # 优势网络经验回放缓冲区容量（每个玩家，适配batch_size=4096）
STRATEGY_MEMORY_CAPACITY=720000  # 策略网络经验回放缓冲区容量（增加到72万，保证新样本有足够保存时间）
QUEUE_MAXSIZE=80000         # 队列最大大小（配合20个Worker，提高缓冲能力）
NEW_SAMPLE_RATIO=0.7        # 新样本占比（0.7=70%新样本+30%老样本，使用numpy向量化加速）
LEARNING_RATE=0.001          # 学习率
POLICY_LAYERS="256 256 256"  # 策略网络结构（3层256节点，6人局状态复杂）
ADVANTAGE_LAYERS="256 256 256"  # 优势网络结构（与策略网络相同）
ADVANTAGE_TRAIN_STEPS=2      # 优势网络训练步骤数
POLICY_TRAIN_STEPS=3        # 策略网络训练步骤数（只在checkpoint时训练）
EVAL_INTERVAL=100            # 评估间隔
CHECKPOINT_INTERVAL=100       # Checkpoint保存间隔
NUM_TEST_GAMES=1000           # 评估时的测试对局数量
MAX_MEMORY_GB=6              # Worker内存限制（每个Worker最多6GB，总计约480GB）
betting_abstraction="fchpa"
blinds="100 200 0 0 0 0"
stack_size=50000

# 保存前缀
SAVE_PREFIX="deepcfr_6p_multi_$(date +%Y%m%d_%H%M%S)"

# 运行训练（后台运行，输出到日志文件）
nohup python deep_cfr_parallel.py \
    --num_players $NUM_PLAYERS \
    --stack_size $stack_size \
    --blinds "$blinds" \
    --num_workers $NUM_WORKERS \
    --num_iterations $NUM_ITERATIONS \
    --num_traversals $NUM_TRAVERSALS \
    --batch_size $BATCH_SIZE \
    --memory_capacity $MEMORY_CAPACITY \
    --strategy_memory_capacity $STRATEGY_MEMORY_CAPACITY \
    --queue_maxsize $QUEUE_MAXSIZE \
    --new_sample_ratio $NEW_SAMPLE_RATIO \
    --advantage_train_steps $ADVANTAGE_TRAIN_STEPS \
    --policy_train_steps $POLICY_TRAIN_STEPS \
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
    --save_prefix $SAVE_PREFIX \
    > train_6p_${SAVE_PREFIX}.log 2>&1 &

# 获取进程ID
PID=$!
echo "训练已启动，进程ID: $PID"
echo "日志文件: train_6p_${SAVE_PREFIX}.log"
echo "模型保存目录: models/$SAVE_PREFIX"
echo ""
echo "查看训练进度: tail -1000f train_6p_${SAVE_PREFIX}.log"
echo "停止训练: kill $PID"

