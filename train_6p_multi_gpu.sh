#!/bin/bash
# 6人场DeepCFR训练脚本（多GPU版本）
# 使用4张GPU，最大化训练速度

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用4张GPU

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate open_spiel

# 训练参数配置
NUM_PLAYERS=6
NUM_WORKERS=10              # Worker数量（优化：从12降到10，减少样本产生速度，降低内存占用）
NUM_ITERATIONS=20000        # 迭代次数（DeepCFR收敛较慢，需要较多迭代）
NUM_TRAVERSALS=1600          # 每次迭代遍历次数（6人场状态复杂，建议1000-1500）
BATCH_SIZE=4096              # 训练批量大小（减少以加快训练速度，多GPU时4096利用率高但训练慢）
MEMORY_CAPACITY=2000000      # 经验回放缓冲区容量（200万，6人场状态复杂，需要更大缓冲区）
QUEUE_MAXSIZE=20000         # 队列最大大小（优化：从200,000降到50,000，减少75%内存占用，降低OOM风险）
LEARNING_RATE=0.001          # 学习率
POLICY_LAYERS="256 256 256"  # 策略网络结构（3层256节点，6人局状态复杂）
ADVANTAGE_LAYERS="256 256 256"  # 优势网络结构（与策略网络相同）
ADVANTAGE_TRAIN_STEPS=1      # 优势网络训练步骤数（每次迭代训练1次，大幅减少以加快迭代速度）
POLICY_TRAIN_STEPS=2         # 策略网络训练步骤数（每次迭代训练2次，减少以加快迭代速度）
EVAL_INTERVAL=50            # 每N次迭代进行一次评估
CHECKPOINT_INTERVAL=100       # Checkpoint保存间隔
NUM_TEST_GAMES=100            # 评估时的测试对局数量
MAX_MEMORY_GB=4              # Worker内存限制（每个Worker最多4GB，防止OOM）

# 保存前缀
SAVE_PREFIX="deepcfr_6p_multi_$(date +%Y%m%d_%H%M%S)"

# 运行训练（后台运行，输出到日志文件）
nohup python deep_cfr_parallel.py \
    --num_players $NUM_PLAYERS \
    --num_workers $NUM_WORKERS \
    --num_iterations $NUM_ITERATIONS \
    --num_traversals $NUM_TRAVERSALS \
    --batch_size $BATCH_SIZE \
    --memory_capacity $MEMORY_CAPACITY \
    --queue_maxsize $QUEUE_MAXSIZE \
    --learning_rate $LEARNING_RATE \
    --policy_layers $POLICY_LAYERS \
    --advantage_layers $ADVANTAGE_LAYERS \
    --eval_interval $EVAL_INTERVAL \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --num_test_games $NUM_TEST_GAMES \
    --use_gpu \
    --gpu_ids 0 1 2 3 4 5 6 7 \
    --skip_nashconv \
    --eval_with_games \
    --save_prefix $SAVE_PREFIX \
    > train_6p_${SAVE_PREFIX}.log 2>&1 &

# 获取进程ID
PID=$!
echo "训练已启动，进程ID: $PID"
echo "日志文件: train_6p_${SAVE_PREFIX}.log"
echo "模型保存目录: models/$SAVE_PREFIX"
echo ""
echo "查看训练进度: tail -f train_6p_${SAVE_PREFIX}.log"
echo "停止训练: kill $PID"

