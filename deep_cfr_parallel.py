#!/usr/bin/env python3
"""
多进程并行 DeepCFR 训练器

架构：
- 多个 Worker 进程并行遍历游戏树，收集样本
- 主进程从共享缓冲区采样，训练神经网络
- 使用共享内存实现进程间高效通信

优势：
- 真正的并行化，充分利用多核 CPU
- 游戏树遍历（CPU 密集）和网络训练（GPU 密集）可以同时进行
- 线性扩展：N 个 Worker 可以获得接近 N 倍的遍历速度
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import time
import signal
import argparse
import logging
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value, Manager
from collections import namedtuple
import queue
import resource

# 配置logging
def setup_logging(log_file=None):
    """配置logging，同时输出到控制台和文件"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # 强制重新配置
    )
    return logging.getLogger(__name__)

# 全局logger（会在main函数中初始化）
logger = None

def get_logger():
    """获取logger，如果未初始化则返回一个简单的logger"""
    global logger
    if logger is None:
        # 如果logger未初始化，创建一个简单的logger（用于Worker进程）
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
        logger = logging.getLogger(__name__)
    return logger

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import pyspiel
from open_spiel.python.pytorch import deep_cfr
from deep_cfr_simple_feature import SimpleFeatureMLP


# 样本数据结构
AdvantageMemory = namedtuple("AdvantageMemory", "info_state iteration advantage action")
StrategyMemory = namedtuple("StrategyMemory", "info_state iteration strategy_action_probs")


class SharedBuffer:
    """共享内存缓冲区，用于进程间通信
    
    使用 Manager 实现跨进程共享的列表
    """
    
    def __init__(self, manager, capacity=1000000):
        self.capacity = capacity
        self._data = manager.list()
        self._add_calls = Value('i', 0)
        self._lock = manager.Lock()
    
    def add(self, element):
        """添加样本（Reservoir Sampling）"""
        with self._lock:
            if len(self._data) < self.capacity:
                self._data.append(element)
            else:
                idx = np.random.randint(0, self._add_calls.value + 1)
                if idx < self.capacity:
                    self._data[idx] = element
            self._add_calls.value += 1
    
    def sample(self, num_samples):
        """采样"""
        with self._lock:
            if len(self._data) < num_samples:
                return list(self._data)
            indices = np.random.choice(len(self._data), num_samples, replace=False)
            return [self._data[i] for i in indices]
    
    def __len__(self):
        return len(self._data)
    
    def clear(self):
        with self._lock:
            while len(self._data) > 0:
                self._data.pop()
            self._add_calls.value = 0


class NetworkWrapper:
    """网络包装器，支持跨进程共享
    
    使用共享内存存储网络参数，Worker 可以读取最新参数
    """
    
    def __init__(self, network, device='cpu'):
        self.network = network
        self.device = device
        self._state_dict = None
    
    def get_state_dict(self):
        """获取网络参数（用于 Worker 同步）"""
        return {k: v.cpu().numpy() for k, v in self.network.state_dict().items()}
    
    def load_state_dict_from_numpy(self, numpy_dict):
        """从 numpy 字典加载参数"""
        state_dict = {k: torch.from_numpy(v) for k, v in numpy_dict.items()}
        self.network.load_state_dict(state_dict)
        self.network = self.network.to(self.device)


def worker_process(
    worker_id,
    game_string,
    num_players,
    embedding_size,
    num_actions,
    advantage_network_layers,
    advantage_queues,  # 每个玩家一个队列
    strategy_queue,
    network_params_queue,  # 接收最新网络参数
    stop_event,
    iteration_counter,
    num_traversals_per_batch,
    device='cpu',
    max_memory_gb=None,  # Worker 内存限制
    parent_pid=None  # 主进程PID，用于检查主进程是否存活
):
    """Worker 进程：并行遍历游戏树
    
    Args:
        worker_id: Worker ID
        game_string: 游戏配置字符串
        num_players: 玩家数量
        embedding_size: 信息状态维度
        num_actions: 动作数量
        advantage_network_layers: 优势网络层配置
        advantage_queues: 优势样本队列（每个玩家一个）
        strategy_queue: 策略样本队列
        network_params_queue: 网络参数队列
        stop_event: 停止信号
        iteration_counter: 当前迭代计数器
        num_traversals_per_batch: 每批遍历次数
        device: 计算设备
    """
    # 设置进程名称
    try:
        import setproctitle
        setproctitle.setproctitle(f"deepcfr_worker_{worker_id}")
    except ImportError:
        pass
    except:
        pass
    
    # 设置内存限制（如果指定）
    if max_memory_gb:
        try:
            max_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
            # 设置虚拟内存限制（RLIMIT_AS）
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            print(f"[Worker {worker_id}] 已设置内存限制: {max_memory_gb}GB")
        except (ValueError, OSError) as e:
            print(f"[Worker {worker_id}] ⚠️ 无法设置内存限制: {e}")
    
    # 设置异常处理
    try:
        # 获取当前进程的内存使用情况
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                print(f"[Worker {worker_id}] 启动，设备: {device}，初始内存: {mem_mb:.1f}MB")
            except:
                print(f"[Worker {worker_id}] 启动，设备: {device}")
        else:
            print(f"[Worker {worker_id}] 启动，设备: {device}")
        
        # 创建游戏
        try:
            game = pyspiel.load_game(game_string)
            root_node = game.new_initial_state()
        except Exception as e:
            print(f"[Worker {worker_id}] ❌ 创建游戏失败: {e}")
            raise
        
        # 创建本地优势网络（用于采样动作）
        advantage_networks = []
        try:
            for _ in range(num_players):
                # 从游戏字符串中解析max_stack
                import re
                game_string = str(game)
                match = re.search(r'stack=([\d\s]+)', game_string)
                max_stack = 2000  # 默认值
                if match:
                    stack_str = match.group(1).strip()
                    stack_values = stack_str.split()
                    if stack_values:
                        try:
                            max_stack = int(stack_values[0])
                        except ValueError:
                            pass
                
                net = SimpleFeatureMLP(
                    embedding_size,
                    list(advantage_network_layers),
                    num_actions,
                    num_players=num_players,
                    max_game_length=game.max_game_length(),
                    max_stack=max_stack
                )
                net = net.to(device)
                net.eval()
                advantage_networks.append(net)
        except Exception as e:
            print(f"[Worker {worker_id}] ❌ 创建优势网络失败: {e}")
            raise
        
        def sample_action_from_advantage(state, player):
            """使用优势网络采样动作"""
            info_state = state.information_state_tensor(player)
            legal_actions = state.legal_actions(player)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
                raw_advantages = advantage_networks[player](state_tensor)[0].cpu().numpy()
            
            advantages = [max(0., advantage) for advantage in raw_advantages]
            cumulative_regret = sum(advantages[action] for action in legal_actions)
            
            matched_regrets = np.array([0.] * num_actions)
            if cumulative_regret > 0.:
                for action in legal_actions:
                    matched_regrets[action] = advantages[action] / cumulative_regret
            else:
                matched_regrets[max(legal_actions, key=lambda a: raw_advantages[a])] = 1
            
            return advantages, matched_regrets
        
        def traverse_game_tree(state, player, iteration):
            """遍历游戏树，收集样本"""
            nonlocal local_strategy_batch
            if state.is_terminal():
                return state.returns()[player]
            
            if state.is_chance_node():
                chance_outcome, chance_proba = zip(*state.chance_outcomes())
                action = np.random.choice(chance_outcome, p=chance_proba)
                return traverse_game_tree(state.child(action), player, iteration)
            
            if state.current_player() == player:
                expected_payoff = {}
                sampled_regret = {}
                
                _, strategy = sample_action_from_advantage(state, player)
                
                for action in state.legal_actions():
                    expected_payoff[action] = traverse_game_tree(
                        state.child(action), player, iteration
                    )
                
                cfv = sum(strategy[a] * expected_payoff[a] for a in state.legal_actions())
                
                for action in state.legal_actions():
                    sampled_regret[action] = expected_payoff[action] - cfv
                
                sampled_regret_arr = [0] * num_actions
                for action in sampled_regret:
                    sampled_regret_arr[action] = sampled_regret[action]
                
                # 发送优势样本
                sample = AdvantageMemory(
                    state.information_state_tensor(),
                    iteration,
                    sampled_regret_arr,
                    action
                )
                
                # --- 修改开始 ---
                # 累积到批次
                if player not in local_advantage_batches:
                    local_advantage_batches[player] = []
                local_advantage_batches[player].append(sample)
                
                # 如果批次满了，发送
                if len(local_advantage_batches[player]) >= batch_size_limit:
                    try:
                        advantage_queues[player].put(local_advantage_batches[player], timeout=0.01)
                        local_advantage_batches[player] = []  # 清空批次
                    except queue.Full:
                        pass  # 队列满了就丢弃整个批次


                return cfv
            else:
                other_player = state.current_player()
                _, strategy = sample_action_from_advantage(state, other_player)
                
                probs = np.array(strategy)
                probs /= probs.sum()
                sampled_action = np.random.choice(range(num_actions), p=probs)
                
                sample = StrategyMemory(
                    state.information_state_tensor(other_player),
                    iteration,
                    strategy
                )
                
                # --- 修改开始 ---
                # 累积到批次
                local_strategy_batch.append(sample)
                
                # 如果批次满了，发送
                if len(local_strategy_batch) >= batch_size_limit:
                    try:
                        strategy_queue.put(local_strategy_batch, timeout=0.01)
                        local_strategy_batch = []  # 清空批次
                    except queue.Full:
                        pass
                
                return traverse_game_tree(state.child(sampled_action), player, iteration)
        
        # 主循环
        last_sync_iteration = 0
        # 本地批次缓冲区（减少 Queue.put 调用频率）
        local_advantage_batches = {}  # {player_id: [samples]}
        local_strategy_batch = []
        batch_size_limit = 200  # 每积累 100 个样本发送一次
        
        # 主进程存活检查：每10次循环检查一次（避免频繁检查影响性能）
        parent_check_counter = 0
        parent_check_interval = 10
        
        def check_parent_alive():
            """检查主进程是否存活"""
            if parent_pid is None:
                return True  # 如果没有传递parent_pid，跳过检查
            
            try:
                # 方法1: 使用os.getppid()检查父进程ID
                # 如果父进程是init (PID 1)，说明主进程已退出
                ppid = os.getppid()
                if ppid == 1:
                    return False
                
                # 方法2: 使用psutil检查指定的主进程是否存活
                if HAS_PSUTIL:
                    try:
                        parent_process = psutil.Process(parent_pid)
                        # 检查进程是否存在且状态不是zombie
                        if parent_process.status() == psutil.STATUS_ZOMBIE:
                            return False
                    except psutil.NoSuchProcess:
                        # 主进程不存在
                        return False
                    except psutil.AccessDenied:
                        # 无法访问，假设存活（可能是权限问题）
                        pass
                
                return True
            except Exception:
                # 检查失败，假设存活（避免误杀）
                return True
        
        while not stop_event.is_set():
            # 定期检查主进程是否存活
            parent_check_counter += 1
            if parent_check_counter >= parent_check_interval:
                parent_check_counter = 0
                if not check_parent_alive():
                    print(f"\n[Worker {worker_id}] 检测到主进程已退出，自动退出...")
                    stop_event.set()  # 设置停止事件
                    break
            
            # 关键修复：先同步网络参数，再读取迭代计数器
            # 这样可以确保Worker使用的网络参数和迭代标记是匹配的
            network_synced = False
            try:
                while True:
                    params = network_params_queue.get_nowait()
                    for player in range(num_players):
                        if player in params:
                            numpy_dict = params[player]
                            state_dict = {k: torch.from_numpy(v) for k, v in numpy_dict.items()}
                            advantage_networks[player].load_state_dict(state_dict)
                            advantage_networks[player] = advantage_networks[player].to(device)
                    network_synced = True
                    last_sync_iteration = iteration_counter.value
            except queue.Empty:
                pass
            
            # 遍历游戏树
            for player in range(num_players):
                for _ in range(num_traversals_per_batch):
                    if stop_event.is_set():
                        break
                    traverse_game_tree(root_node.clone(), player, current_iteration)
                
            # 强制刷新缓冲区：无论是否达到 batch_limit，都将手中的样本发送出去
            # 这防止了在多玩家游戏中，某些玩家的样本积累太慢导致的主进程饥饿
            for p in list(local_advantage_batches.keys()):
                batch = local_advantage_batches[p]
                if batch:
                    try:
                        advantage_queues[p].put(batch, timeout=0.01)
                    except queue.Full:
                        pass # 队列满就算了
                # 清空该玩家的缓冲区
                local_advantage_batches[p] = []
            
            if local_strategy_batch:
                try:
                    strategy_queue.put(local_strategy_batch, timeout=0.01)
                    local_strategy_batch = []
                except queue.Full:
                    pass
        
    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] 收到中断信号，退出...")
    except Exception as e:
        print(f"\n[Worker {worker_id}] 发生异常: {e}")
        import traceback
        traceback.print_exc()
        # 不重新抛出异常，让进程正常退出
        # 这样可以避免异常传播导致其他问题
    finally:
        print(f"[Worker {worker_id}] 停止")


class ParallelDeepCFRSolver:
    """多进程并行 DeepCFR 求解器
    
    使用多个 Worker 进程并行遍历游戏树，主进程训练网络。
    """
    
    def __init__(
        self,
        game,
        num_workers=4,
        policy_network_layers=(128, 128),
        advantage_network_layers=(128, 128),
        num_iterations=100,
        num_traversals=20,
        learning_rate=1e-4,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=1000000,
        device='cuda',
        gpu_ids=None,  # 多 GPU 支持
        sync_interval=1,  # 每多少次迭代同步一次网络参数
        max_memory_gb=None,  # 最大内存限制（GB），None 表示不限制
        queue_maxsize=50000,  # 队列最大大小（降低以减少内存占用）
    ):
        self.game = game
        self.num_workers = num_workers
        self.num_players = game.num_players()
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.learning_rate = learning_rate
        self.batch_size_advantage = batch_size_advantage
        self.batch_size_strategy = batch_size_strategy
        self.memory_capacity = memory_capacity
        self.sync_interval = sync_interval
        self.max_memory_gb = max_memory_gb
        self.queue_maxsize = queue_maxsize
        
        # 内存监控
        self._last_memory_check = 0
        self._memory_check_interval = 60  # 每60秒检查一次内存
        
        # 缓冲区清理参数
        self._buffer_cleanup_threshold = 0.95  # 缓冲区清理阈值：95%（提高到95%，避免频繁清理）
        self._buffer_keep_ratio = 0.75  # 清理后保留比例：75%（删除25%，固定比例）
        
        # 队列监控和动态调整
        self._queue_stats = {
            'last_queue_sizes': {},  # {queue_name: size}
            'queue_growth_rates': {},  # {queue_name: samples/sec}
            'last_check_time': time.time(),
            'collection_counts': {},  # {queue_name: count} 用于统计收集速度
        }
        self._adaptive_max_collect = {
            'base': 20000,  # 基础值
            'min': 5000,    # 最小值
            'max': 100000,  # 最大值（根据队列大小动态调整）
            'current': 20000,  # 当前值
        }
        
        # 多 GPU 设置
        self.gpu_ids = gpu_ids
        self.use_multi_gpu = gpu_ids is not None and len(gpu_ids) > 1 and torch.cuda.is_available()
        
        if self.use_multi_gpu:
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
            print(f"  多 GPU 模式: {gpu_ids}")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 游戏信息
        self._root_node = game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_actions = game.num_distinct_actions()
        
        # 游戏字符串（用于 Worker 创建游戏）
        self._game_string = str(game)
        
        # 从游戏配置中解析max_stack（用于归一化下注统计特征）
        self._max_stack = self._parse_max_stack_from_game_string(self._game_string)
        
        # 网络层配置
        self._policy_network_layers = policy_network_layers
        self._advantage_network_layers = advantage_network_layers
        
        # 创建网络
        self._create_networks()
        
        # 多进程组件
        self._manager = None
        self._workers = []
        self._advantage_queues = []
        self._strategy_queue = None
        self._network_params_queues = []
        self._stop_event = None
        self._iteration_counter = None
        
        # 本地缓冲区
        self._advantage_memories = [
            deep_cfr.ReservoirBuffer(memory_capacity) for _ in range(self.num_players)
        ]
        self._strategy_memories = deep_cfr.ReservoirBuffer(memory_capacity)
        
        self._iteration = 1
    
    def _parse_max_stack_from_game_string(self, game_string):
        """从游戏字符串中解析max_stack值
        
        Args:
            game_string: 游戏配置字符串，例如 "universal_poker(...,stack=2000 2000 2000,...)"
        
        Returns:
            max_stack: 单个玩家的最大筹码量（默认2000）
        """
        import re
        # 匹配 stack=后面的值
        match = re.search(r'stack=([\d\s]+)', game_string)
        if match:
            stack_str = match.group(1).strip()
            # 解析第一个玩家的筹码量（所有玩家应该相同）
            stack_values = stack_str.split()
            if stack_values:
                try:
                    max_stack = int(stack_values[0])
                    return max_stack
                except ValueError:
                    pass
        # 如果解析失败，返回默认值2000
        return 2000
    
    def _create_networks(self):
        """创建神经网络"""
        # 策略网络
        policy_net = SimpleFeatureMLP(
            self._embedding_size,
            list(self._policy_network_layers),
            self._num_actions,
            num_players=self.num_players,
            max_game_length=self.game.max_game_length(),
            max_stack=self._max_stack
        )
        
        # 验证策略网络输入维度
        actual_input_size = policy_net.mlp.model[0]._weight.shape[1]
        expected_input_size = self._embedding_size + 1  # 1维手动特征（手牌强度）
        assert actual_input_size == expected_input_size, \
            f"策略网络输入维度错误: 期望 {expected_input_size}，实际 {actual_input_size}"
        
        # 多 GPU 包装
        if self.use_multi_gpu:
            self._policy_network = nn.DataParallel(policy_net, device_ids=self.gpu_ids)
            self._policy_network = self._policy_network.to(self.device)
        else:
            self._policy_network = policy_net.to(self.device)
        
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=self.learning_rate
        )
        
        # 优势网络（每个玩家一个）
        self._advantage_networks = []
        self._optimizer_advantages = []
        for player in range(self.num_players):
            net = SimpleFeatureMLP(
                self._embedding_size,
                list(self._advantage_network_layers),
                self._num_actions,
                num_players=self.num_players,
                max_game_length=self.game.max_game_length(),
                max_stack=self._max_stack
            )
            
            # 验证优势网络输入维度
            actual_input_size = net.mlp.model[0]._weight.shape[1]
            expected_input_size = self._embedding_size + 1  # 1维手动特征（手牌强度）
            assert actual_input_size == expected_input_size, \
                f"玩家 {player} 优势网络输入维度错误: 期望 {expected_input_size}，实际 {actual_input_size}"
            
            # 多 GPU 包装
            if self.use_multi_gpu:
                net = nn.DataParallel(net, device_ids=self.gpu_ids)
                net = net.to(self.device)
            else:
                net = net.to(self.device)
            
            self._advantage_networks.append(net)
            self._optimizer_advantages.append(
                torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            )
        
        self._loss_advantages = nn.MSELoss(reduction="mean")
    
    def _start_workers(self):
        """启动 Worker 进程"""
        mp.set_start_method('spawn', force=True)
        
        self._manager = Manager()
        self._stop_event = Event()
        self._iteration_counter = Value('i', 1)
        
        # 创建队列（降低 maxsize 以减少内存占用）
        self._advantage_queues = [Queue(maxsize=self.queue_maxsize) for _ in range(self.num_players)]
        self._strategy_queue = Queue(maxsize=self.queue_maxsize)
        self._network_params_queues = [Queue(maxsize=10) for _ in range(self.num_workers)]
        
        # 计算每个 Worker 的遍历次数
        # 关键修正：不再一次性分配 huge number，而是分配一个小批次，让 Worker 快速响应
        # 主进程会通过控制收集样本的数量来保证总遍历次数
        traversals_per_worker = 10  # 每个 Worker 每次只跑 10 次遍历，然后检查同步
        
        # 获取主进程PID，传递给worker用于存活检查
        import os
        main_process_pid = os.getpid()
        
        # 启动 Worker
        print(f"  正在平滑启动 {self.num_workers} 个 Worker (每组5个)...")
        for i in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    i,
                    self._game_string,
                    self.num_players,
                    self._embedding_size,
                    self._num_actions,
                    self._advantage_network_layers,
                    self._advantage_queues,
                    self._strategy_queue,
                    self._network_params_queues[i],
                    self._stop_event,
                    self._iteration_counter,
                    traversals_per_worker,
                    'cpu',  # Worker 在 CPU 上运行
                    self.max_memory_gb,  # Worker 内存限制
                    main_process_pid,  # 主进程PID，用于检查主进程是否存活
                ),
                daemon=True  # 设置为守护进程，主进程退出时自动杀死
            )
            p.start()
            self._workers.append(p)
            
            # 平滑启动：每启动 5 个 Worker 稍微暂停一下，避免瞬间 IO/CPU 拥堵
            if (i + 1) % 5 == 0:
                print(f"    已启动 {i + 1}/{self.num_workers}...", end="\r", flush=True)
                time.sleep(1)
        
        print(f"\n  ✓ 已启动 {self.num_workers} 个 Worker 进程")
        
        # 显示内存使用情况
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                print(f"  主进程内存使用: {mem_mb:.1f}MB")
                
                # 估算总内存需求
                # 实际内存占用包括：
                # 1. 样本数据本身：info_state (numpy数组，可能几百到几千个float32) + iteration + advantage/strategy
                # 2. Python 对象开销：namedtuple、list、numpy数组对象等
                # 3. 队列积压：如果 Worker 产生速度 > 消费速度
                # 4. Worker 进程：每个 Worker 的网络副本、游戏状态等
                # 
                # 保守估算：每个样本约 5-10KB（包括 Python 对象开销）
                # 对于6人局，memory_capacity=1,000,000：
                #   - 优势样本：6 × 1,000,000 × 5KB = 30GB
                #   - 策略样本：1 × 1,000,000 × 5KB = 5GB
                #   - 总计：约 35GB（不包括队列和 Worker）
                sample_size_kb = 5  # 保守估算，实际可能更大
                estimated_memory_gb = (
                    self.memory_capacity * self.num_players * sample_size_kb +  # 优势样本
                    self.memory_capacity * sample_size_kb +  # 策略样本
                    self.queue_maxsize * (self.num_players + 1) * sample_size_kb +  # 队列积压（最坏情况）
                    self.num_workers * 500  # Worker 进程开销（每个 Worker 约 500MB）
                ) / 1024 / 1024
                print(f"  估算总内存需求: {estimated_memory_gb:.2f}GB")
                print(f"    - 优势样本缓冲区: {self.memory_capacity * self.num_players * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - 策略样本缓冲区: {self.memory_capacity * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - 队列积压（最坏情况）: {self.queue_maxsize * (self.num_players + 1) * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - Worker 进程: {self.num_workers * 500 / 1024 / 1024:.2f}GB")
                
                # 获取系统总内存
                try:
                    total_mem = psutil.virtual_memory().total / 1024 / 1024 / 1024
                    available_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
                    print(f"  系统内存: {total_mem:.1f}GB 总计, {available_mem:.1f}GB 可用")
                    
                    if estimated_memory_mb / 1024 > available_mem * 0.8:
                        print(f"  ⚠️ 警告: 估算内存需求 ({estimated_memory_mb/1024:.1f}GB) 接近可用内存 ({available_mem:.1f}GB)")
                        print(f"  建议: 减少 --memory_capacity 或 --num_workers")
                except:
                    pass
            except Exception as e:
                print(f"  ⚠️ 获取内存信息失败: {e}")
        else:
            print(f"  ⚠️ 无法获取内存信息（psutil 未安装，建议安装: pip install psutil）")
    
    def _stop_workers(self):
        """停止 Worker 进程"""
        self._stop_event.set()
        
        # 清空所有队列，防止 Worker 阻塞在 put 操作
        def drain_queue(q):
            try:
                while not q.empty():
                    q.get_nowait()
            except:
                pass
        
        for q in self._advantage_queues:
            drain_queue(q)
        drain_queue(self._strategy_queue)
        for q in self._network_params_queues:
            drain_queue(q)
        
        # 等待 Worker 退出
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        
        self._workers = []
        print("所有 Worker 已停止")
    
    def _sync_network_params(self):
        """同步网络参数到所有 Worker"""
        params = {}
        for player in range(self.num_players):
            # 处理 DataParallel 包装
            net = self._advantage_networks[player]
            if isinstance(net, nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            
            params[player] = {
                k: v.cpu().numpy() 
                for k, v in state_dict.items()
            }
        
        for q in self._network_params_queues:
            try:
                q.put_nowait(params)
            except queue.Full:
                pass
    
    def _cleanup_queue_backlog(self):
        """清理队列积压（安全，不影响样本收集进度）
        
        队列积压清理可以安全地在样本收集循环中执行，因为：
        - 队列中的样本还没有添加到缓冲区
        - 清理队列不会影响已收集的样本数量
        - 队列积压会占用内存，需要及时清理
        """
        # 检查队列是否积压（提高阈值到90%，减少清理频率）
        queue_backlog = False
        for q in self._advantage_queues:
            if q.qsize() > self.queue_maxsize * 0.90:  # 从85%提高到90%
                queue_backlog = True
                break
        if self._strategy_queue.qsize() > self.queue_maxsize * 0.90:  # 从85%提高到90%
            queue_backlog = True
        
        if not queue_backlog:
            return
        
        total_advantage_queue = sum(q.qsize() for q in self._advantage_queues)
        strategy_queue_size = self._strategy_queue.qsize()
        
        # 清理队列中的积压样本（减少丢弃比例：从25%降到10%，只清理超出90%的部分）
        for player, q in enumerate(self._advantage_queues):
            queue_size = q.qsize()
            if queue_size > self.queue_maxsize * 0.90:
                # 只丢弃超出90%的部分，而不是整个队列的10%
                # 这样可以减少样本丢失
                target_size = int(self.queue_maxsize * 0.90)
                to_discard = queue_size - target_size
                for _ in range(to_discard):
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                if to_discard > 0:
                    get_logger().info(f"      玩家 {player} 队列清理: {queue_size} -> {q.qsize()} (丢弃了 {to_discard} 个积压样本)")
        
        if strategy_queue_size > self.queue_maxsize * 0.90:
            target_size = int(self.queue_maxsize * 0.90)
            to_discard = strategy_queue_size - target_size
            for _ in range(to_discard):
                try:
                    self._strategy_queue.get_nowait()
                except queue.Empty:
                    break
            if to_discard > 0:
                get_logger().info(f"      策略队列清理: {strategy_queue_size} -> {self._strategy_queue.qsize()} (丢弃了 {to_discard} 个积压样本)")
    
    def _cleanup_buffers(self, force=False):
        """清理缓冲区（不应该在样本收集循环中执行）
        
        缓冲区清理会删除已收集的样本，导致收集进度倒退。
        应该在样本收集完成后执行。
        
        Args:
            force: 强制清理，即使缓冲区未满
        """
        current_time = time.time()
        
        # 每60秒检查一次内存，避免频繁检查
        if not force and current_time - self._last_memory_check < self._memory_check_interval:
            return
        
        self._last_memory_check = current_time
        
        # 获取内存信息（如果有 psutil）
        mem_gb = None
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_gb = mem_info.rss / 1024 / 1024 / 1024
            except:
                pass
        
        # 检查缓冲区是否接近满（使用可配置的阈值）
        buffer_near_full = False
        total_advantage_samples = sum(len(m) for m in self._advantage_memories)
        advantage_threshold = self.memory_capacity * self._buffer_cleanup_threshold
        if total_advantage_samples >= advantage_threshold:
            buffer_near_full = True
        if len(self._strategy_memories) >= self.memory_capacity * self._buffer_cleanup_threshold:
            buffer_near_full = True
        
        # 检查内存使用情况（优先级最高）
        memory_critical = False
        if mem_gb and self.max_memory_gb and mem_gb > self.max_memory_gb * 0.9:
            memory_critical = True
        
        should_cleanup = False
        cleanup_reason = ""
        
        # 内存使用过高时，立即清理
        if memory_critical:
            should_cleanup = True
            cleanup_reason = f"内存使用过高 ({mem_gb:.2f}GB / {self.max_memory_gb}GB) - 立即清理"
        elif buffer_near_full:
            should_cleanup = True
            total_advantage = sum(len(m) for m in self._advantage_memories)
            max_advantage = max(len(m) for m in self._advantage_memories)
            cleanup_reason = f"缓冲区接近满（优势样本: {total_advantage:,}, 最大单玩家: {max_advantage:,}, 策略样本: {len(self._strategy_memories):,}）"
        elif force:
            should_cleanup = True
            cleanup_reason = "强制清理"
        
        if not should_cleanup:
            return
        
        get_logger().info(f"\n  ⚠️ {cleanup_reason}，清理旧样本...")
        if mem_gb and self.max_memory_gb and mem_gb > self.max_memory_gb * 0.9:
            get_logger().warning(f"  ⚠️ 注意：建议减少 --memory_capacity 或 --num_workers")
        
        # 初始化变量（用于记录清理前后的状态）
        samples_before_total = 0
        samples_after_total = 0
        samples_removed = 0
        mem_before_cleanup = None
        
        try:
            # 清理ReservoirBuffer（当缓冲区接近满、内存过高或强制清理时）
            # 内存过高时也需要清理，即使缓冲区未满
            if buffer_near_full or memory_critical or force:
                # 记录清理前的样本数量和内存（用于计算实际内存变化）
                samples_before_advantage = sum(len(m) for m in self._advantage_memories)
                samples_before_strategy = len(self._strategy_memories)
                samples_before_total = samples_before_advantage + samples_before_strategy
                
                # 在清理操作真正开始前记录内存（删除样本前）
                if HAS_PSUTIL:
                    try:
                        process = psutil.Process()
                        mem_info_before = process.memory_info()
                        mem_before_cleanup = mem_info_before.rss / 1024 / 1024 / 1024
                    except:
                        pass
                
                # 检查总优势样本数是否超标
                total_advantage = sum(len(m) for m in self._advantage_memories)
                advantage_threshold = self.memory_capacity * self._buffer_cleanup_threshold
                
                # 如果内存过高，使用更低的阈值（80%）来清理更多样本
                if memory_critical:
                    advantage_threshold = self.memory_capacity * 0.80
                
                # 决定是否需要清理优势样本
                need_advantage_cleanup = total_advantage >= advantage_threshold
                
                if need_advantage_cleanup and total_advantage > 0:
                    # 计算总目标样本数（容量的75%）
                    keep_ratio = self._buffer_keep_ratio  # 固定保留75%
                    target_total = int(self.memory_capacity * keep_ratio)  # 总目标：3,000,000
                    
                    # 计算减少比例：如果总样本数超过目标，按比例减少
                    if total_advantage > target_total:
                        reduction_ratio = target_total / total_advantage
                    else:
                        reduction_ratio = keep_ratio
                    
                    # 清理每个玩家的优势样本（随机删除，保持Reservoir Sampling的随机性）
                    for player in range(self.num_players):
                        buffer = self._advantage_memories[player]
                        data = buffer._data
                        original_len = len(data)
                        if original_len > 0:
                            # 按比例计算每个玩家应该保留的数量
                            keep_count = max(1000, int(original_len * reduction_ratio))
                            
                            if keep_count < original_len:
                                # 随机选择要保留的样本（保持随机性，而不是删除最旧的）
                                keep_indices = set(np.random.choice(original_len, keep_count, replace=False))
                                
                                removed_count = 0
                                # 从后往前删除，避免索引变化
                                for idx in range(original_len - 1, -1, -1):
                                    if idx not in keep_indices:
                                        del data[idx]
                                        removed_count += 1
                                
                                # 关键修复：调整_add_calls以保持合理的替换概率
                                # 当缓冲区接近满时，我们希望新样本有合理的替换概率
                                # 设置_add_calls为当前长度的2-3倍，这样替换概率约为 capacity/(2-3*capacity) = 33-50%
                                # 这比原来的极低概率要好得多
                                new_len = len(data)
                                if buffer._add_calls > new_len * 10:
                                    # 如果_add_calls太大，重置为合理值
                                    buffer._add_calls = new_len * 2
                                # 否则保持_add_calls不变（如果它已经是合理值）
                                
                                get_logger().info(f"      玩家 {player} 优势样本: {original_len:,} -> {new_len:,} (随机删除 {removed_count:,} 个, 保留比例: {keep_ratio*100:.0f}%, _add_calls: {buffer._add_calls:,})")
                
                # 清理策略样本（随机删除，保持Reservoir Sampling的随机性）
                # 如果内存过高，即使未达到阈值也要清理
                strategy_threshold = self.memory_capacity * self._buffer_cleanup_threshold
                if memory_critical:
                    # 内存过高时，使用更低的阈值（80%）来清理更多样本
                    strategy_threshold = self.memory_capacity * 0.80
                
                if len(self._strategy_memories) >= strategy_threshold:
                    buffer = self._strategy_memories
                    original_len = len(buffer._data)
                    # 固定保留容量的75%
                    keep_ratio = self._buffer_keep_ratio  # 固定保留75%
                    keep_count = max(1000, int(self.memory_capacity * keep_ratio))  # 保留容量的75%
                    
                    if original_len > keep_count:
                        # 随机选择要保留的样本（保持随机性，而不是删除最旧的）
                        keep_indices = set(np.random.choice(original_len, keep_count, replace=False))
                        
                        # 直接删除不需要的样本（从后往前删除，避免索引变化）
                        removed_count = 0
                        for idx in range(original_len - 1, -1, -1):
                            if idx not in keep_indices:
                                del buffer._data[idx]
                                removed_count += 1
                    else:
                        removed_count = 0
                    
                    # 关键修复：调整_add_calls以保持合理的替换概率
                    new_len = len(buffer._data)
                    if buffer._add_calls > new_len * 10:
                        # 如果_add_calls太大，重置为合理值
                        buffer._add_calls = new_len * 2
                    # 否则保持_add_calls不变（如果它已经是合理值）
                    
                    get_logger().info(f"      策略样本: {original_len:,} -> {new_len:,} (随机删除 {removed_count:,} 个, 保留比例: {keep_ratio*100:.0f}%, _add_calls: {buffer._add_calls:,})")
            
            # 记录清理后的样本数量（只有在执行了清理操作时才记录）
            if buffer_near_full or memory_critical or force:
                samples_after_advantage = sum(len(m) for m in self._advantage_memories)
                samples_after_strategy = len(self._strategy_memories)
                samples_after_total = samples_after_advantage + samples_after_strategy
                
                # 计算样本数量变化
                samples_removed = samples_before_total - samples_after_total
            
            # 强制 Python 垃圾回收（多次调用以确保释放）
            import gc
            # 清理所有代的内存
            collected = gc.collect()
            if collected > 0:
                get_logger().debug(f"  垃圾回收: 释放了 {collected} 个对象")
            
            # 等待一小段时间，让内存分配器有机会释放内存
            time.sleep(0.1)
            
            # 再次强制垃圾回收
            collected2 = gc.collect()
            if collected2 > 0:
                get_logger().debug(f"  二次垃圾回收: 释放了 {collected2} 个对象")
            
            # 检查清理后的内存
            if HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    new_mem_info = process.memory_info()
                    new_mem_gb = new_mem_info.rss / 1024 / 1024 / 1024
                    
                    # 计算实际内存变化（清理前后）
                    actual_mem_change_mb = None
                    if mem_before_cleanup is not None:
                        actual_mem_change_mb = (new_mem_gb - mem_before_cleanup) * 1024
                    
                    # 计算基于样本数量的估算内存变化
                    # 假设每个样本平均占用约1KB内存（包括数据结构开销）
                    estimated_mem_freed_mb = 0
                    if samples_removed > 0:
                        estimated_mem_freed_mb = samples_removed * 1.0 / 1024  # KB转MB
                    
                    # 显示内存变化信息
                    if mem_before_cleanup is not None and samples_before_total > 0:
                        if actual_mem_change_mb < 0:
                            # 实际释放了内存
                            get_logger().info(f"  ✓ 清理完成，内存使用: {new_mem_gb:.2f}GB")
                            get_logger().info(f"     样本变化: {samples_before_total:,} -> {samples_after_total:,} (删除了 {samples_removed:,} 个样本)")
                            get_logger().info(f"     实际内存变化: 释放了 {abs(actual_mem_change_mb):.2f}MB")
                            if estimated_mem_freed_mb > 0:
                                get_logger().info(f"     估算内存释放: {estimated_mem_freed_mb:.2f}MB (基于样本数量)")
                        else:
                            # 内存反而增加（可能是清理过程中的临时对象或worker进程继续添加样本）
                            get_logger().warning(f"  ⚠️ 清理完成，内存使用: {new_mem_gb:.2f}GB")
                            get_logger().info(f"     样本变化: {samples_before_total:,} -> {samples_after_total:,} (删除了 {samples_removed:,} 个样本)")
                            get_logger().warning(f"     实际内存变化: 增加了 {actual_mem_change_mb:.2f}MB (可能是清理过程中的临时对象或worker进程继续添加样本)")
                            if estimated_mem_freed_mb > 0:
                                get_logger().info(f"     估算内存释放: {estimated_mem_freed_mb:.2f}MB (基于样本数量，但实际内存可能因临时对象而增加)")
                            get_logger().info(f"     注意：虽然实际内存可能增加，但样本数量已减少，后续内存增长会更慢")
                    elif mem_gb:
                        # 使用旧的mem_gb（清理开始前的内存）
                        freed_mb = (mem_gb - new_mem_gb) * 1024
                        if freed_mb > 0:
                            get_logger().info(f"  ✓ 清理完成，内存使用: {new_mem_gb:.2f}GB (释放了 {freed_mb:.2f}MB)")
                        else:
                            get_logger().warning(f"  ⚠️ 清理完成，内存使用: {new_mem_gb:.2f}GB (内存变化: {freed_mb:.2f}MB)")
                            if samples_before_total > 0:
                                get_logger().info(f"     样本变化: {samples_before_total:,} -> {samples_after_total:,} (删除了 {samples_removed:,} 个样本)")
                                if estimated_mem_freed_mb > 0:
                                    get_logger().info(f"     估算内存释放: {estimated_mem_freed_mb:.2f}MB (基于样本数量)")
                    else:
                        get_logger().info(f"  ✓ 清理完成，内存使用: {new_mem_gb:.2f}GB")
                        if samples_before_total > 0:
                            get_logger().info(f"     样本变化: {samples_before_total:,} -> {samples_after_total:,} (删除了 {samples_removed:,} 个样本)")
                except:
                    pass
        except Exception as e:
            get_logger().error(f"  ⚠️ 内存清理失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_and_cleanup_memory(self, force=False, cleanup_buffers=True):
        """检查内存使用情况，必要时清理旧样本
        
        关键修复：
        1. 分离队列积压清理和缓冲区清理
        2. 队列积压清理可以安全地在样本收集循环中执行
        3. 缓冲区清理应该在样本收集完成后执行，避免删除正在收集的样本
        
        Args:
            force: 强制清理，即使内存使用不高
            cleanup_buffers: 是否清理缓冲区（默认True，在收集循环中应设为False）
        """
        # 总是清理队列积压（安全，不影响收集进度）
        self._cleanup_queue_backlog()
        
        # 可选清理缓冲区（在收集循环中应设为False）
        if cleanup_buffers:
            self._cleanup_buffers(force=force)
    
    def _update_adaptive_max_collect(self):
        """动态调整max_collect，根据队列积压情况和CPU使用率自适应调整消费速度
        
        判断逻辑（优先级从高到低）：
        1. **队列使用率**（主要指标）：队列积压情况，反映消费速度是否足够
           - 使用率 > 80%：消费速度不够，需要增加max_collect
           - 使用率 < 30%：消费速度过快，可以减少max_collect节省CPU
        2. **队列增长率**（次要指标）：Worker产生速度，反映是否需要更快消费
           - 增长率 > 100样本/秒：队列在快速增长，需要增加消费速度
        3. **CPU使用率**（限制指标）：主进程CPU使用情况，避免过度消费导致CPU过载
           - CPU使用率 > 90%：减少max_collect，避免CPU过载
           - CPU使用率 < 50%：可以适当增加max_collect
        
        max_collect的范围：[min, max]，其中max = queue_maxsize * 0.5（避免一次性消费太多）
        """
        current_time = time.time()
        stats = self._queue_stats
        
        # 计算所有队列的最大使用率
        max_queue_usage = 0.0
        total_queue_size = 0
        total_queue_capacity = 0
        
        for player in range(self.num_players):
            q_size = self._advantage_queues[player].qsize()
            usage = q_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
            max_queue_usage = max(max_queue_usage, usage)
            total_queue_size += q_size
            total_queue_capacity += self.queue_maxsize
        
        strategy_q_size = self._strategy_queue.qsize()
        strategy_usage = strategy_q_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
        max_queue_usage = max(max_queue_usage, strategy_usage)
        total_queue_size += strategy_q_size
        total_queue_capacity += self.queue_maxsize
        
        # 计算队列增长率（如果队列大小在增长）
        time_delta = current_time - stats['last_check_time']
        if time_delta > 1.0:  # 至少间隔1秒才更新
            # 更新增长率
            for player in range(self.num_players):
                queue_name = f'advantage_{player}'
                current_size = self._advantage_queues[player].qsize()
                if queue_name in stats['last_queue_sizes']:
                    old_size = stats['last_queue_sizes'][queue_name]
                    growth = (current_size - old_size) / time_delta
                    stats['queue_growth_rates'][queue_name] = growth
                stats['last_queue_sizes'][queue_name] = current_size
            
            # 策略队列
            queue_name = 'strategy'
            current_size = self._strategy_queue.qsize()
            if queue_name in stats['last_queue_sizes']:
                old_size = stats['last_queue_sizes'][queue_name]
                growth = (current_size - old_size) / time_delta
                stats['queue_growth_rates'][queue_name] = growth
            stats['last_queue_sizes'][queue_name] = current_size
            
            stats['last_check_time'] = current_time
        
        # 计算平均队列增长率
        avg_growth_rate = 0.0
        if stats['queue_growth_rates']:
            avg_growth_rate = sum(stats['queue_growth_rates'].values()) / len(stats['queue_growth_rates'])
        
        # 获取CPU使用率（如果可用）
        cpu_percent = None
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.1)  # 非阻塞，快速获取
                # 如果获取失败，尝试获取系统CPU使用率
                if cpu_percent is None or cpu_percent == 0:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
            except:
                pass
        
        # 动态调整max_collect
        adaptive = self._adaptive_max_collect
        base = adaptive['base']
        min_val = adaptive['min']
        # 修复：max_val不应该限制为队列大小的50%，应该允许一次性消费更多
        # 队列大小5000时，max_val应该是队列大小的4-10倍，确保能快速消费
        max_val = min(adaptive['max'], max(self.queue_maxsize * 4, 20000))  # 至少队列大小的4倍，或20000
        
        # 1. 根据队列使用率调整（主要指标）
        if max_queue_usage > 0.80:
            # 队列使用率 > 80%，增加消费速度
            factor = 1.0 + (max_queue_usage - 0.80) * 2.0  # 最多增加到2倍
            new_max_collect = int(base * factor)
        elif max_queue_usage < 0.30:
            # 队列使用率 < 30%，减少消费速度（节省CPU）
            factor = 0.5 + (max_queue_usage / 0.30) * 0.5  # 最少减少到0.5倍
            new_max_collect = int(base * factor)
        else:
            # 正常范围，保持基础值
            new_max_collect = base
        
        # 2. 根据队列增长率调整（次要指标）
        if avg_growth_rate > 100:  # 每秒增长超过100个样本
            growth_factor = min(1.5, 1.0 + avg_growth_rate / 1000.0)  # 最多1.5倍
            new_max_collect = int(new_max_collect * growth_factor)
        
        # 3. 根据CPU使用率调整（限制指标，避免CPU过载）
        # 注意：如果队列使用率很高（>90%），优先处理队列积压，即使CPU高也要增加消费速度
        if cpu_percent is not None:
            if max_queue_usage > 0.90:
                # 队列使用率 > 90%，优先处理队列积压，即使CPU高也要增加消费速度
                # 但如果CPU非常高（>95%），适当限制，避免系统过载
                if cpu_percent > 95:
                    # CPU > 95%，稍微限制，但不要减少太多
                    cpu_factor = max(0.8, 1.0 - (cpu_percent - 95) / 10.0)  # 最多减少到0.8倍
                    new_max_collect = int(new_max_collect * cpu_factor)
                # 否则，队列积压优先，不限制消费速度
            elif cpu_percent > 90:
                # 队列使用率不高，但CPU使用率 > 90%，减少消费速度，避免CPU过载
                cpu_factor = max(0.5, 1.0 - (cpu_percent - 90) / 20.0)  # 最多减少到0.5倍
                new_max_collect = int(new_max_collect * cpu_factor)
            elif cpu_percent < 50 and max_queue_usage > 0.50:
                # CPU使用率 < 50% 且队列使用率 > 50%，可以适当增加消费速度
                cpu_factor = min(1.2, 1.0 + (50 - cpu_percent) / 100.0)  # 最多增加1.2倍
                new_max_collect = int(new_max_collect * cpu_factor)
        
        # 限制在合理范围内
        new_max_collect = max(min_val, min(max_val, new_max_collect))
        adaptive['current'] = new_max_collect
        
        # 记录调整原因（用于调试）
        adaptive['last_adjustment'] = {
            'queue_usage': max_queue_usage,
            'growth_rate': avg_growth_rate,
            'cpu_percent': cpu_percent,
            'final_value': new_max_collect,
        }
        
        return new_max_collect
    
    def _collect_samples(self, timeout=0.1):
        """从队列收集样本
        
        关键修复：
        1. 只清理队列积压（不影响样本收集进度）
        2. 缓冲区清理在样本收集完成后执行，避免删除正在收集的样本
        3. 动态调整max_collect，根据队列积压情况自适应调整消费速度
        """
        # 只清理队列积压（安全，不影响收集进度）
        # 缓冲区清理在样本收集完成后执行
        self._check_and_cleanup_memory(cleanup_buffers=False)
        
        # 动态调整max_collect
        max_collect = self._update_adaptive_max_collect()
        
        # 检查 Worker 状态（增强版：检查退出码）
        dead_workers = []
        for i, p in enumerate(self._workers):
            if not p.is_alive():
                exit_code = p.exitcode
                dead_workers.append((i, exit_code))
        
        if dead_workers:
            worker_info = ", ".join([f"Worker {wid} (退出码: {ec})" for wid, ec in dead_workers])
            error_msg = f"检测到 Worker 已死亡: {worker_info}。训练无法继续。"
            # 如果退出码不为0，可能是OOM或其他严重错误
            for _, exit_code in dead_workers:
                if exit_code != 0:
                    error_msg += f"\n  可能的错误原因: 退出码 {exit_code} 通常表示进程被系统杀死（如OOM）"
            raise RuntimeError(error_msg)

        # 收集优势样本（批量处理，提高效率）
        # 注意：队列积压清理已在 _collect_samples() 中执行，不影响收集进度
        total_collected_advantage = 0
        for player in range(self.num_players):
            collected_count = 0
            batch_list = []
            # 先批量获取，减少锁竞争
            while collected_count < max_collect:
                try:
                    batch = self._advantage_queues[player].get_nowait()
                    batch_list.append(batch)
                    if isinstance(batch, list):
                        collected_count += len(batch)
                    else:
                        collected_count += 1
                except queue.Empty:
                    break
            
            # 批量添加到缓冲区
            for batch in batch_list:
                if isinstance(batch, list):
                    for sample in batch:
                        self._advantage_memories[player].add(sample)
                    total_collected_advantage += len(batch)
                else:
                    self._advantage_memories[player].add(batch)
                    total_collected_advantage += 1
        
        # 收集策略样本（批量处理，提高效率）
        # 注意：队列积压清理已在 _collect_samples() 中执行，不影响收集进度
        collected_count = 0
        batch_list = []
        # 先批量获取，减少锁竞争
        while collected_count < max_collect:
            try:
                batch = self._strategy_queue.get_nowait()
                batch_list.append(batch)
                if isinstance(batch, list):
                    collected_count += len(batch)
                else:
                    collected_count += 1
            except queue.Empty:
                break
        
        # 批量添加到缓冲区
        total_collected_strategy = 0
        for batch in batch_list:
            if isinstance(batch, list):
                for sample in batch:
                    self._strategy_memories.add(sample)
                total_collected_strategy += len(batch)
            else:
                self._strategy_memories.add(batch)
                total_collected_strategy += 1
        
        # 记录收集统计（用于调试和监控）
        if total_collected_advantage > 0 or total_collected_strategy > 0:
            self._queue_stats['collection_counts']['advantage'] = total_collected_advantage
            self._queue_stats['collection_counts']['strategy'] = total_collected_strategy
    
    def _learn_advantage_network(self, player):
        """训练优势网络"""
        num_samples = len(self._advantage_memories[player])
        if num_samples < 32:  # 最少需要 32 个样本才训练
            return None
        
        # 使用实际样本数和 batch_size 的较小值
        actual_batch_size = min(num_samples, self.batch_size_advantage)
        samples = self._advantage_memories[player].sample(actual_batch_size)
        
        info_states = []
        advantages = []
        iterations = []
        for s in samples:
            info_states.append(s.info_state)
            advantages.append(s.advantage)
            iterations.append([s.iteration])
        
        self._optimizer_advantages[player].zero_grad()
        advantages_tensor = torch.FloatTensor(np.array(advantages)).to(self.device)
        iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(self.device)
        outputs = self._advantage_networks[player](
            torch.FloatTensor(np.array(info_states)).to(self.device)
        )
        loss = self._loss_advantages(iters * outputs, iters * advantages_tensor)
        loss.backward()
        self._optimizer_advantages[player].step()
        
        return loss.detach().cpu().numpy()
    
    def _learn_strategy_network(self):
        """训练策略网络"""
        num_samples = len(self._strategy_memories)
        if num_samples < 32:  # 最少需要 32 个样本才训练
            return None
        
        # 使用实际样本数和 batch_size 的较小值
        actual_batch_size = min(num_samples, self.batch_size_strategy)
        samples = self._strategy_memories.sample(actual_batch_size)
        
        info_states = []
        action_probs = []
        iterations = []
        for s in samples:
            info_states.append(s.info_state)
            action_probs.append(s.strategy_action_probs)
            iterations.append([s.iteration])
        
        self._optimizer_policy.zero_grad()
        iters = torch.FloatTensor(np.sqrt(np.array(iterations))).to(self.device)
        ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs))).to(self.device)
        logits = self._policy_network(
            torch.FloatTensor(np.array(info_states)).to(self.device)
        )
        outputs = self._policy_sm(logits)
        loss = self._loss_policy(iters * outputs, iters * ac_probs)
        loss.backward()
        self._optimizer_policy.step()
        
        return loss.detach().cpu().numpy()
    
    def solve(self, verbose=True, eval_interval=10, checkpoint_interval=0, 
              model_dir=None, save_prefix=None, game=None, start_iteration=0,
              eval_with_games=False, num_test_games=50):
        """运行并行 DeepCFR 训练
        
        Args:
            verbose: 是否显示详细信息
            eval_interval: 评估间隔
            checkpoint_interval: checkpoint 保存间隔（0=不保存）
            model_dir: 模型保存目录
            save_prefix: 保存文件前缀
            game: 游戏实例（用于保存 checkpoint）
            start_iteration: 起始迭代次数（用于恢复训练）
            eval_with_games: 是否在评估时运行测试对局
        
        Returns:
            policy_network: 训练好的策略网络
            advantage_losses: 优势网络损失历史
            policy_loss: 策略网络最终损失
        """
        print("=" * 70)
        print("并行 DeepCFR 训练")
        print("=" * 70)
        print(f"  Worker 数量: {self.num_workers}")
        print(f"  迭代次数: {self.num_iterations}")
        if start_iteration > 0:
            print(f"  从迭代 {start_iteration + 1} 恢复")
        print(f"  每次迭代遍历次数: {self.num_traversals}")
        print(f"  设备: {self.device}")
        print()
        
        # 启动 Worker
        self._start_workers()
        
        advantage_losses = {p: [] for p in range(self.num_players)}
        start_time = time.time()
        
        try:
            # 等待 Worker 启动并开始产生样本
            print("  等待 Worker 启动...", end="", flush=True)
            warmup_time = 0
            max_warmup = 30  # 最多等待 30 秒
            while warmup_time < max_warmup:
                time.sleep(1)
                warmup_time += 1
                self._collect_samples()
                total_samples = sum(len(m) for m in self._advantage_memories)
                if total_samples > 0:
                    print(f" 就绪 (耗时 {warmup_time} 秒，已收集 {total_samples} 个样本)")
                    break
                print(".", end="", flush=True)
            else:
                print(f" 警告: Worker 启动超时，继续训练...")
            
            for iteration in range(start_iteration, self.num_iterations):
                iter_start = time.time()
                
                # 更新迭代计数器
                self._iteration_counter.value = iteration + 1
                
                # 动态收集样本：直到收集到足够数量的新样本
                # 这样可以确保每次迭代的数据量是恒定的，不受 Worker 速度影响
                # 同时通过循环 sleep(1) 避免了主进程长时间无响应
                
                current_total_samples = sum(len(m) for m in self._advantage_memories)
                # 目标：本轮新增 num_traversals 个样本
                # 注意：由于可能有多个 Worker 同时提交，可能会略多一点，没关系
                target_total_samples = current_total_samples + self.num_traversals
                
                # 设置一个超时保护（例如 10 分钟），防止 Worker 全部挂死导致主进程死循环
                collection_start_time = time.time()
                last_sample_count = current_total_samples
                no_progress_count = 0  # 连续无进展次数
                
                # 关键修复：样本收集循环中只清理队列积压，不清理缓冲区
                # 优化：根据队列状态动态调整sleep时间，队列满时不sleep，队列空时才sleep
                while True:
                    # 检查队列状态，决定是否需要sleep
                    queue_sizes = [q.qsize() for q in self._advantage_queues]
                    strategy_queue_size = self._strategy_queue.qsize()
                    total_queue_size = sum(queue_sizes) + strategy_queue_size
                    max_queue_size = max(max(queue_sizes) if queue_sizes else 0, strategy_queue_size)
                    queue_usage = max_queue_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
                    
                    self._collect_samples()  # 内部只清理队列积压
                    new_current_samples = sum(len(m) for m in self._advantage_memories)
                    
                    # 检查是否达标
                    if new_current_samples >= target_total_samples:
                        break
                    
                    # 检查是否有进展
                    if new_current_samples > last_sample_count:
                        last_sample_count = new_current_samples
                        no_progress_count = 0
                    else:
                        no_progress_count += 1
                    
                    # 检查超时 (10分钟)
                    elapsed_time = time.time() - collection_start_time
                    if elapsed_time > 600:
                        collected = new_current_samples - current_total_samples
                        if verbose:
                            print(f"\n  ⚠️ 警告: 样本收集超时 (已收集 {collected}/{self.num_traversals})")
                            # 诊断信息
                            print(f"    诊断信息:")
                            print(f"      - 耗时: {elapsed_time:.1f}秒")
                            print(f"      - 当前优势样本总数: {new_current_samples:,}")
                            print(f"      - 策略样本总数: {len(self._strategy_memories):,}")
                            
                            # 检查 Worker 状态
                            alive_workers = sum(1 for p in self._workers if p.is_alive())
                            print(f"      - 存活的 Worker: {alive_workers}/{self.num_workers}")
                            
                            # 检查队列状态
                            queue_sizes = [q.qsize() for q in self._advantage_queues]
                            total_queue_size = sum(queue_sizes)
                            print(f"      - 队列中待处理样本: {total_queue_size:,}")
                            
                            # 检查内存使用情况
                            if HAS_PSUTIL:
                                try:
                                    process = psutil.Process()
                                    mem_info = process.memory_info()
                                    mem_mb = mem_info.rss / 1024 / 1024
                                    mem_percent = process.memory_percent()
                                    print(f"      - 主进程内存使用: {mem_mb:.1f}MB ({mem_percent:.1f}%)")
                                    
                                    # 检查系统内存
                                    sys_mem = psutil.virtual_memory()
                                    print(f"      - 系统内存: {sys_mem.percent:.1f}% 已使用 ({sys_mem.used/1024/1024/1024:.1f}GB / {sys_mem.total/1024/1024/1024:.1f}GB)")
                                    
                                    if sys_mem.percent > 90:
                                        print(f"      ⚠️ 系统内存使用率过高 ({sys_mem.percent:.1f}%)，可能导致 OOM")
                                except:
                                    pass
                            
                            # 如果 Worker 全部死亡，抛出异常
                            if alive_workers == 0:
                                raise RuntimeError("所有 Worker 进程已死亡，无法继续训练。")
                            
                            # 如果队列为空且 Worker 存活，可能是 Worker 陷入死锁或内存不足
                            if total_queue_size == 0 and alive_workers > 0:
                                print(f"      ⚠️ 队列为空但 Worker 存活，可能陷入死锁或内存不足")
                                print(f"      建议: 检查系统日志 (dmesg | grep -i oom) 查看是否有 OOM 杀死进程")
                        break
                    
                    # 如果连续多次无进展，提前警告
                    if no_progress_count >= 20 and verbose:  # 10秒无进展（20次 × 0.5秒）
                        print(f"\n  ⚠️ 警告: 连续 {no_progress_count * 0.5:.1f}秒无样本收集进展")
                        no_progress_count = 0  # 重置计数器，避免重复打印
                    
                    # 优化：根据队列状态动态调整sleep时间
                    # 队列使用率 > 80% 时不sleep，队列空时才sleep，避免消费速度不够快
                    if queue_usage > 0.80:
                        # 队列满了，不sleep，立即继续消费
                        pass
                    elif total_queue_size == 0:
                        # 队列为空，sleep 0.5秒，避免CPU空转
                        time.sleep(0.5)
                    else:
                        # 队列有数据但不满，sleep 0.1秒，平衡消费速度和CPU使用率
                        time.sleep(0.1)
                
                # 关键修复：样本收集完成后，清理缓冲区（避免在收集过程中删除样本）
                # 这样可以避免清理操作导致收集进度倒退
                self._check_and_cleanup_memory(cleanup_buffers=True)
                
                # 训练优势网络
                for player in range(self.num_players):
                    loss = self._learn_advantage_network(player)
                    if loss is not None:
                        advantage_losses[player].append(loss)
                
                # 训练策略网络
                # 为了加速 checkpoint 保存时的策略网络更新，我们在每次迭代中增量训练策略网络
                # 这样可以分摊计算成本，使得 checkpoint 时策略网络已经接近就绪
                        policy_loss = self._learn_strategy_network()
                
                # 同步网络参数到 Worker
                if (iteration + 1) % self.sync_interval == 0:
                    self._sync_network_params()
                
                self._iteration += 1
                
                iter_time = time.time() - iter_start
                
                if verbose:
                    # 显示队列状态和消费速度信息
                    queue_info = []
                    max_queue_usage = 0.0
                    for player in range(self.num_players):
                        q_size = self._advantage_queues[player].qsize()
                        usage = q_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
                        max_queue_usage = max(max_queue_usage, usage)
                        if q_size > 0:
                            queue_info.append(f"玩家{player}:{q_size}")
                    
                    strategy_q_size = self._strategy_queue.qsize()
                    strategy_usage = strategy_q_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
                    max_queue_usage = max(max_queue_usage, strategy_usage)
                    
                    queue_status = ""
                    if queue_info or strategy_q_size > 0:
                        queue_status = f" | 队列: {', '.join(queue_info)}"
                        if strategy_q_size > 0:
                            queue_status += f",策略:{strategy_q_size}"
                        
                        # 显示调整信息
                        adj_info = self._adaptive_max_collect.get('last_adjustment', {})
                        cpu_info = ""
                        if adj_info.get('cpu_percent') is not None:
                            cpu_info = f", CPU:{adj_info['cpu_percent']:.0f}%"
                        growth_info = ""
                        if adj_info.get('growth_rate', 0) > 0:
                            growth_info = f", 增长率:{adj_info['growth_rate']:.0f}/s"
                        
                        queue_status += f" (使用率:{max_queue_usage*100:.0f}%{growth_info}{cpu_info}, max_collect:{self._adaptive_max_collect['current']:,})"
                    
                    get_logger().info(f"  迭代 {iteration + 1}/{self.num_iterations} "
                          f"(耗时: {iter_time:.2f}秒) | "
                          f"优势样本: {sum(len(m) for m in self._advantage_memories):,} | "
                          f"策略样本: {len(self._strategy_memories):,}{queue_status}")
                
                if (iteration + 1) % eval_interval == 0:
                    print()
                    for player, losses in advantage_losses.items():
                        if losses:
                            print(f"    玩家 {player} 损失: {losses[-1]:.6f}")
                    
                    # 运行评估
                    if game is not None:
                        try:
                            from training_evaluator import quick_evaluate
                            print(f"  评估训练效果...", end="", flush=True)
                            eval_result = quick_evaluate(
                                game,
                                self,
                                include_test_games=eval_with_games,
                                num_test_games=num_test_games,
                                max_depth=None,
                                verbose=True  # 启用详细输出以查看错误
                            )
                            get_logger().info(" 完成")
                            
                            # 打印简要评估信息
                            metrics = eval_result['metrics']
                            print(f"    策略熵: {metrics.get('avg_entropy', 0):.4f} | "
                                  f"策略缓冲区: {len(self._strategy_memories):,} | "
                                  f"优势样本: {sum(len(m) for m in self._advantage_memories):,}")
                            
                            if eval_with_games and eval_result.get('test_results'):
                                test_results = eval_result['test_results']
                                num_games = test_results.get('games_played', 0)
                                mode = test_results.get('mode', 'unknown')
                                num_players = test_results.get('num_players', 0)
                                
                                # 获取大盲注值
                                bb = None
                                try:
                                    game_string = str(game)
                                    blind_match = re.search(r'blind=([\d\s]+)', game_string)
                                    if blind_match:
                                        blinds = [int(x) for x in blind_match.group(1).strip().split()]
                                        bb = max([b for b in blinds if b > 0], default=None)
                                except:
                                    pass
                                
                                if num_games > 0:
                                    if mode == "self_play":
                                        # 自对弈模式：显示所有位置
                                        print(f"    测试对局: {num_games} 局 (自对弈)")
                                        for i in range(num_players):
                                            avg_return = test_results.get(f'player{i}_avg_return', 0)
                                            win_rate = test_results.get(f'player{i}_win_rate', 0) * 100
                                            if bb is not None and bb > 0:
                                                bb_value = avg_return / bb
                                                print(f"      玩家{i}: 平均回报 {avg_return:.2f} ({bb_value:+.2f} BB), 胜率 {win_rate:.1f}%")
                                            else:
                                                print(f"      玩家{i}: 平均回报 {avg_return:.2f}, 胜率 {win_rate:.1f}%")
                                    else:
                                        # vs_random模式：显示所有位置使用训练策略时的表现
                                        print(f"    测试对局: {num_games} 局 (vs Random, 随机位置)")
                                        # 显示各位置的表现
                                        position_stats = []
                                        for i in range(num_players):
                                            trained_count = test_results.get(f'player{i}_trained_count', 0)
                                            if trained_count > 0:
                                                avg_return = test_results.get(f'player{i}_trained_avg_return', 0)
                                                win_rate = test_results.get(f'player{i}_trained_win_rate', 0) * 100
                                                if bb is not None and bb > 0:
                                                    bb_value = avg_return / bb
                                                    position_stats.append(f"玩家{i}: {trained_count}局, 回报{avg_return:.0f} ({bb_value:+.2f}BB), 胜率{win_rate:.0f}%")
                                                else:
                                                    position_stats.append(f"玩家{i}: {trained_count}局, 回报{avg_return:.0f}, 胜率{win_rate:.0f}%")
                                        
                                        if position_stats:
                                            print(f"      {' | '.join(position_stats)}")
                                        
                                        # 显示总体统计
                                        overall_avg_return = test_results.get('player0_avg_return', 0)
                                        overall_win_rate = test_results.get('player0_win_rate', 0) * 100
                                        if bb is not None and bb > 0:
                                            bb_value = overall_avg_return / bb
                                            print(f"      总体: 平均回报 {overall_avg_return:.2f} ({bb_value:+.2f} BB), 胜率 {overall_win_rate:.1f}%")
                                        else:
                                            print(f"      总体: 平均回报 {overall_avg_return:.2f}, 胜率 {overall_win_rate:.1f}%")
                        except ImportError:
                            pass  # training_evaluator 不可用
                        except Exception as e:
                            print(f" 评估失败: {e}")
                
                    # 保存 checkpoint
                    if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                        if model_dir and save_prefix and game:
                            get_logger().info(f"\n  💾 保存 checkpoint (迭代 {iteration + 1})...")
                            try:
                                # 优化：checkpoint时只训练1次，提升保存速度
                                get_logger().info("    正在训练策略网络 (用于 Checkpoint)...")
                                policy_loss = self._learn_strategy_network()
                                if policy_loss is not None:
                                    get_logger().info(f"    完成 (Loss: {policy_loss:.6f})")
                                else:
                                    get_logger().info("    完成 (无足够样本训练)")
                                
                                save_checkpoint(self, game, model_dir, save_prefix, iteration + 1)
                                get_logger().info("  ✓ Checkpoint 已保存")
                            except Exception as e:
                                print(f" 失败: {e}")
            
            print()
            
            # 训练策略网络
            print("  训练策略网络...")
            policy_loss = self._learn_strategy_network()
            
            total_time = time.time() - start_time
            print(f"\n  ✓ 训练完成！总耗时: {total_time:.2f} 秒")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 训练被用户中断")
            if model_dir and save_prefix and game:
                get_logger().info(f"  💾 保存中断时的 checkpoint (迭代 {self._iteration})...")
                try:
                    save_checkpoint(self, game, model_dir, save_prefix, self._iteration)
                    get_logger().info(f"  ✓ Checkpoint 已保存")
                except Exception as e:
                    get_logger().error(f"  ✗ 保存失败: {e}")
        finally:
            # 停止 Worker
            self._stop_workers()
        
        return self._policy_network, advantage_losses, policy_loss
    
    def action_probabilities(self, state, player_id=None):
        """计算动作概率（用于推理）"""
        del player_id
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        with torch.no_grad():
            logits = self._policy_network(
                torch.FloatTensor(info_state_vector).to(self.device)
            )
            probs = self._policy_sm(logits).cpu().numpy()
        
        # 确保只返回合法动作的概率，并重新归一化
        action_probs = {action: float(probs[0][action]) for action in legal_actions}
        total_prob = sum(action_probs.values())
        
        if total_prob > 1e-10:
            # 重新归一化
            action_probs = {a: p / total_prob for a, p in action_probs.items()}
        else:
            # 如果所有概率都接近0，使用均匀分布
            action_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            
        return action_probs


def load_checkpoint(solver, model_dir, save_prefix, game):
    """从 checkpoint 加载网络权重
    
    Args:
        solver: ParallelDeepCFRSolver 实例
        model_dir: 模型目录
        save_prefix: 保存前缀
        game: 游戏实例
        
    Returns:
        start_iteration: 恢复的迭代次数
    """
    import glob
    import re
    
    # 查找最新的 checkpoint
    checkpoint_root = os.path.join(model_dir, "checkpoints")
    
    latest_file = None
    max_iter = 0
    
    # 优先从 checkpoints 目录加载
    if os.path.exists(checkpoint_root):
        # 尝试新的目录结构: checkpoints/iter_X/prefix_policy_iterX.pt
        iter_dirs = glob.glob(os.path.join(checkpoint_root, "iter_*"))
        for d in iter_dirs:
            match = re.search(r'iter_(\d+)$', d)
            if match:
                iter_num = int(match.group(1))
                policy_file = os.path.join(d, f"{save_prefix}_policy_network_iter{iter_num}.pt")
                if os.path.exists(policy_file) and iter_num > max_iter:
                    max_iter = iter_num
                    latest_file = policy_file
        
        # 如果没找到（或者是旧结构），尝试旧的扁平结构: checkpoints/prefix_policy_iterX.pt
        if latest_file is None:
            policy_files = glob.glob(os.path.join(checkpoint_root, f"{save_prefix}_policy_network_iter*.pt"))
            for f in policy_files:
                match = re.search(r'_iter(\d+)\.pt$', f)
                if match:
                    iter_num = int(match.group(1))
                    if iter_num > max_iter:
                        max_iter = iter_num
                        latest_file = f
            
        if latest_file:
            print(f"  找到 checkpoint: 迭代 {max_iter}")
            policy_path = latest_file
            start_iteration = max_iter
        else:
            policy_path = None
            start_iteration = 0
    else:
        # ... (后续逻辑不变)
        policy_path = None
        start_iteration = 0

    if policy_path is None: # 如果 checkpoint 没找到，尝试加载最终模型
        policy_path = os.path.join(model_dir, f"{save_prefix}_policy_network.pt")
        if os.path.exists(policy_path):
            print(f"  找到最终模型")
            start_iteration = 0  # 最终模型没有迭代信息
        else:
            policy_path = None
            start_iteration = 0
    
    if policy_path is None or not os.path.exists(policy_path):
        print(f"  ✗ 未找到可加载的模型")
        return 0
    
    # 加载策略网络
    print(f"  加载策略网络: {policy_path}")
    policy_state = torch.load(policy_path, map_location=solver.device)
    policy_net = solver._policy_network
    if isinstance(policy_net, nn.DataParallel):
        policy_net.module.load_state_dict(policy_state)
    else:
        policy_net.load_state_dict(policy_state)
    print(f"  ✓ 策略网络已加载")
    
    # 加载优势网络
    # 确定优势网络所在的目录
    if start_iteration > 0:
        # 检查是新结构还是旧结构
        if "iter_" in os.path.dirname(policy_path):
            # 新结构: checkpoints/iter_X/
            adv_dir = os.path.dirname(policy_path)
        else:
            # 旧结构: checkpoints/
            adv_dir = checkpoint_root
    else:
        adv_dir = model_dir

    for player in range(game.num_players()):
        if start_iteration > 0:
            adv_path = os.path.join(adv_dir, f"{save_prefix}_advantage_player_{player}_iter{start_iteration}.pt")
        else:
            adv_path = os.path.join(adv_dir, f"{save_prefix}_advantage_player_{player}.pt")
        
        if os.path.exists(adv_path):
            # ... (加载逻辑不变)
            adv_state = torch.load(adv_path, map_location=solver.device)
            adv_net = solver._advantage_networks[player]
            if isinstance(adv_net, nn.DataParallel):
                adv_net.module.load_state_dict(adv_state)
            else:
                adv_net.load_state_dict(adv_state)
            print(f"  ✓ 玩家 {player} 优势网络已加载")
        else:
            print(f"  ⚠️ 玩家 {player} 优势网络未找到: {adv_path}")
    
    return start_iteration


def create_save_directory(save_prefix, save_dir="models"):
    """创建保存目录"""
    import time as time_module
    base_dir = save_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    model_dir = os.path.join(base_dir, save_prefix)
    if os.path.exists(model_dir):
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        model_dir = f"{model_dir}_{timestamp}"
    
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_checkpoint(solver, game, model_dir, save_prefix, iteration, is_final=False):
    """保存 checkpoint"""
    if is_final:
        suffix = ""
        checkpoint_dir = model_dir
    else:
        suffix = f"_iter{iteration}"
        # 将每个 iteration 的 checkpoint 放入独立子目录
        checkpoint_dir = os.path.join(model_dir, "checkpoints", f"iter_{iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存策略网络（处理 DataParallel）
    policy_path = os.path.join(checkpoint_dir, f"{save_prefix}_policy_network{suffix}.pt")
    policy_net = solver._policy_network
    if isinstance(policy_net, nn.DataParallel):
        torch.save(policy_net.module.state_dict(), policy_path)
    else:
        torch.save(policy_net.state_dict(), policy_path)
    
    # 保存优势网络（处理 DataParallel）
    for player in range(game.num_players()):
        advantage_path = os.path.join(checkpoint_dir, f"{save_prefix}_advantage_player_{player}{suffix}.pt")
        adv_net = solver._advantage_networks[player]
        if isinstance(adv_net, nn.DataParallel):
            torch.save(adv_net.module.state_dict(), advantage_path)
        else:
            torch.save(adv_net.state_dict(), advantage_path)
    
    return checkpoint_dir


def main():
    global logger
    # 初始化logging（输出到stdout，nohup会捕获）
    logger = setup_logging()
    
    # 注册信号处理，确保被 kill 时也能清理子进程
    def signal_handler(signum, frame):
        get_logger().info(f"\n接收到信号 {signum}，正在清理并退出...")
        # 注意：这里不能直接调用 solver._stop_workers() 因为 solver 不在作用域内
        # 但由于 worker 进程已设置为 daemon=True，主进程退出时它们会自动被系统清理
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="多进程并行 DeepCFR 训练")
    parser.add_argument("--num_players", type=int, default=2, help="玩家数量")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker 进程数量")
    parser.add_argument("--num_iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--num_traversals", type=int, default=40, help="每次迭代遍历次数")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--advantage_layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--memory_capacity", type=int, default=1000000,
                        help="经验回放缓冲区容量（默认: 1000000）")
    parser.add_argument("--max_memory_gb", type=float, default=None,
                        help="最大内存限制（GB），超过此限制会自动清理旧样本（默认: 不限制）")
    parser.add_argument("--queue_maxsize", type=int, default=50000,
                        help="队列最大大小，降低可减少内存占用（默认: 50000）")
    parser.add_argument("--betting_abstraction", type=str, default="fcpa")
    parser.add_argument("--save_prefix", type=str, default="deepcfr_parallel")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=0, 
                        help="Checkpoint 保存间隔（0=不保存中间checkpoint）")
    parser.add_argument("--skip_nashconv", action="store_true", 
                        help="跳过 NashConv 计算（6人局强烈建议）")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="使用 GPU 训练")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="使用的 GPU ID 列表（例如 --gpu_ids 0 1 2 3）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定目录恢复训练（例如 --resume models/deepcfr_parallel_6p）")
    parser.add_argument("--eval_with_games", action="store_true",
                        help="评估时运行测试对局")
    parser.add_argument("--num_test_games", type=int, default=50,
                        help="评估时的测试对局数量（默认: 50）")
    parser.add_argument("--blinds", type=str, default=None,
                        help="盲注配置，格式：'小盲 大盲' 或 '50 100 0 0 0 0'（多人场完整配置）。如果不指定，将根据玩家数量自动生成")
    parser.add_argument("--stack_size", type=int, default=None,
                        help="每个玩家的初始筹码（默认: 2000）。如果不指定，将使用默认值2000")
    
    args = parser.parse_args()
    
    # 处理恢复训练配置（必须在创建游戏之前）
    if args.resume:
        model_dir = args.resume
        if not os.path.exists(model_dir):
            print(f"✗ 恢复目录不存在: {model_dir}")
            import sys
            sys.exit(1)
        
        # 尝试从 config.json 读取配置
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                resume_config = json.load(f)
            
            print(f"从 {model_dir} 恢复训练")
            
            # 自动覆盖关键参数，确保网络结构一致
            # 注意：命令行显式指定的参数优先级应该更高，但为了简化续训，这里默认使用 config 中的值
            # 除非用户想要改变某些训练超参数（如 batch_size, learning_rate）
            
            if 'num_players' in resume_config:
                args.num_players = resume_config['num_players']
            if 'policy_layers' in resume_config:
                args.policy_layers = resume_config['policy_layers']
            if 'advantage_layers' in resume_config:
                args.advantage_layers = resume_config['advantage_layers']
            if 'betting_abstraction' in resume_config:
                args.betting_abstraction = resume_config['betting_abstraction']
            if 'save_prefix' in resume_config:
                args.save_prefix = resume_config['save_prefix']
            if 'num_traversals' in resume_config:
                args.num_traversals = resume_config['num_traversals']
            if 'blinds' in resume_config and args.blinds is None:
                args.blinds = resume_config['blinds']
            if 'stack_size' in resume_config and args.stack_size is None:
                args.stack_size = resume_config['stack_size']
                
            print(f"  自动加载配置: {args.num_players}人局, 策略层{args.policy_layers}, 优势层{args.advantage_layers}")
            print(f"  save_prefix: {args.save_prefix}")
        else:
            print(f"⚠️ 未找到 config.json，使用命令行参数")

    # 创建游戏
    num_players = args.num_players
    
    # 处理盲注配置
    if args.blinds is not None:
        # 如果用户指定了盲注，直接使用
        blinds_str = args.blinds
        print(f"  使用指定的盲注配置: {blinds_str}")
    else:
        # 否则根据玩家数量自动生成
        if num_players == 2:
            blinds_str = "100 50"
        else:
            blinds_list = ["50", "100"] + ["0"] * (num_players - 2)
            blinds_str = " ".join(blinds_list)
        print(f"  使用默认盲注配置: {blinds_str}")
    
    # 处理筹码配置
    stack_size = args.stack_size if args.stack_size is not None else 2000
    stacks_str = " ".join([str(stack_size)] * num_players)
    print(f"  每个玩家初始筹码: {stack_size}")
    
    # 处理行动顺序配置
    if num_players == 2:
        first_player_str = "2 1 1 1"
    else:
        first_player_str = " ".join(["3"] + ["1"] * 3)
    
    game_string = (
        f"universal_poker("
        f"betting=nolimit,"
        f"numPlayers={num_players},"
        f"numRounds=4,"
        f"blind={blinds_str},"
        f"stack={stacks_str},"
        f"numHoleCards=2,"
        f"numBoardCards=0 3 1 1,"
        f"firstPlayer={first_player_str},"
        f"numSuits=4,"
        f"numRanks=13,"
        f"bettingAbstraction={args.betting_abstraction}"
        f")"
    )
    
    # 设置设备
    gpu_ids = None
    if args.use_gpu and torch.cuda.is_available():
        if args.gpu_ids is not None and len(args.gpu_ids) > 0:
            gpu_ids = args.gpu_ids
            device = f"cuda:{gpu_ids[0]}"
            if len(gpu_ids) > 1:
                print(f"使用多 GPU: {gpu_ids}")
                for gid in gpu_ids:
                    print(f"  GPU {gid}: {torch.cuda.get_device_name(gid)}")
            else:
                print(f"使用 GPU: {torch.cuda.get_device_name(gpu_ids[0])}")
        else:
            device = "cuda:0"
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("使用 CPU")
    
    print(f"创建游戏: {game_string}")
    game = pyspiel.load_game(game_string)
    
    # 处理恢复训练
    start_iteration = 0
    if args.resume:
        # 目录检查已在前面完成
        model_dir = args.resume
    else:
        # 创建新的保存目录
        model_dir = create_save_directory(args.save_prefix, args.save_dir)
    
    print(f"模型保存目录: {model_dir}")
    if args.checkpoint_interval > 0:
        print(f"Checkpoint 保存间隔: 每 {args.checkpoint_interval} 次迭代")
    
    # 创建求解器
    solver = ParallelDeepCFRSolver(
        game,
        num_workers=args.num_workers,
        policy_network_layers=tuple(args.policy_layers),
        advantage_network_layers=tuple(args.advantage_layers),
        num_iterations=args.num_iterations,
        num_traversals=args.num_traversals,
        learning_rate=args.learning_rate,
        batch_size_advantage=args.batch_size,
        batch_size_strategy=args.batch_size,
        memory_capacity=args.memory_capacity,
        device=device,
        gpu_ids=gpu_ids,
        max_memory_gb=args.max_memory_gb,
        queue_maxsize=args.queue_maxsize,
    )
    
    # 显示内存配置
    if args.max_memory_gb:
        print(f"  内存限制: {args.max_memory_gb}GB")
    print(f"  队列大小: {args.queue_maxsize}")
    
    # 立即保存配置（方便在训练过程中查看或恢复）
    if not args.resume:
        import json
        config_path = os.path.join(model_dir, "config.json")
        config = {
            'num_players': num_players,
            'num_workers': args.num_workers,
            'num_iterations': args.num_iterations,
            'num_traversals': args.num_traversals,
            'policy_layers': args.policy_layers,
            'advantage_layers': args.advantage_layers,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'memory_capacity': args.memory_capacity,
            'max_memory_gb': args.max_memory_gb,
            'queue_maxsize': args.queue_maxsize,
            'betting_abstraction': args.betting_abstraction,
            'blinds': blinds_str,
            'stack_size': stack_size,
            'device': device,
            'gpu_ids': gpu_ids,
            'game_string': game_string,
            'multi_gpu': gpu_ids is not None and len(gpu_ids) > 1,
            'parallel': True,
            'use_feature_transform': True,
            'use_simple_feature': True,
            'save_prefix': args.save_prefix,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ 配置已保存: {config_path}")
    
    # 如果是恢复训练，加载 checkpoint
    if args.resume:
        print(f"\n加载 checkpoint...")
        start_iteration = load_checkpoint(solver, model_dir, args.save_prefix, game)
        if start_iteration > 0:
            print(f"  ✓ 将从迭代 {start_iteration + 1} 继续训练")
        else:
            print(f"  ⚠️ 未找到有效 checkpoint，从头开始训练")
    
    # 训练（带 checkpoint 支持）
    policy_network, advantage_losses, policy_loss = solver.solve(
        verbose=True,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        model_dir=model_dir,
        save_prefix=args.save_prefix,
        game=game,
        start_iteration=start_iteration,
        eval_with_games=args.eval_with_games,
        num_test_games=args.num_test_games,
    )
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    save_checkpoint(solver, game, model_dir, args.save_prefix, args.num_iterations, is_final=True)
    print(f"  ✓ 策略网络已保存: {os.path.join(model_dir, f'{args.save_prefix}_policy_network.pt')}")
    for player in range(num_players):
        print(f"  ✓ 玩家 {player} 优势网络已保存")
    
    # 保存配置
    import json
    config_path = os.path.join(model_dir, "config.json")
    config = {
        'num_players': num_players,
        'num_workers': args.num_workers,
        'num_iterations': args.num_iterations,
        'num_traversals': args.num_traversals,
        'policy_layers': args.policy_layers,
        'advantage_layers': args.advantage_layers,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'memory_capacity': args.memory_capacity,
        'max_memory_gb': args.max_memory_gb,
        'queue_maxsize': args.queue_maxsize,
        'betting_abstraction': args.betting_abstraction,
        'device': device,
        'gpu_ids': gpu_ids,
        'game_string': game_string,
        'multi_gpu': gpu_ids is not None and len(gpu_ids) > 1,
        'parallel': True,
        'use_feature_transform': True,
        'use_simple_feature': True,
        'save_prefix': args.save_prefix,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ 配置已保存: {config_path}")
    
    # NashConv 计算（可选）
    if not args.skip_nashconv:
        print(f"\n计算 NashConv...")
        if num_players > 2:
            print(f"  ⚠️ 警告: {num_players} 人游戏的 NashConv 计算可能非常慢或不可行")
            print(f"  建议: 使用 --skip_nashconv 跳过")
        try:
            from open_spiel.python import policy
            
            average_policy = policy.tabular_policy_from_callable(
                game, solver.action_probabilities
            )
            pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
            conv = pyspiel.nash_conv(game, pyspiel_policy, use_cpp_br=True)
            print(f"  ✓ NashConv: {conv:.6f}")
        except Exception as e:
            print(f"  ✗ NashConv 计算失败: {e}")
            print(f"  建议: 使用 --skip_nashconv 跳过")
    else:
        print(f"\n  ⏭️ 跳过 NashConv 计算")
    
    print("\n" + "=" * 70)
    print("✓ 并行 DeepCFR 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

