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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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


class RandomReplacementBuffer:
    """Reservoir Sampling 缓冲区（NumPy 优化版）
    
    使用 NumPy 数组存储数据，彻底避免 Python 对象访问瓶颈。
    - info_states: [capacity, info_state_size] float32
    - iterations: [capacity] int32
    - advantages/action_probs: [capacity, num_actions] float32
    
    优化效果：数据准备时间从 ~700ms 降到 ~5ms（140x 加速）
    """
    
    def __init__(self, capacity, info_state_size=None, num_actions=None, buffer_type='advantage'):
        """初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
            info_state_size: 信息状态维度（首次添加时自动检测）
            num_actions: 动作数量（首次添加时自动检测）
            buffer_type: 'advantage' 或 'strategy'
        """
        self._capacity = capacity
        self._buffer_type = buffer_type
        self._size = 0  # 当前有效样本数
        self._add_calls = 0  # 添加计数器（Reservoir Sampling必需）
        
        # 延迟初始化（首次添加时确定维度）
        self._info_state_size = info_state_size
        self._num_actions = num_actions
        self._initialized = False
        
        # NumPy 数组（延迟初始化）
        self._info_states = None
        self._iterations = None
        self._advantages = None  # 用于 advantage buffer
        self._action_probs = None  # 用于 strategy buffer
    
    def _lazy_init(self, element):
        """延迟初始化：根据第一个样本确定维度"""
        if self._initialized:
            return
        
        # 自动检测维度
        if self._info_state_size is None:
            self._info_state_size = len(element.info_state)
        
        if self._buffer_type == 'advantage':
            if self._num_actions is None:
                self._num_actions = len(element.advantage)
            # 预分配数组
            self._info_states = np.zeros((self._capacity, self._info_state_size), dtype=np.float32)
            self._iterations = np.zeros(self._capacity, dtype=np.int32)
            self._advantages = np.zeros((self._capacity, self._num_actions), dtype=np.float32)
        else:  # strategy
            if self._num_actions is None:
                self._num_actions = len(element.strategy_action_probs)
            # 预分配数组
            self._info_states = np.zeros((self._capacity, self._info_state_size), dtype=np.float32)
            self._iterations = np.zeros(self._capacity, dtype=np.int32)
            self._action_probs = np.zeros((self._capacity, self._num_actions), dtype=np.float32)
        
        self._initialized = True
    
    def add(self, element):
        """添加样本（FIFO替换，确保新样本不会被随机替换掉）"""
        if self._capacity == 0:
            return
        
        # 延迟初始化
        self._lazy_init(element)
        
        # 确定写入位置
        if self._size < self._capacity:
            idx = self._size
            self._size += 1
        else:
            # FIFO替换：循环替换，确保新样本不会被随机替换掉
            # 这样可以保持样本的时效性，新样本不会被过早替换
            idx = self._add_calls % self._capacity
        
        # 写入数据（NumPy 数组操作，无 GIL）
        self._info_states[idx] = element.info_state
        self._iterations[idx] = element.iteration
        
        if self._buffer_type == 'advantage':
            adv = element.advantage
            if len(adv) >= self._num_actions:
                self._advantages[idx] = adv[:self._num_actions]
            else:
                self._advantages[idx, :len(adv)] = adv
                self._advantages[idx, len(adv):] = 0
        else:  # strategy
            probs = element.strategy_action_probs
            if len(probs) >= self._num_actions:
                self._action_probs[idx] = probs[:self._num_actions]
            else:
                self._action_probs[idx, :len(probs)] = probs
                self._action_probs[idx, len(probs):] = 0
        
        self._add_calls += 1
    
    def sample(self, num_samples, current_iteration=None, new_sample_ratio=0.5, new_sample_window=0):
        """采样（返回 NumPy 数组，而不是 Python 对象列表）
        
        修改：支持按比例分层采样（新样本/老样本），用于提升“新样本占比”，降低分布漂移带来的训练抖动。
        - 新样本：iteration ∈ [current_iteration - new_sample_window, current_iteration]
          - new_sample_window=0 时退化为 iteration == current_iteration（旧行为）
        - 老样本：其他 iteration
        
        返回:
            dict: {
                'info_states': [batch, info_state_size],
                'iterations': [batch],
                'advantages': [batch, num_actions] (仅 advantage buffer),
                'action_probs': [batch, num_actions] (仅 strategy buffer),
                'new_sample_count': int
            }
        """
        if self._size == 0 or not self._initialized:
            return self._empty_result()
        
        actual_num_samples = min(num_samples, self._size)
        
        # 如果提供 current_iteration，则优先分层采样，确保新样本占比
        if current_iteration is not None and new_sample_ratio is not None:
            try:
                ratio = float(new_sample_ratio)
            except Exception:
                ratio = 0.5
            ratio = max(0.0, min(1.0, ratio))
            try:
                window = int(new_sample_window) if new_sample_window is not None else 0
            except Exception:
                window = 0
            window = max(0, window)
            return self._stratified_sample_numpy(actual_num_samples, int(current_iteration), ratio, window)
        
        # 否则随机采样
        return self._random_sample_numpy(actual_num_samples, current_iteration, new_sample_window=new_sample_window)
    
    def _empty_result(self):
        """返回空结果"""
        if self._buffer_type == 'advantage':
            return {
                'info_states': np.array([], dtype=np.float32).reshape(0, self._info_state_size or 1),
                'iterations': np.array([], dtype=np.int32),
                'advantages': np.array([], dtype=np.float32).reshape(0, self._num_actions or 1),
                'new_sample_count': 0
            }
        else:
            return {
                'info_states': np.array([], dtype=np.float32).reshape(0, self._info_state_size or 1),
                'iterations': np.array([], dtype=np.int32),
                'action_probs': np.array([], dtype=np.float32).reshape(0, self._num_actions or 1),
                'new_sample_count': 0
            }
    
    def _random_sample_numpy(self, num_samples, current_iteration=None, new_sample_window=0):
        """随机采样（纯 NumPy，无 GIL）
        
        修改：统计新样本数量，用于日志输出
        """
        indices = np.random.choice(self._size, num_samples, replace=False)
        
        # 统计新样本数量（用于日志输出）
        new_sample_count = 0
        if current_iteration is not None:
            try:
                window = int(new_sample_window) if new_sample_window is not None else 0
            except Exception:
                window = 0
            window = max(0, window)
            sampled_iterations = self._iterations[indices]
            if window <= 0:
                new_sample_count = np.sum(sampled_iterations == current_iteration)
            else:
                low = current_iteration - window
                new_sample_count = np.sum((sampled_iterations >= low) & (sampled_iterations <= current_iteration))
        
        return self._gather_by_indices(indices, new_sample_count=new_sample_count)
    
    def _stratified_sample_numpy(self, num_samples, current_iteration, new_sample_ratio, new_sample_window=0):
        """分层采样（纯 NumPy 向量化，无 GIL）"""
        # 向量化分离新旧样本（纯 NumPy，释放 GIL）
        valid_iterations = self._iterations[:self._size]
        if new_sample_window is None:
            new_sample_window = 0
        try:
            window = int(new_sample_window)
        except Exception:
            window = 0
        window = max(0, window)
        if window <= 0:
            new_mask = valid_iterations == current_iteration
        else:
            low = current_iteration - window
            new_mask = (valid_iterations >= low) & (valid_iterations <= current_iteration)
        new_indices = np.where(new_mask)[0]
        old_indices = np.where(~new_mask)[0]
        
        num_new_available = len(new_indices)
        num_old_available = len(old_indices)
        
        # 计算目标采样数量：优先满足 new_sample_ratio，但保证尽可能凑满 num_samples
        desired_new = int(num_samples * new_sample_ratio)
        desired_new = max(0, min(desired_new, num_new_available))
        desired_old = num_samples - desired_new
        desired_old = max(0, min(desired_old, num_old_available))
        
        # 如果老样本不足，用剩余的新样本补齐（允许超过 new_sample_ratio）
        remaining = num_samples - (desired_new + desired_old)
        if remaining > 0 and num_new_available > desired_new:
            extra_new = min(remaining, num_new_available - desired_new)
            desired_new += extra_new
            remaining -= extra_new
        
        # 如果新样本不足，再用剩余的老样本补齐
        if remaining > 0 and num_old_available > desired_old:
            extra_old = min(remaining, num_old_available - desired_old)
            desired_old += extra_old
            remaining -= extra_old
        
        # NumPy 随机采样（释放 GIL）
        sampled_indices = []
        
        if desired_new > 0 and num_new_available > 0:
            if desired_new >= num_new_available:
                sampled_indices.append(new_indices)
            else:
                sampled_indices.append(np.random.choice(new_indices, desired_new, replace=False))
        
        if desired_old > 0 and num_old_available > 0:
            if desired_old >= num_old_available:
                sampled_indices.append(old_indices)
            else:
                sampled_indices.append(np.random.choice(old_indices, desired_old, replace=False))
        
        if len(sampled_indices) == 0:
            return self._empty_result()
        
        all_indices = np.concatenate(sampled_indices)
        # 打散索引，避免“新样本/老样本分块”带来的潜在序列偏差
        if all_indices.shape[0] > 1:
            np.random.shuffle(all_indices)
        return self._gather_by_indices(all_indices, new_sample_count=desired_new)
    
    def _gather_by_indices(self, indices, new_sample_count=0):
        """根据索引收集数据（纯 NumPy 切片，无 GIL）"""
        result = {
            'info_states': self._info_states[indices].copy(),
            'iterations': self._iterations[indices].copy(),
            'new_sample_count': new_sample_count
        }
        
        if self._buffer_type == 'advantage':
            result['advantages'] = self._advantages[indices].copy()
        else:
            result['action_probs'] = self._action_probs[indices].copy()
        
        return result
    
    def sample_legacy(self, num_samples, current_iteration=None, new_sample_ratio=0.5, new_sample_window=0):
        """兼容旧接口：返回 namedtuple 列表（慢，仅用于兼容）"""
        result = self.sample(num_samples, current_iteration, new_sample_ratio, new_sample_window)
        
        if len(result['info_states']) == 0:
            return []
        
        samples = []
        for i in range(len(result['info_states'])):
            if self._buffer_type == 'advantage':
                samples.append(AdvantageMemory(
                    info_state=result['info_states'][i],
                    iteration=int(result['iterations'][i]),
                    advantage=result['advantages'][i],
                    action=0
                ))
            else:
                samples.append(StrategyMemory(
                    info_state=result['info_states'][i],
                    iteration=int(result['iterations'][i]),
                    strategy_action_probs=result['action_probs'][i]
                ))
        
        return samples
    
    def clear(self):
        """清空缓冲区"""
        self._size = 0
        self._add_calls = 0
        # 不需要重新分配数组，只需重置大小
    
    def __len__(self):
        return self._size
    
    def __iter__(self):
        """兼容旧接口：迭代器（慢，仅用于兼容）"""
        for i in range(self._size):
            if self._buffer_type == 'advantage':
                yield AdvantageMemory(
                    info_state=self._info_states[i],
                    iteration=int(self._iterations[i]),
                    advantage=self._advantages[i],
                    action=0
                )
            else:
                yield StrategyMemory(
                    info_state=self._info_states[i],
                    iteration=int(self._iterations[i]),
                    strategy_action_probs=self._action_probs[i]
                )


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
    parent_pid=None,  # 主进程PID，用于检查主进程是否存活
    # 🧯 advantage/regret 目标稳健化参数（主进程传入）
    advantage_target_scale: float = 1.0,           # 归一化尺度（通常=max_stack）
    advantage_target_clip: Optional[float] = 5.0   # 归一化后裁剪阈值（None表示不裁剪）
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
            """使用优势网络采样动作（Regret Matching）
            
            注意：优势网络在初始化时创建，之后会持续学习（默认不会重新初始化）。
            即使网络权重在初始时是随机的，也会使用它进行预测，这与原始open_spiel的实现一致。
            
            Args:
                state: 游戏状态
                player: 玩家ID
            
            Returns:
                advantages: 优势值列表
                matched_regrets: 策略概率分布
            """
            info_state = state.information_state_tensor(player)
            legal_actions = state.legal_actions(player)
            
            # 总是使用优势网络（即使权重是随机初始化的）
            # 这与原始open_spiel的实现一致
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
            """遍历游戏树，收集样本
            
            Args:
                state: 游戏状态
                player: 玩家ID
                iteration: 当前迭代次数
            """
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
                    # 修复：学习完整的遗憾值（包括负值），与原始OpenSpiel一致
                    # 理由：优势网络需要学习完整的遗憾值，采样时再截断为0（Regret Matching标准做法）
                    # 如果只学习正遗憾值，Fold等动作的负遗憾值会被忽略，导致无法学习到真实价值
                    sampled_regret_arr[action] = sampled_regret[action]

                # ==========================
                # 🧯 Advantage/Regret 目标稳健化
                # ==========================
                # regret/advantage 以筹码单位计通常重尾且量级大，直接用 MSE 会导致 loss 极大且训练震荡/平台。
                # 推荐：按 max_stack 归一化到 O(1) 的尺度，并对归一化后的目标做裁剪。
                try:
                    if advantage_target_scale and float(advantage_target_scale) > 0:
                        inv_scale = 1.0 / float(advantage_target_scale)
                        sampled_regret_arr = [float(x) * inv_scale for x in sampled_regret_arr]
                        if advantage_target_clip is not None:
                            c = float(advantage_target_clip)
                            if c > 0:
                                sampled_regret_arr = [max(-c, min(c, float(x))) for x in sampled_regret_arr]
                except Exception:
                    # 归一化失败不影响主流程（但会回退为原始尺度）
                    pass
                
                # 调试输出：收集样本数据（只收集前20个样本）
                if hasattr(traverse_game_tree, '_debug_count'):
                    traverse_game_tree._debug_count += 1
                else:
                    traverse_game_tree._debug_count = 1
                
                if traverse_game_tree._debug_count <= 20:
                    try:
                        import json
                        info_state_vec = state.information_state_tensor()
                        if isinstance(info_state_vec, np.ndarray):
                            info_state_list = info_state_vec.tolist()[:10]
                        else:
                            info_state_list = list(info_state_vec)[:10]
                        
                        debug_data = {
                            'sample_id': traverse_game_tree._debug_count,
                            'info_state': info_state_list,  # 只保存前10维
                            'sampled_regret': {str(k): float(v) for k, v in sampled_regret.items()},
                            'sampled_regret_arr_old': [float(sampled_regret.get(a, 0)) for a in range(num_actions)],
                            'sampled_regret_arr_new': [float(x) for x in sampled_regret_arr],
                            'strategy': strategy.tolist() if hasattr(strategy, 'tolist') else list(strategy),
                            'cfv': float(cfv),
                            'expected_payoff': {str(k): float(v) for k, v in expected_payoff.items()},
                        }
                        debug_file = '/tmp/deepcfr_debug_samples.jsonl'
                        with open(debug_file, 'a') as f:
                            f.write(json.dumps(debug_data) + '\n')
                    except Exception as e:
                        pass  # 忽略调试输出错误，不影响训练
                
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
                        # 队列满了，实现FIFO替换：先丢弃最旧的样本，再添加新样本
                        try:
                            # 尝试丢弃一个旧样本（FIFO）
                            advantage_queues[player].get_nowait()
                            # 丢弃成功后，再尝试put新样本
                            try:
                                advantage_queues[player].put(local_advantage_batches[player], timeout=0.01)
                                local_advantage_batches[player] = []  # 清空批次
                            except queue.Full:
                                # 如果还是满了，说明队列大小可能有问题，直接丢弃新样本
                                pass
                        except queue.Empty:
                            # 队列为空（不应该发生，但处理一下）
                            pass


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
                        # 队列满了，实现FIFO替换：先丢弃最旧的样本，再添加新样本
                        try:
                            # 尝试丢弃一个旧样本（FIFO）
                            strategy_queue.get_nowait()
                            # 丢弃成功后，再尝试put新样本
                            try:
                                strategy_queue.put(local_strategy_batch, timeout=0.01)
                                local_strategy_batch = []  # 清空批次
                            except queue.Full:
                                # 如果还是满了，说明队列大小可能有问题，直接丢弃新样本
                                pass
                        except queue.Empty:
                            # 队列为空（不应该发生，但处理一下）
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
                    current_iteration = iteration_counter.value
                    traverse_game_tree(root_node.clone(), player, current_iteration)
                
            # 强制刷新缓冲区：无论是否达到 batch_limit，都将手中的样本发送出去
            # 这防止了在多玩家游戏中，某些玩家的样本积累太慢导致的主进程饥饿
            for p in list(local_advantage_batches.keys()):
                batch = local_advantage_batches[p]
                if batch:
                    try:
                        advantage_queues[p].put(batch, timeout=0.01)
                    except queue.Full:
                        # 队列满了，实现FIFO替换：先丢弃最旧的样本，再添加新样本
                        try:
                            # 尝试丢弃一个旧样本（FIFO）
                            advantage_queues[p].get_nowait()
                            # 丢弃成功后，再尝试put新样本
                            try:
                                advantage_queues[p].put(batch, timeout=0.01)
                            except queue.Full:
                                # 如果还是满了，说明队列大小可能有问题，直接丢弃新样本
                                pass
                        except queue.Empty:
                            # 队列为空（不应该发生，但处理一下）
                            pass
                # 清空该玩家的缓冲区
                local_advantage_batches[p] = []
            
            if local_strategy_batch:
                try:
                    strategy_queue.put(local_strategy_batch, timeout=0.01)
                    local_strategy_batch = []
                except queue.Full:
                    # 队列满了，实现FIFO替换：先丢弃最旧的样本，再添加新样本
                    try:
                        # 尝试丢弃一个旧样本（FIFO）
                        strategy_queue.get_nowait()
                        # 丢弃成功后，再尝试put新样本
                        try:
                            strategy_queue.put(local_strategy_batch, timeout=0.01)
                            local_strategy_batch = []
                        except queue.Full:
                            # 如果还是满了，说明队列大小可能有问题，直接丢弃新样本
                            pass
                    except queue.Empty:
                        # 队列为空（不应该发生，但处理一下）
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
        strategy_memory_capacity=None,  # 策略网络缓冲区容量（None表示使用memory_capacity）
        device='cuda',
        gpu_ids=None,  # 多 GPU 支持
        sync_interval=1,  # 每多少次迭代同步一次网络参数
        max_memory_gb=None,  # 最大内存限制（GB），None 表示不限制
        queue_maxsize=50000,  # 队列最大大小（降低以减少内存占用）
        new_sample_ratio=0.5,  # 新样本占比（分层加权采样，默认0.5即50%）
        new_sample_window=0,   # 新样本窗口（最近W轮算“新”）；0表示仅当前轮
        # 切换条件参数
        switch_threshold_win_rate_strict=0.25,  # 严格胜率阈值（25%）
        switch_threshold_win_rate_relaxed=0.20,  # 宽松胜率阈值（20%）
        switch_threshold_avg_return_strict=0.0,  # 严格平均收益阈值（0 BB）
        switch_threshold_avg_return_relaxed=10.0,  # 宽松平均收益阈值（10 BB）
        switch_stable_iterations=10,  # 稳定性检查的迭代次数
        switch_win_rate_std=0.05,  # 胜率标准差阈值（5%）
        switch_avg_return_std=10.0,  # 平均收益标准差阈值（10 BB）
        transition_iterations=1000,  # 过渡阶段的迭代次数
        reinitialize_advantage_networks=False,  # 是否重新初始化优势网络（默认关闭，避免每次迭代重置导致学习不稳定）
        advantage_network_train_steps=1,  # 优势网络训练步数（每次迭代）
        policy_network_train_steps=1,  # 策略网络训练步数（每次迭代）
        # 🧯 advantage/regret 训练稳健化（建议开启）
        advantage_target_scale: Optional[float] = None,   # None: 自动取 game.stack 的最大值
        advantage_target_clip: Optional[float] = None,    # 归一化后裁剪阈值（None不裁剪，默认不裁剪）
        advantage_loss: str = "mse",                    # "huber" 或 "mse"
        huber_delta: float = 1.0                          # Huber delta（归一化后单位）
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
        # 策略网络缓冲区容量（如果未指定，使用memory_capacity）
        self.strategy_memory_capacity = strategy_memory_capacity if strategy_memory_capacity is not None else memory_capacity
        self.sync_interval = sync_interval
        self.max_memory_gb = max_memory_gb
        self.queue_maxsize = queue_maxsize
        self.new_sample_ratio = new_sample_ratio  # 新样本占比（分层加权采样）
        self.new_sample_window = new_sample_window  # 新样本窗口（最近W轮）
        
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
            'base': 50000,  # 基础值（从20000增加到50000，提高消费速度）
            'min': 25000,   # 最小值（从10000提高到25000，避免max_collect过低导致样本收集变慢）
            'max': 200000,  # 最大值（从100000增加到200000，允许更快消费）
            'current': 50000,  # 当前值
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

        # 🧯 advantage/regret 目标稳健化配置
        self._advantage_target_scale = float(advantage_target_scale) if advantage_target_scale is not None else float(self._max_stack)
        if self._advantage_target_scale <= 0:
            self._advantage_target_scale = 1.0
        self._advantage_target_clip = advantage_target_clip
        self._advantage_loss_type = str(advantage_loss).lower()
        self._huber_delta = float(huber_delta) if huber_delta is not None else 1.0
        if self._huber_delta <= 0:
            self._huber_delta = 1.0
        
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
        
        # 本地缓冲区（使用随机替换策略，NumPy 优化版）
        # 优化：使用 NumPy 数组存储，避免 Python 对象访问瓶颈
        self._advantage_memories = [
            RandomReplacementBuffer(
                memory_capacity, 
                info_state_size=self._embedding_size,
                num_actions=self._num_actions,
                buffer_type='advantage'
            ) for _ in range(self.num_players)
        ]
        self._strategy_memories = RandomReplacementBuffer(
            self.strategy_memory_capacity,
            info_state_size=self._embedding_size,
            num_actions=self._num_actions,
            buffer_type='strategy'
        )

        self._iteration = 1
        
        # 切换条件参数
        self.expected_win_rate_random = 1.0 / self.num_players  # 随机策略期望胜率
        self.switch_threshold_win_rate_strict = switch_threshold_win_rate_strict
        self.switch_threshold_win_rate_relaxed = switch_threshold_win_rate_relaxed
        self.switch_threshold_avg_return_strict = switch_threshold_avg_return_strict
        self.switch_threshold_avg_return_relaxed = switch_threshold_avg_return_relaxed
        self.switch_stable_iterations = switch_stable_iterations
        self.switch_win_rate_std = switch_win_rate_std
        self.switch_avg_return_std = switch_avg_return_std
        self.transition_iterations = transition_iterations
        
        # 切换状态（已废弃，保留用于兼容性）
        self.switch_start_iteration = None
        self.win_rate_history = []
        self.avg_return_history = []
        
        # 优势网络重新初始化（与单进程一致）
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        
        # 网络训练步数
        self._advantage_network_train_steps = advantage_network_train_steps
        self._policy_network_train_steps = policy_network_train_steps
    
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
            stack_values = stack_str.split()
            if stack_values:
                try:
                    # 取最大值，兼容不等深度的配置
                    max_stack = max(int(x) for x in stack_values)
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
        expected_input_size = self._embedding_size + 27  # 23维手动特征（增强版本）
        assert actual_input_size == expected_input_size, \
            f"策略网络输入维度错误: 期望 {expected_input_size}，实际 {actual_input_size}"
        
        # 多 GPU 分配优化：如果GPU数量 > 玩家数量，策略网络可以分配到额外的GPU
        # 这样可以与优势网络并行训练，提高训练速度
        if self.use_multi_gpu and len(self.gpu_ids) > self.num_players:
            # GPU数量充足，策略网络分配到额外的GPU（最后一个GPU）
            policy_gpu_id = self.gpu_ids[-1]
            self._policy_network = policy_net.to(torch.device(f"cuda:{policy_gpu_id}"))
            get_logger().info(f"  策略网络分配到 GPU {policy_gpu_id}（与优势网络并行训练）")
        elif self.use_multi_gpu:
            # GPU数量不足，使用DataParallel在多个GPU上并行训练
            self._policy_network = nn.DataParallel(policy_net, device_ids=self.gpu_ids)
            self._policy_network = self._policy_network.to(self.device)
        else:
            self._policy_network = policy_net.to(self.device)
        
        self._policy_sm = nn.Softmax(dim=-1)
        self._loss_policy = nn.MSELoss(reduction="mean")
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
            expected_input_size = self._embedding_size + 27  # 23维手动特征（增强版本）
            assert actual_input_size == expected_input_size, \
                f"玩家 {player} 优势网络输入维度错误: 期望 {expected_input_size}，实际 {actual_input_size}"
            
            # 多 GPU 分配优化：如果GPU数量 >= 玩家数量，将不同优势网络分配到不同GPU
            # 否则使用DataParallel在多个GPU上并行训练单个网络
            if self.use_multi_gpu and len(self.gpu_ids) >= self.num_players:
                # GPU数量充足，每个玩家分配到不同的GPU（真正的并行）
                gpu_id = self.gpu_ids[player % len(self.gpu_ids)]
                net = net.to(torch.device(f"cuda:{gpu_id}"))
                get_logger().info(f"  玩家 {player} 优势网络分配到 GPU {gpu_id}")
            elif self.use_multi_gpu:
                # GPU数量不足，使用DataParallel在多个GPU上并行训练
                # 注意：DataParallel在多线程环境下可能死锁，需要小心使用
                # 如果GPU数量 < 玩家数量，所有网络共享GPU，多线程训练可能导致死锁
                # 解决方案：不使用DataParallel，而是将网络分配到不同的GPU（循环分配）
                # 这样可以避免多线程死锁，同时充分利用多GPU资源
                gpu_id = self.gpu_ids[player % len(self.gpu_ids)]
                net = net.to(torch.device(f"cuda:{gpu_id}"))
                get_logger().info(f"  玩家 {player} 优势网络分配到 GPU {gpu_id}（循环分配，避免DataParallel死锁）")
            else:
                net = net.to(self.device)
            
            self._advantage_networks.append(net)
            self._optimizer_advantages.append(
                torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            )
        
        # 优势网络损失：默认用 Huber（对重尾目标更稳），可选 MSE
        self._loss_advantages = self._build_advantage_loss()
    
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
        traversals_per_worker = 100  # 每个 Worker 每次只跑 10 次遍历，然后检查同步
        
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
                    self._advantage_target_scale,
                    self._advantage_target_clip,
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
                #   - 策略样本：1 × strategy_memory_capacity × 5KB
                #   - 总计：约 35GB（不包括队列和 Worker）
                sample_size_kb = 5  # 保守估算，实际可能更大
                estimated_memory_gb = (
                    self.memory_capacity * self.num_players * sample_size_kb +  # 优势样本
                    self.strategy_memory_capacity * sample_size_kb +  # 策略样本
                    self.queue_maxsize * (self.num_players + 1) * sample_size_kb +  # 队列积压（最坏情况）
                    self.num_workers * 500  # Worker 进程开销（每个 Worker 约 500MB）
                ) / 1024 / 1024
                print(f"  估算总内存需求: {estimated_memory_gb:.2f}GB")
                print(f"    - 优势样本缓冲区: {self.memory_capacity * self.num_players * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - 策略样本缓冲区: {self.strategy_memory_capacity * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - 队列积压（最坏情况）: {self.queue_maxsize * (self.num_players + 1) * sample_size_kb / 1024 / 1024:.2f}GB")
                print(f"    - Worker 进程: {self.num_workers * 500 / 1024 / 1024:.2f}GB")
                
                # 获取系统总内存
                try:
                    total_mem = psutil.virtual_memory().total / 1024 / 1024 / 1024
                    available_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
                    print(f"  系统内存: {total_mem:.1f}GB 总计, {available_mem:.1f}GB 可用")
                    
                    if estimated_memory_gb > available_mem * 0.8:
                        print(f"  ⚠️ 警告: 估算内存需求 ({estimated_memory_gb:.1f}GB) 接近可用内存 ({available_mem:.1f}GB)")
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
    
    def reinitialize_advantage_network(self, player):
        """重新初始化优势网络（与单进程一致）"""
        net = self._advantage_networks[player]
        # 处理 DataParallel 包装
        if isinstance(net, nn.DataParallel):
            # DataParallel包装的网络，调用reset方法
            net.module.reset()
        else:
            # 直接调用reset方法（SimpleFeatureMLP有reset方法）
            net.reset()
        # 重新创建优化器
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self.learning_rate)
    
    def _sync_network_params(self):
        """同步网络参数到所有 Worker（始终同步，移除两阶段机制）"""
        sync_start_time = time.time()
        
        params_start = time.time()
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
        params_time = time.time() - params_start
        
        queue_start = time.time()
        for q in self._network_params_queues:
            try:
                q.put_nowait(params)
            except queue.Full:
                pass
        queue_time = time.time() - queue_start
        
        total_sync_time = time.time() - sync_start_time
        get_logger().info(f"  网络参数同步耗时: {total_sync_time*1000:.1f}ms (参数提取: {params_time*1000:.1f}ms, 队列发送: {queue_time*1000:.1f}ms)")
    
    def _cleanup_buffers(self, force=False):
        """清理缓冲区（已禁用）
        
        由于使用随机替换策略，缓冲区满了之后新样本会自动随机替换旧样本，
        因此不需要主动清理缓冲区。此方法已禁用，仅保留接口以保持兼容性。
        
        Args:
            force: 强制清理，即使内存使用不高（已忽略）
        """
        # 清理逻辑已移除，完全依赖随机替换策略
        return
    
    def _check_and_cleanup_memory(self, force=False, cleanup_buffers=True):
        """检查内存使用情况（队列清理已移除）
        
        注意：
        1. 队列清理已完全移除，因为Worker进程已经实现了FIFO替换
        2. FIFO替换：当队列满了时，Worker进程会先get一个旧样本（丢弃），再put新样本
        3. 队列大小保持稳定（maxsize），不需要主进程清理
        4. 缓冲区清理已禁用，完全依赖随机替换策略
        
        Args:
            force: 强制清理，即使内存使用不高（已忽略，仅保留接口兼容性）
            cleanup_buffers: 是否清理缓冲区（已忽略，缓冲区清理已禁用）
        """
        # 队列清理已完全移除，因为Worker进程已经实现了FIFO替换
        # FIFO替换：当队列满了时，Worker进程会先get一个旧样本（丢弃），再put新样本
        # 队列大小保持稳定（maxsize），不需要主进程清理
        
        # 缓冲区清理已禁用，完全依赖随机替换策略
        # if cleanup_buffers:
        #     self._cleanup_buffers(force=force)
    
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
        # 修复：使用非阻塞方式获取CPU使用率，避免阻塞
        cpu_percent = None
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                # 使用interval=None，非阻塞获取（返回上次调用后的平均值）
                cpu_percent = process.cpu_percent(interval=None)
                # 如果获取失败，尝试获取系统CPU使用率
                if cpu_percent is None or cpu_percent == 0:
                    cpu_percent = psutil.cpu_percent(interval=None)
            except:
                pass
        
        # 动态调整max_collect
        adaptive = self._adaptive_max_collect
        base = adaptive['base']
        min_val = adaptive['min']
        # 优化：当队列使用率很高时，允许更高的max_collect
        # 队列使用率100%时，允许一次性消费队列大小的10倍，快速清空队列
        if max_queue_usage >= 0.99:
            # 队列几乎满时，允许更高的max_collect（队列大小的10倍）
            max_val = min(adaptive['max'], max(self.queue_maxsize * 10, 50000))
        else:
            # 正常情况，至少队列大小的4倍，或20000
            max_val = min(adaptive['max'], max(self.queue_maxsize * 4, 20000))
        
        # 1. 根据队列使用率调整（主要指标）
        # 优化：当队列使用率100%时，大幅增加消费速度（最多10倍），快速清空队列
        if max_queue_usage >= 0.99:
            # 队列使用率 >= 99%（几乎满），大幅增加消费速度
            factor = 1.0 + (max_queue_usage - 0.80) * 20.0  # 最多增加到10倍（0.99时约3.8倍，1.0时10倍）
            new_max_collect = int(base * factor)
        elif max_queue_usage > 0.80:
            # 队列使用率 > 80%，增加消费速度
            factor = 1.0 + (max_queue_usage - 0.80) * 5.0  # 最多增加到2倍（0.80时1倍，1.0时2倍）
            new_max_collect = int(base * factor)
        elif max_queue_usage < 0.30:
            # 队列使用率 < 30%，减少消费速度（节省CPU）
            factor = 0.5 + (max_queue_usage / 0.30) * 0.5  # 最少减少到0.5倍
            new_max_collect = int(base * factor)
        else:
            # 正常范围，保持基础值
            new_max_collect = base
        
        # 2. 根据队列增长率调整（次要指标）
        # 优化：降低增长率阈值，更敏感地响应队列增长
        if avg_growth_rate > 50:  # 每秒增长超过50个样本（从100降低到50）
            growth_factor = min(2.0, 1.0 + avg_growth_rate / 500.0)  # 最多2倍（从1.5倍提高到2倍）
            new_max_collect = int(new_max_collect * growth_factor)
        
        # 3. 根据CPU使用率调整（限制指标，避免CPU过载）
        # 优化：当队列中有样本时（使用率>5%），优先消费队列，不因CPU高而减速
        # 这避免了"CPU高→max_collect减半→消费变慢→队列空→忙等待→CPU仍高"的恶性循环
        if cpu_percent is not None:
            if max_queue_usage > 0.05:
                # 队列中有样本（使用率>5%），优先消费，不减少max_collect
                # 只有在CPU极高（>98%）时才适当限制
                if cpu_percent > 98:
                    cpu_factor = max(0.9, 1.0 - (cpu_percent - 98) / 20.0)  # 最多减少到0.9倍
                    new_max_collect = int(new_max_collect * cpu_factor)
                # 否则不限制，优先清空队列
            elif cpu_percent > 95:
                # 队列几乎空（<5%）且CPU极高，适当减少消费速度（避免忙等待）
                cpu_factor = max(0.7, 1.0 - (cpu_percent - 95) / 15.0)  # 最多减少到0.7倍
                new_max_collect = int(new_max_collect * cpu_factor)
            elif cpu_percent < 50 and max_queue_usage > 0.30:
                # CPU使用率 < 50% 且队列使用率 > 30%，可以适当增加消费速度
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
    
    def _collect_samples(self, timeout=0.1, current_iteration=None):
        """从队列收集样本
        
        关键修复：
        1. 只清理队列积压（不影响样本收集进度）
        2. 缓冲区清理在样本收集完成后执行，避免删除正在收集的样本
        3. 动态调整max_collect，根据队列积压情况自适应调整消费速度
        4. 在样本添加到缓冲区时，由主进程标记样本的迭代次数（不依赖Worker进程读取iteration_counter）
        
        Args:
            timeout: 超时时间（未使用，保留兼容性）
            current_iteration: 当前迭代次数（用于标记新样本）
        
        Returns:
            total_collected: 本次收集的总样本数（优势样本 + 策略样本）
        """
        collect_start_time = time.time()
        
        # 队列清理已完全移除，因为Worker进程已经实现了FIFO替换
        # FIFO替换：当队列满了时，Worker进程会先get一个旧样本（丢弃），再put新样本
        # 队列大小保持稳定（maxsize），不需要主进程清理
        cleanup_time = 0.0
        
        # 动态调整max_collect
        update_start = time.time()
        max_collect = self._update_adaptive_max_collect()
        update_time = time.time() - update_start
        
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
        # 注意：队列清理已完全移除，因为Worker进程已经实现了FIFO替换
        advantage_collect_start = time.time()
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
            
            # 批量添加到缓冲区（优化：减少方法调用开销）
            # 关键修复：在样本添加到缓冲区时，由主进程标记样本的迭代次数
            # 这样就不依赖Worker进程读取iteration_counter的值了
            memory = self._advantage_memories[player]
            for batch in batch_list:
                if isinstance(batch, list):
                    # 批量添加，减少方法调用次数
                    for sample in batch:
                        # 如果提供了current_iteration，更新样本的iteration字段
                        if current_iteration is not None:
                            # 创建新的样本对象，更新iteration字段
                            updated_sample = AdvantageMemory(
                                sample.info_state,
                                current_iteration,  # 使用主进程的current_iteration
                                sample.advantage,
                                sample.action
                            )
                            memory.add(updated_sample)
                        else:
                            memory.add(sample)
                    total_collected_advantage += len(batch)
                else:
                    # 单个样本
                    if current_iteration is not None:
                        updated_sample = AdvantageMemory(
                            batch.info_state,
                            current_iteration,  # 使用主进程的current_iteration
                            batch.advantage,
                            batch.action
                        )
                        memory.add(updated_sample)
                    else:
                        memory.add(batch)
                    total_collected_advantage += 1
        advantage_collect_time = time.time() - advantage_collect_start
        
        # 收集策略样本（批量处理，提高效率）
        # 注意：队列清理已完全移除，因为Worker进程已经实现了FIFO替换
        strategy_collect_start = time.time()
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
        
        # 批量添加到缓冲区（优化：减少方法调用开销）
        # 关键修复：在样本添加到缓冲区时，由主进程标记样本的迭代次数
        # 这样就不依赖Worker进程读取iteration_counter的值了
        total_collected_strategy = 0
        memory = self._strategy_memories
        for batch in batch_list:
            if isinstance(batch, list):
                # 批量添加，减少方法调用次数
                for sample in batch:
                    # 如果提供了current_iteration，更新样本的iteration字段
                    if current_iteration is not None:
                        # 创建新的样本对象，更新iteration字段
                        updated_sample = StrategyMemory(
                            sample.info_state,
                            current_iteration,  # 使用主进程的current_iteration
                            sample.strategy_action_probs
                        )
                        memory.add(updated_sample)
                    else:
                        memory.add(sample)
                total_collected_strategy += len(batch)
            else:
                # 单个样本
                if current_iteration is not None:
                    updated_sample = StrategyMemory(
                        batch.info_state,
                        current_iteration,  # 使用主进程的current_iteration
                        batch.strategy_action_probs
                    )
                    memory.add(updated_sample)
                else:
                    memory.add(batch)
                total_collected_strategy += 1
        strategy_collect_time = time.time() - strategy_collect_start
        
        # 记录收集统计（用于调试和监控）
        if total_collected_advantage > 0 or total_collected_strategy > 0:
            self._queue_stats['collection_counts']['advantage'] = total_collected_advantage
            self._queue_stats['collection_counts']['strategy'] = total_collected_strategy
        
        # 记录耗时
        total_collect_time = time.time() - collect_start_time
        total_collected = total_collected_advantage + total_collected_strategy
        # if total_collected_advantage > 0 or total_collected_strategy > 0:
        #     get_logger().info(f"  样本收集耗时: {total_collect_time*1000:.1f}ms (清理: {cleanup_time*1000:.1f}ms, 调整: {update_time*1000:.1f}ms, 优势: {advantage_collect_time*1000:.1f}ms, 策略: {strategy_collect_time*1000:.1f}ms)")
        
        return total_collected
    
    def _learn_advantage_network(self, player, current_iteration=None):
        """训练优势网络（NumPy 优化版）
        
        Args:
            player: 玩家ID
            current_iteration: 当前迭代次数（用于统计新样本比例）
        
        优化：使用 NumPy 数组直接采样，避免 Python 对象遍历瓶颈。
        数据准备时间从 ~700ms 降到 ~5ms（140x 加速）。
        """
        train_start_time = time.time()
        
        num_samples = len(self._advantage_memories[player])
        if num_samples < 32:  # 最少需要 32 个样本才训练
            return None
        
        # 使用实际样本数和 batch_size 的较小值
        sample_start = time.time()
        actual_batch_size = min(num_samples, self.batch_size_advantage)
        
        # 优化：直接获取 NumPy 数组，无需遍历 Python 对象
        sample_result = self._advantage_memories[player].sample(
            actual_batch_size, 
            current_iteration=current_iteration,
            new_sample_ratio=self.new_sample_ratio,
            new_sample_window=self.new_sample_window
        )
        sample_time = time.time() - sample_start
        
        data_prep_start = time.time()
        
        # 优化：直接使用 NumPy 数组，无需遍历
        if isinstance(sample_result, dict):
            # 新接口：直接返回 NumPy 数组
            info_states = sample_result['info_states']
            advantages = sample_result['advantages']
            iterations_arr = sample_result['iterations'].reshape(-1, 1).astype(np.float32)
            new_sample_count = sample_result['new_sample_count']
            num_samples_batch = len(info_states)
        else:
            # 兼容旧接口：sample_result 是 namedtuple 列表
            samples = sample_result
            if len(samples) > 0:
                num_samples_batch = len(samples)
                expected_len = self._num_actions
                
                info_states = np.empty((num_samples_batch, len(samples[0].info_state)), dtype=np.float32)
                advantages = np.empty((num_samples_batch, expected_len), dtype=np.float32)
                iterations_arr = np.empty((num_samples_batch, 1), dtype=np.float32)
                
                new_sample_count = 0
                for i, s in enumerate(samples):
                    info_states[i] = s.info_state
                    adv = s.advantage
                    if len(adv) >= expected_len:
                        advantages[i] = adv[:expected_len]
                    else:
                        advantages[i, :len(adv)] = adv
                        advantages[i, len(adv):] = 0
                    iterations_arr[i, 0] = s.iteration
                if current_iteration is not None and hasattr(s, 'iteration'):
                    try:
                        w = int(self.new_sample_window) if self.new_sample_window is not None else 0
                    except Exception:
                        w = 0
                    w = max(0, w)
                    if w <= 0:
                        if s.iteration == current_iteration:
                            new_sample_count += 1
                    else:
                        if (s.iteration >= (current_iteration - w)) and (s.iteration <= current_iteration):
                            new_sample_count += 1
            else:
                info_states = np.array([], dtype=np.float32)
                advantages = np.array([], dtype=np.float32)
                iterations_arr = np.array([], dtype=np.float32).reshape(0, 1)
                new_sample_count = 0
                num_samples_batch = 0
        
        # sqrt(iteration)
        iterations = np.sqrt(iterations_arr)
        
        # 统计新样本和老样本比例
        if current_iteration is not None and num_samples_batch > 0:
            old_sample_count = num_samples_batch - new_sample_count
            new_ratio = new_sample_count / num_samples_batch * 100
            get_logger().info(f"    玩家 {player} 优势网络训练样本: 新样本 {new_sample_count}/{num_samples_batch} ({new_ratio:.1f}%), 老样本 {old_sample_count}/{num_samples_batch} ({100-new_ratio:.1f}%)")
        
        data_prep_time = time.time() - data_prep_start
        
        # 修复：如果samples为空，直接返回None，避免空tensor训练
        if num_samples_batch == 0:
            return None
        
        # 优化：获取网络所在的设备（支持多GPU分配）
        network = self._advantage_networks[player]
        if isinstance(network, nn.DataParallel):
            # DataParallel包装的网络，使用主设备
            device = next(network.parameters()).device
        else:
            # 单个GPU上的网络，使用网络所在的设备
            device = next(network.parameters()).device
        
        # 优化：预计算手动特征，避免在 forward() 中重复计算
        # 这是关键优化！原来每次 forward() 都要遍历所有样本计算特征，非常慢
        feature_calc_start = time.time()
        info_states_tensor = torch.from_numpy(info_states).to(device)
        
        # 预计算特征并拼接到输入
        if hasattr(network, 'extract_manual_features'):
            # SimpleFeatureMLP 需要计算手动特征
            with torch.no_grad():  # 特征计算不需要梯度
                manual_features = network.extract_manual_features(info_states_tensor)
            # 拼接特征到输入
            info_states_tensor = torch.cat([info_states_tensor, manual_features], dim=1)
        feature_calc_time = time.time() - feature_calc_start
        
        # 创建其他 tensor
        advantages_tensor = torch.from_numpy(advantages).to(device)
        iters = torch.from_numpy(iterations).to(device)

        # 训练期可观测性：打印 advantage label 分布（确认归一化/裁剪是否生效）
        if current_iteration is not None and current_iteration % 1000 == 0 and player == 0:
            try:
                flat = np.asarray(advantages, dtype=np.float32).reshape(-1)
                abs_flat = np.abs(flat)
                p50, p90, p99 = np.quantile(abs_flat, [0.5, 0.9, 0.99]).tolist()
                mx = float(abs_flat.max()) if abs_flat.size else 0.0
                get_logger().info(
                    f"  🧯 advantage targets stats(abs) @iter={current_iteration}: "
                    f"p50={p50:.4f}, p90={p90:.4f}, p99={p99:.4f}, max={mx:.4f} | "
                    f"scale={self._advantage_target_scale:.1f}, clip={self._advantage_target_clip}, loss={self._advantage_loss_type}"
                )
            except Exception:
                pass
        
        # 修复：DataParallel在多线程环境下可能死锁，需要设置环境变量
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        # 如果使用DataParallel，可能需要设置torch.set_num_threads
        if isinstance(network, nn.DataParallel):
            torch.set_num_threads(1)
        
        forward_backward_start = time.time()
        # 多次训练（与原始OpenSpiel一致）
        final_loss = None
        for step in range(self._advantage_network_train_steps):
            self._optimizer_advantages[player].zero_grad()
            try:
                outputs = network(info_states_tensor)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
            loss = self._loss_advantages(iters * outputs, iters * advantages_tensor)
            
            # 调试输出已禁用（影响性能）
            
            loss.backward()
            self._optimizer_advantages[player].step()
            final_loss = loss  # 保存最后一次训练的损失
        
        # 优化：确保CUDA操作完成，避免多线程竞争
        # 注意：synchronize不接受device对象，需要获取设备索引
        if device.type == 'cuda':
            device_index = device.index if device.index is not None else 0
            torch.cuda.synchronize(device_index)
        forward_backward_time = time.time() - forward_backward_start
        
        total_train_time = time.time() - train_start_time
        # 开启详细耗时日志（用于性能调优）
        if total_train_time > 1.0:  # 只记录超过1秒的训练
            get_logger().warning(f"  ⚠️ 玩家 {player} 训练慢: {total_train_time:.2f}秒 (采样: {sample_time*1000:.0f}ms, 数据准备: {data_prep_time*1000:.0f}ms, 特征计算: {feature_calc_time*1000:.0f}ms, 前向反向: {forward_backward_time*1000:.0f}ms, 样本数: {num_samples_batch})")
        
        # 修复：确保返回标量（MSELoss返回标量tensor，.numpy()可能返回0维数组）
        if final_loss is not None:
            loss_value = final_loss.detach().cpu().numpy()
            return float(loss_value) if np.isscalar(loss_value) else float(loss_value.item())
        return None

    def _build_advantage_loss(self):
        """构建优势网络损失函数（默认 Huber）"""
        if self._advantage_loss_type == "mse":
            return nn.MSELoss(reduction="mean")

        delta = float(self._huber_delta)

        def huber(pred, target):
            diff = pred - target
            abs_diff = torch.abs(diff)
            quad = torch.clamp(abs_diff, max=delta)
            lin = abs_diff - quad
            loss = 0.5 * (quad ** 2) / delta + lin
            return loss.mean()

        return huber
    
    def _learn_strategy_network(self, current_iteration=None):
        """训练策略网络（NumPy 优化版）
        
        Args:
            current_iteration: 当前迭代次数（用于统计新样本比例）
        
        优化：使用 NumPy 数组直接采样，避免 Python 对象遍历瓶颈。
        """
        train_start_time = time.time()
        
        num_samples = len(self._strategy_memories)
        if num_samples < 32:  # 最少需要 32 个样本才训练
            return None
        
        # 使用实际样本数和 batch_size 的较小值
        sample_start = time.time()
        actual_batch_size = min(num_samples, self.batch_size_strategy)
        sample_result = self._strategy_memories.sample(
            actual_batch_size,
            current_iteration=current_iteration,
            new_sample_ratio=self.new_sample_ratio,
            new_sample_window=self.new_sample_window
        )
        sample_time = time.time() - sample_start
        
        data_prep_start = time.time()
        
        # 优化：直接使用 NumPy 数组，无需遍历
        if isinstance(sample_result, dict):
            # 新接口：直接返回 NumPy 数组
            info_states = sample_result['info_states']
            action_probs = sample_result['action_probs']
            iterations_arr = sample_result['iterations'].astype(np.float32)
            new_sample_count = sample_result['new_sample_count']
            num_samples_batch = len(info_states)
            
            # sqrt(iteration)，保持 2D 形状 [batch_size, 1]
            iterations = np.sqrt(iterations_arr.reshape(-1, 1))
            
            # 统计新样本和老样本比例
            if current_iteration is not None and num_samples_batch > 0:
                old_sample_count = num_samples_batch - new_sample_count
                new_ratio = new_sample_count / num_samples_batch * 100
                get_logger().info(f"    策略网络训练样本: 新样本 {new_sample_count}/{num_samples_batch} ({new_ratio:.1f}%), 老样本 {old_sample_count}/{num_samples_batch} ({100-new_ratio:.1f}%)")
        else:
            # 兼容旧接口：sample_result 是 namedtuple 列表
            samples = sample_result
            
            # 统计新样本和老样本比例
            if current_iteration is not None and len(samples) > 0:
                try:
                    w = int(self.new_sample_window) if self.new_sample_window is not None else 0
                except Exception:
                    w = 0
                w = max(0, w)
                if w <= 0:
                    new_samples = sum(1 for s in samples if hasattr(s, 'iteration') and s.iteration == current_iteration)
                else:
                    low = current_iteration - w
                    new_samples = sum(1 for s in samples if hasattr(s, 'iteration') and (s.iteration >= low) and (s.iteration <= current_iteration))
                old_samples = len(samples) - new_samples
                new_ratio = new_samples / len(samples) * 100 if len(samples) > 0 else 0
                get_logger().info(f"    策略网络训练样本: 新样本 {new_samples}/{len(samples)} ({new_ratio:.1f}%), 老样本 {old_samples}/{len(samples)} ({100-new_ratio:.1f}%)")
            
            if len(samples) > 0:
                info_states = np.array([s.info_state for s in samples], dtype=np.float32)
                action_probs_list = [s.strategy_action_probs for s in samples]
                if len(action_probs_list) > 0:
                    first_probs = action_probs_list[0]
                    if isinstance(first_probs, (list, np.ndarray)):
                        expected_len = self._num_actions
                        action_probs = np.array([
                            np.array(probs, dtype=np.float32)[:expected_len] if len(probs) >= expected_len
                            else np.pad(np.array(probs, dtype=np.float32), (0, expected_len - len(probs)), 
                                       mode='constant', constant_values=0)
                            for probs in action_probs_list
                        ], dtype=np.float32)
                    else:
                        action_probs = np.array(action_probs_list, dtype=np.float32)
                        if action_probs.ndim == 1:
                            action_probs = action_probs[:, np.newaxis]
                    if action_probs.ndim > 2:
                        action_probs = np.squeeze(action_probs)
                else:
                    action_probs = np.array([], dtype=np.float32)
                iterations = np.sqrt(np.array([[s.iteration] for s in samples], dtype=np.float32))
                num_samples_batch = len(samples)
            else:
                info_states = np.array([], dtype=np.float32)
                action_probs = np.array([], dtype=np.float32)
                iterations = np.array([], dtype=np.float32).reshape(0, 1)
                num_samples_batch = 0
        
        data_prep_time = time.time() - data_prep_start
        
        # 修复：如果samples为空，直接返回None，避免空tensor训练
        if num_samples_batch == 0:
            return None
        
        # 优化：获取策略网络所在的设备（支持多GPU分配）
        network = self._policy_network
        if isinstance(network, nn.DataParallel):
            # DataParallel包装的网络，使用主设备
            device = next(network.parameters()).device
        else:
            # 单个GPU上的网络，使用网络所在的设备
            device = next(network.parameters()).device
        
        # 优化：批量创建tensor，减少CPU-GPU数据传输次数，并分配到正确的设备
        iters = torch.from_numpy(iterations).to(device)
        ac_probs = torch.from_numpy(action_probs).to(device)
        info_states_tensor = torch.from_numpy(info_states).to(device)
        
        forward_backward_start = time.time()
        # 多次训练（与原始OpenSpiel一致）
        final_loss = None
        for step in range(self._policy_network_train_steps):
            self._optimizer_policy.zero_grad()
            logits = network(info_states_tensor)
            outputs = self._policy_sm(logits)
            loss = self._loss_policy(iters * outputs, iters * ac_probs)
            loss.backward()
            self._optimizer_policy.step()
            final_loss = loss  # 保存最后一次训练的损失
        
        # 优化：确保CUDA操作完成，避免多线程竞争
        # 注意：synchronize不接受device对象，需要获取设备索引
        if device.type == 'cuda':
            device_index = device.index if device.index is not None else 0
            torch.cuda.synchronize(device_index)
        forward_backward_time = time.time() - forward_backward_start
        
        total_train_time = time.time() - train_start_time
        get_logger().info(f"  策略网络训练耗时: {total_train_time*1000:.1f}ms (采样: {sample_time*1000:.1f}ms, 数据准备: {data_prep_time*1000:.1f}ms, 前向反向: {forward_backward_time*1000:.1f}ms)")
        
        # 修复：确保返回标量（MSELoss返回标量tensor，.numpy()可能返回0维数组）
        if final_loss is not None:
            loss_value = final_loss.detach().cpu().numpy()
            return float(loss_value) if np.isscalar(loss_value) else float(loss_value.item())
        return None
    
    def should_start_transition(self, iteration, advantage_losses, win_rate=None, avg_return=None):
        """判断是否应该开始过渡阶段
        
        Args:
            iteration: 当前迭代次数
            advantage_losses: 优势网络损失值历史（dict，key为玩家ID）
            win_rate: 当前迭代的胜率（vs Random）
            avg_return: 当前迭代的平均收益（vs Random，单位：BB）
        
        Returns:
            bool: 是否应该开始过渡阶段
        """
        if self.switch_start_iteration is not None:
            return False  # 已经开始过渡阶段
        
        if win_rate is None or avg_return is None:
            return False  # 缺少评估数据
        
        # 记录历史
        self.win_rate_history.append(win_rate)
        self.avg_return_history.append(avg_return)
        
        # 检查是否有足够的历史
        if len(self.win_rate_history) < self.switch_stable_iterations:
            return False
        
        # 检查胜率和平均收益条件
        recent_win_rates = self.win_rate_history[-self.switch_stable_iterations:]
        recent_avg_returns = self.avg_return_history[-self.switch_stable_iterations:]
        
        avg_win_rate = np.mean(recent_win_rates)
        min_win_rate = min(recent_win_rates)
        avg_return_value = np.mean(recent_avg_returns)
        min_avg_return = min(recent_avg_returns)
        
        # 检查是否满足严格条件或宽松条件
        strict_condition = (
            avg_win_rate >= self.switch_threshold_win_rate_strict and
            min_win_rate >= self.switch_threshold_win_rate_strict and
            avg_return_value >= self.switch_threshold_avg_return_strict and
            min_avg_return >= self.switch_threshold_avg_return_strict
        )
        
        relaxed_condition = (
            avg_win_rate >= self.switch_threshold_win_rate_relaxed and
            min_win_rate >= self.switch_threshold_win_rate_relaxed and
            avg_return_value >= self.switch_threshold_avg_return_relaxed and
            min_avg_return >= self.switch_threshold_avg_return_relaxed
        )
        
        if not (strict_condition or relaxed_condition):
            return False
        
        # 检查稳定性
        std_win_rate = np.std(recent_win_rates)
        std_avg_return = np.std(recent_avg_returns)
        
        if std_win_rate >= self.switch_win_rate_std:
            return False
        
        if std_avg_return >= self.switch_avg_return_std:
            return False
        
        # 满足所有条件，可以开始过渡阶段
        self.switch_start_iteration = iteration
        print(f"\n🎯 开始过渡阶段（迭代 {iteration + 1}）")
        print(f"   - 随机策略期望胜率: {self.expected_win_rate_random*100:.2f}%")
        print(f"   - 平均胜率: {avg_win_rate*100:.1f}% (vs Random)")
        print(f"   - 最小胜率: {min_win_rate*100:.1f}% (vs Random)")
        print(f"   - 胜率提升: {(avg_win_rate - self.expected_win_rate_random)*100:.1f}% ({(avg_win_rate / self.expected_win_rate_random - 1)*100:.1f}% 相对提升)")
        print(f"   - 平均收益: {avg_return_value:.2f} BB (vs Random)")
        print(f"   - 最小收益: {min_avg_return:.2f} BB (vs Random)")
        print(f"   - 收益标准差: {std_avg_return:.2f} BB")
        
        if strict_condition:
            print(f"   - 满足严格条件（胜率 > 25% 且 收益 > 0 BB）")
        else:
            print(f"   - 满足宽松条件（胜率 > 20% 且 收益 > 10 BB）")
        print()
        
        return True
    
    def solve(self, verbose=True, eval_interval=10, checkpoint_interval=0, 
              model_dir=None, save_prefix=None, game=None, start_iteration=0,
              eval_with_games=False, num_test_games=1000, training_state=None):
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
            training_state: 训练状态字典（用于恢复多阶段训练状态）
        
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
        
        # 恢复多阶段训练状态（如果提供）
        if training_state is not None:
            self.switch_start_iteration = training_state.get('switch_start_iteration')
            self.win_rate_history = training_state.get('win_rate_history', [])
            self.avg_return_history = training_state.get('avg_return_history', [])
            # 恢复迭代计数器
            if start_iteration > 0:
                self._iteration = start_iteration + 1
            if verbose:
                if self.switch_start_iteration is not None:
                    print(f"  ✓ 恢复多阶段训练状态:")
                    print(f"    - switch_start_iteration: {self.switch_start_iteration}")
                    print(f"    - win_rate_history长度: {len(self.win_rate_history)}")
                    print(f"    - avg_return_history长度: {len(self.avg_return_history)}")
                    # 第一阶段和第二阶段通过switch_start_iteration区分
                    if self.switch_start_iteration is None:
                        print(f"    - 当前阶段：第一阶段（网络权重随机初始化）")
                    else:
                        print(f"    - 当前阶段：第二阶段（自博弈，switch_start_iteration={self.switch_start_iteration}）")
                else:
                    print(f"  ✓ 恢复多阶段训练状态: 仍在第一阶段（完全随机策略）")
                print()
        
        # 启动 Worker
        self._start_workers()
        
        advantage_losses = {p: [] for p in range(self.num_players)}
        policy_losses = []  # 策略网络损失历史
        start_time = time.time()
        
        try:
            # 等待 Worker 启动并开始产生样本
            print("  等待 Worker 启动...", end="", flush=True)
            warmup_time = 0
            max_warmup = 30  # 最多等待 30 秒
            # 修复：在warmup期间也检查Worker状态，避免卡住
            while warmup_time < max_warmup:
                # 检查Worker状态
                dead_workers = [i for i, p in enumerate(self._workers) if not p.is_alive()]
                if dead_workers:
                    worker_info = ", ".join([f"Worker {wid}" for wid in dead_workers])
                    raise RuntimeError(f"检测到 Worker 已死亡: {worker_info}。训练无法继续。")
                
                time.sleep(1)
                warmup_time += 1
                collected = self._collect_samples(current_iteration=None)  # warmup阶段不需要标记迭代次数
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
                
                collection_start_time = time.time()
                current_total_samples = sum(len(m) for m in self._advantage_memories)
                # 目标：本轮新增 num_traversals 个样本
                # 注意：由于可能有多个 Worker 同时提交，可能会略多一点，没关系
                target_total_samples = current_total_samples + self.num_traversals
                
                # 设置一个超时保护（例如 10 分钟），防止 Worker 全部挂死导致主进程死循环
                last_sample_count = current_total_samples
                
                # 关键修复：样本收集循环中只清理队列积压，不清理缓冲区
                # 优化：根据队列状态动态调整sleep时间，队列满时不sleep，队列空时才sleep
                collected_in_this_iteration = 0  # 本次迭代收集的样本数
                loop_count = 0
                no_progress_count = 0  # 连续无进展次数
                last_warning_time = 0  # 上次警告时间，避免重复打印
                while True:
                    loop_count += 1
                    # 检查队列状态，决定是否需要sleep
                    queue_sizes = [q.qsize() for q in self._advantage_queues]
                    strategy_queue_size = self._strategy_queue.qsize()
                    total_queue_size = sum(queue_sizes) + strategy_queue_size
                    max_queue_size = max(max(queue_sizes) if queue_sizes else 0, strategy_queue_size)
                    queue_usage = max_queue_size / self.queue_maxsize if self.queue_maxsize > 0 else 0
                    
                    # 收集样本，返回本次收集的样本数
                    # 关键修复：传递current_iteration，让主进程标记样本的迭代次数
                    collected = self._collect_samples(current_iteration=self._iteration_counter.value)  # 内部只清理队列积压
                    collected_in_this_iteration += collected
                    
                    # 检查是否达标：检查本次迭代收集的样本数，而不是缓冲区总样本数
                    # 因为缓冲区满了之后，虽然会随机替换，但总样本数不会增加
                    if collected_in_this_iteration >= self.num_traversals:
                        break
                    
                    # 检查是否有进展：如果本次收集了样本，说明有进展
                    # 优化：区分"队列空"和"队列满但收集失败"两种情况
                    if collected > 0:
                        no_progress_count = 0
                    else:
                        # 只有在队列不为空时才视为"无进展"
                        # 如果队列为空，说明 Worker 可能暂时没有产生样本，这是正常的
                        if total_queue_size > 0:
                            # 修复：如果队列使用率较高（>=80%），不应该触发警告
                            # 因为队列中有样本，只是收集速度慢，这是正常情况
                            # 队列使用率 >= 99%：队列满了，Worker无法继续添加样本，队列大小不会增加
                            # 队列使用率 >= 80%：队列中有样本，只是收集速度慢，不应该触发警告
                            if queue_usage >= 0.80:
                                # 队列使用率较高，重置计数器（这是正常情况，不是问题）
                                # 队列中有样本，只是收集速度慢，不应该触发警告
                                no_progress_count = 0
                            else:
                                # 队列使用率较低，但队列中有样本，可能是收集速度慢
                                # 增加计数器，但阈值可以适当放宽
                                no_progress_count += 1
                        else:
                            # 队列为空，重置计数器（这是正常情况，不是问题）
                            no_progress_count = 0
                    
                    # 检查超时 (10分钟)
                    elapsed_time = time.time() - collection_start_time
                    if elapsed_time > 600:
                        if verbose:
                            print(f"\n  ⚠️ 警告: 样本收集超时 (已收集 {collected_in_this_iteration}/{self.num_traversals})")
                            # 诊断信息
                            print(f"    诊断信息:")
                            print(f"      - 耗时: {elapsed_time:.1f}秒")
                            current_total_samples = sum(len(m) for m in self._advantage_memories)
                            print(f"      - 当前优势样本总数: {current_total_samples:,}")
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
                    
                    # 如果连续多次无进展，提前警告（优化：避免重复打印）
                    # 修复：使用时间间隔控制，每30秒最多警告一次
                    # 修复：增加诊断信息，记录max_collect和实际收集的样本数
                    current_time = time.time()
                    if no_progress_count >= 20 and verbose:  # 10秒无进展（20次 × 0.5秒）
                        if current_time - last_warning_time > 30:  # 每30秒最多警告一次
                            # 获取当前的max_collect值
                            current_max_collect = self._adaptive_max_collect.get('current', 'N/A')
                            print(f"\n  ⚠️ 警告: 连续 {no_progress_count * 0.5:.1f}秒无样本收集进展")
                            print(f"    诊断信息:")
                            print(f"      - 队列大小: {total_queue_size:,}")
                            print(f"      - 队列使用率: {queue_usage:.1%}")
                            print(f"      - max_collect: {current_max_collect}")
                            print(f"      - 本次收集样本数: {collected}")
                            print(f"      - 本次迭代已收集: {collected_in_this_iteration}/{self.num_traversals}")
                            print(f"      - 耗时: {time.time() - collection_start_time:.1f}秒")
                            last_warning_time = current_time
                    
                    # 优化：根据队列状态动态调整sleep时间
                    # 队列使用率 > 80% 时不sleep，队列空时才sleep，避免消费速度不够快
                    if queue_usage > 0.80:
                        # 队列满了，不sleep，立即继续消费
                        sleep_time = 0.0
                    elif total_queue_size == 0:
                        # 队列为空，sleep 0.5秒，避免CPU空转
                        sleep_time = 0.5
                    elif queue_usage > 0.50:
                        # 队列使用率 > 50%，sleep时间缩短到0.01秒，加快消费速度
                        sleep_time = 0.01
                    else:
                        # 队列有数据但使用率较低，sleep 0.1秒，平衡消费速度和CPU使用率
                        sleep_time = 0.1
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # 清理队列积压（缓冲区清理已禁用，完全依赖随机替换策略）
                cleanup_final_start = time.time()
                self._check_and_cleanup_memory(cleanup_buffers=False)
                cleanup_final_time = time.time() - cleanup_final_start
                
                collection_total_time = time.time() - collection_start_time
                
                # 训练优势网络（并行优化）
                # 关键修复：第一阶段（完全随机策略）仍然训练网络（用于checkpoint和学习样本）
                # 但不同步到Worker，让Worker继续使用随机策略，避免自博弈
                advantage_train_start = time.time()
                player_train_times = {}  # 记录每个玩家的训练时间
                
                # 优化：使用多线程并行训练多个优势网络
                # PyTorch的CUDA操作是异步的，多线程可以并行执行，充分利用GPU资源
                
                def train_advantage(player):
                    """训练单个玩家的优势网络"""
                    player_start_time = time.time()
                    try:
                        # 重新初始化优势网络（如果启用）
                        # 注意：默认关闭重新初始化，因为每次迭代重置会导致：
                        # 1. 优势网络无法持续学习，生成的策略样本质量差
                        # 2. 策略网络学习差的策略样本，形成恶性循环
                        # 3. Fold概率接近0，All-In概率过高，策略不稳定
                        if self._reinitialize_advantage_networks:
                            self.reinitialize_advantage_network(player)
                        
                        # 注意：样本的iteration字段记录的是创建时的iteration_counter.value
                        # Worker进程读取iteration_counter.value，样本记录的iteration = iteration_counter.value
                        # 训练时，应该使用self._iteration_counter.value来匹配样本的iteration字段
                        # 使用self._iteration_counter.value而不是iteration+1，确保与样本的iteration字段匹配
                        result = self._learn_advantage_network(player, current_iteration=self._iteration_counter.value)
                        player_train_time = time.time() - player_start_time
                        player_train_times[player] = player_train_time
                        return result, player_train_time
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise
                
                # 并行训练（最多使用min(玩家数, GPU数, 4)个线程）
                # 如果GPU数量 >= 玩家数量，每个玩家已经在不同GPU上，可以完全并行
                # 否则使用多线程并行训练（CUDA操作是异步的，可以并行执行）
                if self.use_multi_gpu:
                    max_workers = min(self.num_players, len(self.gpu_ids))
                else:
                    max_workers = min(self.num_players, 6)
                
                # 策略网络训练已移到checkpoint部分，这里只训练优势网络
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交优势网络训练任务
                    advantage_futures = {executor.submit(train_advantage, player): player 
                                       for player in range(self.num_players)}
                    
                    # 等待优势网络训练完成
                    completed_count = 0
                    pending_players = set(range(self.num_players))
                    try:
                        for future in as_completed(advantage_futures, timeout=300):  # 添加总超时，避免无限等待
                            player = advantage_futures[future]
                            completed_count += 1
                            pending_players.discard(player)
                            try:
                                result = future.result()
                                if result is not None:
                                    if isinstance(result, tuple):
                                        loss, player_time = result
                                    else:
                                        loss = result
                                        player_time = player_train_times.get(player, 0)
                                    if loss is not None:
                                        advantage_losses[player].append(loss)
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                get_logger().warning(f"玩家 {player} 优势网络训练失败: {e}")
                    except TimeoutError:
                        # 取消剩余任务
                        for future in advantage_futures:
                            if not future.done():
                                future.cancel()
                        raise RuntimeError(f"优势网络训练超时，已完成 {completed_count}/{self.num_players} 个玩家")
                
                advantage_train_time = time.time() - advantage_train_start
                
                # 输出每个玩家的训练时间（用于验证并行训练是否生效）
                if verbose and player_train_times:
                    player_times_str = ", ".join([f"玩家{i}: {player_train_times.get(i, 0):.2f}秒" 
                                                  for i in range(self.num_players)])
                    get_logger().info(f"    各玩家优势网络训练时间: {player_times_str}")
                    # 计算并行效率
                    max_player_time = max(player_train_times.values()) if player_train_times else 0
                    total_sequential_time = sum(player_train_times.values()) if player_train_times else 0
                    if max_player_time > 0 and advantage_train_time > 0:
                        # 正确的并行效率计算：
                        # 1. 理想并行时间 = max_player_time（所有玩家并行执行，总时间等于最慢的那个）
                        # 2. 实际并行时间 = advantage_train_time（实际测量的总时间）
                        # 3. 并行效率 = (理想并行时间 / 实际并行时间) * 100
                        ideal_parallel_time = max_player_time
                        parallel_efficiency = (ideal_parallel_time / advantage_train_time) * 100
                        
                        # 计算加速比
                        speedup = total_sequential_time / advantage_train_time if advantage_train_time > 0 else 0
                        theoretical_max_speedup = self.num_players
                        speedup_efficiency = (speedup / theoretical_max_speedup) * 100 if theoretical_max_speedup > 0 else 0
                        
                        get_logger().info(f"    并行效率: {parallel_efficiency:.1f}% (理想时间: {ideal_parallel_time:.2f}秒, 实际时间: {advantage_train_time:.2f}秒)")
                        get_logger().info(f"    加速比: {speedup:.2f}x (串行: {total_sequential_time:.2f}秒, 并行: {advantage_train_time:.2f}秒, 理论最大: {theoretical_max_speedup}x, 效率: {speedup_efficiency:.1f}%)")
                        
                        if parallel_efficiency < 70:
                            get_logger().warning(f"    ⚠️ 并行效率较低 ({parallel_efficiency:.1f}%)，可能并行训练未完全生效")
                        elif speedup_efficiency < 70:
                            get_logger().warning(f"    ⚠️ 加速比效率较低 ({speedup_efficiency:.1f}%)，可能并行训练未完全生效")
                
                # 策略网络训练：每次迭代都训练，确保与优势网络同步更新
                # 修改：从只在checkpoint时训练改为每次迭代都训练
                # 这样可以确保策略网络和优势网络同步更新，避免策略样本不一致的问题
                strategy_train_start = time.time()
                policy_loss = self._learn_strategy_network(current_iteration=self._iteration_counter.value)
                strategy_train_time = time.time() - strategy_train_start
                if policy_loss is not None:
                    # 保存策略损失（用于checkpoint时显示）
                    if not hasattr(self, '_last_policy_loss'):
                        self._last_policy_loss = []
                    self._last_policy_loss.append(policy_loss)
                
                # 同步网络参数到 Worker（始终同步，移除两阶段机制）
                # 修改：像单进程训练一样，始终使用训练后的网络，避免两阶段问题
                sync_start = time.time()
                if (iteration + 1) % self.sync_interval == 0:
                    self._sync_network_params()
                sync_time = time.time() - sync_start
                
                self._iteration += 1
                
                iter_time = time.time() - iter_start
                
                # 记录各环节耗时
                if verbose:
                    get_logger().info(f"  迭代 {iteration + 1} 各环节耗时:")
                    get_logger().info(f"    - 样本收集: {collection_total_time:.2f}秒")
                    get_logger().info(f"    - 优势网络训练: {advantage_train_time:.2f}秒")
                    get_logger().info(f"    - 策略网络训练: {strategy_train_time:.2f}秒")
                    if (iteration + 1) % self.sync_interval == 0:
                        get_logger().info(f"    - 网络同步: {sync_time:.2f}秒")
                    get_logger().info(f"    - 其他(清理等): {cleanup_final_time:.2f}秒")
                    get_logger().info(f"    - 迭代总耗时: {iter_time:.2f}秒")
                
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
                
                # 评估：只在checkpoint时评估（优化：减少评估时间）
                # 评估逻辑已移到checkpoint部分
                
                # 保存 checkpoint（同时训练策略网络和评估）
                if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                        if model_dir and save_prefix and game:
                            get_logger().info(f"\n  💾 保存 checkpoint (迭代 {iteration + 1})...")
                            try:
                                # 1. 训练策略网络（checkpoint时额外训练一次，确保策略网络充分学习）
                                get_logger().info("    正在训练策略网络 (用于 Checkpoint，额外训练)...")
                                # 注意：样本的iteration字段记录的是创建时的iteration_counter.value
                                # Worker进程读取iteration_counter.value，样本记录的iteration = iteration_counter.value
                                # 训练时，应该使用self._iteration_counter.value来匹配样本的iteration字段
                                # 使用self._iteration_counter.value而不是iteration+1，确保与样本的iteration字段匹配
                                # 使用最后一次迭代的策略损失（如果存在），否则重新训练
                                if hasattr(self, '_last_policy_loss') and self._last_policy_loss:
                                    policy_loss = self._last_policy_loss[-1]
                                    get_logger().info(f"    使用最后一次迭代的策略损失: MSE={policy_loss/self._iteration_counter.value:.6f} (原始: {policy_loss:.2f})")
                                else:
                                    policy_loss = self._learn_strategy_network(current_iteration=self._iteration_counter.value)
                                    if policy_loss is not None:
                                        # 保存策略损失
                                        policy_losses.append(policy_loss)
                                        # 打印归一化的损失值（MSE）
                                        current_iter = iteration + 1
                                        mse = policy_loss / current_iter if current_iter > 0 else policy_loss
                                        get_logger().info(f"    完成 (MSE: {mse:.6f}, 原始: {policy_loss:.2f})")
                                    else:
                                        get_logger().info("    完成 (无足够样本训练)")
                                
                                # 2. 运行评估（checkpoint时评估）
                                print()
                                # 打印优势网络损失
                                current_iter = iteration + 1
                                for player, losses in advantage_losses.items():
                                    if not losses:
                                        continue
                                    raw_loss = losses[-1]
                                    if self._advantage_loss_type == "mse":
                                        # 归一化：除以iteration（因为损失值 = iteration * MSE）
                                        mse = raw_loss / current_iter if current_iter > 0 else raw_loss
                                        print(f"    玩家 {player} 优势网络损失: MSE={mse:.2f} (原始: {raw_loss:.2f})")
                                    else:
                                        # Huber 等稳健损失不再做“除以iteration”的归一化展示
                                        print(f"    玩家 {player} 优势网络损失: Loss({self._advantage_loss_type})={raw_loss:.6f}")
                                
                                # 打印归一化的策略网络损失（除以iteration得到MSE）
                                if policy_losses:
                                    raw_policy_loss = policy_losses[-1]
                                    # 归一化：除以iteration（因为损失值 = iteration * MSE）
                                    # 归一化后的值就是MSE本身
                                    mse = raw_policy_loss / current_iter if current_iter > 0 else raw_policy_loss
                                    print(f"    策略网络损失: MSE={mse:.6f} (原始: {raw_policy_loss:.2f})")
                                
                                # 运行评估
                                if game is not None:
                                    try:
                                        from training_evaluator import quick_evaluate, evaluate_with_test_games, _get_action_name
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
                                                    
                                                    # 打印测试对局中的动作平均占比（自对弈模式）
                                                    if test_results.get('action_statistics'):
                                                        action_stats = test_results['action_statistics']
                                                        total_count = sum(s['count'] for s in action_stats.values())
                                                        if total_count > 0:
                                                            print(f"    动作统计 (测试对局):")
                                                            # 按占比排序
                                                            sorted_actions = sorted(action_stats.items(), 
                                                                                  key=lambda x: x[1]['percentage'], 
                                                                                  reverse=True)
                                                            action_info = []
                                                            for action, stats in sorted_actions:
                                                                count = stats['count']
                                                                percentage = stats['percentage']
                                                                avg_prob = stats['avg_probability']
                                                                action_name = _get_action_name(action, game)
                                                                action_info.append(f"{action_name}: {percentage:.1f}% (平均概率: {avg_prob:.3f})")
                                                            print(f"      {' | '.join(action_info)}")
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
                                                    
                                                    # 打印测试对局中的动作平均占比
                                                    if test_results.get('action_statistics'):
                                                        action_stats = test_results['action_statistics']
                                                        total_count = sum(s['count'] for s in action_stats.values())
                                                        if total_count > 0:
                                                            print(f"    动作统计 (测试对局):")
                                                            # 按占比排序
                                                            sorted_actions = sorted(action_stats.items(), 
                                                                                  key=lambda x: x[1]['percentage'], 
                                                                                  reverse=True)
                                                            action_info = []
                                                            for action, stats in sorted_actions:
                                                                count = stats['count']
                                                                percentage = stats['percentage']
                                                                avg_prob = stats['avg_probability']
                                                                action_name = _get_action_name(action, game)
                                                                action_info.append(f"{action_name}: {percentage:.1f}% (平均概率: {avg_prob:.3f})")
                                                            print(f"      {' | '.join(action_info)}")

                                                    # ===== 额外评估：不止 vs Random（只用于诊断，不回灌训练）=====
                                                    try:
                                                        extra_opponents = getattr(self, "_eval_extra_opponents", "snapshot") or ""
                                                        extra_opponents = [s.strip().lower() for s in str(extra_opponents).split(",") if s.strip()]
                                                    except Exception:
                                                        extra_opponents = ["snapshot"]

                                                    try:
                                                        extra_games = int(getattr(self, "_eval_extra_games", 200))
                                                    except Exception:
                                                        extra_games = 200
                                                    extra_games = max(10, extra_games)

                                                    def _print_overall_line(label, tr):
                                                        ng = tr.get('games_played', 0)
                                                        if ng <= 0:
                                                            print(f"    {label}: 无有效对局")
                                                            return
                                                        overall_avg_return = tr.get('player0_avg_return', 0)
                                                        overall_win_rate = tr.get('player0_win_rate', 0) * 100
                                                        if bb is not None and bb > 0:
                                                            bb_value2 = overall_avg_return / bb
                                                            print(f"    {label}: {ng}局 | 总体: 平均回报 {overall_avg_return:.2f} ({bb_value2:+.2f} BB), 胜率 {overall_win_rate:.1f}%")
                                                        else:
                                                            print(f"    {label}: {ng}局 | 总体: 平均回报 {overall_avg_return:.2f}, 胜率 {overall_win_rate:.1f}%")
                                                        if tr.get('action_statistics'):
                                                            action_stats2 = tr['action_statistics']
                                                            sorted_actions2 = sorted(action_stats2.items(),
                                                                                    key=lambda x: x[1]['percentage'],
                                                                                    reverse=True)
                                                            parts = []
                                                            for action, st in sorted_actions2[:8]:
                                                                parts.append(f"{_get_action_name(action, game)}: {st['percentage']:.1f}% (p={st['avg_probability']:.3f})")
                                                            print(f"      动作: {' | '.join(parts)}")

                                                    if extra_opponents:
                                                        # 1) vs Snapshot：用旧checkpoint策略当对手
                                                        if "snapshot" in extra_opponents and model_dir and save_prefix:
                                                            try:
                                                                gap = int(getattr(self, "_eval_snapshot_gap", 1000))
                                                            except Exception:
                                                                gap = 1000
                                                            gap = max(1, gap)
                                                            snap_iter = max(1, (iteration + 1) - gap)
                                                            snap_path = os.path.join(
                                                                model_dir, "checkpoints", f"iter_{snap_iter}",
                                                                f"{save_prefix}_policy_network_iter{snap_iter}.pt"
                                                            )
                                                            if os.path.exists(snap_path):
                                                                try:
                                                                    import torch
                                                                    snap_solver = ParallelDeepCFRSolver(
                                                                        game,
                                                                        num_workers=0,  # 仅用于出动作，不启动worker
                                                                        policy_network_layers=tuple(self._policy_network_layers),
                                                                        advantage_network_layers=tuple(self._advantage_network_layers),
                                                                        num_iterations=1,
                                                                        num_traversals=1,
                                                                        learning_rate=self.learning_rate,
                                                                        batch_size_advantage=32,
                                                                        batch_size_strategy=32,
                                                                        memory_capacity=1,
                                                                        strategy_memory_capacity=1,
                                                                        device="cpu",
                                                                        gpu_ids=None,
                                                                        new_sample_ratio=self.new_sample_ratio,
                                                                        new_sample_window=self.new_sample_window,
                                                                        advantage_target_scale=self._advantage_target_scale,
                                                                        advantage_target_clip=self._advantage_target_clip,
                                                                        advantage_loss=self._advantage_loss_type,
                                                                        huber_delta=self._huber_delta
                                                                    )
                                                                    sd = torch.load(snap_path, map_location="cpu")
                                                                    snap_solver._policy_network.load_state_dict(sd)
                                                                    snap_solver._policy_network.eval()
                                                                    tr_snap = evaluate_with_test_games(
                                                                        game,
                                                                        self,
                                                                        num_games=extra_games,
                                                                        verbose=False,
                                                                        mode="vs_random",
                                                                        opponent_solver=snap_solver,
                                                                        opponent_strategy="solver"
                                                                    )
                                                                    _print_overall_line(f"测试对局: {extra_games} 局 (vs Snapshot@{snap_iter})", tr_snap)
                                                                except Exception as e:
                                                                    print(f"    测试对局: vs Snapshot 失败: {e}")
                                                            else:
                                                                print(f"    测试对局: snapshot 不存在（期望 {snap_path}）")

                                                        # 2) vs Tight：能fold就fold，否则call/check
                                                        if "tight" in extra_opponents:
                                                            tr_tight = evaluate_with_test_games(
                                                                game,
                                                                self,
                                                                num_games=extra_games,
                                                                verbose=False,
                                                                mode="vs_random",
                                                                opponent_solver=None,
                                                                opponent_strategy="tight"
                                                            )
                                                            _print_overall_line(f"测试对局: {extra_games} 局 (vs Tight)", tr_tight)

                                                        # 3) vs Call：优先call/check
                                                        if "call" in extra_opponents:
                                                            tr_call = evaluate_with_test_games(
                                                                game,
                                                                self,
                                                                num_games=extra_games,
                                                                verbose=False,
                                                                mode="vs_random",
                                                                opponent_solver=None,
                                                                opponent_strategy="call"
                                                            )
                                                            _print_overall_line(f"测试对局: {extra_games} 局 (vs Call)", tr_call)
                                                        
                                                        # 4) vs 指定Checkpoint：使用指定的checkpoint作为对手
                                                        eval_checkpoint_path = getattr(self, "_eval_checkpoint_path", None)
                                                        if eval_checkpoint_path and os.path.exists(eval_checkpoint_path):
                                                            try:
                                                                import torch
                                                                import json
                                                                import glob
                                                                import re
                                                                
                                                                # 查找策略网络文件
                                                                policy_files = glob.glob(os.path.join(eval_checkpoint_path, "*_policy_network*.pt"))
                                                                if not policy_files:
                                                                    # 尝试从父目录查找
                                                                    parent_dir = os.path.dirname(eval_checkpoint_path)
                                                                    policy_files = glob.glob(os.path.join(parent_dir, "*_policy_network*.pt"))
                                                                
                                                                if policy_files:
                                                                    # 选择最新的checkpoint文件
                                                                    latest_file = None
                                                                    max_iter = 0
                                                                    for f in policy_files:
                                                                        match = re.search(r'_iter(\d+)\.pt$', f)
                                                                        if match:
                                                                            iter_num = int(match.group(1))
                                                                            if iter_num > max_iter:
                                                                                max_iter = iter_num
                                                                                latest_file = f
                                                                    if not latest_file:
                                                                        latest_file = policy_files[0]
                                                                    
                                                                    # 加载配置
                                                                    config_path = os.path.join(eval_checkpoint_path, "config.json")
                                                                    if not os.path.exists(config_path):
                                                                        parent_dir = os.path.dirname(eval_checkpoint_path)
                                                                        config_path = os.path.join(parent_dir, "config.json")
                                                                    
                                                                    checkpoint_solver = ParallelDeepCFRSolver(
                                                                        game,
                                                                        num_workers=0,
                                                                        policy_network_layers=tuple(self._policy_network_layers),
                                                                        advantage_network_layers=tuple(self._advantage_network_layers),
                                                                        num_iterations=1,
                                                                        num_traversals=1,
                                                                        learning_rate=self.learning_rate,
                                                                        batch_size_advantage=32,
                                                                        batch_size_strategy=32,
                                                                        memory_capacity=1,
                                                                        strategy_memory_capacity=1,
                                                                        device="cpu",
                                                                        gpu_ids=None,
                                                                        new_sample_ratio=self.new_sample_ratio,
                                                                        new_sample_window=self.new_sample_window,
                                                                        advantage_target_scale=self._advantage_target_scale,
                                                                        advantage_target_clip=self._advantage_target_clip,
                                                                        advantage_loss=self._advantage_loss_type,
                                                                        huber_delta=self._huber_delta
                                                                    )
                                                                    sd = torch.load(latest_file, map_location="cpu")
                                                                    checkpoint_solver._policy_network.load_state_dict(sd)
                                                                    checkpoint_solver._policy_network.eval()
                                                                    
                                                                    checkpoint_name = os.path.basename(eval_checkpoint_path)
                                                                    tr_checkpoint = evaluate_with_test_games(
                                                                        game,
                                                                        self,
                                                                        num_games=extra_games,
                                                                        verbose=False,
                                                                        mode="vs_random",
                                                                        opponent_solver=checkpoint_solver,
                                                                        opponent_strategy="solver"
                                                                    )
                                                                    _print_overall_line(f"测试对局: {extra_games} 局 (vs Checkpoint: {checkpoint_name})", tr_checkpoint)
                                                                else:
                                                                    print(f"    测试对局: 找不到checkpoint文件 ({eval_checkpoint_path})")
                                                            except Exception as e:
                                                                print(f"    测试对局: vs Checkpoint 失败: {e}")
                                                                import traceback
                                                                traceback.print_exc()
                                                    
                                                    # 检查是否应该开始过渡阶段
                                                    win_rate = test_results.get('player0_win_rate', None)
                                                    avg_return = test_results.get('player0_avg_return', None)
                                                    
                                                    # 转换为BB单位（如果需要）
                                                    if avg_return is not None and bb is not None and bb > 0:
                                                        avg_return_bb = avg_return / bb
                                                    else:
                                                        avg_return_bb = avg_return
                                                    
                                                    # 检查是否应该开始过渡阶段
                                                    self.should_start_transition(iteration, advantage_losses, win_rate, avg_return_bb)
                                    except ImportError:
                                        pass  # training_evaluator 不可用
                                    except Exception as e:
                                        print(f" 评估失败: {e}")
                                
                                # 3. 保存checkpoint
                                save_checkpoint(self, game, model_dir, save_prefix, iteration + 1)
                                get_logger().info("  ✓ Checkpoint 已保存")
                            except Exception as e:
                                print(f" 失败: {e}")
            
            print()
            
            # 训练策略网络（最终训练，使用最后一次迭代号）
            print("  训练策略网络...")
            final_iteration = self.num_iterations - 1 if start_iteration < self.num_iterations else start_iteration
            policy_loss = self._learn_strategy_network(current_iteration=final_iteration)
            
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
        
        # 返回策略网络、优势网络损失历史和策略网络损失历史
        final_policy_loss = policy_losses[-1] if policy_losses else None
        return self._policy_network, advantage_losses, final_policy_loss
    
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
    """从 checkpoint 加载网络权重和多阶段训练状态
    
    Args:
        solver: ParallelDeepCFRSolver 实例
        model_dir: 模型目录
        save_prefix: 保存前缀
        game: 游戏实例
        
    Returns:
        start_iteration: 恢复的迭代次数
        training_state: 训练状态字典（包含switch_start_iteration、win_rate_history、avg_return_history）
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
        return 0, None
    
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
            training_state_dir = adv_dir
        else:
            # 旧结构: checkpoints/
            adv_dir = checkpoint_root
            training_state_dir = checkpoint_root
    else:
        adv_dir = model_dir
        training_state_dir = model_dir

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
    
    # 加载多阶段训练状态
    training_state = None
    if start_iteration > 0:
        training_state_path = os.path.join(training_state_dir, "training_state.json")
        if os.path.exists(training_state_path):
            import json
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            print(f"  ✓ 多阶段训练状态已加载:")
            print(f"    - switch_start_iteration: {training_state.get('switch_start_iteration')}")
            print(f"    - win_rate_history长度: {len(training_state.get('win_rate_history', []))}")
            print(f"    - avg_return_history长度: {len(training_state.get('avg_return_history', []))}")
        else:
            print(f"  ⚠️ 未找到训练状态文件: {training_state_path}，将使用默认值")
    
    return start_iteration, training_state


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
    
    # 保存多阶段训练状态
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    import json
    training_state = {
        'switch_start_iteration': solver.switch_start_iteration,
        'win_rate_history': solver.win_rate_history,
        'avg_return_history': solver.avg_return_history,
        'iteration': iteration
    }
    with open(training_state_path, 'w') as f:
        json.dump(training_state, f, indent=2)
    
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
                        help="优势网络经验回放缓冲区容量（每个玩家，默认: 1000000）")
    parser.add_argument("--strategy_memory_capacity", type=int, default=None,
                        help="策略网络经验回放缓冲区容量（所有玩家共享，默认: 使用memory_capacity）")
    parser.add_argument("--max_memory_gb", type=float, default=None,
                        help="最大内存限制（GB），超过此限制会自动清理旧样本（默认: 不限制）")
    parser.add_argument("--queue_maxsize", type=int, default=50000,
                        help="队列最大大小，降低可减少内存占用（默认: 50000）")
    parser.add_argument("--new_sample_ratio", type=float, default=0.5,
                        help="新样本占比（分层加权采样，默认0.5即50%）")
    parser.add_argument("--new_sample_window", type=int, default=0,
                        help="新样本窗口（最近W轮算“新”）；0表示仅当前轮。建议 10~50")
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
    parser.add_argument("--num_test_games", type=int, default=1000,
                        help="评估时的测试对局数量（默认: 1000）")
    parser.add_argument("--eval_extra_opponents", type=str, default="snapshot",
                        help="额外评估对手（逗号分隔）：snapshot,tight,call。例如 'snapshot,tight'。空字符串表示不额外评估")
    parser.add_argument("--eval_extra_games", type=int, default=200,
                        help="额外评估每种对手的对局数量（默认: 200，避免评估过慢）")
    parser.add_argument("--eval_snapshot_gap", type=int, default=1000,
                        help="snapshot评估使用的旧checkpoint间隔（默认: 1000，例如用 iter-(gap) 作为对手）")
    parser.add_argument("--eval_checkpoint_path", type=str, default=None,
                        help="指定checkpoint路径作为评测对手（例如：models/deepcfr_6p_multi_20260116_171819/checkpoints/iter_114200）")
    parser.add_argument("--blinds", type=str, default=None,
                        help="盲注配置，格式：'小盲 大盲' 或 '50 100 0 0 0 0'（多人场完整配置）。如果不指定，将根据玩家数量自动生成")
    parser.add_argument("--stack_size", type=int, default=None,
                        help="每个玩家的初始筹码（默认: 2000）。如果不指定，将使用默认值2000")
    parser.add_argument("--advantage_train_steps", type=int, default=1,
                        help="优势网络训练步数（每次迭代，默认: 1）")
    parser.add_argument("--policy_train_steps", type=int, default=1,
                        help="策略网络训练步数（每次迭代，默认: 1）")
    
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
            if 'strategy_memory_capacity' in resume_config and args.strategy_memory_capacity is None:
                args.strategy_memory_capacity = resume_config['strategy_memory_capacity']
            if 'advantage_train_steps' in resume_config:
                args.advantage_train_steps = resume_config['advantage_train_steps']
            if 'policy_train_steps' in resume_config:
                args.policy_train_steps = resume_config['policy_train_steps']
            if 'new_sample_ratio' in resume_config:
                args.new_sample_ratio = resume_config['new_sample_ratio']
                
            print(f"  自动加载配置: {args.num_players}人局, 策略层{args.policy_layers}, 优势层{args.advantage_layers}")
            print(f"  save_prefix: {args.save_prefix}")
            print(f"  优势网络训练步数: {args.advantage_train_steps}, 策略网络训练步数: {args.policy_train_steps}")
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
        strategy_memory_capacity=args.strategy_memory_capacity,
        device=device,
        gpu_ids=gpu_ids,
        max_memory_gb=args.max_memory_gb,
        queue_maxsize=args.queue_maxsize,
        new_sample_ratio=args.new_sample_ratio,
        new_sample_window=args.new_sample_window,
        advantage_network_train_steps=args.advantage_train_steps,
        policy_network_train_steps=args.policy_train_steps,
    )

    # 评估扩展参数（仅用于 checkpoint 评估打印，不影响训练样本）
    solver._eval_extra_opponents = args.eval_extra_opponents
    solver._eval_extra_games = args.eval_extra_games
    solver._eval_snapshot_gap = args.eval_snapshot_gap
    solver._eval_checkpoint_path = args.eval_checkpoint_path
    
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
            'strategy_memory_capacity': args.strategy_memory_capacity,
            'max_memory_gb': args.max_memory_gb,
            'queue_maxsize': args.queue_maxsize,
            'new_sample_ratio': args.new_sample_ratio,
            'new_sample_window': args.new_sample_window,
            'advantage_train_steps': args.advantage_train_steps,
            'policy_train_steps': args.policy_train_steps,
            'eval_extra_opponents': args.eval_extra_opponents,
            'eval_extra_games': args.eval_extra_games,
            'eval_snapshot_gap': args.eval_snapshot_gap,
            'eval_checkpoint_path': args.eval_checkpoint_path,
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
    training_state = None
    if args.resume:
        print(f"\n加载 checkpoint...")
        start_iteration, training_state = load_checkpoint(solver, model_dir, args.save_prefix, game)
        if start_iteration > 0:
            print(f"  ✓ 将从迭代 {start_iteration + 1} 继续训练")
        else:
            print(f"  ⚠️ 未找到有效 checkpoint，从头开始训练")
    else:
        start_iteration = 0
    
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
        training_state=training_state,
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
        'new_sample_ratio': args.new_sample_ratio,
        'advantage_train_steps': args.advantage_train_steps,
        'policy_train_steps': args.policy_train_steps,
        'strategy_memory_capacity': args.strategy_memory_capacity,
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

