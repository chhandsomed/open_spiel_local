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
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value, Manager
from collections import namedtuple
import queue

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
    device='cpu'
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
    import setproctitle
    try:
        setproctitle.setproctitle(f"deepcfr_worker_{worker_id}")
    except:
        pass
    
    print(f"[Worker {worker_id}] 启动，设备: {device}")
    
    # 创建游戏
    game = pyspiel.load_game(game_string)
    root_node = game.new_initial_state()
    
    # 创建本地优势网络（用于采样动作）
    advantage_networks = []
    for _ in range(num_players):
        net = SimpleFeatureMLP(
            embedding_size,
            list(advantage_network_layers),
            num_actions,
            num_players=num_players,
            max_game_length=game.max_game_length()
        )
        net = net.to(device)
        net.eval()
        advantage_networks.append(net)
    
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
            try:
                advantage_queues[player].put_nowait(sample)
            except queue.Full:
                pass  # 队列满了就丢弃
            
            return cfv
        else:
            other_player = state.current_player()
            _, strategy = sample_action_from_advantage(state, other_player)
            
            probs = np.array(strategy)
            probs /= probs.sum()
            sampled_action = np.random.choice(range(num_actions), p=probs)
            
            # 发送策略样本
            sample = StrategyMemory(
                state.information_state_tensor(other_player),
                iteration,
                strategy
            )
            try:
                strategy_queue.put_nowait(sample)
            except queue.Full:
                pass
            
            return traverse_game_tree(state.child(sampled_action), player, iteration)
    
    # 主循环
    last_sync_iteration = 0
    
    while not stop_event.is_set():
        current_iteration = iteration_counter.value
        
        # 检查是否需要同步网络参数
        try:
            while True:
                params = network_params_queue.get_nowait()
                for player in range(num_players):
                    if player in params:
                        numpy_dict = params[player]
                        state_dict = {k: torch.from_numpy(v) for k, v in numpy_dict.items()}
                        advantage_networks[player].load_state_dict(state_dict)
                        advantage_networks[player] = advantage_networks[player].to(device)
                last_sync_iteration = current_iteration
        except queue.Empty:
            pass
        
        # 遍历游戏树
        for player in range(num_players):
            for _ in range(num_traversals_per_batch):
                if stop_event.is_set():
                    break
                traverse_game_tree(root_node.clone(), player, current_iteration)
    
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
    
    def _create_networks(self):
        """创建神经网络"""
        # 策略网络
        policy_net = SimpleFeatureMLP(
            self._embedding_size,
            list(self._policy_network_layers),
            self._num_actions,
            num_players=self.num_players,
            max_game_length=self.game.max_game_length()
        )
        
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
        for _ in range(self.num_players):
            net = SimpleFeatureMLP(
                self._embedding_size,
                list(self._advantage_network_layers),
                self._num_actions,
                num_players=self.num_players,
                max_game_length=self.game.max_game_length()
            )
            
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
        
        # 创建队列
        self._advantage_queues = [Queue(maxsize=100000) for _ in range(self.num_players)]
        self._strategy_queue = Queue(maxsize=100000)
        self._network_params_queues = [Queue(maxsize=10) for _ in range(self.num_workers)]
        
        # 计算每个 Worker 的遍历次数
        traversals_per_worker = max(1, self.num_traversals // self.num_workers)
        
        # 启动 Worker
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
                )
            )
            p.start()
            self._workers.append(p)
        
        print(f"已启动 {self.num_workers} 个 Worker 进程")
    
    def _stop_workers(self):
        """停止 Worker 进程"""
        self._stop_event.set()
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
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
    
    def _collect_samples(self, timeout=0.1):
        """从队列收集样本"""
        # 收集优势样本
        for player in range(self.num_players):
            while True:
                try:
                    sample = self._advantage_queues[player].get_nowait()
                    self._advantage_memories[player].add(sample)
                except queue.Empty:
                    break
        
        # 收集策略样本
        while True:
            try:
                sample = self._strategy_queue.get_nowait()
                self._strategy_memories.add(sample)
            except queue.Empty:
                break
    
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
              model_dir=None, save_prefix=None, game=None):
        """运行并行 DeepCFR 训练
        
        Args:
            verbose: 是否显示详细信息
            eval_interval: 评估间隔
            checkpoint_interval: checkpoint 保存间隔（0=不保存）
            model_dir: 模型保存目录
            save_prefix: 保存文件前缀
            game: 游戏实例（用于保存 checkpoint）
        
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
            
            for iteration in range(self.num_iterations):
                iter_start = time.time()
                
                # 更新迭代计数器
                self._iteration_counter.value = iteration + 1
                
                # 等待 Worker 收集样本（根据遍历次数动态调整）
                # 每次遍历大约需要 0.2-0.5 秒，总共需要 num_traversals / num_workers 次
                wait_time = max(0.5, (self.num_traversals / self.num_workers) * 0.3)
                time.sleep(wait_time)
                
                # 收集样本
                self._collect_samples()
                
                # 训练优势网络
                for player in range(self.num_players):
                    loss = self._learn_advantage_network(player)
                    if loss is not None:
                        advantage_losses[player].append(loss)
                
                # 同步网络参数到 Worker
                if (iteration + 1) % self.sync_interval == 0:
                    self._sync_network_params()
                
                self._iteration += 1
                
                iter_time = time.time() - iter_start
                
                if verbose:
                    print(f"\r  迭代 {iteration + 1}/{self.num_iterations} "
                          f"(耗时: {iter_time:.2f}秒) | "
                          f"优势样本: {sum(len(m) for m in self._advantage_memories):,} | "
                          f"策略样本: {len(self._strategy_memories):,}", end="")
                
                if (iteration + 1) % eval_interval == 0:
                    print()
                    for player, losses in advantage_losses.items():
                        if losses:
                            print(f"    玩家 {player} 损失: {losses[-1]:.6f}")
                
                # 保存 checkpoint
                if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                    if model_dir and save_prefix and game:
                        print(f"\n  💾 保存 checkpoint (迭代 {iteration + 1})...", end="", flush=True)
                        try:
                            save_checkpoint(self, game, model_dir, save_prefix, iteration + 1)
                            print(" 完成")
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
                print(f"  💾 保存中断时的 checkpoint (迭代 {self._iteration})...")
                try:
                    save_checkpoint(self, game, model_dir, save_prefix, self._iteration)
                    print(f"  ✓ Checkpoint 已保存")
                except Exception as e:
                    print(f"  ✗ 保存失败: {e}")
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
        return {action: probs[0][action] for action in legal_actions}


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
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
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
    parser = argparse.ArgumentParser(description="多进程并行 DeepCFR 训练")
    parser.add_argument("--num_players", type=int, default=2, help="玩家数量")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker 进程数量")
    parser.add_argument("--num_iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--num_traversals", type=int, default=40, help="每次迭代遍历次数")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--advantage_layers", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--memory_capacity", type=int, default=1000000)
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
    
    args = parser.parse_args()
    
    # 创建游戏
    num_players = args.num_players
    if num_players == 2:
        blinds_str = "100 50"
        first_player_str = "2 1 1 1"
    else:
        blinds_list = ["50", "100"] + ["0"] * (num_players - 2)
        blinds_str = " ".join(blinds_list)
        first_player_str = " ".join(["3"] + ["1"] * 3)
    
    stacks_str = " ".join(["2000"] * num_players)
    
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
    
    # 创建保存目录
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
    )
    
    # 训练（带 checkpoint 支持）
    policy_network, advantage_losses, policy_loss = solver.solve(
        verbose=True,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        model_dir=model_dir,
        save_prefix=args.save_prefix,
        game=game,
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
        'betting_abstraction': args.betting_abstraction,
        'device': device,
        'gpu_ids': gpu_ids,
        'game_string': game_string,
        'multi_gpu': gpu_ids is not None and len(gpu_ids) > 1,
        'parallel': True,
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

