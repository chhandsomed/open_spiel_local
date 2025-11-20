#!/usr/bin/env python3
"""训练过程评估模块

提供轻量级的评估指标，用于在训练过程中评估训练效果，
无需计算耗时的 NashConv。
"""

import numpy as np
import torch
import pyspiel
from collections import defaultdict


def compute_policy_entropy(probs_dict):
    """计算策略熵（衡量策略的随机性）
    
    Args:
        probs_dict: {action: probability} 字典
    
    Returns:
        float: 策略熵
    """
    probs = np.array(list(probs_dict.values()))
    probs = probs[probs > 0]  # 只考虑非零概率
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs + 1e-10))


def evaluate_policy_quality(
    game,
    deep_cfr_solver,
    num_sample_states=100,
    verbose=True
):
    """评估策略质量（轻量级方法）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        num_sample_states: 采样状态数量
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 1. 策略熵统计
    entropies = []
    states_sampled = 0
    
    try:
        # 采样一些状态，计算策略熵
        for _ in range(num_sample_states):
            state = game.new_initial_state()
            depth = 0
            max_depth = 10  # 限制深度，避免太深
            
            while not state.is_terminal() and depth < max_depth:
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action = np.random.choice([a for a, _ in outcomes], 
                                             p=[p for _, p in outcomes])
                    state = state.child(action)
                else:
                    player = state.current_player()
                    probs = deep_cfr_solver.action_probabilities(state, player)
                    entropy = compute_policy_entropy(probs)
                    entropies.append(entropy)
                    states_sampled += 1
                    
                    # 根据策略采样动作
                    actions = list(probs.keys())
                    probabilities = list(probs.values())
                    action = np.random.choice(actions, p=probabilities)
                    state = state.child(action)
                depth += 1
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 策略熵计算出错: {e}")
    
    if entropies:
        metrics['avg_entropy'] = np.mean(entropies)
        metrics['std_entropy'] = np.std(entropies)
        metrics['min_entropy'] = np.min(entropies)
        metrics['max_entropy'] = np.max(entropies)
        metrics['states_sampled'] = states_sampled
    else:
        metrics['avg_entropy'] = 0.0
        metrics['states_sampled'] = 0
    
    # 2. 缓冲区大小（探索的状态数量）
    try:
        strategy_buffer_size = len(deep_cfr_solver.strategy_buffer)
        metrics['strategy_buffer_size'] = strategy_buffer_size
        
        advantage_buffer_sizes = {}
        for player in range(game.num_players()):
            if player < len(deep_cfr_solver.advantage_buffers):
                size = len(deep_cfr_solver.advantage_buffers[player])
                advantage_buffer_sizes[player] = size
        metrics['advantage_buffer_sizes'] = advantage_buffer_sizes
        metrics['total_advantage_samples'] = sum(advantage_buffer_sizes.values())
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 缓冲区大小计算出错: {e}")
        metrics['strategy_buffer_size'] = 0
        metrics['advantage_buffer_sizes'] = {}
        metrics['total_advantage_samples'] = 0
    
    return metrics


def play_test_game(
    game,
    deep_cfr_solver,
    opponent_strategy="random",
    verbose=False
):
    """测试对局（与随机策略对局）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        opponent_strategy: 对手策略 ("random" 或 "uniform")
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含对局结果的字典
    """
    state = game.new_initial_state()
    returns = None
    
    try:
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = np.random.choice([a for a, _ in outcomes], 
                                         p=[p for _, p in outcomes])
                state = state.child(action)
            else:
                player = state.current_player()
                legal_actions = state.legal_actions()
                
                if player == 0:  # 使用训练的策略
                    probs = deep_cfr_solver.action_probabilities(state, player)
                    # 确保所有合法动作都有概率
                    action_probs = {a: probs.get(a, 0.0) for a in legal_actions}
                    # 归一化
                    total = sum(action_probs.values())
                    if total > 0:
                        action_probs = {a: p/total for a, p in action_probs.items()}
                    else:
                        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
                    actions = list(action_probs.keys())
                    probabilities = list(action_probs.values())
                    action = np.random.choice(actions, p=probabilities)
                else:  # 对手使用随机策略
                    action = np.random.choice(legal_actions)
                
                state = state.child(action)
        
        returns = state.returns()
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 测试对局出错: {e}")
        return None
    
    return {
        'returns': returns,
        'player0_return': returns[0] if returns else 0.0,
        'player1_return': returns[1] if returns else 0.0,
    }


def evaluate_with_test_games(
    game,
    deep_cfr_solver,
    num_games=100,
    verbose=True
):
    """通过测试对局评估策略
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        num_games: 测试对局数量
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含测试结果的字典
    """
    results = {
        'player0_returns': [],
        'player1_returns': [],
        'player0_wins': 0,
        'player1_wins': 0,
        'draws': 0,
    }
    
    for _ in range(num_games):
        game_result = play_test_game(game, deep_cfr_solver, verbose=False)
        if game_result:
            p0_return = game_result['player0_return']
            p1_return = game_result['player1_return']
            results['player0_returns'].append(p0_return)
            results['player1_returns'].append(p1_return)
            
            if p0_return > p1_return:
                results['player0_wins'] += 1
            elif p1_return > p0_return:
                results['player1_wins'] += 1
            else:
                results['draws'] += 1
    
    if results['player0_returns']:
        results['player0_avg_return'] = np.mean(results['player0_returns'])
        results['player0_std_return'] = np.std(results['player0_returns'])
        results['player0_win_rate'] = results['player0_wins'] / num_games
    else:
        results['player0_avg_return'] = 0.0
        results['player0_win_rate'] = 0.0
    
    return results


def print_evaluation_summary(metrics, test_results=None, iteration=None):
    """打印评估摘要"""
    print("\n" + "=" * 70)
    if iteration is not None:
        print(f"训练评估 (迭代 {iteration})")
    else:
        print("训练评估")
    print("=" * 70)
    
    # 策略质量指标
    print("\n[策略质量]")
    if metrics.get('states_sampled', 0) > 0:
        print(f"  平均策略熵: {metrics.get('avg_entropy', 0):.4f}")
        print(f"  策略熵范围: [{metrics.get('min_entropy', 0):.4f}, {metrics.get('max_entropy', 0):.4f}]")
        print(f"  采样状态数: {metrics.get('states_sampled', 0)}")
    else:
        print("  ⚠️ 无法计算策略熵（采样失败）")
    
    # 缓冲区大小
    print("\n[探索进度]")
    print(f"  策略缓冲区: {metrics.get('strategy_buffer_size', 0):,} 样本")
    if metrics.get('advantage_buffer_sizes'):
        for player, size in metrics['advantage_buffer_sizes'].items():
            print(f"  玩家{player}优势缓冲区: {size:,} 样本")
    print(f"  总优势样本: {metrics.get('total_advantage_samples', 0):,}")
    
    # 测试对局结果
    if test_results:
        print("\n[测试对局] (vs 随机策略)")
        print(f"  玩家0平均收益: {test_results.get('player0_avg_return', 0):.4f} "
              f"(±{test_results.get('player0_std_return', 0):.4f})")
        print(f"  玩家0胜率: {test_results.get('player0_win_rate', 0)*100:.2f}%")
        print(f"  对局数: {len(test_results.get('player0_returns', []))}")
    
    print("=" * 70)


def quick_evaluate(
    game,
    deep_cfr_solver,
    include_test_games=False,
    num_test_games=50,
    verbose=True
):
    """快速评估（轻量级）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        include_test_games: 是否包含测试对局
        num_test_games: 测试对局数量
        verbose: 是否显示详细信息
    
    Returns:
        dict: 评估结果
    """
    # 策略质量评估
    metrics = evaluate_policy_quality(game, deep_cfr_solver, 
                                      num_sample_states=50, 
                                      verbose=verbose)
    
    # 测试对局（可选）
    test_results = None
    if include_test_games:
        if verbose:
            print("  进行测试对局...", end="", flush=True)
        test_results = evaluate_with_test_games(
            game, deep_cfr_solver, num_games=num_test_games, verbose=verbose
        )
        if verbose:
            print(" 完成")
    
    return {
        'metrics': metrics,
        'test_results': test_results
    }


