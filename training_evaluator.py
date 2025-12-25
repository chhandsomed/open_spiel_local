#!/usr/bin/env python3
"""训练过程评估模块

提供轻量级的评估指标，用于在训练过程中评估训练效果，
无需计算耗时的 NashConv。
"""

import numpy as np
import torch
import pyspiel
import re
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
    max_depth=None,
    verbose=True
):
    """评估策略质量（轻量级方法）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        num_sample_states: 采样状态数量
        max_depth: 最大采样深度（None 时自动设置）
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 自动设置 max_depth（基于游戏特性）
    if max_depth is None:
        try:
            # 尝试获取游戏的最大长度
            max_game_length = game.max_game_length()
            # 设置为最大长度的 75%，确保能覆盖大部分状态
            # 但不超过 30，防止评估时间过长
            max_depth = min(int(max_game_length * 0.75), 30)
        except:
            # 如果无法获取，使用默认值
            # 德州扑克：4轮 × 平均每轮3-4步 ≈ 15
            max_depth = 15
    
    # 1. 策略熵统计
    entropies = []
    states_sampled = 0
    
    try:
        # 采样一些状态，计算策略熵
        for sample_idx in range(num_sample_states):
            try:
                state = game.new_initial_state()
                depth = 0
                
                while not state.is_terminal() and depth < max_depth:
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        if not outcomes:
                            break
                        action = np.random.choice([a for a, _ in outcomes], 
                                                 p=[p for _, p in outcomes])
                        state = state.child(action)
                    else:
                        player = state.current_player()
                        legal_actions = state.legal_actions()
                        
                        if len(legal_actions) == 0:
                            break
                        
                        # 获取策略概率
                        try:
                            probs = deep_cfr_solver.action_probabilities(state, player)
                            
                            # 确保所有合法动作都有概率
                            action_probs = {a: probs.get(a, 0.0) for a in legal_actions}
                            
                            # 归一化（防止概率和不为1）
                            total = sum(action_probs.values())
                            if total > 0:
                                # 修复: 确保总和严格为 1.0，避免浮点误差
                                # 先做除法
                                action_probs = {a: p/total for a, p in action_probs.items()}
                                # 再重新归一化，处理最后的微小误差
                                total_v2 = sum(action_probs.values())
                                if abs(total_v2 - 1.0) > 1e-9:
                                    # 找出最大概率的动作，把误差加到它身上
                                    max_a = max(action_probs, key=action_probs.get)
                                    diff = 1.0 - total_v2
                                    action_probs[max_a] += diff
                            else:
                                # 如果所有概率为0，使用均匀分布
                                action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
                            
                            # 计算熵
                            entropy = compute_policy_entropy(action_probs)
                            if entropy > 0:  # 只记录有效的熵值
                                entropies.append(entropy)
                                states_sampled += 1
                            
                            # 根据策略采样动作继续
                            actions = list(action_probs.keys())
                            probabilities = list(action_probs.values())
                            action = np.random.choice(actions, p=probabilities)
                            state = state.child(action)
                        except Exception as e:
                            # 如果获取策略失败，跳过这个状态
                            if verbose:
                                print(f"    采样 {sample_idx}: 获取策略失败: {e}")
                            break
                    
                    depth += 1
                    
                    # 如果已经采样到足够的状态，提前退出
                    if states_sampled >= num_sample_states:
                        break
                        
            except Exception as e:
                # 单个采样失败不影响整体
                if verbose:
                    print(f"    采样 {sample_idx} 失败: {e}")
                continue
                
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 策略熵计算出错: {e}")
            import traceback
            traceback.print_exc()
    
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
        # 兼容 ParallelDeepCFRSolver 和标准 DeepCFRSolver
        if hasattr(deep_cfr_solver, 'strategy_buffer'):
            # 标准 OpenSpiel DeepCFR
            strategy_buffer_size = len(deep_cfr_solver.strategy_buffer)
            advantage_buffers = deep_cfr_solver.advantage_buffers
        elif hasattr(deep_cfr_solver, '_strategy_memories'):
            # 自定义 ParallelDeepCFRSolver
            strategy_buffer_size = len(deep_cfr_solver._strategy_memories)
            advantage_buffers = deep_cfr_solver._advantage_memories
        else:
            strategy_buffer_size = 0
            advantage_buffers = []
            
        metrics['strategy_buffer_size'] = strategy_buffer_size
        
        advantage_buffer_sizes = {}
        for player in range(game.num_players()):
            if player < len(advantage_buffers):
                size = len(advantage_buffers[player])
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
    verbose=False,
    use_trained_for_all=True,
    trained_player=None
):
    """测试对局（自对弈或与随机策略对局）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        opponent_strategy: 对手策略 ("random" 或 "uniform")
        verbose: 是否显示详细信息
        use_trained_for_all: 是否所有玩家都使用训练策略（自对弈模式）
        trained_player: 指定哪个玩家使用训练策略（None时在vs_random模式下随机选择）
    
    Returns:
        dict: 包含对局结果的字典
    """
    state = game.new_initial_state()
    returns = None
    num_players = game.num_players()
    max_steps = 1000  # 防止无限循环
    
    # 如果未指定trained_player且不是自对弈模式，随机选择一个玩家使用训练策略
    if not use_trained_for_all and trained_player is None:
        trained_player = np.random.randint(0, num_players)
    
    try:
        steps = 0
        while not state.is_terminal():
            if steps >= max_steps:
                if verbose:
                    print(f"  ⚠️ 测试对局超时（超过 {max_steps} 步）")
                return None
            
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                if not outcomes:
                    if verbose:
                        print(f"  ⚠️ 测试对局出错: chance节点没有可用动作")
                    return None
                action = np.random.choice([a for a, _ in outcomes], 
                                         p=[p for _, p in outcomes])
                state = state.child(action)
            else:
                player = state.current_player()
                legal_actions = state.legal_actions()
                
                if not legal_actions:
                    if verbose:
                        print(f"  ⚠️ 测试对局出错: 玩家 {player} 没有合法动作")
                    return None
                
                # 决定是否使用训练策略
                use_trained = use_trained_for_all or (player == trained_player)
                
                if use_trained:  # 使用训练的策略
                    try:
                        probs = deep_cfr_solver.action_probabilities(state, player)
                        # 确保所有合法动作都有概率
                        action_probs = {a: probs.get(a, 0.0) for a in legal_actions}
                        # 归一化
                        total = sum(action_probs.values())
                        if total > 0:
                            action_probs = {a: p/total for a, p in action_probs.items()}
                            
                            # 修复: 确保总和严格为 1.0
                            total_v2 = sum(action_probs.values())
                            if abs(total_v2 - 1.0) > 1e-9:
                                max_a = max(action_probs, key=action_probs.get)
                                diff = 1.0 - total_v2
                                action_probs[max_a] += diff
                        else:
                            # 如果所有概率为0，使用均匀分布
                            action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
                        # 使用确定性策略：选择概率最高的动作（如果相同则选择第一个）
                        best_action = max(legal_actions, key=lambda a: action_probs.get(a, 0.0))
                        action = best_action
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠️ 获取动作概率失败: {e}")
                        # 使用固定策略：选择第一个合法动作
                        action = legal_actions[0]
                else:  # 对手使用随机策略
                    # 随机选择合法动作
                    action = np.random.choice(legal_actions)
                
                state = state.child(action)
            
            steps += 1
        
        returns = state.returns()
    except Exception as e:
        if verbose:
            import traceback
            print(f"  ⚠️ 测试对局出错: {e}")
            traceback.print_exc()
        return None
    
    # 返回所有玩家的收益和使用的训练玩家
    result = {
        'returns': returns,
        'num_players': num_players,
        'trained_player': trained_player if not use_trained_for_all else None,
    }
    if returns:
        for i in range(min(len(returns), num_players)):
            result[f'player{i}_return'] = returns[i]
    return result


def evaluate_with_test_games(
    game,
    deep_cfr_solver,
    num_games=100,
    verbose=True,
    mode="self_play"  # "self_play" or "vs_random"
):
    """通过测试对局评估策略
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        num_games: 测试对局数量
        verbose: 是否显示详细信息
        mode: "self_play" (自对弈) 或 "vs_random" (随机位置用模型，其他用随机)
    
    Returns:
        dict: 包含测试结果的字典
    """
    num_players = game.num_players()
    use_trained_for_all = (mode == "self_play")
    
    # 初始化每个玩家的结果（包括作为训练玩家时的统计）
    results = {
        'num_players': num_players,
        'games_played': 0,
        'mode': mode,
    }
    for i in range(num_players):
        results[f'player{i}_returns'] = []
        results[f'player{i}_wins'] = 0
        # 当该玩家使用训练策略时的统计
        results[f'player{i}_trained_returns'] = []
        results[f'player{i}_trained_wins'] = 0
        results[f'player{i}_trained_count'] = 0
    
    failed_games = 0
    error_messages = {}  # 记录错误类型和次数
    for game_idx in range(num_games):
        game_result = play_test_game(
            game, 
            deep_cfr_solver, 
            verbose=verbose and failed_games < 3, 
            use_trained_for_all=use_trained_for_all
        )
        if game_result:
            results['games_played'] += 1
            returns = game_result.get('returns', [])
            trained_player = game_result.get('trained_player')
            
            # 记录每个玩家的收益
            for i in range(min(len(returns), num_players)):
                results[f'player{i}_returns'].append(returns[i])
            
            # 如果使用训练策略的玩家不是所有玩家，记录该玩家的表现
            if trained_player is not None:
                results[f'player{trained_player}_trained_returns'].append(returns[trained_player])
                results[f'player{trained_player}_trained_count'] += 1
            
            # 找出赢家（收益最高的玩家）
            if returns:
                max_return = max(returns)
                winners = [i for i, r in enumerate(returns) if r == max_return]
                if len(winners) == 1:
                    results[f'player{winners[0]}_wins'] += 1
                    # 如果赢家是使用训练策略的玩家，记录
                    if trained_player is not None and winners[0] == trained_player:
                        results[f'player{trained_player}_trained_wins'] += 1
        else:
            failed_games += 1
            # 记录错误（简化版，不打印详细堆栈）
            error_key = "未知错误"
            if failed_games <= 3 and verbose:
                print(f"  ⚠️ 对局 {game_idx + 1} 失败")
            if error_key not in error_messages:
                error_messages[error_key] = 0
            error_messages[error_key] += 1
    
    if failed_games > 0:
        results['failed_games'] = failed_games
        results['error_summary'] = error_messages
        if verbose:
            print(f"  ⚠️ 总共 {failed_games}/{num_games} 局失败")
    
    # 计算统计信息
    for i in range(num_players):
        player_returns = results[f'player{i}_returns']
        if player_returns:
            results[f'player{i}_avg_return'] = np.mean(player_returns)
            results[f'player{i}_std_return'] = np.std(player_returns)
            results[f'player{i}_win_rate'] = results[f'player{i}_wins'] / max(results['games_played'], 1)
        else:
            results[f'player{i}_avg_return'] = 0.0
            results[f'player{i}_win_rate'] = 0.0
        
        # 计算使用训练策略时的统计
        trained_returns = results[f'player{i}_trained_returns']
        trained_count = results[f'player{i}_trained_count']
        if trained_count > 0:
            results[f'player{i}_trained_avg_return'] = np.mean(trained_returns)
            results[f'player{i}_trained_win_rate'] = results[f'player{i}_trained_wins'] / trained_count
        else:
            results[f'player{i}_trained_avg_return'] = 0.0
            results[f'player{i}_trained_win_rate'] = 0.0
    
    # 兼容旧接口（使用训练策略时的平均表现）
    if not use_trained_for_all:
        # 计算所有位置使用训练策略时的平均表现
        all_trained_returns = []
        all_trained_wins = 0
        total_trained_games = 0
        for i in range(num_players):
            all_trained_returns.extend(results[f'player{i}_trained_returns'])
            all_trained_wins += results[f'player{i}_trained_wins']
            total_trained_games += results[f'player{i}_trained_count']
        
        if total_trained_games > 0:
            results['player0_avg_return'] = np.mean(all_trained_returns)  # 所有位置的平均
            results['player0_win_rate'] = all_trained_wins / total_trained_games
        else:
            results['player0_avg_return'] = 0.0
            results['player0_win_rate'] = 0.0
    else:
        # 自对弈模式，使用玩家0的统计
        results['player0_returns'] = results.get('player0_returns', [])
        results['player0_avg_return'] = results.get('player0_avg_return', 0.0)
        results['player0_win_rate'] = results.get('player0_win_rate', 0.0)
    
    return results


def _get_big_blind_from_game(game):
    """从游戏对象中获取大盲注（BB）值
    
    Args:
        game: OpenSpiel 游戏对象
        
    Returns:
        int: 大盲注值，如果无法获取则返回 None
    """
    try:
        # 尝试从游戏字符串中解析
        game_string = str(game)
        blind_match = re.search(r'blind=([\d\s]+)', game_string)
        if blind_match:
            blinds = [int(x) for x in blind_match.group(1).strip().split()]
            # 找到最大的非零盲注（即大盲）
            bb = max([b for b in blinds if b > 0], default=None)
            return bb
    except:
        pass
    return None


def _format_return_with_bb(return_value, bb=None):
    """格式化回报显示，同时显示筹码和BB单位
    
    Args:
        return_value: 回报值（筹码）
        bb: 大盲注值，如果为None则不显示BB
        
    Returns:
        str: 格式化后的字符串
    """
    if bb is not None and bb > 0:
        bb_value = return_value / bb
        return f"{return_value:.2f} ({bb_value:+.2f} BB)"
    else:
        return f"{return_value:.2f}"


def print_evaluation_summary(metrics, test_results=None, iteration=None, game=None):
    """打印评估摘要
    
    Args:
        metrics: 评估指标
        test_results: 测试对局结果
        iteration: 迭代次数
        game: 游戏对象（用于获取BB值）
    """
    print("\n" + "=" * 70)
    if iteration is not None:
        print(f"训练评估 (迭代 {iteration})")
    else:
        print("训练评估")
    print("=" * 70)
    
    # 获取大盲注值
    bb = None
    if game is not None:
        bb = _get_big_blind_from_game(game)
    
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
        mode = test_results.get('mode', 'unknown')
        num_players = test_results.get('num_players', 0)
        mode_str = "自对弈" if mode == "self_play" else "vs 随机策略（随机位置）"
        print(f"\n[测试对局] ({mode_str})")
        print(f"  对局数: {test_results.get('games_played', 0)}")
        if bb is not None:
            print(f"  大盲注: {bb} 筹码")
        
        if mode == "self_play":
            # 自对弈模式：显示所有玩家的表现
            print(f"  各位置表现:")
            for i in range(num_players):
                avg_return = test_results.get(f'player{i}_avg_return', 0)
                win_rate = test_results.get(f'player{i}_win_rate', 0) * 100
                std_return = test_results.get(f'player{i}_std_return', 0)
                return_str = _format_return_with_bb(avg_return, bb)
                std_str = _format_return_with_bb(std_return, bb) if std_return > 0 else f"{std_return:.2f}"
                print(f"    玩家{i}: 平均回报 {return_str} (±{std_str}), 胜率 {win_rate:.1f}%")
        else:
            # vs_random模式：显示所有位置使用训练策略时的表现
            print(f"  各位置使用训练策略时的表现:")
            total_trained_games = 0
            for i in range(num_players):
                trained_count = test_results.get(f'player{i}_trained_count', 0)
                if trained_count > 0:
                    avg_return = test_results.get(f'player{i}_trained_avg_return', 0)
                    win_rate = test_results.get(f'player{i}_trained_win_rate', 0) * 100
                    total_trained_games += trained_count
                    return_str = _format_return_with_bb(avg_return, bb)
                    print(f"    玩家{i}: {trained_count}局, 平均回报 {return_str}, 胜率 {win_rate:.1f}%")
            
            # 显示总体统计
            if total_trained_games > 0:
                overall_avg_return = test_results.get('player0_avg_return', 0)
                overall_win_rate = test_results.get('player0_win_rate', 0) * 100
                return_str = _format_return_with_bb(overall_avg_return, bb)
                print(f"  总体（所有位置）: 平均回报 {return_str}, 胜率 {overall_win_rate:.1f}%")
    
    print("=" * 70)


def quick_evaluate(
    game,
    deep_cfr_solver,
    include_test_games=False,
    num_test_games=50,
    max_depth=None,
    verbose=True
):
    """快速评估（轻量级）
    
    Args:
        game: OpenSpiel 游戏对象
        deep_cfr_solver: DeepCFRSolver 实例
        include_test_games: 是否包含测试对局
        num_test_games: 测试对局数量
        max_depth: 最大采样深度（None 时自动设置）
        verbose: 是否显示详细信息
    
    Returns:
        dict: 评估结果
    """
    # 策略质量评估
    metrics = evaluate_policy_quality(game, deep_cfr_solver, 
                                      num_sample_states=50,
                                      max_depth=max_depth,
                                      verbose=verbose)
    
    # 测试对局（可选）
    test_results = None
    if include_test_games:
        # 默认进行 vs Random 测试，因为更能体现进步
        # 自对弈在对称游戏中收益为0，且由于位置优势，Player 0 (SB) 可能会亏钱
        if verbose:
            print("  进行测试对局 (vs Random)...", end="", flush=True)
        test_results = evaluate_with_test_games(
            game, 
            deep_cfr_solver, 
            num_games=num_test_games, 
            verbose=verbose,
            mode="vs_random"  # 改为 vs Random
        )
        if verbose:
            if test_results and test_results.get('games_played', 0) < num_test_games:
                failed = num_test_games - test_results.get('games_played', 0)
                print(f" 完成 ({failed}/{num_test_games} 局失败)")
            else:
                print(" 完成")
    
    return {
        'metrics': metrics,
        'test_results': test_results
    }


