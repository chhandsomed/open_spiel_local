#!/usr/bin/env python3
"""完整的模型评估脚本

支持：
- 加载已训练的模型进行全面评估
- 与多种对手策略对局（随机、均匀、固定策略等）
- 详细的统计信息（均值、标准差、置信区间等）
- 保存评估结果到文件
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import argparse
import json
import torch
import numpy as np
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python.pytorch.deep_cfr import MLP
from collections import defaultdict


def load_policy_network(model_path, embedding_size, num_actions, layers=(64, 64), device='cpu'):
    """加载策略网络"""
    print(f"加载策略网络: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ✗ 文件不存在: {model_path}")
        return None
    
    network = MLP(embedding_size, list(layers), num_actions)
    network = network.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)
    network.eval()
    print(f"  ✓ 策略网络加载成功")
    return network


def get_action_from_network(state, policy_network, device):
    """使用策略网络获取动作"""
    info_state = state.information_state_tensor()
    legal_actions = state.legal_actions()
    
    info_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
    
    with torch.no_grad():
        logits = policy_network(info_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    action_probs = {a: float(probs[a]) for a in legal_actions}
    total = sum(action_probs.values())
    if total > 1e-10:
        action_probs = {a: p/total for a, p in action_probs.items()}
    else:
        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
    
    actions = list(action_probs.keys())
    probabilities = np.array([action_probs[a] for a in actions])
    probabilities = probabilities / probabilities.sum()
    action = np.random.choice(actions, p=probabilities)
    
    return action, action_probs


def get_opponent_action(state, opponent_strategy="random"):
    """获取对手动作"""
    legal_actions = state.legal_actions()
    
    if opponent_strategy == "random":
        return np.random.choice(legal_actions)
    elif opponent_strategy == "uniform":
        # 均匀分布（实际上和random一样）
        return np.random.choice(legal_actions)
    elif opponent_strategy == "call":
        # 总是跟注（如果可能）
        if 1 in legal_actions:  # 假设1是跟注
            return 1
        return legal_actions[0]
    elif opponent_strategy == "fold":
        # 总是弃牌（如果可能）
        if 0 in legal_actions:  # 假设0是弃牌
            return 0
        return legal_actions[0]
    else:
        return np.random.choice(legal_actions)


def action_to_string(action):
    """将动作编号转换为字符串"""
    action_map = {
        0: "Fold",
        1: "Call/Check",
        2: "Bet/Raise",
        3: "All-in",
        4: "Half-pot"
    }
    return action_map.get(action, f"Action_{action}")


def format_game_state(state, show_all_hands=False):
    """格式化显示游戏状态信息"""
    try:
        state_struct = state.to_struct()
        
        # 获取基本信息
        num_players = state.num_players()
        current_player = state.current_player() if not state.is_terminal() else -1
        
        # 获取手牌和公共牌
        player_hands = getattr(state_struct, 'player_hands', [])
        board_cards = getattr(state_struct, 'board_cards', '')
        pot_size = getattr(state_struct, 'pot_size', 0)
        player_contributions = getattr(state_struct, 'player_contributions', [])
        betting_history = getattr(state_struct, 'betting_history', '')
        
        lines = []
        lines.append("  " + "=" * 60)
        
        # 显示公共牌
        if board_cards:
            lines.append(f"  公共牌: {board_cards}")
        else:
            lines.append(f"  公共牌: (未发牌)")
        
        # 显示底池
        lines.append(f"  底池: {pot_size}")
        
        # 显示每个玩家的信息
        for p in range(num_players):
            hand_str = player_hands[p] if p < len(player_hands) else "??"
            contribution = player_contributions[p] if p < len(player_contributions) else 0
            
            if show_all_hands or state.is_terminal():
                # 显示所有玩家的手牌（游戏结束时）
                lines.append(f"  玩家 {p}: 手牌={hand_str}, 投入={contribution}")
            else:
                # 游戏进行中，只显示当前玩家或已弃牌玩家的手牌
                if p == current_player:
                    lines.append(f"  玩家 {p} [当前行动]: 手牌={hand_str}, 投入={contribution}")
                else:
                    lines.append(f"  玩家 {p}: 投入={contribution}")
        
        # 显示下注历史
        if betting_history:
            lines.append(f"  下注历史: {betting_history}")
        
        lines.append("  " + "=" * 60)
        
        return "\n".join(lines)
    except Exception as e:
        # 如果获取状态信息失败，返回简化信息
        return f"  状态信息获取失败: {e}"


def play_game(game, policy_network, device, opponent_strategy="random", verbose=False, show_full_game=False):
    """玩一局游戏
    
    Args:
        game: 游戏对象
        policy_network: 策略网络
        device: 设备
        opponent_strategy: 对手策略
        verbose: 是否显示基本信息
        show_full_game: 是否显示完整的对局流程（包括手牌、公共牌等）
    """
    state = game.new_initial_state()
    action_history = []
    last_board_cards_count = 0
    last_round_displayed = -1
    
    if show_full_game:
        print("\n" + "=" * 70)
        print("开始新对局")
        print("=" * 70)
        # 显示初始状态（Preflop）
        print("\n--- Preflop ---")
        print(format_game_state(state, show_all_hands=False))
    
    while not state.is_terminal():
        if state.is_chance_node():
            # 机会节点（发牌）
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
            # 玩家行动节点：检测是否进入新的一轮
            if show_full_game:
                try:
                    state_struct = state.to_struct()
                    board_cards_str = getattr(state_struct, 'board_cards', '')
                    current_board_cards_count = len(board_cards_str) // 2  # 每张牌2个字符
                    
                    # 根据已发公共牌数判断当前轮次
                    # numBoardCards=0 3 1 1 表示：
                    # Round 0 (Preflop): 0张
                    # Round 1 (Flop): 3张（累计）
                    # Round 2 (Turn): 4张（累计，3+1）
                    # Round 3 (River): 5张（累计，3+1+1）
                    if current_board_cards_count != last_board_cards_count:
                        # 公共牌数量变化，可能是新轮次开始
                        if current_board_cards_count == 0:
                            round_name = "Preflop"
                            round_num = 0
                        elif current_board_cards_count <= 3:
                            round_name = "Flop"
                            round_num = 1
                        elif current_board_cards_count == 4:
                            round_name = "Turn"
                            round_num = 2
                        elif current_board_cards_count == 5:
                            round_name = "River"
                            round_num = 3
                        else:
                            round_name = f"Round {current_board_cards_count}"
                            round_num = current_board_cards_count
                        
                        # 只在轮次变化时显示（避免重复显示）
                        if round_num != last_round_displayed:
                            print(f"\n--- {round_name} ---")
                            print(format_game_state(state, show_all_hands=False))
                            last_round_displayed = round_num
                        
                        last_board_cards_count = current_board_cards_count
                except:
                    pass
            
            player = state.current_player()
            legal_actions = state.legal_actions()
            
            if player == 0:  # 使用模型策略
                action, action_probs = get_action_from_network(state, policy_network, device)
            else:  # 对手策略
                action = get_opponent_action(state, opponent_strategy)
                action_probs = None
            
            action_str = action_to_string(action)
            
            if show_full_game:
                print(f"\n  玩家 {player} 行动: {action_str} (动作编号: {action})")
                if action_probs and player == 0:
                    # 显示模型的动作概率分布
                    prob_str = ", ".join([f"{action_to_string(a)}: {p:.3f}" 
                                         for a, p in sorted(action_probs.items(), 
                                                           key=lambda x: x[1], reverse=True)[:3]])
                    print(f"    动作概率分布 (前3): {prob_str}")
            
            action_history.append((player, action, action_str))
            state = state.child(action)
    
    # 游戏结束，显示最终结果
    returns = state.returns()
    
    if show_full_game:
        print("\n" + "=" * 70)
        print("对局结束 - 最终状态")
        print("=" * 70)
        print(format_game_state(state, show_all_hands=True))
        
        # 显示最终手牌和牌型（如果可用）
        try:
            state_struct = state.to_struct()
            best_hand_rank_types = getattr(state_struct, 'best_hand_rank_types', [])
            best_five_card_hands = getattr(state_struct, 'best_five_card_hands', [])
            
            print("\n最终牌型:")
            for p in range(game.num_players()):
                rank_type = best_hand_rank_types[p] if p < len(best_hand_rank_types) else "Unknown"
                best_hand = best_five_card_hands[p] if p < len(best_five_card_hands) else "Unknown"
                print(f"  玩家 {p}: {rank_type} ({best_hand})")
        except:
            pass
        
        print("\n最终收益:")
        for p in range(game.num_players()):
            print(f"  玩家 {p}: {returns[p]:.2f}")
        
        print("\n动作历史:")
        for i, (player, action, action_str) in enumerate(action_history, 1):
            print(f"  {i}. 玩家 {player}: {action_str}")
        
        print("=" * 70)
    elif verbose:
        print(f"  对局结束，收益: {returns}")
    
    return returns


def evaluate_against_opponent(game, policy_network, device, opponent_strategy="random", 
                             num_games=100, verbose=True, show_full_game=False, show_first_n_games=0):
    """评估模型对特定对手的表现
    
    Args:
        game: 游戏对象
        policy_network: 策略网络
        device: 设备
        opponent_strategy: 对手策略
        num_games: 对局数量
        verbose: 是否显示基本信息
        show_full_game: 是否显示完整的对局流程
        show_first_n_games: 显示前N局完整流程（0表示不显示，-1表示显示所有）
    """
    if verbose:
        print(f"\n评估对局 ({opponent_strategy} 对手, {num_games} 局)...")
    
    results = {
        'returns': [],
        'wins': 0,
        'losses': 0,
        'draws': 0,
    }
    
    for i in range(num_games):
        if verbose and (i + 1) % 20 == 0:
            print(f"  进行第 {i + 1}/{num_games} 局...")
        
        # 决定是否显示完整对局流程
        should_show = show_full_game or (show_first_n_games > 0 and i < show_first_n_games)
        
        returns = play_game(game, policy_network, device, opponent_strategy, 
                          verbose=False, show_full_game=should_show)
        results['returns'].append(returns[0])  # 玩家0的收益
        
        if returns[0] > returns[1]:
            results['wins'] += 1
        elif returns[0] < returns[1]:
            results['losses'] += 1
        else:
            results['draws'] += 1
    
    # 计算统计信息
    returns_array = np.array(results['returns'])
    results['avg_return'] = float(np.mean(returns_array))
    results['std_return'] = float(np.std(returns_array))
    results['min_return'] = float(np.min(returns_array))
    results['max_return'] = float(np.max(returns_array))
    results['win_rate'] = results['wins'] / num_games
    results['loss_rate'] = results['losses'] / num_games
    results['draw_rate'] = results['draws'] / num_games
    
    # 计算置信区间（95%）
    if len(returns_array) > 1:
        try:
            from scipy import stats
            confidence_interval = stats.t.interval(0.95, len(returns_array)-1, 
                                                   loc=results['avg_return'],
                                                   scale=stats.sem(returns_array))
            results['confidence_interval_95'] = [float(confidence_interval[0]), 
                                               float(confidence_interval[1])]
        except ImportError:
            # 如果没有 scipy，使用简化的置信区间估算
            import math
            std_error = results['std_return'] / math.sqrt(len(returns_array))
            t_value = 1.96  # 95% 置信区间的近似值
            margin = t_value * std_error
            results['confidence_interval_95'] = [
                float(results['avg_return'] - margin),
                float(results['avg_return'] + margin)
            ]
    else:
        results['confidence_interval_95'] = [results['avg_return'], results['avg_return']]
    
    return results


def print_evaluation_results(all_results, model_name):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print(f"模型评估结果: {model_name}")
    print("=" * 70)
    
    for opponent, results in all_results.items():
        print(f"\n[对手策略: {opponent}]")
        print(f"  平均收益: {results['avg_return']:.4f} (±{results['std_return']:.4f})")
        print(f"  收益范围: [{results['min_return']:.4f}, {results['max_return']:.4f}]")
        if 'confidence_interval_95' in results:
            ci = results['confidence_interval_95']
            print(f"  95%置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  胜率: {results['win_rate']*100:.2f}%")
        print(f"  负率: {results['loss_rate']*100:.2f}%")
        print(f"  平局率: {results['draw_rate']*100:.2f}%")
    
    print("=" * 70)


def save_results(all_results, model_name, output_path):
    """保存评估结果到JSON文件"""
    output_data = {
        'model_name': model_name,
        'evaluation_results': all_results,
        'summary': {}
    }
    
    # 计算总体摘要
    for opponent, results in all_results.items():
        output_data['summary'][opponent] = {
            'avg_return': results['avg_return'],
            'win_rate': results['win_rate'],
            'num_games': len(results['returns'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n评估结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="完整的模型评估脚本")
    parser.add_argument("--model_prefix", type=str, default="deepcfr_texas",
                       help="模型文件前缀")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[64, 64],
                       help="策略网络层大小")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    parser.add_argument("--num_games", type=int, default=100,
                       help="每个对手的对局数量")
    parser.add_argument("--opponents", type=str, nargs="+", 
                       default=["random", "call", "fold"],
                       help="对手策略列表: random, call, fold")
    parser.add_argument("--num_players", type=int, default=2,
                       help="玩家数量（必须与训练时一致）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON文件路径（可选）")
    parser.add_argument("--show_full_game", action="store_true",
                       help="显示完整的对局流程（包括手牌、公共牌、动作等）")
    parser.add_argument("--show_first_n_games", type=int, default=0,
                       help="显示前N局完整流程（默认0，不显示）")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("完整模型评估")
    print("=" * 70)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建游戏（必须与训练时完全一致）
    print(f"\n[1/3] 创建游戏 ({args.num_players}人场)...")
    num_players = args.num_players
    blinds_str = " ".join(["100"] * num_players)
    stacks_str = " ".join(["2000"] * num_players)
    
    # 配置 firstPlayer: Preflop 是大盲后第一个玩家，后续轮次是小盲
    # 注意: universal_poker 使用 1-indexed，所以 player 0 = 1, player 2 = 3
    if num_players == 2:
        # 2人场: Preflop 是 button (player 1), 后续是小盲 (player 0)
        first_player_str = " ".join(["2"] + ["1"] * 3)  # round 0: player 1 (button), rounds 1-3: player 0 (SB)
    else:
        # 多人场: Preflop 是 UTG (player 2), 后续是小盲 (player 0)
        # 注意: 1-indexed，所以 player 2 = 3, player 0 = 1
        first_player_str = " ".join(["3"] + ["1"] * 3)  # round 0: player 2 (UTG), rounds 1-3: player 0 (SB)
    
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
        f"numRanks=13"
        f")"
    )
    game = pyspiel.load_game(game_string)
    print(f"  ✓ 游戏创建成功: {game.get_type().short_name}")
    
    # 获取游戏参数
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    print(f"  信息状态大小: {embedding_size}")
    print(f"  动作数量: {num_actions}")
    
    # 加载模型
    print(f"\n[2/3] 加载模型...")
    policy_path = f"{args.model_prefix}_policy_network.pt"
    policy_network = load_policy_network(
        policy_path,
        embedding_size,
        num_actions,
        layers=tuple(args.policy_layers),
        device=device
    )
    
    if policy_network is None:
        print("\n✗ 模型加载失败，退出")
        return
    
    # 评估
    print(f"\n[3/3] 评估模型...")
    all_results = {}
    
    for opponent in args.opponents:
        results = evaluate_against_opponent(
            game, policy_network, device, 
            opponent_strategy=opponent,
            num_games=args.num_games,
            verbose=True,
            show_full_game=args.show_full_game,
            show_first_n_games=args.show_first_n_games
        )
        all_results[opponent] = results
    
    # 打印结果
    print_evaluation_results(all_results, args.model_prefix)
    
    # 保存结果
    if args.output:
        save_results(all_results, args.model_prefix, args.output)
    else:
        # 默认保存路径
        default_output = f"{args.model_prefix}_evaluation.json"
        save_results(all_results, args.model_prefix, default_output)
    
    print("\n" + "=" * 70)
    print("✓ 评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

