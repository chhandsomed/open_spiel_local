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


def play_game(game, policy_network, device, opponent_strategy="random", verbose=False):
    """玩一局游戏"""
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
            player = state.current_player()
            if player == 0:  # 使用模型策略
                action, _ = get_action_from_network(state, policy_network, device)
            else:  # 对手策略
                action = get_opponent_action(state, opponent_strategy)
            state = state.child(action)
    
    returns = state.returns()
    return returns


def evaluate_against_opponent(game, policy_network, device, opponent_strategy="random", 
                             num_games=100, verbose=True):
    """评估模型对特定对手的表现"""
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
        
        returns = play_game(game, policy_network, device, opponent_strategy, verbose=False)
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
            verbose=True
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

