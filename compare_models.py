#!/usr/bin/env python3
"""模型对比评估脚本

支持：
- 对比两个或多个训练好的模型
- 模型之间的直接对局
- 与相同对手的对比评估
- 生成对比报告
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
    if not os.path.exists(model_path):
        return None
    
    network = MLP(embedding_size, list(layers), num_actions)
    network = network.to(device)
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)
    network.eval()
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
        action_probs = {a: p / total for a, p in action_probs.items()}
    else:
        # 所有概率接近 0 时，退化为均匀策略
        action_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    # 使用 argmax 选择动作（确定性策略）。
    # 为了在概率完全相同时有确定性行为，按动作编号最小的打破平局。
    best_action = max(sorted(action_probs.keys()), key=lambda a: action_probs[a])
    action = best_action
    
    return action


def play_game_models(game, model0, model1, device, verbose=False):
    """两个模型对局"""
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
            player = state.current_player()
            if player == 0:
                action = get_action_from_network(state, model0, device)
            else:
                action = get_action_from_network(state, model1, device)
            state = state.child(action)
    
    returns = state.returns()
    return returns


def compare_models_direct(game, model0, model1, device, num_games=100, verbose=True):
    """直接对比两个模型"""
    if verbose:
        print(f"\n直接对局 ({num_games} 局)...")
    
    results = {
        'model0_returns': [],
        'model1_returns': [],
        'model0_wins': 0,
        'model1_wins': 0,
        'draws': 0,
    }
    
    for i in range(num_games):
        if verbose and (i + 1) % 20 == 0:
            print(f"  进行第 {i + 1}/{num_games} 局...")
        
        returns = play_game_models(game, model0, model1, device, verbose=False)
        results['model0_returns'].append(returns[0])
        results['model1_returns'].append(returns[1])
        
        if returns[0] > returns[1]:
            results['model0_wins'] += 1
        elif returns[0] < returns[1]:
            results['model1_wins'] += 1
        else:
            results['draws'] += 1
    
    # 计算统计信息
    m0_returns = np.array(results['model0_returns'])
    m1_returns = np.array(results['model1_returns'])
    
    results['model0_avg'] = float(np.mean(m0_returns))
    results['model0_std'] = float(np.std(m0_returns))
    results['model1_avg'] = float(np.mean(m1_returns))
    results['model1_std'] = float(np.std(m1_returns))
    results['model0_win_rate'] = results['model0_wins'] / num_games
    results['model1_win_rate'] = results['model1_wins'] / num_games
    results['draw_rate'] = results['draws'] / num_games
    
    return results


def evaluate_against_opponent(game, model, device, opponent_strategy="random", 
                             num_games=100):
    """评估模型对特定对手的表现"""
    results = {'returns': [], 'wins': 0, 'losses': 0, 'draws': 0}
    
    for _ in range(num_games):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = np.random.choice([a for a, _ in outcomes], 
                                         p=[p for _, p in outcomes])
                state = state.child(action)
            else:
                player = state.current_player()
                legal_actions = state.legal_actions()
                
                if player == 0:
                    action = get_action_from_network(state, model, device)
                else:
                    if opponent_strategy == "random":
                        action = np.random.choice(legal_actions)
                    elif opponent_strategy == "call" and 1 in legal_actions:
                        action = 1
                    elif opponent_strategy == "fold" and 0 in legal_actions:
                        action = 0
                    else:
                        action = np.random.choice(legal_actions)
                
                state = state.child(action)
        
        returns = state.returns()
        results['returns'].append(returns[0])
        
        if returns[0] > returns[1]:
            results['wins'] += 1
        elif returns[0] < returns[1]:
            results['losses'] += 1
        else:
            results['draws'] += 1
    
    returns_array = np.array(results['returns'])
    results['avg_return'] = float(np.mean(returns_array))
    results['win_rate'] = results['wins'] / num_games
    
    return results


def compare_against_opponents(game, models, model_names, device, opponents, 
                              num_games=100, verbose=True):
    """对比多个模型对相同对手的表现"""
    if verbose:
        print(f"\n对比评估（对 {len(opponents)} 种对手，每种 {num_games} 局）...")
    
    comparison_results = {}
    
    for opponent in opponents:
        if verbose:
            print(f"\n  对手: {opponent}")
        opponent_results = {}
        
        for model, name in zip(models, model_names):
            if verbose:
                print(f"    评估 {name}...", end="", flush=True)
            results = evaluate_against_opponent(
                game, model, device, opponent, num_games, verbose=False
            )
            opponent_results[name] = results
            if verbose:
                print(f" 完成 (平均收益: {results['avg_return']:.4f}, 胜率: {results['win_rate']*100:.1f}%)")
        
        comparison_results[opponent] = opponent_results
    
    return comparison_results


def print_comparison_results(direct_results, opponent_results, model_names):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("模型对比结果")
    print("=" * 70)
    
    # 直接对局结果
    if direct_results:
        print(f"\n[直接对局]")
        print(f"  {model_names[0]} vs {model_names[1]}")
        print(f"    {model_names[0]}: 平均收益 {direct_results['model0_avg']:.4f} (±{direct_results['model0_std']:.4f}), "
              f"胜率 {direct_results['model0_win_rate']*100:.2f}%")
        print(f"    {model_names[1]}: 平均收益 {direct_results['model1_avg']:.4f} (±{direct_results['model1_std']:.4f}), "
              f"胜率 {direct_results['model1_win_rate']*100:.2f}%")
        print(f"    平局率: {direct_results['draw_rate']*100:.2f}%")
    
    # 对相同对手的对比
    if opponent_results:
        print(f"\n[对相同对手的对比]")
        for opponent, results in opponent_results.items():
            print(f"\n  对手: {opponent}")
            sorted_models = sorted(results.items(), key=lambda x: x[1]['avg_return'], reverse=True)
            for rank, (name, res) in enumerate(sorted_models, 1):
                print(f"    {rank}. {name}: 平均收益 {res['avg_return']:.4f}, "
                      f"胜率 {res['win_rate']*100:.2f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="模型对比评估脚本")
    parser.add_argument("--model1_prefix", type=str, required=True,
                       help="第一个模型文件前缀")
    parser.add_argument("--model2_prefix", type=str, required=True,
                       help="第二个模型文件前缀")
    parser.add_argument("--model1_name", type=str, default=None,
                       help="第一个模型名称（用于显示）")
    parser.add_argument("--model2_name", type=str, default=None,
                       help="第二个模型名称（用于显示）")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[64, 64],
                       help="策略网络层大小（两个模型必须相同）")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    parser.add_argument("--num_games", type=int, default=100,
                       help="对局数量")
    parser.add_argument("--opponents", type=str, nargs="+", 
                       default=["random"],
                       help="对手策略列表（用于对比评估）")
    parser.add_argument("--num_players", type=int, default=2,
                       help="玩家数量（必须与训练时一致）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    model_names = [
        args.model1_name or args.model1_prefix,
        args.model2_name or args.model2_prefix
    ]
    
    print("=" * 70)
    print("模型对比评估")
    print("=" * 70)
    print(f"\n模型1: {model_names[0]} ({args.model1_prefix})")
    print(f"模型2: {model_names[1]} ({args.model2_prefix})")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建游戏
    print(f"\n[1/4] 创建游戏 ({args.num_players}人场)...")
    num_players = args.num_players
    # 配置 blinds 和 firstPlayer
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
        f"numRanks=13"
        f")"
    )
    game = pyspiel.load_game(game_string)
    print(f"  ✓ 游戏创建成功")
    
    # 获取游戏参数
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    # 加载模型
    print(f"\n[2/4] 加载模型...")
    model1_path = f"{args.model1_prefix}_policy_network.pt"
    model2_path = f"{args.model2_prefix}_policy_network.pt"
    
    model1 = load_policy_network(model1_path, embedding_size, num_actions,
                               tuple(args.policy_layers), device)
    model2 = load_policy_network(model2_path, embedding_size, num_actions,
                               tuple(args.policy_layers), device)
    
    if model1 is None or model2 is None:
        print("\n✗ 模型加载失败，退出")
        if model1 is None:
            print(f"  模型1不存在: {model1_path}")
        if model2 is None:
            print(f"  模型2不存在: {model2_path}")
        return
    
    print(f"  ✓ 两个模型加载成功")
    
    # 直接对局
    print(f"\n[3/4] 直接对局...")
    direct_results = compare_models_direct(
        game, model1, model2, device, args.num_games, verbose=True
    )
    
    # 对相同对手的对比
    print(f"\n[4/4] 对相同对手的对比评估...")
    opponent_results = compare_against_opponents(
        game, [model1, model2], model_names, device, 
        args.opponents, args.num_games, verbose=True
    )
    
    # 打印结果
    print_comparison_results(direct_results, opponent_results, model_names)
    
    # 保存结果
    if args.output:
        output_data = {
            'model1': model_names[0],
            'model2': model_names[1],
            'direct_comparison': direct_results,
            'opponent_comparison': opponent_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n对比结果已保存到: {args.output}")
    
    print("\n" + "=" * 70)
    print("✓ 对比评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

