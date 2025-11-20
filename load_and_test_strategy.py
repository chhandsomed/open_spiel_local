#!/usr/bin/env python3
"""加载并测试保存的策略

示例用法:
    python load_and_test_strategy.py --strategy_path texas_holdem_strategy.pkl
"""

import argparse
import pickle
import random

import numpy as np
import pyspiel
# 导入 pokerkit_wrapper 以注册游戏
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability


def load_strategy(strategy_path):
    """加载保存的策略"""
    try:
        with open(strategy_path, "rb") as f:
            loaded_policy = pickle.load(f)
        print(f"成功加载策略: {strategy_path}")
        return loaded_policy
    except Exception as e:
        print(f"错误: 无法加载策略: {e}")
        return None


def test_strategy(game, loaded_policy, num_games=100):
    """测试策略性能"""
    print(f"\n测试策略 ({num_games} 局游戏)...")
    
    # 转换为 pyspiel 策略
    pyspiel_policy = policy.python_policy_to_pyspiel_policy(loaded_policy)
    
    # 计算 exploitability
    print("计算 NashConv...")
    conv = exploitability.nash_conv(game, loaded_policy)
    print(f"NashConv: {conv:.6f}")
    
    # 模拟游戏
    total_returns = [0.0] * game.num_players()
    
    for i in range(num_games):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                # 随机选择机会节点
                outcomes, probs = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes, p=probs)
            else:
                # 使用策略选择动作
                action_probs = pyspiel_policy.action_probabilities(state)
                if action_probs:
                    actions = list(action_probs.keys())
                    probs = list(action_probs.values())
                    action = np.random.choice(actions, p=probs)
                else:
                    # 如果没有策略，随机选择
                    legal_actions = state.legal_actions()
                    action = random.choice(legal_actions)
            
            state.apply_action(action)
        
        # 记录回报
        returns = state.returns()
        for player in range(game.num_players()):
            total_returns[player] += returns[player]
    
    # 打印平均回报
    print(f"\n平均回报 ({num_games} 局):")
    for player in range(game.num_players()):
        avg_return = total_returns[player] / num_games
        print(f"  玩家 {player}: {avg_return:.2f}")
    
    return conv, total_returns


def play_interactive(game, loaded_policy):
    """交互式游戏（用于测试）"""
    print("\n交互式游戏模式")
    print("输入 'q' 退出")
    
    pyspiel_policy = policy.python_policy_to_pyspiel_policy(loaded_policy)
    
    while True:
        state = game.new_initial_state()
        print("\n" + "=" * 50)
        print("新游戏开始")
        print("=" * 50)
        
        while not state.is_terminal():
            print(f"\n当前状态:")
            print(f"  当前玩家: {state.current_player()}")
            print(f"  合法动作: {state.legal_actions()}")
            
            if state.is_chance_node():
                print("  机会节点，自动处理...")
                outcomes, probs = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes, p=probs)
            else:
                # 获取策略推荐的动作
                action_probs = pyspiel_policy.action_probabilities(state)
                if action_probs:
                    best_action = max(action_probs, key=action_probs.get)
                    print(f"  策略推荐动作: {best_action} (概率: {action_probs[best_action]:.4f})")
                
                user_input = input("  输入动作 (或 'q' 退出): ").strip()
                if user_input.lower() == 'q':
                    return
                
                try:
                    action = int(user_input)
                    if action not in state.legal_actions():
                        print(f"  无效动作，使用策略推荐的动作: {best_action}")
                        action = best_action
                except ValueError:
                    print(f"  无效输入，使用策略推荐的动作: {best_action}")
                    action = best_action
            
            state.apply_action(action)
            print(f"  执行动作: {action}")
        
        # 游戏结束
        returns = state.returns()
        print(f"\n游戏结束!")
        print(f"  回报: {returns}")
        
        user_input = input("\n继续游戏? (y/n): ").strip().lower()
        if user_input != 'y':
            break


def main():
    parser = argparse.ArgumentParser(description="加载并测试保存的策略")
    parser.add_argument(
        "--strategy_path",
        type=str,
        default="texas_holdem_strategy.pkl",
        help="策略文件路径 (默认: texas_holdem_strategy.pkl)",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=6,
        help="玩家数量 (默认: 6)",
    )
    parser.add_argument(
        "--blinds",
        type=str,
        default="5 10",
        help="盲注大小 (默认: '5 10')",
    )
    parser.add_argument(
        "--stack_size",
        type=int,
        default=2000,
        help="初始筹码 (默认: 2000)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="测试游戏数量 (默认: 100)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式游戏模式",
    )
    
    args = parser.parse_args()
    
    # 加载策略
    loaded_policy = load_strategy(args.strategy_path)
    if loaded_policy is None:
        return
    
    # 加载相同的游戏配置
    stack_sizes = " ".join([str(args.stack_size)] * args.num_players)
    game = pyspiel.load_game(
        "python_pokerkit_wrapper",
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": args.num_players,
            "blinds": args.blinds,
            "stack_sizes": stack_sizes,
            "num_streets": 4,
        }
    )
    
    # 测试策略
    if args.interactive:
        play_interactive(game, loaded_policy)
    else:
        test_strategy(game, loaded_policy, args.num_games)


if __name__ == "__main__":
    main()

