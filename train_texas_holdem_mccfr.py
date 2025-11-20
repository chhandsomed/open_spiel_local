#!/usr/bin/env python3
"""使用 MCCFR 训练多人无限注德州扑克策略

示例用法:
    python train_texas_holdem_mccfr.py --num_players 6 --iterations 10000
"""

import argparse
import pickle
import sys

import pyspiel
# 导入 pokerkit_wrapper 以注册游戏
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python.algorithms import exploitability


def train_mccfr(
    num_players=6,
    iterations=10000,
    sampling="external",
    blinds="5 10",
    stack_size=2000,
    save_path="texas_holdem_strategy.pkl",
    print_freq=1000,
):
    """训练多人无限注德州扑克策略
    
    Args:
        num_players: 玩家数量 (2-10)
        iterations: 训练迭代次数
        sampling: 采样方式 ("external" 或 "outcome")
        blinds: 盲注大小，格式 "小盲 大盲"
        stack_size: 每个玩家的初始筹码
        save_path: 策略保存路径
        print_freq: 打印频率
    """
    # 配置游戏参数
    stack_sizes = " ".join([str(stack_size)] * num_players)
    
    print(f"配置游戏:")
    print(f"  玩家数: {num_players}")
    print(f"  盲注: {blinds}")
    print(f"  初始筹码: {stack_size} (每个玩家)")
    print(f"  迭代次数: {iterations}")
    print(f"  采样方式: {sampling}")
    
    # 加载游戏
    try:
        game = pyspiel.load_game(
            "python_pokerkit_wrapper",
            {
                "variant": "NoLimitTexasHoldem",
                "num_players": num_players,
                "blinds": blinds,
                "stack_sizes": stack_sizes,
                "num_streets": 4,
            }
        )
    except Exception as e:
        print(f"错误: 无法加载游戏: {e}")
        print("提示: 确保已安装 pokerkit: pip install pokerkit")
        sys.exit(1)
    
    print(f"\n游戏信息:")
    print(f"  游戏类型: {game.get_type().short_name}")
    print(f"  最大动作数: {game.num_distinct_actions()}")
    print(f"  信息集数量: {game.max_game_length()}")
    
    # 创建求解器
    if sampling == "external":
        solver = external_mccfr.ExternalSamplingSolver(
            game, external_mccfr.AverageType.SIMPLE
        )
    elif sampling == "outcome":
        solver = outcome_mccfr.OutcomeSamplingSolver(game)
    else:
        raise ValueError(f"未知的采样方式: {sampling}")
    
    # 训练循环
    print(f"\n开始训练...")
    best_conv = float("inf")
    
    for i in range(iterations):
        solver.iteration()
        
        if (i + 1) % print_freq == 0 or i == iterations - 1:
            policy = solver.average_policy()
            conv = exploitability.nash_conv(game, policy)
            best_conv = min(best_conv, conv)
            
            print(f"迭代 {i+1}/{iterations}: NashConv = {conv:.6f} (最佳: {best_conv:.6f})")
    
    # 保存策略
    final_policy = solver.average_policy()
    try:
        with open(save_path, "wb") as f:
            pickle.dump(final_policy, f)
        print(f"\n策略已保存到: {save_path}")
        print(f"最终 NashConv: {best_conv:.6f}")
    except Exception as e:
        print(f"警告: 无法保存策略: {e}")
    
    return final_policy, solver


def main():
    parser = argparse.ArgumentParser(
        description="使用 MCCFR 训练多人无限注德州扑克策略"
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=6,
        help="玩家数量 (默认: 6)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="训练迭代次数 (默认: 10000)",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="external",
        choices=["external", "outcome"],
        help="采样方式: external 或 outcome (默认: external)",
    )
    parser.add_argument(
        "--blinds",
        type=str,
        default="5 10",
        help="盲注大小，格式 '小盲 大盲' (默认: '5 10')",
    )
    parser.add_argument(
        "--stack_size",
        type=int,
        default=2000,
        help="每个玩家的初始筹码 (默认: 2000)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="texas_holdem_strategy.pkl",
        help="策略保存路径 (默认: texas_holdem_strategy.pkl)",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1000,
        help="打印频率 (默认: 1000)",
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.num_players < 2 or args.num_players > 10:
        print("错误: 玩家数量必须在 2-10 之间")
        sys.exit(1)
    
    if args.iterations < 1:
        print("错误: 迭代次数必须大于 0")
        sys.exit(1)
    
    # 训练
    train_mccfr(
        num_players=args.num_players,
        iterations=args.iterations,
        sampling=args.sampling,
        blinds=args.blinds,
        stack_size=args.stack_size,
        save_path=args.save_path,
        print_freq=args.print_freq,
    )


if __name__ == "__main__":
    main()

