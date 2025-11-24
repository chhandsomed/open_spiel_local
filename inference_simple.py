#!/usr/bin/env python3
"""简化的 DeepCFR 推理脚本

直接加载模型权重，避免创建完整求解器（避免 PyTorch 导入问题）
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
import numpy as np
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python.pytorch.deep_cfr import MLP


def load_policy_network(model_path, embedding_size, num_actions, layers=(64, 64), device='cpu'):
    """直接加载策略网络"""
    print(f"  加载策略网络: {model_path}")
    
    # 创建网络结构
    network = MLP(embedding_size, list(layers), num_actions)
    network = network.to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        print(f"  ✓ 策略网络加载成功")
        return network
    else:
        print(f"  ✗ 文件不存在: {model_path}")
        return None


def get_action_from_network(state, policy_network, device):
    """使用策略网络获取动作"""
    # 获取信息状态张量
    info_state = state.information_state_tensor()
    legal_actions = state.legal_actions()
    
    # 转换为张量
    info_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
    
    # 前向传播
    with torch.no_grad():
        logits = policy_network(info_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # 获取合法动作的概率
    action_probs = {a: float(probs[a]) for a in legal_actions}
    
    # 归一化
    total = sum(action_probs.values())
    if total > 1e-10:  # 避免除零
        action_probs = {a: p/total for a, p in action_probs.items()}
    else:
        # 如果所有概率都是0，使用均匀分布
        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
    
    # 采样动作
    actions = list(action_probs.keys())
    probabilities = np.array([action_probs[a] for a in actions])
    # 确保概率和为1（处理浮点误差）
    probabilities = probabilities / probabilities.sum()
    action = np.random.choice(actions, p=probabilities)
    
    return action, action_probs


def play_game_simple(game, policy_network, device, verbose=True):
    """使用模型玩一局游戏（简化版）"""
    state = game.new_initial_state()
    
    if verbose:
        print("\n  游戏开始...")
    
    while not state.is_terminal():
        if state.is_chance_node():
            # 机会节点：随机选择
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
            # 玩家节点：使用模型策略
            player = state.current_player()
            action, action_probs = get_action_from_network(state, policy_network, device)
            
            if verbose:
                print(f"    玩家 {player}: 动作 {action}")
            
            state = state.child(action)
    
    returns = state.returns()
    if verbose:
        print(f"  游戏结束，收益: {returns}")
    
    return returns


def test_model_simple(game, policy_network, device, num_games=10):
    """测试模型（简化版）"""
    print(f"\n[2/2] 测试模型 ({num_games} 局游戏)...")
    
    results = {
        'returns': [],
        'wins': [0] * game.num_players(),
        'total_returns': [0.0] * game.num_players(),
    }
    
    for i in range(num_games):
        if (i + 1) % 5 == 0:
            print(f"  进行第 {i + 1}/{num_games} 局...")
        
        returns = play_game_simple(game, policy_network, device, verbose=False)
        results['returns'].append(returns)
        
        for player in range(game.num_players()):
            results['total_returns'][player] += returns[player]
            if returns[player] > 0:
                results['wins'][player] += 1
    
    # 打印结果
    print(f"\n  测试结果:")
    for player in range(game.num_players()):
        avg_return = results['total_returns'][player] / num_games
        win_rate = results['wins'][player] / num_games * 100
        print(f"    玩家 {player}:")
        print(f"      平均收益: {avg_return:.4f}")
        print(f"      胜率: {win_rate:.1f}%")
        print(f"      总收益: {results['total_returns'][player]:.2f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简化的 DeepCFR 推理")
    parser.add_argument("--model_prefix", type=str, default="deepcfr_texas",
                       help="模型文件前缀")
    parser.add_argument("--num_games", type=int, default=10,
                       help="测试游戏数量")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[64, 64],
                       help="策略网络层大小")
    parser.add_argument("--num_players", type=int, default=2,
                       help="玩家数量（必须与训练时一致）")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepCFR 简化推理")
    print("=" * 70)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建游戏（必须与训练时完全一致）
    print(f"\n[0/2] 创建游戏 ({args.num_players}人场)...")
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
    print(f"  ✓ 游戏创建成功: {game.get_type().short_name}")
    
    # 获取游戏参数
    state = game.new_initial_state()
    embedding_size = len(state.information_state_tensor(0))
    num_actions = game.num_distinct_actions()
    
    print(f"  信息状态大小: {embedding_size}")
    print(f"  动作数量: {num_actions}")
    
    # 加载模型
    print(f"\n[1/2] 加载模型...")
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
    
    # 测试模型
    test_model_simple(game, policy_network, device, num_games=args.num_games)
    
    print("\n" + "=" * 70)
    print("✓ 推理完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

