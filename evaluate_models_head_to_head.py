#!/usr/bin/env python3
"""模型对战评估脚本 (Head-to-Head Evaluation)

让两个不同的 DeepCFR 模型在 6 人德州扑克中进行对战。
支持交替座位或指定位置。
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import pyspiel
from open_spiel.python.pytorch.deep_cfr import MLP

# 尝试导入自定义特征类
try:
    from deep_cfr_simple_feature import DeepCFRSimpleFeature
    from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
    HAVE_CUSTOM_FEATURES = True
except ImportError:
    HAVE_CUSTOM_FEATURES = False
    print("注意: 未找到自定义特征模块，仅支持标准 MLP 模型")


def load_config(model_dir):
    """加载配置文件（支持从父目录查找）"""
    # 1. 先尝试从当前目录加载
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️ 无法读取配置文件 {config_path}: {e}")
    
    # 2. 如果是 checkpoint 子目录，尝试从父目录加载
    if "checkpoints" in model_dir:
        parent_dir = os.path.dirname(model_dir)
        # 如果父目录还是 checkpoints，再往上一级
        if "checkpoints" in parent_dir:
            main_dir = os.path.dirname(parent_dir)
        else:
            main_dir = parent_dir
        
        config_path = os.path.join(main_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  ⚠️ 无法读取配置文件 {config_path}: {e}")
    
    return None


def load_model_network(model_dir, game, device):
    """加载单个模型的策略网络（支持 checkpoint 格式）"""
    print(f"  加载模型: {model_dir}")
    
    # 尝试从当前目录加载 config
    config = load_config(model_dir)
    
    # 如果当前目录没有 config，尝试从父目录加载（checkpoint 子目录的情况）
    if not config:
        parent_dir = os.path.dirname(model_dir)
        if "checkpoints" in model_dir:
            # 尝试从主模型目录加载
            main_dir = os.path.dirname(parent_dir) if "checkpoints" in parent_dir else parent_dir
            config = load_config(main_dir)
            if config:
                print(f"    ✓ 从主目录加载配置文件: {os.path.join(main_dir, 'config.json')}")
    
    # 确定前缀
    save_prefix = "deepcfr_texas"
    if config and 'save_prefix' in config:
        save_prefix = config['save_prefix']
    
    # 寻找策略网络文件
    import glob
    import re
    
    policy_path = None
    
    # 1. 尝试最终模型格式: prefix_policy_network.pt
    policy_path = os.path.join(model_dir, f"{save_prefix}_policy_network.pt")
    if not os.path.exists(policy_path):
        # 2. 尝试 checkpoint 格式: prefix_policy_network_iterN.pt
        pt_files = glob.glob(os.path.join(model_dir, "*_policy_network*.pt"))
        if pt_files:
            # 如果是 checkpoint 格式，选择最新的
            checkpoint_files = [f for f in pt_files if "_iter" in os.path.basename(f)]
            if checkpoint_files:
                # 提取迭代号，选择最大的
                max_iter = 0
                latest_file = None
                for f in checkpoint_files:
                    match = re.search(r'_iter(\d+)\.pt$', f)
                    if match:
                        iter_num = int(match.group(1))
                        if iter_num > max_iter:
                            max_iter = iter_num
                            latest_file = f
                if latest_file:
                    policy_path = latest_file
                    print(f"    ✓ 找到 checkpoint: 迭代 {max_iter}")
            else:
                # 如果找到文件但不是 checkpoint 格式，使用第一个
                policy_path = pt_files[0]
                print(f"    ✓ 找到模型文件: {os.path.basename(policy_path)}")
        else:
            # 3. 尝试旧命名
            fallback_path = os.path.join(model_dir, "deepcfr_texas_policy_network.pt")
            if os.path.exists(fallback_path):
                policy_path = fallback_path
            else:
                print(f"  ✗ 找不到策略网络文件")
                return None, None
    
    if not policy_path or not os.path.exists(policy_path):
        print(f"  ✗ 策略网络文件不存在: {policy_path}")
        return None, None

    # 确定模型结构参数
    policy_layers = [64, 64]
    use_simple_feature = False
    use_feature_transform = False
    transformed_size = 150
    use_hybrid_transform = True
    betting_abstraction = 'fcpa'

    if config:
        policy_layers = config.get('policy_layers', [64, 64])
        use_simple_feature = config.get('use_simple_feature', False)
        use_feature_transform = config.get('use_feature_transform', False)
        transformed_size = config.get('transformed_size', 150)
        use_hybrid_transform = config.get('use_hybrid_transform', True)
        betting_abstraction = config.get('betting_abstraction', 'fcpa')
    
    # 创建网络实例
    network = None
    if use_simple_feature and HAVE_CUSTOM_FEATURES:
        print("    类型: Simple Feature")
        solver = DeepCFRSimpleFeature(
            game,
            policy_network_layers=tuple(policy_layers),
            advantage_network_layers=(32, 32),
            device=device
        )
        network = solver._policy_network
    elif use_feature_transform and HAVE_CUSTOM_FEATURES:
        print("    类型: Feature Transform")
        solver = DeepCFRWithFeatureTransform(
            game,
            policy_network_layers=tuple(policy_layers),
            advantage_network_layers=(32, 32),
            transformed_size=transformed_size,
            use_hybrid_transform=use_hybrid_transform,
            device=device
        )
        network = solver._policy_network
    else:
        print("    类型: Standard MLP")
        state = game.new_initial_state()
        embedding_size = len(state.information_state_tensor(0))
        num_actions = game.num_distinct_actions()
        network = MLP(embedding_size, list(policy_layers), num_actions)
        network = network.to(device)

    # 加载权重
    try:
        network.load_state_dict(torch.load(policy_path, map_location=device))
        network.eval()
        print(f"    ✓ 权重加载成功")
        return network, betting_abstraction
    except Exception as e:
        print(f"    ✗ 权重加载失败: {e}")
        return None, None


def get_action(state, network, device):
    """从网络获取动作"""
    info_state = state.information_state_tensor()
    legal_actions = state.legal_actions()
    
    info_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
    
    with torch.no_grad():
        logits = network(info_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # 过滤非法动作并归一化
    action_probs = {a: float(probs[a]) for a in legal_actions}
    total = sum(action_probs.values())
    if total > 1e-10:
        action_probs = {a: p/total for a, p in action_probs.items()}
    else:
        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
    
    actions = list(action_probs.keys())
    probabilities = np.array([action_probs[a] for a in actions])
    probabilities = probabilities / probabilities.sum()
    
    return np.random.choice(actions, p=probabilities)


def play_match(game, model_a, model_b, device, seat_assignment, num_games=100):
    """进行一组对战
    
    Args:
        seat_assignment: 列表，长度为 num_players，值为 'A' 或 'B'
    """
    print(f"\n开始对战: {num_games} 局")
    print(f"座位安排: {seat_assignment}")
    
    stats = {
        'A': {'return': 0.0, 'wins': 0},
        'B': {'return': 0.0, 'wins': 0}
    }
    
    # 记录每个座位的收益，用于分析位置优势
    seat_stats = [{'A_return': 0, 'B_return': 0, 'count': 0} for _ in range(game.num_players())]
    
    for i in range(num_games):
        if (i+1) % 50 == 0:
            print(f"  进度: {i+1}/{num_games}")
            
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = np.random.choice([a for a, _ in outcomes], 
                                         p=[p for _, p in outcomes])
                state = state.child(action)
            else:
                player = state.current_player()
                model_type = seat_assignment[player]
                network = model_a if model_type == 'A' else model_b
                
                action = get_action(state, network, device)
                state = state.child(action)
        
        returns = state.returns()
        
        # 统计结果
        for p, ret in enumerate(returns):
            model_type = seat_assignment[p]
            stats[model_type]['return'] += ret
            if ret > 0:
                stats[model_type]['wins'] += 1
            
            # 记录座位统计
            seat_stats[p]['count'] += 1
            if model_type == 'A':
                seat_stats[p]['A_return'] += ret
            else:
                seat_stats[p]['B_return'] += ret

    return stats, seat_stats


def main():
    parser = argparse.ArgumentParser(description="模型对战评估")
    parser.add_argument("--model_a", type=str, required=True, help="模型 A 的目录")
    parser.add_argument("--model_b", type=str, required=True, help="模型 B 的目录")
    parser.add_argument("--num_games", type=int, default=1000, help="总对局数")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="使用 GPU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载配置以创建游戏
    # 我们假设两个模型的游戏配置必须一致，以 Model A 为准
    config_a = load_config(args.model_a)
    config_b = load_config(args.model_b)
    
    if not config_a:
        print("错误: 无法加载模型 A 的配置")
        return

    # 检查 betting_abstraction 兼容性
    ba_a = config_a.get('betting_abstraction', 'fcpa')
    ba_b = config_b.get('betting_abstraction', 'fcpa') if config_b else 'fcpa'
    
    if ba_a != ba_b:
        print(f"⚠️ 警告: 模型下注抽象不一致! A={ba_a}, B={ba_b}")
        print("这可能导致非法动作或维度错误。建议仅对比相同配置的模型。")
        print("按 Enter 继续，或 Ctrl+C 退出...")
        input()

    # 创建游戏
    print("\n[1/3] 创建游戏环境...")
    # 优先使用 game_string
    game = None
    game_string = config_a.get('game_string')
    if game_string:
        try:
            game = pyspiel.load_game(game_string)
            print(f"  使用 game_string 创建成功: {game.get_type().short_name}")
        except Exception:
            game = None
            
    if game is None:
        print("  使用手动配置创建...")
        # 回退到手动创建
        num_players = config_a.get('num_players', 6)
        blinds_str = "50 100 0 0 0 0" if num_players == 6 else "100 50"
        first_player_str = "3 1 1 1" if num_players == 6 else "2 1 1 1"
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
            f"bettingAbstraction={ba_a}"
            f")"
        )
        game = pyspiel.load_game(game_string)

    # 2. 加载模型
    print("\n[2/3] 加载模型网络...")
    network_a, ba_check_a = load_model_network(args.model_a, game, device)
    network_b, ba_check_b = load_model_network(args.model_b, game, device)

    if network_a is None or network_b is None:
        print("错误: 模型加载失败")
        return

    # 3. 进行对战
    print("\n[3/3] 开始评估...")
    
    # 模式 1: 交替座位 (A B A B A B)
    seats_alt = ['A', 'B'] * (game.num_players() // 2)
    if len(seats_alt) < game.num_players(): seats_alt.append('A') # 处理奇数
    
    # 模式 2: 反向交替 (B A B A B A) - 消除位置优势干扰
    seats_alt_rev = ['B', 'A'] * (game.num_players() // 2)
    if len(seats_alt_rev) < game.num_players(): seats_alt_rev.append('B')

    total_games = args.num_games
    half_games = total_games // 2
    
    print(f"总局数: {total_games}")
    print("为了公平，将进行两轮测试，交换座位配置。")

    # 第一轮
    stats1, seats1 = play_match(game, network_a, network_b, device, seats_alt, half_games)
    
    # 第二轮
    stats2, seats2 = play_match(game, network_a, network_b, device, seats_alt_rev, half_games)
    
    # 汇总结果
    total_a_return = stats1['A']['return'] + stats2['A']['return']
    total_b_return = stats1['B']['return'] + stats2['B']['return']
    total_a_wins = stats1['A']['wins'] + stats2['A']['wins']
    total_b_wins = stats1['B']['wins'] + stats2['B']['wins']
    
    # 因为每局有多个 A 和 多个 B，我们需要计算“每个玩家位置的平均”
    # 6人局中，每局有 3 个 A 和 3 个 B
    num_a_players = seats_alt.count('A')
    num_b_players = seats_alt.count('B')
    
    # 总样本数 = 局数 * 该模型的玩家数
    total_samples_a = total_games * num_a_players
    total_samples_b = total_games * num_b_players
    
    avg_return_a = total_a_return / total_samples_a
    avg_return_b = total_b_return / total_samples_b
    
    win_rate_a = total_a_wins / total_samples_a * 100
    win_rate_b = total_b_wins / total_samples_b * 100
    
    print("\n" + "="*60)
    print("最终评估结果 (Model A vs Model B)")
    print("="*60)
    print(f"Model A: {os.path.basename(args.model_a)}")
    print(f"Model B: {os.path.basename(args.model_b)}")
    print("-" * 60)
    
    print(f"{'指标':<20} {'Model A':<20} {'Model B':<20} {'差值 (A-B)':<20}")
    print("-" * 60)
    print(f"{'平均收益 (bb/hand)':<20} {avg_return_a:>8.4f} {'':<10} {avg_return_b:>8.4f} {'':<10} {avg_return_a - avg_return_b:>+8.4f}")
    print(f"{'胜率 (%)':<20} {win_rate_a:>8.2f}% {'':<9} {win_rate_b:>8.2f}% {'':<9} {win_rate_a - win_rate_b:>+8.2f}%")
    print(f"{'总收益':<20} {total_a_return:>10.2f} {'':<8} {total_b_return:>10.2f}")
    
    print("\n结论:")
    if avg_return_a > avg_return_b:
        print("🏆 Model A 表现更好")
    else:
        print("🏆 Model B 表现更好")
        
    print("\n(注: 这是一个零和游戏，两者的平均收益之和应该接近 0)")


if __name__ == "__main__":
    main()
