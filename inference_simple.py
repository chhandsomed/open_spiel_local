#!/usr/bin/env python3
"""简化的 DeepCFR 推理脚本

直接加载模型权重，支持多种模型架构（简单特征、复杂特征转换、标准MLP）
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
import numpy as np
import json
import sys
import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python.pytorch.deep_cfr import MLP

# 尝试导入自定义特征类
try:
    from deep_cfr_simple_feature import DeepCFRSimpleFeature
    from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
    HAVE_CUSTOM_FEATURES = True
except ImportError:
    HAVE_CUSTOM_FEATURES = False
    print("注意: 未找到自定义特征模块 (deep_cfr_simple_feature.py 等)，只能加载标准 MLP 模型")


def load_config(model_prefix):
    """尝试加载配置文件"""
    # 尝试找到配置文件
    # case 1: model_prefix 是完整路径前缀，如 models/dir/prefix
    dir_name = os.path.dirname(model_prefix)
    config_path = os.path.join(dir_name, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️ 无法读取配置文件 {config_path}: {e}")
    
    return None


def load_policy_network(model_path, game, config=None, device='cpu'):
    """加载策略网络，根据配置自动选择模型结构"""
    print(f"  加载策略网络: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ✗ 文件不存在: {model_path}")
        return None

    # 默认参数
    policy_layers = [64, 64]
    use_simple_feature = False
    use_feature_transform = False
    transformed_size = 150
    use_hybrid_transform = True
    
    # 从配置读取
    if config:
        policy_layers = config.get('policy_layers', [64, 64])
        use_simple_feature = config.get('use_simple_feature', False)
        use_feature_transform = config.get('use_feature_transform', False)
        transformed_size = config.get('transformed_size', 150)
        use_hybrid_transform = config.get('use_hybrid_transform', True)
        
        # 兼容性检查
        if use_simple_feature and not HAVE_CUSTOM_FEATURES:
            print("  ✗ 模型使用了 Simple Feature，但未找到对应代码模块")
            return None
        if use_feature_transform and not use_simple_feature and not HAVE_CUSTOM_FEATURES:
            print("  ✗ 模型使用了 Feature Transform，但未找到对应代码模块")
            return None

    # 1. 简单特征版本 (Simple Feature)
    if use_simple_feature and HAVE_CUSTOM_FEATURES:
        print("  模式: Simple Feature (直接拼接特征)")
        try:
            # 我们只需要创建一个临时的 solver 来获取网络结构
            # 不需要实际运行 solver
            solver = DeepCFRSimpleFeature(
                game,
                policy_network_layers=tuple(policy_layers),
                advantage_network_layers=(32, 32), # 占位
                device=device
            )
            network = solver._policy_network
        except Exception as e:
            print(f"  ✗ 创建 SimpleFeature 模型失败: {e}")
            return None

    # 2. 复杂特征转换版本 (Feature Transform)
    elif use_feature_transform and HAVE_CUSTOM_FEATURES:
        print("  模式: Feature Transform (复杂特征层)")
        try:
            solver = DeepCFRWithFeatureTransform(
                game,
                policy_network_layers=tuple(policy_layers),
                advantage_network_layers=(32, 32),
                transformed_size=transformed_size,
                use_hybrid_transform=use_hybrid_transform,
                device=device
            )
            network = solver._policy_network
        except Exception as e:
            print(f"  ✗ 创建 FeatureTransform 模型失败: {e}")
            return None

    # 3. 标准版本 (Standard MLP)
    else:
        print("  模式: Standard MLP")
        state = game.new_initial_state()
        embedding_size = len(state.information_state_tensor(0))
        num_actions = game.num_distinct_actions()
        network = MLP(embedding_size, list(policy_layers), num_actions)
        network = network.to(device)

    # 加载权重
    try:
        state_dict = torch.load(model_path, map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        print(f"  ✓ 策略网络加载成功")
        return network
    except RuntimeError as e:
        print(f"  ✗ 权重加载失败: {e}")
        print(f"  提示: 模型结构可能不匹配 (Layers: {policy_layers})")
        return None


def get_action_from_network(state, policy_network, device):
    """使用策略网络获取动作 (兼容普通 MLP 和带特征提取的网络)"""
    info_state = state.information_state_tensor()
    legal_actions = state.legal_actions()
    
    # 转换为张量
    info_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
    
    # 前向传播
    with torch.no_grad():
        # 检查网络是否需要特殊输入处理 (例如 SimpleFeatureMLP)
        # 通常这些网络重写了 forward 方法来处理 raw input
        logits = policy_network(info_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # 获取合法动作的概率
    action_probs = {a: float(probs[a]) for a in legal_actions}
    
    # 归一化
    total = sum(action_probs.values())
    if total > 1e-10:
        action_probs = {a: p/total for a, p in action_probs.items()}
    else:
        action_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
    
    # 采样动作
    actions = list(action_probs.keys())
    probabilities = np.array([action_probs[a] for a in actions])
    probabilities = probabilities / probabilities.sum()
    action = np.random.choice(actions, p=probabilities)
    
    return action, action_probs


def play_game_simple(game, policy_network, device, verbose=True):
    """使用模型玩一局游戏"""
    state = game.new_initial_state()
    
    if verbose:
        print("\n  游戏开始...")
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = np.random.choice([a for a, _ in outcomes], 
                                     p=[p for _, p in outcomes])
            state = state.child(action)
        else:
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
    """测试模型"""
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
    parser.add_argument("--model_dir", type=str, default=None,
                       help="模型目录路径 (例如: models/deepcfr_texas_6p)，推荐使用")
    parser.add_argument("--model_prefix", type=str, default=None,
                       help="模型文件路径前缀 (例如: models/run1/deepcfr_texas)，兼容旧版")
    parser.add_argument("--num_games", type=int, default=10,
                       help="测试游戏数量")
    parser.add_argument("--num_players", type=int, default=None,
                       help="玩家数量 (如果不指定，尝试从配置读取，默认6)")
    parser.add_argument("--betting_abstraction", type=str, default=None,
                       help="下注抽象 (如果不指定，尝试从配置读取，默认为 fcpa)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepCFR 简化推理")
    print("=" * 70)
    
    # 处理 model_dir 和 model_prefix 参数
    if args.model_dir:
        # 新方式：只传目录，从 config.json 读取 save_prefix
        model_dir = args.model_dir
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            save_prefix = config.get('save_prefix', 'deepcfr_texas')
            args.model_prefix = os.path.join(model_dir, save_prefix)
            print(f"  ✓ 找到并加载配置文件 config.json")
            print(f"  模型前缀: {save_prefix}")
        else:
            # 尝试自动检测模型文件
            import glob
            pt_files = glob.glob(os.path.join(model_dir, "*_policy_network.pt"))
            if pt_files:
                # 从文件名推断 prefix
                policy_file = os.path.basename(pt_files[0])
                save_prefix = policy_file.replace("_policy_network.pt", "")
                args.model_prefix = os.path.join(model_dir, save_prefix)
                config = load_config(args.model_prefix)
                print(f"  ⚠️ 未找到 config.json，从文件名推断前缀: {save_prefix}")
            else:
                print(f"  ✗ 目录中未找到模型文件: {model_dir}")
                sys.exit(1)
    elif args.model_prefix:
        # 兼容旧方式
        config = load_config(args.model_prefix)
        if config:
            print("  ✓ 找到并加载配置文件 config.json")
    else:
        print("  ✗ 请指定 --model_dir 或 --model_prefix")
        sys.exit(1)
    
    # 如果还没有 config，显示警告
    if config is None:
        print("  ⚠️ 未找到配置文件，将使用默认参数或命令行参数")

    # 确定 num_players
    if args.num_players is not None:
        num_players = args.num_players
    elif config and 'num_players' in config:
        num_players = config['num_players']
    else:
        num_players = 6  # 默认为6人场
        print(f"  注意: 未指定玩家数量，默认为 {num_players}")

    # 确定 betting_abstraction
    betting_abstraction = "fcpa" # 默认值
    if args.betting_abstraction is not None:
        betting_abstraction = args.betting_abstraction
    elif config and 'betting_abstraction' in config:
        betting_abstraction = config['betting_abstraction']
    
    print(f"  下注抽象: {betting_abstraction}")

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建游戏
    print(f"\n[0/2] 创建游戏 ({num_players}人场)...")
    
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
        f"bettingAbstraction={betting_abstraction}"
        f")"
    )
    try:
        game = pyspiel.load_game(game_string)
        print(f"  ✓ 游戏创建成功: {game.get_type().short_name}")
    except Exception as e:
        print(f"  ✗ 游戏创建失败: {e}")
        return
    
    # 加载模型
    print(f"\n[1/2] 加载模型...")
    policy_path = f"{args.model_prefix}_policy_network.pt"
    
    policy_network = load_policy_network(
        policy_path,
        game,
        config=config,
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
