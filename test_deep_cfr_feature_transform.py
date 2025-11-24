#!/usr/bin/env python3
"""
测试 deep_cfr_with_feature_transform.py 的完整功能

运行方式：
conda activate open_spiel
python test_deep_cfr_feature_transform.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pyspiel
from deep_cfr_with_feature_transform import (
    calculate_hand_strength_features,
    calculate_position_advantage,
    HybridFeatureTransform,
    DeepCFRWithFeatureTransform
)


def test_hand_strength_features():
    """测试手牌强度特征（应该只有1维：起手牌强度）"""
    print("=" * 60)
    print("测试1: 手牌强度特征")
    print("=" * 60)
    
    batch_size = 3
    hole_cards = torch.zeros(batch_size, 52)
    board_cards = torch.zeros(batch_size, 52)
    
    # 测试用例1: AA (最强起手牌)
    hole_cards[0, 48] = 1.0  # As
    hole_cards[0, 49] = 1.0  # Ah
    
    # 测试用例2: AKs (强起手牌)
    hole_cards[1, 48] = 1.0  # As
    hole_cards[1, 44] = 1.0  # Ks
    
    # 测试用例3: 72o (最弱起手牌)
    hole_cards[2, 0] = 1.0   # 2s
    hole_cards[2, 30] = 1.0  # 7h
    
    features = calculate_hand_strength_features(hole_cards, board_cards)
    
    print(f"输入形状: hole_cards={hole_cards.shape}, board_cards={board_cards.shape}")
    print(f"输出形状: {features.shape}")
    print(f"期望维度: 1")
    print(f"实际维度: {features.shape[1]}")
    print(f"✓ 维度匹配: {features.shape[1] == 1}")
    
    print(f"\n起手牌强度值:")
    for i in range(batch_size):
        strength = features[i, 0].item()
        print(f"  样本{i}: {strength:.4f}")
    
    # 验证强度顺序：AA > AKs > 72o
    assert features[0, 0] > features[1, 0], "AA应该比AKs强"
    assert features[1, 0] > features[2, 0], "AKs应该比72o强"
    print("✓ 强度顺序验证通过")
    print()


def test_position_advantage():
    """测试位置优势特征（应该是4维）"""
    print("=" * 60)
    print("测试2: 位置优势特征")
    print("=" * 60)
    
    batch_size = 6
    player_indices = torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.]])  # UTG, MP, CO, BTN, SB, BB
    
    features = calculate_position_advantage(player_indices, num_players=6)
    
    print(f"输入形状: {player_indices.shape}")
    print(f"输出形状: {features.shape}")
    print(f"期望维度: 4")
    print(f"实际维度: {features.shape[1]}")
    print(f"✓ 维度匹配: {features.shape[1] == 4}")
    
    print(f"\n位置特征值:")
    positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    for i, pos in enumerate(positions):
        pos_adv = features[i, 0].item()
        is_early = features[i, 1].item()
        is_late = features[i, 2].item()
        is_blind = features[i, 3].item()
        print(f"  {pos}: 优势={pos_adv:.1f}, 早期={is_early:.0f}, 后期={is_late:.0f}, 盲注={is_blind:.0f}")
    
    # 验证位置优势值
    assert features[3, 0] > features[2, 0], "BTN应该比CO强"
    assert features[2, 0] > features[1, 0], "CO应该比MP强"
    assert features[1, 0] > features[0, 0], "MP应该比UTG强"
    print("✓ 位置优势值验证通过")
    print()


def test_hybrid_feature_transform():
    """测试混合特征转换层"""
    print("=" * 60)
    print("测试3: HybridFeatureTransform")
    print("=" * 60)
    
    raw_input_size = 266
    transformed_size = 150
    num_players = 6
    max_game_length = 52
    
    transform = HybridFeatureTransform(
        raw_input_size=raw_input_size,
        transformed_size=transformed_size,
        num_players=num_players,
        max_game_length=max_game_length,
        use_normalization=True
    )
    
    print(f"手动特征维度: {transform.manual_feature_size}")
    print(f"期望维度: 7 (4位置 + 1起手牌强度 + 2下注)")
    print(f"✓ 维度匹配: {transform.manual_feature_size == 7}")
    
    # 测试批量输入
    batch_size = 4
    x = torch.randn(batch_size, raw_input_size)
    
    with torch.no_grad():
        output = transform(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"期望输出形状: ({batch_size}, {transformed_size})")
    print(f"✓ 输出形状匹配: {output.shape == (batch_size, transformed_size)}")
    
    # 测试单样本输入
    x_single = torch.randn(raw_input_size)
    with torch.no_grad():
        output_single = transform(x_single)
    
    print(f"\n单样本输入形状: {x_single.shape}")
    print(f"单样本输出形状: {output_single.shape}")
    print(f"期望输出形状: (1, {transformed_size})")
    print(f"✓ 单样本输出形状匹配: {output_single.shape == (1, transformed_size)}")
    print()


def test_deep_cfr_with_feature_transform():
    """测试 DeepCFRWithFeatureTransform"""
    print("=" * 60)
    print("测试4: DeepCFRWithFeatureTransform")
    print("=" * 60)
    
    # 创建游戏
    game_config = {
        "numPlayers": 6,
        "numBoardCards": "0 3 1 1",  # Preflop, Flop, Turn, River
        "numRanks": 13,
        "numSuits": 4,
        "firstPlayer": "2",  # UTG
        "stack": "20000 20000 20000 20000 20000 20000",
        "blind": "50 100 0 0 0 0",  # SB, BB, UTG, MP, CO, BTN
        "numHoleCards": 2,
        "numRounds": 4,
        "betting": "nolimit",
        "maxRaises": "3",
    }
    game = pyspiel.load_game("universal_poker", game_config)
    
    print(f"游戏: {game.get_type().short_name}")
    print(f"玩家数量: {game.num_players()}")
    
    # 创建 DeepCFRWithFeatureTransform
    solver = DeepCFRWithFeatureTransform(
        game,
        policy_network_layers=(256, 256),
        advantage_network_layers=(128, 128),
        transformed_size=150,
        use_hybrid_transform=True,  # 使用混合特征转换
        num_iterations=10,
        num_traversals=5,
        learning_rate=1e-4,
        device=torch.device("cpu")
    )
    
    print(f"转换后的特征大小: {solver._transformed_size}")
    print(f"使用混合转换: {solver._use_hybrid_transform}")
    
    # 测试策略网络
    state = game.new_initial_state()
    while not state.is_terminal() and not state.is_chance_node():
        legal_actions = state.legal_actions()
        if legal_actions:
            state.apply_action(legal_actions[0])
        else:
            break
    
    if not state.is_terminal():
        info_state = state.information_state_tensor(0)
        info_state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0))
        
        with torch.no_grad():
            # 测试策略网络
            output = solver._policy_network(info_state_tensor)
            probs = solver._policy_sm(output)
            
            print(f"\n策略网络测试:")
            print(f"  输入形状: {info_state_tensor.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  概率形状: {probs.shape}")
            print(f"  动作数量: {game.num_distinct_actions()}")
            print(f"  ✓ 输出维度匹配: {output.shape[1] == game.num_distinct_actions()}")
            
            # 测试优势网络
            advantage_output = solver._advantage_networks[0](info_state_tensor)
            print(f"\n优势网络测试:")
            print(f"  输入形状: {info_state_tensor.shape}")
            print(f"  输出形状: {advantage_output.shape}")
            print(f"  ✓ 输出维度匹配: {advantage_output.shape[1] == game.num_distinct_actions()}")
    
    print()


def test_feature_extraction():
    """测试手动特征提取"""
    print("=" * 60)
    print("测试5: 手动特征提取")
    print("=" * 60)
    
    raw_input_size = 266
    num_players = 6
    max_game_length = 52
    
    transform = HybridFeatureTransform(
        raw_input_size=raw_input_size,
        transformed_size=150,
        num_players=num_players,
        max_game_length=max_game_length
    )
    
    # 创建模拟信息状态
    batch_size = 2
    x = torch.zeros(batch_size, raw_input_size)
    
    # 设置玩家位置（玩家0）
    x[0, 0] = 1.0
    x[1, 3] = 1.0  # BTN
    
    # 设置手牌（AA）
    x[0, 48] = 1.0  # As
    x[0, 49] = 1.0  # Ah
    x[1, 48] = 1.0  # As
    x[1, 49] = 1.0  # Ah
    
    # 提取手动特征
    with torch.no_grad():
        manual_features = transform.extract_manual_features(x)
    
    print(f"输入形状: {x.shape}")
    print(f"手动特征形状: {manual_features.shape}")
    print(f"期望维度: {transform.manual_feature_size}")
    print(f"实际维度: {manual_features.shape[1]}")
    print(f"✓ 维度匹配: {manual_features.shape[1] == transform.manual_feature_size}")
    
    print(f"\n手动特征值（样本0，UTG位置，AA手牌）:")
    print(f"  位置优势: {manual_features[0, 0]:.4f}")
    print(f"  是否早期: {manual_features[0, 1]:.4f}")
    print(f"  是否后期: {manual_features[0, 2]:.4f}")
    print(f"  是否盲注: {manual_features[0, 3]:.4f}")
    print(f"  起手牌强度: {manual_features[0, 4]:.4f}")
    print(f"  最大下注: {manual_features[0, 5]:.4f}")
    print(f"  总下注: {manual_features[0, 6]:.4f}")
    
    print(f"\n手动特征值（样本1，BTN位置，AA手牌）:")
    print(f"  位置优势: {manual_features[1, 0]:.4f}")
    print(f"  是否早期: {manual_features[1, 1]:.4f}")
    print(f"  是否后期: {manual_features[1, 2]:.4f}")
    print(f"  是否盲注: {manual_features[1, 3]:.4f}")
    print(f"  起手牌强度: {manual_features[1, 4]:.4f}")
    
    # 验证位置优势：BTN应该比UTG强
    assert manual_features[1, 0] > manual_features[0, 0], "BTN位置优势应该比UTG强"
    # 验证起手牌强度：AA应该相同
    assert abs(manual_features[0, 4] - manual_features[1, 4]) < 1e-5, "相同手牌的强度应该相同"
    print("✓ 特征值验证通过")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DeepCFR with Feature Transform - 完整测试")
    print("=" * 60 + "\n")
    
    try:
        test_hand_strength_features()
        test_position_advantage()
        test_hybrid_feature_transform()
        test_deep_cfr_with_feature_transform()
        test_feature_extraction()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

