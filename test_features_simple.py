#!/usr/bin/env python3
"""
测试增强特征的值和分布（简化版）
"""

import sys
import os
import torch
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_cfr_simple_feature import (
    SimpleFeatureMLP,
    calculate_preflop_hand_strength,
    evaluate_hand_strength_from_bits,
    check_made_hands,
    check_draws,
    calculate_board_features,
    card_index_to_rank_suit
)

def card_str_to_index(card_str):
    """将牌字符串转换为索引"""
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    rank = rank_map[card_str[0]]
    suit = suit_map[card_str[1]]
    return rank * 4 + suit

def create_bits_from_cards(hole_cards_str, board_cards_str=None):
    """从手牌字符串创建bits tensor"""
    hole_bits = torch.zeros(52)
    board_bits = torch.zeros(52)
    
    for card_str in hole_cards_str.split():
        idx = card_str_to_index(card_str)
        hole_bits[idx] = 1.0
    
    if board_cards_str:
        for card_str in board_cards_str.split():
            idx = card_str_to_index(card_str)
            board_bits[idx] = 1.0
    
    return hole_bits, board_bits

def test_features():
    """测试特征值"""
    print("=" * 80)
    print("测试增强特征的值和分布")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        {
            "name": "Preflop - 强牌 (AA)",
            "hole_cards": "As Ah",
            "board_cards": None,
            "description": "起手最强牌"
        },
        {
            "name": "Preflop - 中等牌 (KQ)",
            "hole_cards": "Kh Qd",
            "board_cards": None,
            "description": "中等起手牌"
        },
        {
            "name": "Flop - 中了对子 (KK on 9-7-2)",
            "hole_cards": "Kh Kd",
            "board_cards": "9c 7h 2s",
            "description": "手牌对子，公共牌无威胁"
        },
        {
            "name": "Flop - 同花听牌 (A♠K♠ on 9♠7♠2♥)",
            "hole_cards": "As Ks",
            "board_cards": "9s 7s 2h",
            "description": "同花听牌"
        },
        {
            "name": "Flop - 顺子听牌 (J-T on 9-8-2)",
            "hole_cards": "Jh Th",
            "board_cards": "9c 8d 2s",
            "description": "两端顺子听牌"
        },
        {
            "name": "Turn - 中了两对 (A-K on A-K-9-7)",
            "hole_cards": "Ah Kd",
            "board_cards": "Ac Kc 9h 7s",
            "description": "顶两对"
        },
        {
            "name": "River - 成同花 (A♠K♠ on 9♠7♠2♠5♠)",
            "hole_cards": "As Ks",
            "board_cards": "9s 7s 2s 5s",
            "description": "同花成牌"
        },
        {
            "name": "River - 弱牌 (7-2 on A-K-Q-J-T)",
            "hole_cards": "7c 2d",
            "board_cards": "Ac Kc Qc Jc Tc",
            "description": "弱牌，公共牌很强"
        },
    ]
    
    print(f"\n{'=' * 80}")
    print("特征值测试结果")
    print(f"{'=' * 80}\n")
    
    all_features = []
    feature_names = [
        "1. 起手牌强度",
        "2. 当前手牌强度",
        "3. 是否中牌",
        "4. 是否中了对子",
        "5. 是否中了三条",
        "6. 是否中了两对",
        "7. 是否同花听牌",
        "8. 同花听牌outs数量",
        "9. 是否顺子听牌",
        "10. 顺子听牌outs数量",
        "11. 听牌成牌概率",
        "12. 公共牌强度",
        "13. 是否同花面",
        "14. 是否顺子面",
        "15. 游戏轮次-Preflop",
        "16. 游戏轮次-Flop",
        "17. 游戏轮次-Turn",
        "18. 游戏轮次-River",
        "19. 手牌强度变化",
        "20. 归一化最大下注",
        "21. 归一化总下注",
        "22. 是否有人加注",
        "23. 是否有人全押",
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['name']}")
        print(f"  描述: {test_case['description']}")
        print(f"  手牌: {test_case['hole_cards']}")
        if test_case['board_cards']:
            print(f"  公共牌: {test_case['board_cards']}")
        else:
            print(f"  公共牌: (Preflop)")
        
        try:
            # 创建bits tensor
            hole_bits, board_bits = create_bits_from_cards(
                test_case['hole_cards'],
                test_case['board_cards']
            )
            
            hole_bits_batch = hole_bits.unsqueeze(0)
            board_bits_batch = board_bits.unsqueeze(0)
            
            # 创建模拟的信息状态tensor（只需要手牌和公共牌部分）
            # 格式：玩家位置(5) + 手牌(52) + 公共牌(52) + 动作序列(52*2) + action_sizings(52)
            info_state = torch.zeros(5 + 52 + 52 + 52*2 + 52)
            info_state[0] = 1.0  # 玩家0
            info_state[5:5+52] = hole_bits
            info_state[5+52:5+104] = board_bits
            
            # 提取特征
            feature_extractor = SimpleFeatureMLP(
                raw_input_size=len(info_state),
                hidden_sizes=[256, 256],
                output_size=5,
                num_players=5,
                max_game_length=52,
                max_stack=50000,
                manual_feature_size=25
            )
            
            with torch.no_grad():
                features = feature_extractor.extract_manual_features(info_state.unsqueeze(0))
            
            features_np = features.squeeze().cpu().numpy()
            all_features.append(features_np)
            
            print(f"\n  特征值:")
            for j, (name, value) in enumerate(zip(feature_names[:len(features_np)], features_np)):
                print(f"    {name:30s}: {value:8.4f}")
            
            # 分析
            print(f"\n  特征分析:")
            print(f"    起手牌强度: {features_np[0]:.4f} (范围: 0-1)")
            print(f"    当前手牌强度: {features_np[1]:.4f} (范围: 0-1)")
            print(f"    强度变化: {features_np[18]:.4f} (当前 - 起手)")
            print(f"    是否中牌: {features_np[2]:.1f} (0=否, 1=是)")
            print(f"    成牌类型: 对子={features_np[3]:.1f}, 三条={features_np[4]:.1f}, 两对={features_np[5]:.1f}")
            print(f"    听牌: 同花={features_np[6]:.1f}, 顺子={features_np[8]:.1f}, 成牌概率={features_np[10]:.4f}")
            print(f"    公共牌: 强度={features_np[11]:.4f}, 同花面={features_np[12]:.1f}, 顺子面={features_np[13]:.1f}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计分析
    if all_features:
        all_features_array = np.array(all_features)
        print(f"\n{'=' * 80}")
        print("特征统计分析")
        print(f"{'=' * 80}\n")
        
        print(f"特征维度: {all_features_array.shape}")
        print(f"\n各特征统计:")
        print(f"{'特征名称':<30} {'均值':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'范围':<10} {'状态':<10}")
        print("-" * 90)
        
        for j, name in enumerate(feature_names[:all_features_array.shape[1]]):
            values = all_features_array[:, j]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            
            # 检查是否在合理范围内
            if j < 2:  # 起手牌强度和当前手牌强度应该在0-1
                range_status = "✓" if (min_val >= 0 and max_val <= 1) else "⚠"
            elif j in [2, 3, 4, 5, 6, 8, 12, 13, 15, 16, 17, 18, 21, 22]:  # 布尔特征应该在0-1
                range_status = "✓" if (min_val >= 0 and max_val <= 1) else "⚠"
            else:
                range_status = "✓" if (min_val >= -0.1 and max_val <= 1.1) else "⚠"
            
            print(f"{name:<30} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f} {range_val:<10.4f} {range_status:<10}")
        
        # 检查起手牌强度是否会被弱化
        print(f"\n{'=' * 80}")
        print("起手牌强度特征分析")
        print(f"{'=' * 80}\n")
        
        preflop_strength = all_features_array[:, 0]
        current_strength = all_features_array[:, 1]
        other_features = all_features_array[:, 2:]  # 其他23维特征
        
        print(f"起手牌强度统计:")
        print(f"  均值: {np.mean(preflop_strength):.4f}")
        print(f"  标准差: {np.std(preflop_strength):.4f}")
        print(f"  范围: [{np.min(preflop_strength):.4f}, {np.max(preflop_strength):.4f}]")
        
        print(f"\n其他特征统计:")
        print(f"  均值范围: [{np.min(np.mean(other_features, axis=0)):.4f}, {np.max(np.mean(other_features, axis=0)):.4f}]")
        print(f"  标准差范围: [{np.min(np.std(other_features, axis=0)):.4f}, {np.max(np.std(other_features, axis=0)):.4f}]")
        
        # 检查特征之间的相关性
        print(f"\n特征相关性分析:")
        # 计算起手牌强度与其他特征的相关性
        correlations = []
        for j in range(other_features.shape[1]):
            corr = np.corrcoef(preflop_strength, other_features[:, j])[0, 1]
            correlations.append((feature_names[j+2], corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"  与起手牌强度相关性最高的特征（前5个）:")
        for name, corr in correlations[:5]:
            print(f"    {name:<30}: {corr:7.4f}")
        
        # 检查特征是否会被归一化弱化
        print(f"\n特征归一化影响分析:")
        feature_norms = np.linalg.norm(all_features_array, axis=0)
        preflop_norm = feature_norms[0]
        other_norms = feature_norms[1:]
        
        print(f"  起手牌强度L2范数: {preflop_norm:.4f}")
        print(f"  其他特征L2范数范围: [{np.min(other_norms):.4f}, {np.max(other_norms):.4f}]")
        print(f"  起手牌强度占比: {preflop_norm / np.sum(feature_norms) * 100:.2f}%")
        
        if preflop_norm / np.sum(feature_norms) < 0.05:
            print(f"  ⚠️  警告: 起手牌强度特征可能被其他特征弱化（占比 < 5%）")
        elif preflop_norm / np.sum(feature_norms) < 0.1:
            print(f"  ⚠️  注意: 起手牌强度特征占比较低（占比 < 10%）")
        else:
            print(f"  ✓ 起手牌强度特征占比合理")
        
        # 检查特征值范围
        print(f"\n特征值范围检查:")
        print(f"  起手牌强度范围: [{np.min(preflop_strength):.4f}, {np.max(preflop_strength):.4f}]")
        print(f"  是否在0-1范围内: {'✓' if np.min(preflop_strength) >= 0 and np.max(preflop_strength) <= 1 else '⚠'}")
        
        # 检查特征是否都能从原始特征提取
        print(f"\n{'=' * 80}")
        print("特征提取来源分析")
        print(f"{'=' * 80}\n")
        print("所有25维特征都可以从原始信息状态中提取：")
        print("  ✓ 起手牌强度: 从手牌bits (52维)提取")
        print("  ✓ 当前手牌强度: 从手牌bits + 公共牌bits (104维)提取")
        print("  ✓ 成牌特征: 从手牌bits + 公共牌bits提取")
        print("  ✓ 听牌特征: 从手牌bits + 公共牌bits提取")
        print("  ✓ 公共牌特征: 从公共牌bits (52维)提取")
        print("  ✓ 游戏轮次: 从公共牌bits数量推断")
        print("  ✓ 下注统计: 从action_sizings提取")
        print("  ✓ 动作特征: 从动作序列bits提取")

if __name__ == "__main__":
    test_features()

