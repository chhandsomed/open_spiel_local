#!/usr/bin/env python3
"""测试卡牌转换函数"""

def convert_user_card_to_openspiel(card_input):
    """将用户输入的牌面格式转换为OpenSpiel的card index
    
    用户输入格式：
    - 数字格式（0-51）：数字已经包含花色信息
      * 花色顺序：方块(Diamond)[0-12] -> 梅花(Clubs)[13-25] -> 红桃(Hearts)[26-38] -> 黑桃(Spade)[39-51]
      * 每个花色内：2~JQKA 对应 0~12（rank）
    - 字符串格式：如 "As", "Kh", "2d", "Tc", "Xh"（传统格式，兼容）
    - 大小王：JL(小王), JB(大王) - 不支持
    
    OpenSpiel格式（suit * 13 + rank）：
    - Diamonds(0-12): suit=0
    - Spades(13-25): suit=1
    - Hearts(26-38): suit=2
    - Clubs(39-51): suit=3
    """
    # 如果是整数，直接转换
    if isinstance(card_input, int):
        user_index = card_input
        
        if user_index < 0 or user_index > 51:
            raise ValueError(f"Invalid card index: {user_index}, must be 0-51")
        
        # 用户输入的花色顺序：方块[0-12] -> 梅花[13-25] -> 红桃[26-38] -> 黑桃[39-51]
        # OpenSpiel顺序：方块[0-12] -> 黑桃[13-25] -> 红桃[26-38] -> 梅花[39-51]
        
        if 0 <= user_index <= 12:
            # 方块：不变
            return user_index  # 0-12
        elif 13 <= user_index <= 25:
            # 用户：梅花[13-25] -> OpenSpiel：梅花[39-51]
            rank = user_index - 13
            return 39 + rank  # 39-51
        elif 26 <= user_index <= 38:
            # 红桃：不变
            return user_index  # 26-38
        elif 39 <= user_index <= 51:
            # 用户：黑桃[39-51] -> OpenSpiel：黑桃[13-25]
            rank = user_index - 39
            return 13 + rank  # 13-25
        else:
            raise ValueError(f"Invalid card index: {user_index}")
    
    # 如果是字符串，处理传统格式或大小王
    elif isinstance(card_input, str):
        card_str = card_input
        card_upper = card_str.upper()
        
        # 处理大小王
        if card_upper == "JL" or card_upper == "JB":
            raise ValueError(f"Joker cards ({card_str}) are not supported in standard poker")
        
        # 检查是否是纯数字字符串（如 "0", "13", "26", "39"）
        if card_str.isdigit():
            return convert_user_card_to_openspiel(int(card_str))
        
        # 传统格式需要card_string_to_index函数，这里简化处理
        raise ValueError(f"Traditional format not supported in this test: {card_str}")
    
    else:
        raise ValueError(f"Invalid card input type: {type(card_input)}, expected int or str")


def card_index_to_string(card_idx):
    """将card index转换为牌面字符串（OpenSpiel格式）"""
    suit_names = ['d', 's', 'h', 'c']  # Diamonds, Spades, Hearts, Clubs
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    suit = card_idx // 13
    rank = card_idx % 13
    
    return rank_names[rank] + suit_names[suit]


def test_conversion():
    """测试用户输入格式到OpenSpiel格式的转换"""
    
    print("=" * 70)
    print("测试卡牌转换函数")
    print("=" * 70)
    print()
    
    # 测试用例：用户输入格式 -> OpenSpiel格式
    test_cases = [
        # (用户输入, 期望的OpenSpiel index, 说明)
        # 方块 [0-12]
        (0, 0, "方块2"),
        (12, 12, "方块A"),
        (6, 6, "方块8"),
        
        # 梅花 [13-25] -> OpenSpiel [39-51]
        (13, 39, "梅花2"),
        (25, 51, "梅花A"),
        (19, 45, "梅花8"),
        
        # 红桃 [26-38] -> OpenSpiel [26-38] (不变)
        (26, 26, "红桃2"),
        (38, 38, "红桃A"),
        (32, 32, "红桃8"),
        
        # 黑桃 [39-51] -> OpenSpiel [13-25]
        (39, 13, "黑桃2"),
        (51, 25, "黑桃A"),
        (45, 19, "黑桃8"),
    ]
    
    print("测试整数输入 (0-51):")
    print("-" * 70)
    all_passed = True
    
    for user_input, expected_openspiel, description in test_cases:
        try:
            result = convert_user_card_to_openspiel(user_input)
            openspiel_str = card_index_to_string(result)
            expected_str = card_index_to_string(expected_openspiel)
            
            status = "✓" if result == expected_openspiel else "✗"
            if result != expected_openspiel:
                all_passed = False
            
            print(f"{status} 用户输入: {user_input:2d} ({description:6s}) -> OpenSpiel: {result:2d} ({openspiel_str:4s}) [期望: {expected_openspiel:2d} ({expected_str:4s})]")
        except Exception as e:
            print(f"✗ 用户输入: {user_input:2d} ({description:6s}) -> 错误: {e}")
            all_passed = False
    
    print()
    
    # 测试字符串输入（数字字符串）
    print("测试字符串输入 (数字字符串):")
    print("-" * 70)
    string_test_cases = [
        ("0", 0, "方块2"),
        ("13", 39, "梅花2"),
        ("26", 26, "红桃2"),
        ("39", 13, "黑桃2"),
    ]
    
    for user_input, expected_openspiel, description in string_test_cases:
        try:
            result = convert_user_card_to_openspiel(user_input)
            openspiel_str = card_index_to_string(result)
            expected_str = card_index_to_string(expected_openspiel)
            
            status = "✓" if result == expected_openspiel else "✗"
            if result != expected_openspiel:
                all_passed = False
            
            print(f"{status} 用户输入: '{user_input}' ({description:6s}) -> OpenSpiel: {result:2d} ({openspiel_str:4s}) [期望: {expected_openspiel:2d} ({expected_str:4s})]")
        except Exception as e:
            print(f"✗ 用户输入: '{user_input}' ({description:6s}) -> 错误: {e}")
            all_passed = False
    
    print()
    
    # 测试边界情况
    print("测试边界情况:")
    print("-" * 70)
    boundary_cases = [
        (-1, "负数"),
        (52, "超出范围"),
        ("JL", "小王"),
        ("JB", "大王"),
    ]
    
    for user_input, description in boundary_cases:
        try:
            result = convert_user_card_to_openspiel(user_input)
            print(f"✗ 用户输入: {user_input} ({description}) -> 应该报错但返回了: {result}")
            all_passed = False
        except (ValueError, TypeError) as e:
            print(f"✓ 用户输入: {user_input} ({description}) -> 正确报错: {e}")
        except Exception as e:
            print(f"✗ 用户输入: {user_input} ({description}) -> 意外错误: {e}")
            all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败！")
    print("=" * 70)
    
    return all_passed


def test_full_deck_conversion():
    """测试完整52张牌的转换"""
    print("\n" + "=" * 70)
    print("测试完整52张牌的转换（每花色显示前5张）")
    print("=" * 70)
    print()
    
    # 按花色分组显示
    suits_info = [
        ("方块", 0, 12, "Diamonds"),
        ("梅花", 13, 25, "Clubs"),
        ("红桃", 26, 38, "Hearts"),
        ("黑桃", 39, 51, "Spades"),
    ]
    
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    for suit_name, start, end, openspiel_suit in suits_info:
        print(f"{suit_name} [{start}-{end}]:")
        for i in range(start, min(start + 5, end + 1)):  # 只显示前5张
            try:
                openspiel_idx = convert_user_card_to_openspiel(i)
                openspiel_str = card_index_to_string(openspiel_idx)
                rank = i % 13
                print(f"  用户: {i:2d} ({rank_names[rank]:2s}) -> OpenSpiel: {openspiel_idx:2d} ({openspiel_str})")
            except Exception as e:
                print(f"  用户: {i:2d} -> 错误: {e}")
        if end - start >= 5:
            print(f"  ... (共 {end - start + 1} 张)")
        print()


if __name__ == "__main__":
    success = test_conversion()
    test_full_deck_conversion()
    exit(0 if success else 1)
