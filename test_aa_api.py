#!/usr/bin/env python3
"""
测试AA手牌和加注历史的API调用
用于验证状态重建逻辑
"""

import requests
import json

# API服务器地址
API_URL = "http://localhost:8826/api/v1/recommend_action"

def test_aa_with_raise():
    """测试AA手牌，包含加注历史"""
    
    # 测试场景：Player 1 持有 AA，历史动作 [0, 2, 0, 0, 0]
    # 动作含义：有人弃牌，有人加注到Pot，然后3个人弃牌
    request_data = {
        "player_id": 1,
        "hole_cards": ["As", "Ah"],  # AA
        "board_cards": [],  # Preflop
        "action_history": [0, 2, 0, 0, 0],  # Fold, Pot, Fold, Fold, Fold
        "action_sizings": [0.0, 350.0, 0.0, 0.0, 0.0],
        "blinds": [50, 100, 0, 0, 0, 0],
        "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
        "dealer_pos": 5
    }
    
    print("=" * 80)
    print("测试场景：AA手牌 + 加注历史")
    print("=" * 80)
    print(f"请求数据:")
    print(json.dumps(request_data, indent=2))
    print()
    
    try:
        response = requests.post(API_URL, json=request_data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n响应数据:")
            print(json.dumps(result, indent=2))
            
            if result.get('success'):
                data = result.get('data', {})
                recommended_action = data.get('recommended_action')
                action_probs = data.get('action_probabilities', {})
                
                print(f"\n推荐动作: {recommended_action}")
                print(f"动作概率分布:")
                action_names = {
                    0: "Fold",
                    1: "Call/Check",
                    2: "Pot",
                    3: "All-in",
                    4: "Half-Pot"
                }
                for action_id, prob in sorted(action_probs.items()):
                    action_name = action_names.get(int(action_id), f"Unknown({action_id})")
                    print(f"  {action_id} ({action_name}): {prob:.4f}")
                
                # 分析结果
                print(f"\n分析:")
                if recommended_action == 0:
                    print("❌ 问题：AA被推荐弃牌！这是不合理的。")
                elif recommended_action == 1:
                    print("⚠️  AA被推荐跟注，可以考虑加注。")
                elif recommended_action in [2, 4]:
                    print("✅ AA被推荐加注，这是合理的。")
                elif recommended_action == 3:
                    print("⚠️  AA被推荐全押，可能过于激进。")
            else:
                print(f"❌ API返回错误: {result.get('error')}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()


def test_aa_no_history():
    """测试AA手牌，无历史动作（Preflop开始）"""
    
    request_data = {
        "player_id": 1,
        "hole_cards": ["As", "Ah"],  # AA
        "board_cards": [],  # Preflop
        "action_history": [],  # 无历史动作
        "action_sizings": [],
        "blinds": [50, 100, 0, 0, 0, 0],
        "stacks": [2000, 2000, 2000, 2000, 2000, 2000],
        "dealer_pos": 5
    }
    
    print("\n" + "=" * 80)
    print("测试场景：AA手牌 + 无历史动作（Preflop开始）")
    print("=" * 80)
    print(f"请求数据:")
    print(json.dumps(request_data, indent=2))
    print()
    
    try:
        response = requests.post(API_URL, json=request_data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result.get('data', {})
                recommended_action = data.get('recommended_action')
                action_probs = data.get('action_probabilities', {})
                
                print(f"\n推荐动作: {recommended_action}")
                print(f"动作概率分布:")
                action_names = {
                    0: "Fold",
                    1: "Call/Check",
                    2: "Pot",
                    3: "All-in",
                    4: "Half-Pot"
                }
                for action_id, prob in sorted(action_probs.items()):
                    action_name = action_names.get(int(action_id), f"Unknown({action_id})")
                    print(f"  {action_id} ({action_name}): {prob:.4f}")
                
                # 分析结果
                print(f"\n分析:")
                if recommended_action == 0:
                    print("❌ 问题：AA在Preflop被推荐弃牌！这是严重错误。")
                elif recommended_action in [2, 4]:
                    print("✅ AA在Preflop被推荐加注，这是合理的。")
                else:
                    print(f"⚠️  AA在Preflop被推荐动作{recommended_action}，可能不是最优。")
                    
    except Exception as e:
        print(f"❌ 请求失败: {e}")


if __name__ == "__main__":
    print("🔍 AA手牌API测试脚本")
    print("=" * 80)
    print("注意：请确保API服务器正在运行 (http://localhost:8826)")
    print("=" * 80)
    print()
    
    # 测试1：AA + 加注历史
    test_aa_with_raise()
    
    # 测试2：AA + 无历史
    test_aa_no_history()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n请查看API服务器日志文件，搜索以下关键词：")
    print("  - 🔍 状态重建验证")
    print("  - 🔍 状态重建调试信息")
    print("  - ⚠️ 警告")
    print("  - ✅ 验证通过")


