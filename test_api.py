#!/usr/bin/env python3
"""测试API服务器"""

import requests
import json
import sys

def test_health_check(base_url):
    """测试健康检查"""
    print("Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_action_mapping(base_url):
    """测试动作映射"""
    print("Testing action mapping...")
    response = requests.get(f"{base_url}/action_mapping")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_recommend_action(base_url):
    """测试推荐动作"""
    print("Testing recommend action...")
    
    # 测试用例：Preflop阶段，当前玩家是P0，手牌As Kh，还没有公共牌
    request_data = {
        "player_id": 0,
        "hole_cards": ["As", "Kh"],
        "board_cards": [],
        "action_history": [],
        "seed": 12345
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    response = requests.post(f"{base_url}/recommend_action", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    # 测试用例：Flop阶段，有公共牌
    request_data2 = {
        "player_id": 0,
        "hole_cards": ["As", "Kh"],
        "board_cards": ["2d", "3c", "4h"],
        "action_history": [1, 1, 1, 1],  # 所有玩家都call
        "seed": 12345
    }
    
    print(f"Request: {json.dumps(request_data2, indent=2)}")
    response2 = requests.post(f"{base_url}/recommend_action", json=request_data2)
    print(f"Status: {response2.status_code}")
    print(f"Response: {json.dumps(response2.json(), indent=2)}")
    print()
    
    # 测试用例：使用自定义盲注和筹码
    request_data3 = {
        "player_id": 0,
        "hole_cards": ["As", "Kh"],
        "board_cards": [],
        "action_history": [],
        "blinds": [50, 100, 0, 0, 0, 0],
        "stacks": [5000, 5000, 5000, 5000, 5000, 5000],
        "seed": 12345
    }
    
    print(f"Request (with custom blinds/stacks): {json.dumps(request_data3, indent=2)}")
    response3 = requests.post(f"{base_url}/recommend_action", json=request_data3)
    print(f"Status: {response3.status_code}")
    print(f"Response: {json.dumps(response3.json(), indent=2)}")
    print()

def test_reload_model(base_url):
    """测试模型重载"""
    print("Testing reload model...")
    
    # 注意：这里需要替换为实际的模型路径
    request_data = {
        "model_dir": "models/deepcfr_stable_run",
        "device": "cpu"
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    response = requests.post(f"{base_url}/reload_model", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <base_url>")
        print("Example: python test_api.py http://localhost:5000/api/v1")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    print("=" * 70)
    print("API Server Test")
    print("=" * 70)
    print()
    
    test_health_check(base_url)
    test_action_mapping(base_url)
    test_recommend_action(base_url)
    # test_reload_model(base_url)  # 取消注释以测试模型重载功能
    
    print("=" * 70)
    print("Test completed")
    print("=" * 70)

if __name__ == '__main__':
    main()
