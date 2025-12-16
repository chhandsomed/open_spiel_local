#!/usr/bin/env python3
"""并发API请求测试脚本

测试API服务器处理并发请求的能力，特别是不同dealer_pos的情况。
"""

import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import json

# API基础URL
API_BASE_URL = "http://localhost:8819/api/v1"

def get_first_player(dealer_pos: int, num_players: int = 6) -> int:
    """获取preflop阶段的firstPlayer"""
    if num_players == 2:
        # Heads Up: SB starts
        return dealer_pos
    else:
        # Ring Game: UTG starts
        utg_pos = (dealer_pos + 3) % num_players
        return utg_pos


def make_request(
    dealer_pos: int,
    player_id: int,
    with_blinds_stacks: bool = True,
    request_id: int = 0,
    use_first_player: bool = True
) -> Tuple[int, int, bool, Dict]:
    """发送API请求
    
    Args:
        dealer_pos: Dealer位置
        player_id: 玩家ID（如果use_first_player=True，会被忽略，使用firstPlayer）
        with_blinds_stacks: 是否传递blinds和stacks
        request_id: 请求ID（用于标识）
        use_first_player: 是否使用firstPlayer作为player_id
    
    Returns:
        (request_id, dealer_pos, success, response_data)
    """
    try:
        # 如果使用firstPlayer，计算正确的player_id
        if use_first_player:
            actual_player_id = get_first_player(dealer_pos)
        else:
            actual_player_id = player_id
        
        request_data = {
            "player_id": actual_player_id,
            "hole_cards": ["As", "Kh"],
            "board_cards": [],
            "action_history": [],
            "action_sizings": [],
            "dealer_pos": dealer_pos,
        }
        
        if with_blinds_stacks:
            request_data["blinds"] = [50, 100, 0, 0, 0, 0]
            request_data["stacks"] = [2000, 2000, 2000, 2000, 2000, 2000]
        
        # 保存请求数据用于显示
        saved_request_data = request_data.copy()
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/recommend_action",
            json=request_data,
            timeout=10
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            success = result.get('success', False)
            data = result.get('data', {})
            return (request_id, dealer_pos, success, {
                'status_code': response.status_code,
                'elapsed_time': elapsed_time,
                'request_data': saved_request_data,
                'recommended_action': data.get('recommended_action'),
                'action_probabilities': data.get('action_probabilities', {}),
                'legal_actions': data.get('legal_actions', []),
                'current_player': data.get('current_player'),
                'error': result.get('error')
            })
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
            return (request_id, dealer_pos, False, {
                'status_code': response.status_code,
                'elapsed_time': elapsed_time,
                'request_data': saved_request_data,
                'error': error_msg
            })
    except Exception as e:
        return (request_id, dealer_pos, False, {
            'error': str(e),
            'elapsed_time': 0,
            'request_data': saved_request_data if 'saved_request_data' in locals() else {}
        })


def test_concurrent_requests(
    num_requests: int = 10,
    with_blinds_stacks: bool = True,
    test_name: str = ""
):
    """测试并发请求
    
    Args:
        num_requests: 并发请求数量
        with_blinds_stacks: 是否传递blinds和stacks
        test_name: 测试名称
    """
    print(f"\n{'='*70}")
    print(f"测试: {test_name}")
    print(f"并发数: {num_requests}")
    print(f"传递blinds/stacks: {with_blinds_stacks}")
    print(f"{'='*70}")
    
    # 创建测试请求：不同的dealer_pos和player_id组合
    requests_list = []
    for i in range(num_requests):
        dealer_pos = i % 6  # 0-5轮换
        player_id = (i * 2) % 6  # 不同的player_id
        requests_list.append((dealer_pos, player_id, i))
    
    print(f"\n请求列表:")
    for dealer_pos, player_id, req_id in requests_list:
        first_player = get_first_player(dealer_pos)
        offset = (first_player - dealer_pos) % 6
        roles = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}
        role = roles.get(offset, f'P{offset}')
        print(f"  请求{req_id}: dealer_pos={dealer_pos}, player_id={first_player} ({role}, firstPlayer)")
    
    # 并发执行请求
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = {
            executor.submit(
                make_request,
                dealer_pos,
                player_id,
                with_blinds_stacks,
                req_id
            ): req_id
            for dealer_pos, player_id, req_id in requests_list
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    
    # 分析结果
    print(f"\n结果分析:")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均耗时: {total_time/num_requests:.2f}秒/请求")
    
    success_count = sum(1 for _, _, success, _ in results if success)
    fail_count = num_requests - success_count
    
    print(f"  成功: {success_count}/{num_requests}")
    print(f"  失败: {fail_count}/{num_requests}")
    
    if fail_count > 0:
        print(f"\n失败的请求:")
        for req_id, dealer_pos, success, data in results:
            if not success:
                print(f"    请求{req_id} (dealer_pos={dealer_pos}): {data.get('error', 'Unknown error')}")
    
    # 检查是否有重复的推荐动作（可能表示使用了错误的游戏配置）
    if success_count > 0:
        recommended_actions = {}
        action_probs_by_dealer = {}
        for req_id, dealer_pos, success, data in results:
            if success:
                action = data.get('recommended_action')
                probs = data.get('action_probabilities', {})
                key = (dealer_pos, action)
                if key not in recommended_actions:
                    recommended_actions[key] = []
                recommended_actions[key].append(req_id)
                
                # 记录每个dealer_pos的动作概率
                if dealer_pos not in action_probs_by_dealer:
                    action_probs_by_dealer[dealer_pos] = []
                action_probs_by_dealer[dealer_pos].append({
                    'action': action,
                    'probs': probs
                })
        
        print(f"\n推荐动作分布:")
        for (dealer_pos, action), req_ids in sorted(recommended_actions.items()):
            action_names = {0: 'Fold', 1: 'Call/Check', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}
            action_name = action_names.get(action, f'Action{action}')
            print(f"  dealer_pos={dealer_pos}, action={action} ({action_name}): {len(req_ids)}个请求 {req_ids}")
        
        # 显示每个dealer_pos的详细动作概率
        print(f"\n各dealer_pos的详细动作概率:")
        for dealer_pos in sorted(action_probs_by_dealer.keys()):
            print(f"\n  dealer_pos={dealer_pos}:")
            for i, result in enumerate(action_probs_by_dealer[dealer_pos]):
                action = result['action']
                probs = result['probs']
                action_names = {0: 'Fold', 1: 'Call/Check', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}
                print(f"    请求{i+1}: 推荐动作={action} ({action_names.get(action, f'Action{action}')})")
                if probs:
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    print(f"      动作概率: {', '.join([f'{action_names.get(int(k), k)}={v:.3f}' for k, v in sorted_probs[:3]])}")
    
    # 详细结果（包含请求和响应）
    print(f"\n详细结果（请求和响应）:")
    for req_id, dealer_pos, success, data in sorted(results, key=lambda x: x[0]):
        status = "✅" if success else "❌"
        elapsed = data.get('elapsed_time', 0)
        action = data.get('recommended_action', 'N/A')
        request_data = data.get('request_data', {})
        
        print(f"\n  {status} 请求{req_id} (dealer_pos={dealer_pos}):")
        print(f"    请求数据:")
        print(f"      player_id: {request_data.get('player_id')}")
        print(f"      dealer_pos: {request_data.get('dealer_pos')}")
        print(f"      hole_cards: {request_data.get('hole_cards')}")
        print(f"      board_cards: {request_data.get('board_cards')}")
        print(f"      action_history: {request_data.get('action_history')}")
        if request_data.get('blinds'):
            print(f"      blinds: {request_data.get('blinds')}")
        if request_data.get('stacks'):
            print(f"      stacks: {request_data.get('stacks')}")
        
        if success:
            print(f"    响应数据:")
            print(f"      推荐动作: {action}")
            probs = data.get('action_probabilities', {})
            if probs:
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                action_names = {0: 'Fold', 1: 'Call/Check', 2: 'Pot', 3: 'All-in', 4: 'Half-Pot'}
                prob_str = ', '.join([f"{action_names.get(int(k), k)}={v:.3f}" for k, v in sorted_probs[:3]])
                print(f"      动作概率: {prob_str}")
            print(f"      合法动作: {data.get('legal_actions', [])}")
            print(f"      当前玩家: {data.get('current_player')}")
        else:
            print(f"    错误: {data.get('error', 'Unknown error')}")
        print(f"    耗时: {elapsed:.3f}s")
    
    return results


def main():
    """主函数"""
    print("="*70)
    print("API并发请求测试")
    print("="*70)
    
    # 检查API服务器是否运行
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✅ API服务器运行正常")
            print(f"   模型已加载: {health.get('model_loaded', False)}")
            print(f"   游戏已加载: {health.get('game_loaded', False)}")
        else:
            print(f"\n❌ API服务器响应异常: {response.status_code}")
            return
    except Exception as e:
        print(f"\n❌ 无法连接到API服务器: {e}")
        print(f"   请确保API服务器运行在 {API_BASE_URL}")
        return
    
    # 测试1: 传递blinds和stacks（应该安全）
    print("\n" + "="*70)
    print("测试1: 传递blinds和stacks（推荐方式）")
    print("="*70)
    results1 = test_concurrent_requests(
        num_requests=10,
        with_blinds_stacks=True,
        test_name="传递blinds和stacks"
    )
    
    # 测试2: 不传递blinds和stacks（可能有风险）
    print("\n" + "="*70)
    print("测试2: 不传递blinds和stacks（使用全局GAME实例）")
    print("="*70)
    results2 = test_concurrent_requests(
        num_requests=10,
        with_blinds_stacks=False,
        test_name="不传递blinds和stacks"
    )
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    success1 = sum(1 for _, _, success, _ in results1 if success)
    success2 = sum(1 for _, _, success, _ in results2 if success)
    
    print(f"\n测试1（传递blinds/stacks）: {success1}/10 成功")
    print(f"测试2（不传递blinds/stacks）: {success2}/10 成功")
    
    if success1 == 10 and success2 == 10:
        print("\n✅ 所有测试通过！")
    elif success1 == 10:
        print("\n⚠️  测试1通过，但测试2有失败（可能因为全局GAME实例冲突）")
    else:
        print("\n❌ 测试失败，请检查API服务器")


if __name__ == '__main__':
    main()

