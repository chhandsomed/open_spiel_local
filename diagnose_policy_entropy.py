#!/usr/bin/env python3
"""诊断策略熵为0的问题"""

import sys
import numpy as np
import torch
import pyspiel

# 添加路径
sys.path.insert(0, '.')

from deep_cfr_simple_feature import DeepCFRSimpleFeature

def compute_entropy(probs_dict):
    """计算策略熵"""
    probs = np.array(list(probs_dict.values()))
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs + 1e-10))

def diagnose_policy_entropy(model_dir):
    """诊断策略熵问题"""
    print("=" * 70)
    print("策略熵诊断")
    print("=" * 70)
    
    # 创建游戏
    game_config = {
        'numPlayers': 6,
        'numBoardCards': '0 3 1 1',
        'numRanks': 13,
        'numSuits': 4,
        'firstPlayer': '2',
        'stack': '2000 2000 2000 2000 2000 2000',
        'blind': '100 100 100 100 100 100',
        'numHoleCards': 2,
        'numRounds': 4,
        'betting': 'nolimit',
        'maxRaises': '3',
    }
    game = pyspiel.load_game('universal_poker', game_config)
    
    # 创建solver（不训练，只用于加载模型）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver = DeepCFRSimpleFeature(
        game,
        policy_network_layers=(64, 64),
        advantage_network_layers=(32, 32),
        num_iterations=1,
        num_traversals=1,
        learning_rate=1e-4,
        device=device
    )
    
    # 加载策略网络
    import os
    policy_path = os.path.join(model_dir, 'deepcfr_texas_policy_network.pt')
    if not os.path.exists(policy_path):
        print(f"❌ 未找到策略网络: {policy_path}")
        return
    
    print(f"\n[1] 加载策略网络: {policy_path}")
    solver._policy_network.load_state_dict(torch.load(policy_path, map_location=device))
    solver._policy_network.eval()
    print("✓ 策略网络加载成功")
    
    # 测试多个状态
    print("\n[2] 测试策略输出")
    entropies = []
    num_states_tested = 0
    num_states_with_zero_entropy = 0
    
    for test_idx in range(20):
        try:
            state = game.new_initial_state()
            depth = 0
            max_depth = 15
            
            # 模拟游戏到玩家决策节点
            while not state.is_terminal() and depth < max_depth:
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    if not outcomes:
                        break
                    action = np.random.choice([a for a, _ in outcomes], 
                                             p=[p for _, p in outcomes])
                    state = state.child(action)
                else:
                    player = state.current_player()
                    legal_actions = state.legal_actions()
                    
                    if len(legal_actions) == 0:
                        break
                    
                    # 获取策略输出
                    with torch.no_grad():
                        info_state = state.information_state_tensor(player)
                        info_state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0)).to(device)
                        logits = solver._policy_network(info_state_tensor)
                        probs_tensor = torch.softmax(logits, dim=-1)
                        probs_np = probs_tensor[0].cpu().numpy()
                    
                    # 构建动作概率字典
                    probs_dict = {}
                    for i, action in enumerate(legal_actions):
                        if action < len(probs_np):
                            probs_dict[action] = float(probs_np[action])
                    
                    # 归一化
                    total = sum(probs_dict.values())
                    if total > 0:
                        probs_dict = {a: p/total for a, p in probs_dict.items()}
                    else:
                        # 如果所有概率为0，使用均匀分布
                        probs_dict = {a: 1.0/len(legal_actions) for a in legal_actions}
                    
                    # 计算熵
                    entropy = compute_entropy(probs_dict)
                    entropies.append(entropy)
                    num_states_tested += 1
                    
                    if entropy == 0.0:
                        num_states_with_zero_entropy += 1
                        if num_states_with_zero_entropy <= 3:  # 只打印前3个
                            print(f"\n  状态 {num_states_tested}: 熵 = 0.0")
                            print(f"    合法动作: {legal_actions}")
                            print(f"    动作概率: {probs_dict}")
                            print(f"    网络输出 (logits): {logits[0].cpu().numpy()}")
                            print(f"    网络输出 (probs): {probs_np}")
                    
                    # 采样动作继续
                    actions = list(probs_dict.keys())
                    probabilities = list(probs_dict.values())
                    action = np.random.choice(actions, p=probabilities)
                    state = state.child(action)
                    
                    if num_states_tested >= 50:  # 限制测试状态数
                        break
                
                depth += 1
                if num_states_tested >= 50:
                    break
            
        except Exception as e:
            print(f"  ⚠️ 测试状态 {test_idx} 时出错: {e}")
            continue
    
    # 分析结果
    print(f"\n[3] 诊断结果")
    print(f"  测试状态数: {num_states_tested}")
    print(f"  熵为0的状态数: {num_states_with_zero_entropy}")
    print(f"  熵为0的比例: {num_states_with_zero_entropy/num_states_tested*100:.1f}%")
    
    if entropies:
        print(f"\n  熵统计:")
        print(f"    平均熵: {np.mean(entropies):.4f}")
        print(f"    最小熵: {np.min(entropies):.4f}")
        print(f"    最大熵: {np.max(entropies):.4f}")
        print(f"    标准差: {np.std(entropies):.4f}")
        
        if np.mean(entropies) == 0.0:
            print(f"\n  ❌ 问题确认: 策略熵确实为0")
            print(f"     可能原因:")
            print(f"     1. 网络输出过于确定（只有一个动作概率接近1）")
            print(f"     2. Softmax 输出异常")
            print(f"     3. 网络未正确训练")
        elif np.mean(entropies) < 0.1:
            print(f"\n  ⚠️ 警告: 策略熵过低（< 0.1）")
            print(f"     策略过于确定，可能无法进行有效探索")
        else:
            print(f"\n  ✓ 策略熵正常")
            print(f"     但评估代码可能有问题，导致显示为0")
    else:
        print(f"\n  ❌ 无法测试任何状态")
        print(f"     可能原因:")
        print(f"     1. 游戏状态访问失败")
        print(f"     2. 网络输出异常导致无法继续")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认使用最新的模型目录
        import glob
        model_dirs = glob.glob("models/deepcfr_texas_*/")
        if model_dirs:
            model_dir = sorted(model_dirs)[-1]  # 最新的
            print(f"使用最新的模型目录: {model_dir}")
        else:
            print("错误: 未找到模型目录")
            print("用法: python diagnose_policy_entropy.py <model_dir>")
            sys.exit(1)
    else:
        model_dir = sys.argv[1]
    
    diagnose_policy_entropy(model_dir)

