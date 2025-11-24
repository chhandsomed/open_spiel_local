#!/usr/bin/env python3
"""
使用 DeepCFR with Feature Transform 训练德州扑克的示例

这个脚本展示了如何在训练中使用特征转换层
"""

import os
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import torch
import pyspiel
from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform


def train_with_feature_transform():
    """使用特征转换层训练"""
    
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
    
    print("=" * 70)
    print("DeepCFR with Feature Transform - 训练示例")
    print("=" * 70)
    print(f"游戏: {game.get_type().short_name}")
    print(f"玩家数量: {game.num_players()}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建带特征转换的 DeepCFR Solver
    print("\n创建 DeepCFR with Feature Transform...")
    solver = DeepCFRWithFeatureTransform(
        game,
        policy_network_layers=(256, 256),
        advantage_network_layers=(128, 128),
        transformed_size=150,  # 特征维度从266降到150
        use_hybrid_transform=True,  # 使用混合特征转换（推荐）
        num_iterations=10,  # 示例：少量迭代
        num_traversals=5,
        learning_rate=1e-4,
        memory_capacity=int(1e6),
        device=device,
    )
    
    print(f"✓ 转换后的特征大小: {solver._transformed_size}")
    print(f"✓ 使用混合转换: {solver._use_hybrid_transform}")
    print(f"✓ 手动特征维度: {solver._policy_network[0].manual_feature_size if hasattr(solver._policy_network, '__getitem__') else 'N/A'}")
    
    # 训练
    print("\n开始训练...")
    num_iterations = 10
    num_traversals = 5
    
    for iteration in range(num_iterations):
        print(f"  迭代 {iteration + 1}/{num_iterations}...", end="", flush=True)
        
        # 对每个玩家进行遍历和学习
        for player in range(game.num_players()):
            for _ in range(num_traversals):
                solver._traverse_game_tree(solver._root_node, player)
            
            # 重新初始化优势网络（如果需要）
            if solver._reinitialize_advantage_networks:
                solver.reinitialize_advantage_network(player)
            
            # 学习优势网络
            loss = solver._learn_advantage_network(player)
            if loss is not None:
                print(f" 玩家{player}损失: {loss:.6f}", end="", flush=True)
        
        # 更新迭代计数
        solver._iteration += 1
        
        # 学习策略网络
        policy_loss = solver._learn_strategy_network()
        if policy_loss is not None:
            print(f" 策略损失: {policy_loss:.6f}", end="", flush=True)
        
        print(" ✓")
    
    print("\n✓ 训练完成！")
    
    # 保存模型（可选）
    print("\n保存模型...")
    model_dir = "models/feature_transform_example"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存策略网络
    policy_path = os.path.join(model_dir, "policy_network.pt")
    torch.save(solver._policy_network.state_dict(), policy_path)
    print(f"  ✓ 策略网络: {policy_path}")
    
    # 保存优势网络
    for player in range(game.num_players()):
        advantage_path = os.path.join(model_dir, f"advantage_player_{player}.pt")
        torch.save(solver._advantage_networks[player].state_dict(), advantage_path)
        print(f"  ✓ 玩家{player}优势网络: {advantage_path}")
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)
    print("\n提示：")
    print("1. 在 train_deep_cfr_texas.py 中导入 DeepCFRWithFeatureTransform")
    print("2. 替换 DeepCFRSolver 为 DeepCFRWithFeatureTransform")
    print("3. 设置 use_hybrid_transform=True 和 transformed_size=150")
    print("4. 其他代码无需修改！")


if __name__ == "__main__":
    train_with_feature_transform()

