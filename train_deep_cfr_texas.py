#!/usr/bin/env python3
"""使用 DeepCFR 训练德州扑克策略"""

import os
# 禁用 torch.compile 以避免导入问题
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import argparse
import sys
import time
import json
import numpy as np
import torch

import pyspiel
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python import policy
from deep_cfr_with_feature_transform import DeepCFRWithFeatureTransform
from deep_cfr_simple_feature import DeepCFRSimpleFeature


def json_serialize(obj):
    """将对象转换为 JSON 可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(item) for item in obj]
    else:
        return obj


def create_save_directory(save_prefix, save_dir="models"):
    """创建保存目录，如果已存在则添加时间戳
    
    Args:
        save_prefix: 保存文件前缀
        save_dir: 基础保存目录（默认 "models"）
    
    Returns:
        str: 创建的目录路径
    """
    # 创建基础目录
    base_dir = save_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 创建模型子目录
    model_dir = os.path.join(base_dir, save_prefix)
    
    # 如果目录已存在，添加时间戳
    if os.path.exists(model_dir):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = f"{model_dir}_{timestamp}"
    
    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir

def train_deep_cfr(
    num_players=2,
    num_iterations=10,
    num_traversals=20,
    policy_layers=(64, 64),
    advantage_layers=(32, 32),
    learning_rate=1e-3,
    memory_capacity=int(1e6),
    save_prefix="deepcfr_texas",
    use_gpu=True,
    skip_nashconv=False,
    eval_interval=10,
    eval_with_games=False,
    save_history=True,
    save_dir="models",
    use_feature_transform=True,  # 新增：是否使用特征转换层
    use_simple_feature=True,  # 新增：是否使用简单版本（直接拼接7维特征，推荐True）
    transformed_size=150,  # 新增：转换后的特征大小（仅用于复杂版本）
    use_hybrid_transform=True,  # 新增：是否使用混合特征转换（仅用于复杂版本）
    betting_abstraction="fcpa", # 新增：下注抽象模式
):
    """使用 DeepCFR 训练德州扑克策略
    
    Args:
        num_players: 玩家数量
        num_iterations: 迭代次数
        num_traversals: 每次迭代的遍历次数
        policy_layers: 策略网络层大小
        advantage_layers: 优势网络层大小
        learning_rate: 学习率
        memory_capacity: 内存容量
        save_prefix: 保存文件前缀
        betting_abstraction: 下注抽象 (fcpa, fchpa, etc.)
    """
    print("=" * 70)
    print("DeepCFR 训练 - 德州扑克")
    print("=" * 70)
    
    # 检查 PyTorch 和 GPU
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    
    # 创建游戏 - 使用 universal_poker 因为 DeepCFR 需要 information_state_tensor
    # pokerkit_wrapper 不支持 tensor，所以不能用于 DeepCFR
    print(f"\n[1/4] 创建游戏 ({num_players}人无限注德州)...")
    print("  注意: 使用 universal_poker (DeepCFR 需要 information_state_tensor)")
    print(f"  下注抽象: {betting_abstraction}")
    sys.stdout.flush()
    
    # 配置 blinds 和 firstPlayer
    if num_players == 2:
        # 2人场 (Heads-Up): 
        # P1 是 Button/SB (行动: Preflop先, Postflop后)
        # P0 是 BB (行动: Preflop后, Postflop先)
        # Blinds: P0=100(BB), P1=50(SB)
        blinds_str = "100 50" 
        first_player_str = "2 1 1 1"  # Preflop: P1(SB), Postflop: P0(BB)
    else:
        # 多人场 (e.g. 6-max):
        # P0 是 SB
        # P1 是 BB
        # P2 是 UTG
        # ...
        # P(N-1) 是 Button
        # Blinds: P0=50(SB), P1=100(BB), Others=0
        blinds_list = ["50", "100"] + ["0"] * (num_players - 2)
        blinds_str = " ".join(blinds_list)
        # Preflop: UTG (P2, index 3) acts first
        # Postflop: SB (P0, index 1) acts first
        # 注意: universal_poker 使用 1-indexed，所以 player 2 = 3, player 0 = 1
        first_player_str = " ".join(["3"] + ["1"] * 3)
    
    # stacks_str 保持一致
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
        f"bettingAbstraction={betting_abstraction}"  # 添加参数
        f")"
    )
    
    print(f"  正在加载游戏...")
    print(f"  游戏字符串: {game_string}")
    sys.stdout.flush()
    
    try:
        game = pyspiel.load_game(game_string)
        print(f"  ✓ 游戏创建成功: {game.get_type().short_name}")
        print(f"  ✓ 动作数量: {game.num_distinct_actions()}")
        sys.stdout.flush()
        
        # 验证 tensor 支持
        print(f"  验证 tensor 支持...")
        sys.stdout.flush()
        state = game.new_initial_state()
        tensor = state.information_state_tensor(0)
        print(f"  ✓ 信息状态张量大小: {len(tensor)}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ 游戏创建失败: {e}")
        print(f"  尝试的游戏字符串: {game_string}")
        import traceback
        traceback.print_exc()
        raise
    
    # 创建求解器
    print(f"\n[2/4] 创建 DeepCFR 求解器...")
    print(f"  策略网络: {policy_layers}")
    print(f"  优势网络: {advantage_layers}")
    print(f"  迭代次数: {num_iterations}")
    print(f"  每次迭代遍历次数: {num_traversals}")
    sys.stdout.flush()
    
    print(f"  正在创建求解器（这可能需要几秒钟）...")
    sys.stdout.flush()
    
    try:
        if use_feature_transform:
            if use_simple_feature:
                # 使用简单版本：直接拼接7维手动特征
                deep_cfr_solver = DeepCFRSimpleFeature(
                    game,
                    policy_network_layers=policy_layers,
                    advantage_network_layers=advantage_layers,
                    num_iterations=num_iterations,
                    num_traversals=num_traversals,
                    learning_rate=learning_rate,
                    batch_size_advantage=None,
                    batch_size_strategy=None,
                    memory_capacity=memory_capacity,
                    device=device,
                )
                print("  ✓ DeepCFR Simple Feature 求解器创建成功（简单版本）")
                print(f"  ✓ 原始信息状态大小: {deep_cfr_solver._embedding_size}")
                print(f"  ✓ 手动特征维度: 7 (直接拼接)")
                print(f"  ✓ MLP输入维度: {deep_cfr_solver._embedding_size + 7}")
                print(f"  ✓ 说明: 信息状态({deep_cfr_solver._embedding_size}维) + 手动特征(7维) -> MLP")
            else:
                # 使用复杂版本：特征转换层
                deep_cfr_solver = DeepCFRWithFeatureTransform(
                    game,
                    policy_network_layers=policy_layers,
                    advantage_network_layers=advantage_layers,
                    transformed_size=transformed_size,
                    use_hybrid_transform=use_hybrid_transform,
                    num_iterations=num_iterations,
                    num_traversals=num_traversals,
                    learning_rate=learning_rate,
                    batch_size_advantage=None,
                    batch_size_strategy=None,
                    memory_capacity=memory_capacity,
                    device=device,
                )
                print("  ✓ DeepCFR with Feature Transform 求解器创建成功（复杂版本）")
                print(f"  ✓ 原始信息状态大小: {deep_cfr_solver._embedding_size} (自动检测，保留)")
                print(f"  ✓ 手动特征维度: 7 (新增)")
                print(f"  ✓ 可学习特征维度: 64 (新增)")
                combined_size = deep_cfr_solver._embedding_size + 7 + 64
                print(f"  ✓ 合并后维度: {combined_size} (原始{deep_cfr_solver._embedding_size} + 手动7 + 可学习64)")
                print(f"  ✓ 最终输出维度: {transformed_size}")
                print(f"  ✓ 使用混合特征转换: {use_hybrid_transform}")
        else:
            # 使用标准 DeepCFR
            deep_cfr_solver = deep_cfr.DeepCFRSolver(
                game,
                policy_network_layers=policy_layers,
                advantage_network_layers=advantage_layers,
                num_iterations=num_iterations,
                num_traversals=num_traversals,
                learning_rate=learning_rate,
                batch_size_advantage=None,
                batch_size_strategy=None,
                memory_capacity=memory_capacity,
                device=device,
            )
            print("  ✓ DeepCFR 求解器创建成功（标准版本）")
        if device.type == "cuda":
            print(f"  ✓ 模型已移到 GPU: {device}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ 求解器创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 训练
    print(f"\n[3/4] 开始训练...")
    print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  配置: {num_iterations} 次迭代, 每次 {num_traversals} 次遍历")
    print(f"  设备: {device}")
    start_time = time.time()
    
    # 训练历史记录
    training_history = {
        'config': {
            'num_players': num_players,
            'num_iterations': num_iterations,
            'num_traversals': num_traversals,
            'policy_layers': list(policy_layers),
            'advantage_layers': list(advantage_layers),
            'learning_rate': learning_rate,
            'memory_capacity': memory_capacity,
            'device': str(device),
            'use_feature_transform': use_feature_transform,
            'use_simple_feature': use_simple_feature if use_feature_transform else None,
            'transformed_size': transformed_size if (use_feature_transform and not use_simple_feature) else None,
            'use_hybrid_transform': use_hybrid_transform if (use_feature_transform and not use_simple_feature) else None,
            'betting_abstraction': betting_abstraction,
        },
        'iterations': [],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    try:
        advantage_losses = {}
        policy_loss = None
        
        # 手动运行迭代以显示进度
        for iteration in range(num_iterations):
            iter_start = time.time()
            print(f"\n  迭代 {iteration + 1}/{num_iterations}...", end="", flush=True)
            
            for player in range(game.num_players()):
                for _ in range(num_traversals):
                    deep_cfr_solver._traverse_game_tree(deep_cfr_solver._root_node, player)
                
                if deep_cfr_solver._reinitialize_advantage_networks:
                    deep_cfr_solver.reinitialize_advantage_network(player)
                
                loss = deep_cfr_solver._learn_advantage_network(player)
                if player not in advantage_losses:
                    advantage_losses[player] = []
                if loss is not None:
                    advantage_losses[player].append(loss)
            
            deep_cfr_solver._iteration += 1
            
            iter_time = time.time() - iter_start
            print(f" 完成 (耗时: {iter_time:.2f}秒)", end="", flush=True)
            
            # 每 eval_interval 次迭代进行评估和打印详细信息
            if (iteration + 1) % eval_interval == 0 or iteration == num_iterations - 1:
                if advantage_losses:
                    for player, losses in advantage_losses.items():
                        if losses and losses[-1] is not None:
                            print(f" | 玩家{player}损失: {losses[-1]:.6f}", end="")
                if device.type == "cuda":
                    gpu_memory = torch.cuda.memory_allocated(0) / 1e9
                    print(f" | GPU内存: {gpu_memory:.2f}GB", end="")
                print()
                
                # 进行评估（轻量级，不计算 NashConv）
                # 注意：评估应该总是运行（如果可用），NashConv是可选的
                try:
                    from training_evaluator import quick_evaluate, print_evaluation_summary
                    print(f"\n  评估训练效果（迭代 {iteration + 1}）...", end="", flush=True)
                    eval_result = quick_evaluate(
                        game,
                        deep_cfr_solver,
                        include_test_games=eval_with_games,
                        num_test_games=50,
                        max_depth=None,  # 自动设置（基于游戏特性）
                        verbose=False
                    )
                    print(" 完成")
                    
                    # 打印简要评估信息
                    metrics = eval_result['metrics']
                    print(f"  策略熵: {metrics.get('avg_entropy', 0):.4f} | "
                          f"策略缓冲区: {metrics.get('strategy_buffer_size', 0):,} | "
                          f"优势样本: {metrics.get('total_advantage_samples', 0):,}")
                    
                    if eval_result.get('test_results'):
                        test = eval_result['test_results']
                        print(f"  测试对局: 玩家0平均收益={test.get('player0_avg_return', 0):.4f}, "
                              f"胜率={test.get('player0_win_rate', 0)*100:.1f}%")
                    
                    # 记录评估结果到历史
                    if save_history:
                        iteration_record = {
                            'iteration': iteration + 1,
                            'time_elapsed': float(time.time() - start_time),
                            'advantage_losses': {str(p): float(losses[-1]) if losses and losses[-1] is not None else None 
                                                 for p, losses in advantage_losses.items()},
                            'metrics': {
                                'avg_entropy': float(metrics.get('avg_entropy', 0)),
                                'strategy_buffer_size': int(metrics.get('strategy_buffer_size', 0)),
                                'total_advantage_samples': int(metrics.get('total_advantage_samples', 0)),
                            }
                        }
                        if eval_result.get('test_results'):
                            test = eval_result['test_results']
                            iteration_record['test_results'] = {
                                'player0_avg_return': float(test.get('player0_avg_return', 0)),
                                'player0_win_rate': float(test.get('player0_win_rate', 0)),
                            }
                        training_history['iterations'].append(iteration_record)
                except ImportError:
                    # 如果评估模块不存在，跳过
                    pass
                except Exception as e:
                    # 评估失败不影响训练
                    print(f"  ⚠️ 评估失败: {e}")
        
        # 训练策略网络
        print("  训练策略网络...", end="", flush=True)
        policy_loss = deep_cfr_solver._learn_strategy_network()
        print(" 完成")
        
        total_time = time.time() - start_time
        print(f"\n  ✓ 训练完成！")
        print(f"  ✓ 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        
        # 记录最终结果
        if save_history:
            training_history['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            training_history['total_time'] = total_time
            training_history['final_losses'] = {}
            if advantage_losses:
                for player, losses in advantage_losses.items():
                    if losses:
                        training_history['final_losses'][f'player_{player}_advantage'] = {
                            'initial': losses[0] if losses[0] is not None else None,
                            'final': losses[-1] if losses[-1] is not None else None,
                        }
            if policy_loss is not None:
                training_history['final_losses']['policy'] = policy_loss
        
        # 打印损失
        if advantage_losses:
            for player, losses in advantage_losses.items():
                if losses and losses[-1] is not None:
                    print(f"  玩家 {player} 优势损失: {losses[0]:.6f} -> {losses[-1]:.6f}")
        
        if policy_loss is not None:
            print(f"  策略损失: {policy_loss:.6f}")
            
    except KeyboardInterrupt:
        print("\n训练被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 保存模型
    print(f"\n[4/4] 保存模型...")
    try:
        # 创建保存目录
        model_dir = create_save_directory(save_prefix, save_dir)
        print(f"  保存目录: {model_dir}")
        
        # DeepCFRSolver 使用 _policy_network (单数) 和 _advantage_networks (复数)
        # 策略网络是所有玩家共享的
        policy_path = os.path.join(model_dir, f"{save_prefix}_policy_network.pt")
        torch.save(
            deep_cfr_solver._policy_network.state_dict(),
            policy_path
        )
        print(f"  ✓ 策略网络已保存: {policy_path}")
        
        # 优势网络是每个玩家一个
        for player in range(game.num_players()):
            advantage_path = os.path.join(model_dir, f"{save_prefix}_advantage_player_{player}.pt")
            torch.save(
                deep_cfr_solver._advantage_networks[player].state_dict(),
                advantage_path
            )
            print(f"  ✓ 玩家 {player} 优势网络已保存: {advantage_path}")
        
        # 计算 NashConv（可选，带资源限制）
        if not skip_nashconv:
            print(f"\n计算 NashConv（这可能需要较长时间）...")
            print(f"  注意: 策略查询已在 GPU 上，游戏树遍历在 C++ 层面")
            print(f"  ⚠️ 警告: 对于大规模游戏，可能消耗大量 CPU 和内存")
            print(f"  如果系统资源不足，建议使用 --skip_nashconv 跳过")
            try:
                # 使用 GPU 加速的 NashConv 计算（带资源限制）
                # 限制资源使用，避免系统卡死
                from nash_conv_gpu import nash_conv_lightweight
                conv = nash_conv_lightweight(
                    game, 
                    deep_cfr_solver,
                    max_cpu_threads=2,  # 限制 CPU 线程数
                    max_memory_gb=8,    # 限制内存使用（8GB）
                    timeout_seconds=600,  # 10 分钟超时
                    verbose=True,
                    device=device
                )
            except ImportError:
                # 回退到标准方法（不推荐，可能消耗大量资源）
                print(f"  ⚠️ 警告: 使用标准方法，可能消耗大量资源")
                print(f"  建议: 使用 --skip_nashconv 跳过，或安装 nash_conv_gpu 模块")
                try:
                    average_policy = policy.tabular_policy_from_callable(
                        game, deep_cfr_solver.action_probabilities
                    )
                    pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
                    conv = pyspiel.nash_conv(game, pyspiel_policy, use_cpp_br=True)
                    print(f"  ✓ NashConv: {conv:.6f}")
                except MemoryError:
                    print(f"  ✗ 内存不足，无法计算 NashConv")
                    print(f"  建议: 使用 --skip_nashconv 跳过")
                except Exception as e:
                    print(f"  ✗ NashConv 计算失败: {e}")
                    print(f"  建议: 使用 --skip_nashconv 跳过")
            except (TimeoutError, MemoryError) as e:
                print(f"\n  ⚠️ NashConv 计算失败: {e}")
                print(f"  建议: 使用 --skip_nashconv 跳过计算")
            except KeyboardInterrupt:
                print(f"\n  ⚠️ NashConv 计算被用户中断")
            except Exception as e:
                print(f"  ⚠️ NashConv 计算失败: {e}")
                print(f"  建议: 使用 --skip_nashconv 跳过计算")
        else:
            print(f"\n  ⏭️ 跳过 NashConv 计算（推荐用于大规模训练）")
        
        # 保存训练历史
        if save_history:
            history_path = os.path.join(model_dir, f"{save_prefix}_training_history.json")
            try:
                # 确保所有数据都是 JSON 可序列化的
                serializable_history = json_serialize(training_history)
                with open(history_path, 'w') as f:
                    json.dump(serializable_history, f, indent=2)
                print(f"\n  ✓ 训练历史已保存: {history_path}")
            except Exception as e:
                print(f"  ⚠️ 训练历史保存失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存训练配置信息
        config_path = os.path.join(model_dir, "config.json")
        try:
            config_info = {
                'save_prefix': save_prefix,
                'num_players': num_players,
                'num_iterations': num_iterations,
                'num_traversals': num_traversals,
                'policy_layers': list(policy_layers),
                'advantage_layers': list(advantage_layers),
                'learning_rate': learning_rate,
                'memory_capacity': memory_capacity,
                'device': str(device),
                'use_feature_transform': use_feature_transform,
                'use_simple_feature': use_simple_feature if use_feature_transform else None,
                'transformed_size': transformed_size if (use_feature_transform and not use_simple_feature) else None,
                'use_hybrid_transform': use_hybrid_transform if (use_feature_transform and not use_simple_feature) else None,
                'betting_abstraction': betting_abstraction,
                'game_string': game_string,
                'training_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(config_path, 'w') as f:
                json.dump(config_info, f, indent=2)
            print(f"  ✓ 训练配置已保存: {config_path}")
        except Exception as e:
            print(f"  ⚠️ 配置保存失败: {e}")
        
    except Exception as e:
        print(f"  ⚠️ 模型保存或评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✓ DeepCFR 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 DeepCFR 训练德州扑克策略")
    parser.add_argument("--num_players", type=int, default=2, help="玩家数量")
    parser.add_argument("--num_iterations", type=int, default=10, help="迭代次数")
    parser.add_argument("--num_traversals", type=int, default=20, help="每次迭代遍历次数")
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[64, 64], help="策略网络层大小")
    parser.add_argument("--advantage_layers", type=int, nargs="+", default=[32, 32], help="优势网络层大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--memory_capacity", type=int, default=int(1e6), help="内存容量")
    parser.add_argument("--save_prefix", type=str, default="deepcfr_texas", help="保存文件前缀")
    parser.add_argument("--save_dir", type=str, default="models", help="模型保存目录（默认: models）")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="使用 GPU")
    parser.add_argument("--skip_nashconv", action="store_true", help="跳过 NashConv 计算")
    parser.add_argument("--eval_interval", type=int, default=10, help="每 N 次迭代进行一次评估")
    parser.add_argument("--eval_with_games", action="store_true", help="评估时包含测试对局")
    parser.add_argument("--save_history", action="store_true", default=True, help="保存训练历史到JSON文件")
    parser.add_argument("--use_feature_transform", action="store_true", default=True, help="使用特征转换层（默认启用）")
    parser.add_argument("--no_feature_transform", dest="use_feature_transform", action="store_false", help="不使用特征转换层")
    parser.add_argument("--use_simple_feature", action="store_true", default=True, help="使用简单版本（直接拼接7维特征，默认启用，推荐）")
    parser.add_argument("--no_simple_feature", dest="use_simple_feature", action="store_false", help="不使用简单版本（使用复杂特征转换层）")
    parser.add_argument("--transformed_size", type=int, default=150, help="转换后的特征大小（仅用于复杂版本，默认150）")
    parser.add_argument("--use_hybrid_transform", action="store_true", default=True, help="使用混合特征转换（仅用于复杂版本，默认启用）")
    parser.add_argument("--no_hybrid_transform", dest="use_hybrid_transform", action="store_false", help="不使用混合特征转换（仅用于复杂版本）")
    parser.add_argument("--betting_abstraction", type=str, default="fcpa", help="下注抽象: fcpa (默认), fchpa (含半池), fc, fullgame")
    
    args = parser.parse_args()
    
    train_deep_cfr(
        num_players=args.num_players,
        num_iterations=args.num_iterations,
        num_traversals=args.num_traversals,
        policy_layers=tuple(args.policy_layers),
        advantage_layers=tuple(args.advantage_layers),
        learning_rate=args.learning_rate,
        memory_capacity=args.memory_capacity,
        save_prefix=args.save_prefix,
        save_dir=args.save_dir,
        use_gpu=args.use_gpu,
        skip_nashconv=args.skip_nashconv,
        eval_interval=args.eval_interval,
        eval_with_games=args.eval_with_games,
        save_history=args.save_history,
        use_feature_transform=args.use_feature_transform,
        use_simple_feature=args.use_simple_feature,
        transformed_size=args.transformed_size,
        use_hybrid_transform=args.use_hybrid_transform,
        betting_abstraction=args.betting_abstraction,
    )
