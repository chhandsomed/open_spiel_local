#!/usr/bin/env python3
"""分析和对比训练历史数据

支持两种模式：
1. 单模型分析: python analyze_training.py <history.json>
2. 双模型对比: python analyze_training.py <history1.json> --compare <history2.json>
"""

import json
import sys
import argparse

def analyze_single_model(json_path):
    """分析单个模型的训练历史"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {json_path}: {e}")
        return

    config = data.get('config', {})
    iterations = data.get('iterations', [])
    final_losses = data.get('final_losses', {})
    
    print("=" * 80)
    print(f"单模型分析: {json_path}")
    print("=" * 80)
    
    # 1. 训练配置
    print("\n【训练配置】")
    print(f"玩家数量: {config.get('num_players', 'N/A')}")
    print(f"迭代次数: {config.get('num_iterations', 'N/A')}")
    print(f"每次迭代遍历次数: {config.get('num_traversals', 'N/A')}")
    print(f"策略网络层: {config.get('policy_layers', 'N/A')}")
    print(f"优势网络层: {config.get('advantage_layers', 'N/A')}")
    print(f"学习率: {config.get('learning_rate', 'N/A')}")
    print(f"下注抽象: {config.get('betting_abstraction', 'fcpa (默认)')}")
    print(f"使用简单特征: {config.get('use_simple_feature', False)}")
    print(f"游戏字符串: {config.get('game_string', 'N/A')}")
    
    if iterations:
        start_time = data.get('start_time', 'N/A')
        end_time = data.get('end_time', 'N/A')
        total_time = data.get('total_time', 0)
        print(f"\n训练时间: {start_time} -> {end_time}")
        print(f"总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.2f} 分钟)")
    
    # 2. 关键指标趋势
    print("\n【关键指标趋势】")
    key_iterations = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    found_iterations = {}
    
    for iter_data in iterations:
        iter_num = iter_data.get('iteration', 0)
        if iter_num in key_iterations:
            found_iterations[iter_num] = iter_data
    
    print(f"{'迭代':<8} {'策略熵':<12} {'策略缓冲区':<15} {'优势样本':<15} {'玩家0收益':<15} {'玩家0胜率':<12}")
    print("-" * 80)
    
    for iter_num in sorted(found_iterations.keys()):
        iter_data = found_iterations[iter_num]
        metrics = iter_data.get('metrics', {})
        test_results = iter_data.get('test_results', {})
        
        entropy = metrics.get('avg_entropy', 0)
        buffer_size = metrics.get('strategy_buffer_size', 0)
        advantage_samples = metrics.get('total_advantage_samples', 0)
        avg_return = test_results.get('player0_avg_return', 0)
        win_rate = test_results.get('player0_win_rate', 0)
        
        print(f"{iter_num:<8} {entropy:<12.4f} {buffer_size:<15,} {advantage_samples:<15,} {avg_return:<15.2f} {win_rate*100:<12.1f}%")
    
    # 3. 优势损失分析
    print("\n【优势损失分析】")
    if iterations:
        first_iter = iterations[0]
        last_iter = iterations[-1]
        
        first_losses = first_iter.get('advantage_losses', {})
        last_losses = last_iter.get('advantage_losses', {})
        
        print(f"{'玩家':<10} {'初始损失':<20} {'最终损失':<20} {'增长倍数':<15}")
        print("-" * 65)
        for player in sorted(first_losses.keys(), key=int):
            p_init = first_losses.get(player, 0)
            p_final = last_losses.get(player, 0)
            ratio = p_final / p_init if p_init > 0 else 0
            print(f"玩家 {player:<5} {p_init:>18,.0f} {p_final:>18,.0f} {ratio:>13.2f}x")

    # 4. 训练效果评估
    print("\n【训练效果评估】")
    if iterations:
        entropies = [i.get('metrics', {}).get('avg_entropy', 0) for i in iterations]
        print(f"策略熵: 初始 {entropies[0]:.4f} -> 最终 {entropies[-1]:.4f} (变化 {entropies[-1]-entropies[0]:.4f})")
        
        test_returns = [i.get('test_results', {}).get('player0_avg_return', 0) for i in iterations if 'test_results' in i]
        if test_returns:
            print(f"玩家0收益: 初始 {test_returns[0]:.2f} -> 最终 {test_returns[-1]:.2f} (平均 {sum(test_returns)/len(test_returns):.2f})")

    # 5. 结论
    print("\n【结论】")
    if iterations:
        last_entropy = iterations[-1].get('metrics', {}).get('avg_entropy', 0)
        if last_entropy > 0.8:
            print("1. 策略仍然较为随机，探索充分但可能未收敛")
        elif last_entropy > 0.5:
            print("1. 策略开始收敛，但仍有探索空间")
        else:
            print("1. 策略已高度确定")


def compare_two_models(path1, path2):
    """对比两个模型的训练历史"""
    try:
        with open(path1, 'r') as f: data1 = json.load(f)
        with open(path2, 'r') as f: data2 = json.load(f)
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        return

    config1 = data1.get('config', {})
    config2 = data2.get('config', {})
    iter1 = data1.get('iterations', [])
    iter2 = data2.get('iterations', [])
    
    print("=" * 100)
    print(f"模型对比分析")
    print(f"模型1: {path1}")
    print(f"模型2: {path2}")
    print("=" * 100)
    
    # 1. 基本信息对比
    print("\n【基本配置对比】")
    print(f"{'指标':<25} {'模型1':<35} {'模型2':<35}")
    print("-" * 100)
    print(f"{'迭代次数':<25} {config1.get('num_iterations', 'N/A'):<35} {config2.get('num_iterations', 'N/A'):<35}")
    print(f"{'每次遍历次数':<25} {config1.get('num_traversals', 'N/A'):<35} {config2.get('num_traversals', 'N/A'):<35}")
    print(f"{'学习率':<25} {config1.get('learning_rate', 'N/A'):<35} {config2.get('learning_rate', 'N/A'):<35}")
    print(f"{'网络层结构':<25} {str(config1.get('policy_layers', 'N/A')):<35} {str(config2.get('policy_layers', 'N/A')):<35}")
    print(f"{'下注抽象':<25} {config1.get('betting_abstraction', 'fcpa'):<35} {config2.get('betting_abstraction', 'fcpa'):<35}")
    
    t1 = data1.get('total_time', 0)
    t2 = data2.get('total_time', 0)
    print(f"{'总训练时间':<25} {t1/60:.1f}分 ({t1/3600:.1f}时):<35} {t2/60:.1f}分 ({t2/3600:.1f}时):<35}")
    
    iter_time1 = t1/max(1, config1.get('num_iterations', 1))
    iter_time2 = t2/max(1, config2.get('num_iterations', 1))
    print(f"{'平均迭代时间':<25} {iter_time1:.2f}秒:<35} {iter_time2:.2f}秒:<35}")

    # 2. 最终状态对比
    print("\n【最终状态对比】")
    if iter1 and iter2:
        last1 = iter1[-1]
        last2 = iter2[-1]
        
        # 策略熵
        e1 = last1.get('metrics', {}).get('avg_entropy', 0)
        e2 = last2.get('metrics', {}).get('avg_entropy', 0)
        print(f"{'策略熵':<25} {e1:.4f}{'':<29} {e2:.4f}{'':<29} (差值: {e2-e1:+.4f})")
        
        # 缓冲区
        b1 = last1.get('metrics', {}).get('strategy_buffer_size', 0)
        b2 = last2.get('metrics', {}).get('strategy_buffer_size', 0)
        print(f"{'策略缓冲区':<25} {b1:,}{'':<29} {b2:,}{'':<29} (差值: {b2-b1:+,})")
        
        # 优势样本
        s1 = last1.get('metrics', {}).get('total_advantage_samples', 0)
        s2 = last2.get('metrics', {}).get('total_advantage_samples', 0)
        print(f"{'优势样本数':<25} {s1:,}{'':<29} {s2:,}{'':<29} (差值: {s2-s1:+,})")
        
        # 测试结果
        t1_res = last1.get('test_results', {})
        t2_res = last2.get('test_results', {})
        if t1_res and t2_res:
            r1 = t1_res.get('player0_avg_return', 0)
            r2 = t2_res.get('player0_avg_return', 0)
            w1 = t1_res.get('player0_win_rate', 0)
            w2 = t2_res.get('player0_win_rate', 0)
            print(f"{'玩家0收益':<25} {r1:.2f}{'':<29} {r2:.2f}{'':<29} (差值: {r2-r1:+.2f})")
            print(f"{'玩家0胜率':<25} {w1*100:.1f}%{'':<29} {w2*100:.1f}%{'':<29} (差值: {(w2-w1)*100:+.1f}%)")

    # 3. 优势损失对比 (仅显示总增长倍数)
    print("\n【优势损失增长倍数对比】")
    if iter1 and iter2:
        l1_first = iter1[0].get('advantage_losses', {})
        l1_last = iter1[-1].get('advantage_losses', {})
        l2_first = iter2[0].get('advantage_losses', {})
        l2_last = iter2[-1].get('advantage_losses', {})
        
        print(f"{'玩家':<10} {'模型1倍数':<20} {'模型2倍数':<20}")
        print("-" * 50)
        for p in sorted(l1_first.keys(), key=int):
            r1 = l1_last.get(p, 0) / l1_first.get(p, 1) if l1_first.get(p, 0) > 0 else 0
            r2 = l2_last.get(p, 0) / l2_first.get(p, 1) if l2_first.get(p, 0) > 0 else 0
            print(f"玩家 {p:<5} {r1:>10.1f}x{'':<10} {r2:>10.1f}x")
            
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="分析训练历史")
    parser.add_argument("path", help="训练历史 JSON 文件路径 (模型1)")
    parser.add_argument("--compare", help="对比的第二个训练历史 JSON 文件路径 (模型2)", default=None)
    
    args = parser.parse_args()
    
    if args.compare:
        compare_two_models(args.path, args.compare)
    else:
        analyze_single_model(args.path)


if __name__ == "__main__":
    main()
