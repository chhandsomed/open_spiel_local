#!/usr/bin/env python3
"""分析训练结果"""

import json
import numpy as np
from pathlib import Path
import sys

# 可选：matplotlib用于绘图
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def analyze_training_history(history_path):
    """分析训练历史"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    iterations = history['iterations']
    config = history['config']
    
    print("=" * 70)
    print("训练效果评估报告")
    print("=" * 70)
    
    # 1. 基本信息
    print("\n[1] 训练配置")
    print(f"  玩家数量: {config['num_players']}")
    print(f"  迭代次数: {config['num_iterations']}")
    print(f"  每次遍历: {config['num_traversals']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  总耗时: {history.get('total_time', 0)/60:.2f} 分钟")
    
    # 2. 损失分析
    print("\n[2] 损失分析")
    if iterations:
        first_iter = iterations[0]
        last_iter = iterations[-1]
        
        print(f"  优势损失变化:")
        for player in range(config['num_players']):
            player_str = str(player)
            if player_str in first_iter['advantage_losses']:
                initial = first_iter['advantage_losses'][player_str]
                final = last_iter['advantage_losses'][player_str]
                growth = (final - initial) / initial * 100 if initial > 0 else 0
                print(f"    玩家{player}: {initial:.0f} -> {final:.0f} (增长 {growth:.1f}%)")
        
        print(f"\n  策略损失: {history['final_losses'].get('policy', 'N/A')}")
        print(f"  ⚠️ 注意: 损失增长是正常的（使用了 sqrt(iteration) 加权）")
    
    # 3. 缓冲区增长
    print("\n[3] 探索进度（缓冲区大小）")
    if iterations:
        buffer_sizes = [it['metrics']['strategy_buffer_size'] for it in iterations]
        advantage_samples = [it['metrics']['total_advantage_samples'] for it in iterations]
        
        print(f"  策略缓冲区: {buffer_sizes[0]:,} -> {buffer_sizes[-1]:,} "
              f"(增长 {((buffer_sizes[-1] - buffer_sizes[0]) / buffer_sizes[0] * 100):.1f}%)")
        print(f"  优势样本: {advantage_samples[0]:,} -> {advantage_samples[-1]:,} "
              f"(增长 {((advantage_samples[-1] - advantage_samples[0]) / advantage_samples[0] * 100):.1f}%)")
        
        # 检查增长趋势
        if len(buffer_sizes) >= 3:
            recent_growth = (buffer_sizes[-1] - buffer_sizes[-2]) / buffer_sizes[-2] * 100
            early_growth = (buffer_sizes[1] - buffer_sizes[0]) / buffer_sizes[0] * 100
            if recent_growth < early_growth * 0.5:
                print(f"  ⚠️ 警告: 缓冲区增长放缓（可能接近饱和）")
            else:
                print(f"  ✓ 缓冲区持续增长（探索正常）")
    
    # 4. 策略熵分析
    print("\n[4] 策略熵分析")
    entropies = [it['metrics']['avg_entropy'] for it in iterations]
    if all(e == 0.0 for e in entropies):
        print(f"  ⚠️ 警告: 策略熵始终为0")
        print(f"    可能原因:")
        print(f"    1. 策略过于确定（只有一个动作的概率为1）")
        print(f"    2. 采样失败（无法访问有效状态）")
        print(f"    3. 网络输出异常（需要检查网络输出）")
        print(f"    建议: 检查 action_probabilities 的输出")
    else:
        print(f"  平均熵: {np.mean(entropies):.4f}")
        print(f"  熵范围: [{np.min(entropies):.4f}, {np.max(entropies):.4f}]")
        if entropies[-1] < entropies[0] * 0.5:
            print(f"  ✓ 熵逐渐降低（策略在收敛）")
        else:
            print(f"  ⚠️ 熵变化不大（策略可能未充分学习）")
    
    # 5. 测试对局结果
    print("\n[5] 测试对局结果（vs 随机策略）")
    if iterations and 'test_results' in iterations[0]:
        returns = [it['test_results']['player0_avg_return'] for it in iterations]
        win_rates = [it['test_results']['player0_win_rate'] * 100 for it in iterations]
        
        print(f"  平均收益: {returns[0]:.2f} -> {returns[-1]:.2f}")
        print(f"  胜率: {win_rates[0]:.1f}% -> {win_rates[-1]:.1f}%")
        
        # 分析趋势
        if len(returns) >= 3:
            recent_avg = np.mean(returns[-3:])
            early_avg = np.mean(returns[:3])
            improvement = recent_avg - early_avg
            
            if improvement > 0:
                print(f"  ✓ 收益改善: {improvement:.2f} (从 {early_avg:.2f} 到 {recent_avg:.2f})")
            else:
                print(f"  ⚠️ 收益下降: {improvement:.2f} (从 {early_avg:.2f} 到 {recent_avg:.2f})")
            
            if win_rates[-1] > 50:
                print(f"  ✓ 胜率超过50%（策略优于随机）")
            elif win_rates[-1] > win_rates[0]:
                print(f"  ⚠️ 胜率有所提高但仍低于50%")
            else:
                print(f"  ⚠️ 胜率未提高（策略可能未学习到有效策略）")
        
        # 计算收益波动
        if len(returns) > 1:
            std = np.std(returns)
            print(f"  收益波动（标准差）: {std:.2f}")
            if std > abs(np.mean(returns)) * 0.5:
                print(f"  ⚠️ 警告: 收益波动较大（策略可能不稳定）")
    
    # 6. 综合评估
    print("\n[6] 综合评估")
    issues = []
    positives = []
    
    # 检查缓冲区增长
    if iterations:
        buffer_growth = (buffer_sizes[-1] - buffer_sizes[0]) / buffer_sizes[0] * 100
        if buffer_growth > 500:
            positives.append("缓冲区大幅增长（探索充分）")
        elif buffer_growth < 100:
            issues.append("缓冲区增长缓慢（可能探索不足）")
    
    # 检查策略熵
    if all(e == 0.0 for e in entropies):
        issues.append("策略熵为0（需要检查策略输出）")
    
    # 检查测试对局
    if iterations and 'test_results' in iterations[0]:
        final_return = iterations[-1]['test_results']['player0_avg_return']
        final_win_rate = iterations[-1]['test_results']['player0_win_rate']
        
        if final_return > 0:
            positives.append("测试对局收益为正")
        else:
            issues.append("测试对局收益为负")
        
        if final_win_rate > 0.5:
            positives.append("胜率超过50%")
        elif final_win_rate > 0.3:
            positives.append("胜率有所提高")
        else:
            issues.append("胜率仍然很低")
    
    if positives:
        print("  ✓ 积极方面:")
        for p in positives:
            print(f"    - {p}")
    
    if issues:
        print("  ⚠️ 需要关注:")
        for i in issues:
            print(f"    - {i}")
    
    if not issues and positives:
        print("  ✓ 训练效果良好！")
    elif len(positives) > len(issues):
        print("  ⚠️ 训练基本正常，但有一些问题需要关注")
    else:
        print("  ⚠️ 训练可能存在问题，建议检查")
    
    # 7. 建议
    print("\n[7] 改进建议")
    if all(e == 0.0 for e in entropies):
        print("  1. 检查策略网络输出，确保概率分布合理")
        print("  2. 增加训练迭代次数")
        print("  3. 调整学习率或网络结构")
    
    if iterations and 'test_results' in iterations[0]:
        final_return = iterations[-1]['test_results']['player0_avg_return']
        if final_return < 0:
            print("  1. 增加训练迭代次数（策略可能还未充分学习）")
            print("  2. 增加每次迭代的遍历次数（num_traversals）")
            print("  3. 调整网络结构或学习率")
    
    if iterations:
        buffer_growth = (buffer_sizes[-1] - buffer_sizes[0]) / buffer_sizes[0] * 100
        if buffer_growth < 200:
            print("  1. 增加遍历次数以探索更多状态")
            print("  2. 检查游戏配置是否正确")
    
    print("\n" + "=" * 70)
    
    # 生成图表
    if HAS_MATPLOTLIB:
        try:
            plot_training_curves(history, history_path)
            print(f"\n✓ 训练曲线图已保存: {Path(history_path).parent / 'training_curves.png'}")
        except Exception as e:
            print(f"\n⚠️ 无法生成图表: {e}")
    else:
        print(f"\n⚠️ matplotlib 未安装，跳过图表生成")

def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    iterations = history['iterations']
    if not iterations:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    iter_nums = [it['iteration'] for it in iterations]
    
    # 1. 优势损失
    ax1 = axes[0, 0]
    for player in range(history['config']['num_players']):
        player_str = str(player)
        losses = [it['advantage_losses'].get(player_str, 0) for it in iterations]
        ax1.plot(iter_nums, losses, label=f'Player {player}', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Advantage Loss')
    ax1.set_title('Advantage Network Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 缓冲区大小
    ax2 = axes[0, 1]
    buffer_sizes = [it['metrics']['strategy_buffer_size'] for it in iterations]
    advantage_samples = [it['metrics']['total_advantage_samples'] for it in iterations]
    ax2.plot(iter_nums, buffer_sizes, label='Strategy Buffer', marker='o', markersize=3)
    ax2.plot(iter_nums, advantage_samples, label='Advantage Samples', marker='s', markersize=3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Buffer Size')
    ax2.set_title('Exploration Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 测试对局收益
    ax3 = axes[1, 0]
    if 'test_results' in iterations[0]:
        returns = [it['test_results']['player0_avg_return'] for it in iterations]
        ax3.plot(iter_nums, returns, label='Avg Return', marker='o', markersize=3, color='green')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Average Return')
        ax3.set_title('Test Game Performance (vs Random)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No test results', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Test Game Performance')
    
    # 4. 胜率
    ax4 = axes[1, 1]
    if 'test_results' in iterations[0]:
        win_rates = [it['test_results']['player0_win_rate'] * 100 for it in iterations]
        ax4.plot(iter_nums, win_rates, label='Win Rate', marker='o', markersize=3, color='blue')
        ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Win Rate (vs Random)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])
    else:
        ax4.text(0.5, 0.5, 'No test results', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Win Rate')
    
    plt.tight_layout()
    output_path = Path(save_path).parent / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认使用最新的训练历史
        import glob
        history_files = glob.glob("models/deepcfr_texas_*/deepcfr_texas_training_history.json")
        if history_files:
            history_path = sorted(history_files)[-1]  # 最新的
            print(f"使用最新的训练历史: {history_path}")
        else:
            print("错误: 未找到训练历史文件")
            print("用法: python analyze_training_results.py <history_path>")
            sys.exit(1)
    else:
        history_path = sys.argv[1]
    
    analyze_training_history(history_path)

