#!/usr/bin/env python3
"""批量评估所有 checkpoint，找出最佳模型

使用方法:
    python evaluate_all_checkpoints.py --model_dir models/deepcfr_parallel_6p --num_games 500 --use_gpu
"""

import os
import sys
import argparse
import json
import glob
import re
import subprocess
from pathlib import Path

def find_all_checkpoints(model_dir):
    """查找所有 checkpoint 目录"""
    checkpoints = []
    
    checkpoint_root = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoint_root):
        print(f"  ✗ 未找到 checkpoints 目录: {checkpoint_root}")
        return checkpoints
    
    # 查找所有 iter_* 目录
    iter_dirs = glob.glob(os.path.join(checkpoint_root, "iter_*"))
    for d in iter_dirs:
        match = re.search(r'iter_(\d+)$', d)
        if match:
            iter_num = int(match.group(1))
            # 检查是否有策略网络文件
            policy_files = glob.glob(os.path.join(d, "*_policy_network_iter*.pt"))
            if policy_files:
                checkpoints.append({
                    'iter': iter_num,
                    'dir': d,
                    'path': d
                })
    
    # 按迭代号排序
    checkpoints.sort(key=lambda x: x['iter'])
    return checkpoints


def evaluate_checkpoint(checkpoint_path, num_games=500, use_gpu=True):
    """评估单个 checkpoint"""
    print(f"\n评估 checkpoint: {os.path.basename(checkpoint_path)}")
    
    # 调用 inference_simple.py 进行评估
    cmd = [
        sys.executable, "inference_simple.py",
        "--model_dir", checkpoint_path,
        "--num_games", str(num_games),
    ]
    if use_gpu:
        cmd.append("--use_gpu")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode != 0:
            print(f"  ✗ 评估失败: {result.stderr[:200]}")
            return None
        
        # 解析输出，提取关键指标
        output = result.stdout
        metrics = {}
        
        # 提取平均收益和胜率
        # 格式: "玩家 0: 平均收益: X.XXXX 胜率: XX.X%"
        for line in output.split('\n'):
            if '平均收益:' in line and '玩家 0' in line:
                # 提取平均收益
                match = re.search(r'平均收益:\s*([-\d.]+)', line)
                if match:
                    metrics['player0_avg_return'] = float(match.group(1))
            
            if '胜率:' in line and '玩家 0' in line:
                # 提取胜率
                match = re.search(r'胜率:\s*([\d.]+)%', line)
                if match:
                    metrics['player0_win_rate'] = float(match.group(1))
        
        # 提取所有玩家的收益（用于计算总体表现）
        player_returns = []
        for i in range(6):  # 6人局
            pattern = f'玩家 {i}:.*?平均收益:\s*([-\d.]+)'
            match = re.search(pattern, output)
            if match:
                player_returns.append(float(match.group(1)))
        
        if player_returns:
            metrics['all_players_returns'] = player_returns
            metrics['avg_return_all'] = sum(player_returns) / len(player_returns)
            metrics['max_return'] = max(player_returns)
            metrics['min_return'] = min(player_returns)
            # 计算收益方差（越小越好，说明策略更平衡）
            metrics['return_variance'] = sum((r - metrics['avg_return_all'])**2 for r in player_returns) / len(player_returns)
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ 评估超时")
        return None
    except Exception as e:
        print(f"  ✗ 评估出错: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="批量评估所有 checkpoint")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="模型目录（例如: models/deepcfr_parallel_6p）")
    parser.add_argument("--num_games", type=int, default=500,
                       help="每个 checkpoint 的测试局数（默认: 500）")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="使用 GPU")
    parser.add_argument("--top_k", type=int, default=5,
                       help="显示前 K 个最佳模型（默认: 5）")
    parser.add_argument("--output", type=str, default=None,
                       help="保存结果到 JSON 文件")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("批量评估所有 Checkpoint")
    print("=" * 70)
    print(f"模型目录: {args.model_dir}")
    print(f"测试局数: {args.num_games} 局/checkpoint")
    print(f"使用 GPU: {args.use_gpu}")
    
    # 查找所有 checkpoint
    print(f"\n[1/3] 查找所有 checkpoint...")
    checkpoints = find_all_checkpoints(args.model_dir)
    
    if not checkpoints:
        print("  ✗ 未找到任何 checkpoint")
        return
    
    print(f"  ✓ 找到 {len(checkpoints)} 个 checkpoint")
    print(f"  迭代范围: {checkpoints[0]['iter']} - {checkpoints[-1]['iter']}")
    
    # 评估每个 checkpoint
    print(f"\n[2/3] 评估 checkpoint（这可能需要一些时间）...")
    results = []
    
    for i, ckpt in enumerate(checkpoints):
        print(f"\n进度: {i+1}/{len(checkpoints)}")
        metrics = evaluate_checkpoint(ckpt['path'], args.num_games, args.use_gpu)
        
        if metrics:
            result = {
                'iter': ckpt['iter'],
                'path': ckpt['path'],
                **metrics
            }
            results.append(result)
            print(f"  ✓ 迭代 {ckpt['iter']}: 玩家0平均收益={metrics.get('player0_avg_return', 'N/A'):.2f}, "
                  f"胜率={metrics.get('player0_win_rate', 'N/A'):.1f}%")
        else:
            print(f"  ✗ 迭代 {ckpt['iter']}: 评估失败")
    
    if not results:
        print("\n  ✗ 所有 checkpoint 评估失败")
        return
    
    # 排序和显示结果
    print(f"\n[3/3] 分析结果...")
    
    # 按玩家0平均收益排序（越高越好）
    results_sorted = sorted(results, key=lambda x: x.get('player0_avg_return', -999), reverse=True)
    
    print("\n" + "=" * 70)
    print(f"评估结果（按玩家0平均收益排序，前 {args.top_k} 名）")
    print("=" * 70)
    print(f"{'排名':<6} {'迭代':<8} {'玩家0收益':<12} {'玩家0胜率':<12} {'收益方差':<12} {'路径'}")
    print("-" * 70)
    
    for i, r in enumerate(results_sorted[:args.top_k], 1):
        iter_num = r['iter']
        avg_return = r.get('player0_avg_return', 0)
        win_rate = r.get('player0_win_rate', 0)
        variance = r.get('return_variance', 0)
        path = os.path.basename(r['path'])
        
        print(f"{i:<6} {iter_num:<8} {avg_return:>10.2f}    {win_rate:>9.1f}%    {variance:>10.2f}    {path}")
    
    # 显示最佳模型
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "=" * 70)
        print("🏆 最佳模型")
        print("=" * 70)
        print(f"迭代: {best['iter']}")
        print(f"路径: {best['path']}")
        print(f"玩家0平均收益: {best.get('player0_avg_return', 0):.2f}")
        print(f"玩家0胜率: {best.get('player0_win_rate', 0):.1f}%")
        if 'all_players_returns' in best:
            print(f"所有玩家收益: {[f'{r:.2f}' for r in best['all_players_returns']]}")
            print(f"收益方差: {best.get('return_variance', 0):.2f} (越小越好，说明策略更平衡)")
    
    # 保存结果
    if args.output:
        output_path = args.output
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果已保存到: {output_path}")
    
    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)
    print("\n使用最佳模型进行推理:")
    if results_sorted:
        best_path = results_sorted[0]['path']
        print(f"  python inference_simple.py --model_dir {best_path} --num_games 1000 --use_gpu")


if __name__ == "__main__":
    main()

