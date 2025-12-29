#!/usr/bin/env python3
"""
清理遗留的 multiprocessing 进程

这些进程可能是之前训练时留下的，主进程异常退出后没有被正确清理。
"""

import os
import sys
import signal
import subprocess

def find_zombie_processes():
    """查找遗留的 multiprocessing 进程"""
    try:
        # 查找所有 multiprocessing 相关进程
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'multiprocessing' in line.lower() and 'python' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    cmd = ' '.join(parts[10:])
                        processes.append({
                            'pid': pid,
                            'cmd': cmd,
                            'full_line': line
                        })
        
        return processes
    except Exception as e:
        print(f"查找进程失败: {e}")
        return []

def kill_process(pid, force=False):
    """终止进程"""
    try:
        if force:
            os.kill(int(pid), signal.SIGKILL)
            print(f"  ✓ 强制终止进程 {pid}")
        else:
            os.kill(int(pid), signal.SIGTERM)
            print(f"  ✓ 终止进程 {pid}")
        return True
    except ProcessLookupError:
        print(f"  ⚠️ 进程 {pid} 不存在")
        return False
    except PermissionError:
        print(f"  ✗ 权限不足，无法终止进程 {pid}（需要 root 权限）")
        return False
    except Exception as e:
        print(f"  ✗ 终止进程 {pid} 失败: {e}")
        return False

def main():
    print("=" * 70)
    print("清理遗留的 multiprocessing 进程")
    print("=" * 70)
    
    processes = find_zombie_processes()
    
    if not processes:
        print("\n✓ 没有找到遗留的 multiprocessing 进程")
        return
    
    print(f"\n找到 {len(processes)} 个遗留进程：")
    print()
    
    # 按类型分组
    resource_trackers = []
    spawn_mains = []
    
    for p in processes:
        if 'resource_tracker' in p['cmd']:
            resource_trackers.append(p)
        elif 'spawn_main' in p['cmd']:
            spawn_mains.append(p)
        else:
            print(f"  未知类型: PID {p['pid']} - {p['cmd'][:80]}")
    
    print(f"  Resource Tracker 进程: {len(resource_trackers)} 个")
    print(f"  Spawn Main 进程: {len(spawn_mains)} 个")
        print()
    
    # 询问是否清理
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        force = True
        print("⚠️ 强制模式：将直接终止所有进程")
    elif len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        print("🔍 仅查看模式：不会实际清理进程")
        print("\n要清理这些进程，请运行：")
        print("  python cleanup_zombie_processes.py --force")
        print("或使用 root 权限：")
        print("  sudo python cleanup_zombie_processes.py --force")
        return
    else:
        # 检查是否在交互式终端
        if sys.stdin.isatty():
            response = input("是否清理这些进程？(y/N): ")
            if response.lower() != 'y':
                print("取消清理")
                return
        force = False
        else:
            print("⚠️ 非交互式环境，使用 --force 参数强制清理")
            print("或使用 --dry-run 参数仅查看")
            return
        force = False
    
    print("\n开始清理...")
    print()
    
    # 先终止 spawn_main 进程（worker 进程）
    killed_count = 0
    for p in spawn_mains:
        print(f"终止 Spawn Main 进程 {p['pid']}...")
        if kill_process(p['pid'], force=force):
            killed_count += 1
    
    # 再终止 resource_tracker 进程
    for p in resource_trackers:
        print(f"终止 Resource Tracker 进程 {p['pid']}...")
        if kill_process(p['pid'], force=force):
            killed_count += 1
    
        print()
    print(f"✓ 已清理 {killed_count}/{len(processes)} 个进程")
    
    # 再次检查
    remaining = find_zombie_processes()
        if remaining:
        print(f"\n⚠️ 仍有 {len(remaining)} 个进程未清理")
        print("可能需要 root 权限或使用 --force 参数")
        print("\n使用 root 权限清理：")
        print("  sudo python cleanup_zombie_processes.py --force")
    else:
        print("\n✓ 所有遗留进程已清理完成")

if __name__ == "__main__":
    main()
