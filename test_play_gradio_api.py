#!/usr/bin/env python3
"""测试 play_gradio_api.py 的API调用功能"""

import requests
import json

API_BASE_URL = "http://localhost:8823/api/v1"

def test_api_connection():
    """测试API连接"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✓ API服务器连接成功")
            return True
        else:
            print(f"✗ API服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API服务器连接失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("测试 play_gradio_api.py 的API连接")
    print("=" * 70)
    print()
    
    if test_api_connection():
        print("\n✓ API服务器可用，可以启动 play_gradio_api.py")
        print("\n启动命令:")
        print("  python play_gradio_api.py")
        print("\n或者:")
        print("  gradio play_gradio_api.py")
    else:
        print("\n✗ 请先启动API服务器:")
        print("  python api_server.py --model_dir models/deepcfr_stable_run --host 0.0.0.0 --port 8823 --device cpu")


