#!/usr/bin/env python3
"""
端口检查和管理工具
"""

import subprocess
import sys
import os

def check_port(port):
    """检查端口是否被占用"""
    try:
        # 使用lsof检查端口占用 (Linux/Mac)
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            return True, pids
        return False, []
    except FileNotFoundError:
        # 如果lsof不可用，使用netstat (更通用)
        try:
            result = subprocess.run(
                ['netstat', '-tlnp'], 
                capture_output=True, 
                text=True
            )
            if f':{port} ' in result.stdout:
                return True, []
            return False, []
        except:
            return False, []

def kill_port_processes(port):
    """终止占用指定端口的进程"""
    occupied, pids = check_port(port)
    if not occupied:
        print(f"端口 {port} 未被占用")
        return True
    
    if not pids:
        print(f"端口 {port} 被占用，但无法获取进程ID")
        return False
    
    print(f"发现端口 {port} 被以下进程占用: {pids}")
    
    for pid in pids:
        try:
            print(f"正在终止进程 {pid}...")
            subprocess.run(['kill', '-9', pid], check=True)
            print(f"✅ 进程 {pid} 已终止")
        except subprocess.CalledProcessError:
            print(f"❌ 无法终止进程 {pid}")
            return False
    
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='端口检查和管理工具')
    parser.add_argument('--port', '-p', type=int, default=7860, help='要检查的端口号')
    parser.add_argument('--kill', '-k', action='store_true', help='终止占用端口的进程')
    parser.add_argument('--range', '-r', help='检查端口范围 (格式: start-end)')
    
    args = parser.parse_args()
    
    if args.range:
        try:
            start, end = map(int, args.range.split('-'))
            print(f"检查端口范围 {start}-{end}:")
            for port in range(start, end + 1):
                occupied, pids = check_port(port)
                status = "占用" if occupied else "可用"
                pid_info = f" (PID: {','.join(pids)})" if pids else ""
                print(f"  端口 {port}: {status}{pid_info}")
        except ValueError:
            print("❌ 端口范围格式错误，请使用 start-end 格式")
            return
    else:
        port = args.port
        occupied, pids = check_port(port)
        
        if occupied:
            pid_info = f" (PID: {','.join(pids)})" if pids else ""
            print(f"端口 {port} 被占用{pid_info}")
            
            if args.kill:
                if kill_port_processes(port):
                    print(f"✅ 端口 {port} 已释放")
                else:
                    print(f"❌ 无法释放端口 {port}")
        else:
            print(f"端口 {port} 可用")

if __name__ == "__main__":
    main()