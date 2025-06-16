#!/usr/bin/env python3
"""
批量运行所有测试脚本
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_script(script_name):
    """运行单个脚本"""
    print(f"\n{'='*60}")
    print(f"运行: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {script_name}")
        print(f"错误信息: {e.stderr}")
        return False

def main():
    """主函数"""
    print("《心境流转》批量测试运行器")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取所有测试脚本
    scripts_dir = Path(__file__).parent
    test_scripts = sorted(scripts_dir.glob("[0-9]*.py"))
    
    print(f"\n找到 {len(test_scripts)} 个测试脚本")
    
    # 运行每个脚本
    success_count = 0
    failed_scripts = []
    
    for script in test_scripts:
        if run_script(script):
            success_count += 1
        else:
            failed_scripts.append(script.name)
    
    # 总结
    print(f"\n{'='*60}")
    print("测试运行完成")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{len(test_scripts)}")
    
    if failed_scripts:
        print(f"❌ 失败: {len(failed_scripts)}")
        for script in failed_scripts:
            print(f"  - {script}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()