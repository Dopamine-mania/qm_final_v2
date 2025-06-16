#!/usr/bin/env python3
"""
03 - 模型适配器测试
测试各种AI模型的硬件适配能力和内存管理
"""

import torch
import psutil
import GPUtil
from datetime import datetime
from pathlib import Path
import json
import time
import gc
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HardwareMonitor:
    """硬件资源监控器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        if not self.gpu_available:
            return {"available": False}
        
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        if gpu:
            return {
                "available": True,
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_used": gpu.memoryUsed,
                "memory_free": gpu.memoryFree,
                "utilization": gpu.load * 100
            }
        return {"available": False}
    
    def get_memory_info(self) -> Dict:
        """获取系统内存信息"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / (1024**3),  # GB
            "used": mem.used / (1024**3),
            "available": mem.available / (1024**3),
            "percent": mem.percent
        }
    
    def get_current_usage(self) -> Dict:
        """获取当前资源使用情况"""
        return {
            "gpu": self.get_gpu_info(),
            "memory": self.get_memory_info(),
            "timestamp": datetime.now().isoformat()
        }

class ModelAdapter:
    """模型适配器基类"""
    
    def __init__(self, model_name: str, model_size_gb: float):
        self.model_name = model_name
        self.model_size_gb = model_size_gb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = HardwareMonitor()
        
    def check_compatibility(self) -> Tuple[bool, str]:
        """检查模型兼容性"""
        gpu_info = self.monitor.get_gpu_info()
        mem_info = self.monitor.get_memory_info()
        
        # 检查GPU内存
        if gpu_info["available"]:
            gpu_free_gb = gpu_info["memory_free"] / 1024
            if gpu_free_gb >= self.model_size_gb:
                return True, f"GPU模式可用 ({gpu_free_gb:.1f}GB可用)"
            
        # 检查系统内存
        if mem_info["available"] >= self.model_size_gb * 2:  # 需要2倍内存
            return True, f"CPU模式可用 ({mem_info['available']:.1f}GB可用)"
        
        return False, "内存不足"
    
    def simulate_loading(self) -> Dict:
        """模拟模型加载"""
        print(f"\n📊 测试 {self.model_name}")
        print("-" * 40)
        
        # 检查兼容性
        compatible, message = self.check_compatibility()
        print(f"兼容性: {'✅' if compatible else '❌'} {message}")
        
        if not compatible:
            return {
                "model": self.model_name,
                "status": "failed",
                "reason": message
            }
        
        # 模拟加载过程
        start_time = time.time()
        start_usage = self.monitor.get_current_usage()
        
        # 创建模拟张量
        try:
            # 模拟占用内存 (使用较小的张量避免真的耗尽内存)
            tensor_size = min(int(self.model_size_gb * 100), 1000)  # MB
            dummy_tensors = []
            
            for i in range(5):
                tensor = torch.randn(tensor_size, 1024, 1024, device=self.device)
                dummy_tensors.append(tensor)
                time.sleep(0.1)
            
            # 获取峰值使用
            peak_usage = self.monitor.get_current_usage()
            
            # 清理
            del dummy_tensors
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            load_time = time.time() - start_time
            
            print(f"✅ 加载成功")
            print(f"  设备: {self.device}")
            print(f"  加载时间: {load_time:.2f}秒")
            
            return {
                "model": self.model_name,
                "status": "success",
                "device": str(self.device),
                "load_time": load_time,
                "memory_usage": {
                    "start": start_usage,
                    "peak": peak_usage
                }
            }
            
        except Exception as e:
            print(f"❌ 加载失败: {str(e)}")
            return {
                "model": self.model_name,
                "status": "failed",
                "reason": str(e)
            }

class AdapterTester:
    """适配器测试器"""
    
    def __init__(self):
        self.monitor = HardwareMonitor()
        self.results = []
        
    def run_tests(self):
        """运行所有测试"""
        print("《心境流转》模型适配器测试")
        print("=" * 50)
        print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 显示系统信息
        self.show_system_info()
        
        # 定义测试模型
        test_models = [
            ("EmotionNet-Small", 0.5),    # 0.5GB
            ("EmotionNet-Base", 2.0),      # 2GB
            ("MusicGen-Small", 4.0),       # 4GB
            ("MusicGen-Large", 8.0),       # 8GB
            ("VideoGen-Base", 10.0),       # 10GB
            ("VideoGen-Pro", 20.0),        # 20GB
            ("MultiModal-Fusion", 40.0)    # 40GB
        ]
        
        # 测试每个模型
        for model_name, size_gb in test_models:
            adapter = ModelAdapter(model_name, size_gb)
            result = adapter.simulate_loading()
            self.results.append(result)
            
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)
        
        # 生成报告
        self.generate_report()
        
    def show_system_info(self):
        """显示系统信息"""
        print("\n🖥️ 系统配置")
        print("-" * 40)
        
        # GPU信息
        gpu_info = self.monitor.get_gpu_info()
        if gpu_info["available"]:
            print(f"GPU: {gpu_info['name']}")
            print(f"显存: {gpu_info['memory_total']/1024:.1f}GB")
            print(f"可用: {gpu_info['memory_free']/1024:.1f}GB")
        else:
            print("GPU: 不可用")
        
        # 内存信息
        mem_info = self.monitor.get_memory_info()
        print(f"内存: {mem_info['total']:.1f}GB")
        print(f"可用: {mem_info['available']:.1f}GB")
        
    def generate_report(self):
        """生成测试报告"""
        print("\n📊 测试报告")
        print("=" * 50)
        
        # 统计结果
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        total_count = len(self.results)
        
        print(f"总测试: {total_count}")
        print(f"成功: {success_count}")
        print(f"失败: {total_count - success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        # 推荐配置
        print("\n💡 推荐配置")
        print("-" * 40)
        
        gpu_info = self.monitor.get_gpu_info()
        if gpu_info["available"]:
            gpu_memory_gb = gpu_info['memory_total'] / 1024
            
            if gpu_memory_gb >= 40:
                print("✅ 旗舰配置: 可运行所有模型")
                print("  - 支持MultiModal-Fusion")
                print("  - 支持并行推理")
            elif gpu_memory_gb >= 20:
                print("✅ 专业配置: 可运行大部分模型")
                print("  - 支持VideoGen-Base")
                print("  - 需要显存优化")
            elif gpu_memory_gb >= 8:
                print("⚠️ 标准配置: 可运行基础模型")
                print("  - 支持MusicGen-Small")
                print("  - 建议使用量化")
            else:
                print("❌ 入门配置: 仅支持小模型")
                print("  - 仅EmotionNet系列")
                print("  - 强烈建议升级")
        
        # 保存结果
        self.save_results()
        
    def save_results(self):
        """保存测试结果"""
        output_dir = Path("outputs/validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "gpu": self.monitor.get_gpu_info(),
                "memory": self.monitor.get_memory_info()
            },
            "test_results": self.results,
            "summary": {
                "total": len(self.results),
                "success": sum(1 for r in self.results if r['status'] == 'success'),
                "failed": sum(1 for r in self.results if r['status'] == 'failed')
            }
        }
        
        output_file = output_dir / "adapter_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存: {output_file}")

def main():
    """主函数"""
    try:
        tester = AdapterTester()
        tester.run_tests()
        
        print("\n" + "=" * 50)
        print("模型适配器测试完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 04_therapy_session_demo.py")
        
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()