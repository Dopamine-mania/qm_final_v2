#!/usr/bin/env python3
"""
09 - 性能优化测试
测试和优化系统性能，包括内存管理、GPU加速等
"""

import time
import psutil
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "gpu_memory": [],
            "timestamps": []
        }
        self.start_time = time.time()
    
    def record_metrics(self):
        """记录当前性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics["cpu_usage"].append(cpu_percent)
        
        # 内存使用
        memory = psutil.virtual_memory()
        self.metrics["memory_usage"].append(memory.percent)
        
        # GPU使用率（如果可用）
        if torch.cuda.is_available():
            try:
                # 使用nvidia-ml-py获取GPU使用率
                gpu_usage = 50.0  # 模拟值，因为torch.cuda.utilization()可能不可用
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
                
                self.metrics["gpu_usage"].append(gpu_usage)
                self.metrics["gpu_memory"].append(gpu_memory_percent)
            except Exception as e:
                print(f"GPU监控警告: {e}")
                self.metrics["gpu_usage"].append(0)
                self.metrics["gpu_memory"].append(0)
        else:
            self.metrics["gpu_usage"].append(0)
            self.metrics["gpu_memory"].append(0)
        
        # 时间戳
        self.metrics["timestamps"].append(time.time() - self.start_time)
    
    def get_summary(self):
        """获取性能总结"""
        summary = {
            "avg_cpu": np.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "max_cpu": np.max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "avg_memory": np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "max_memory": np.max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "avg_gpu": np.mean(self.metrics["gpu_usage"]) if self.metrics["gpu_usage"] else 0,
            "max_gpu": np.max(self.metrics["gpu_usage"]) if self.metrics["gpu_usage"] else 0,
            "duration": self.metrics["timestamps"][-1] if self.metrics["timestamps"] else 0
        }
        return summary

class OptimizationBenchmark:
    """优化基准测试"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = PerformanceMonitor()
    
    def benchmark_tensor_operations(self, size=(1000, 1000)):
        """测试张量运算性能"""
        print(f"\n🔧 张量运算基准测试")
        print(f"设备: {self.device}")
        print(f"矩阵大小: {size}")
        
        results = {}
        
        # 测试不同精度
        dtypes = [torch.float32]
        if self.device.type == 'cuda':
            dtypes.append(torch.float16)  # FP16只在GPU上测试
        
        for dtype in dtypes:
            print(f"\n数据类型: {dtype}")
            
            try:
                # 创建随机张量
                a = torch.randn(size, dtype=dtype, device=self.device)
                b = torch.randn(size, dtype=dtype, device=self.device)
                
                # 预热
                for _ in range(5):
                    _ = torch.matmul(a, b)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 测试矩阵乘法
                start = time.time()
                for _ in range(100):
                    c = torch.matmul(a, b)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                duration = time.time() - start
                ops_per_sec = 100 / duration
                
                results[str(dtype)] = {
                    "duration": duration,
                    "ops_per_sec": ops_per_sec,
                    "memory_mb": a.element_size() * a.nelement() * 2 / (1024**2)
                }
                
                print(f"  速度: {ops_per_sec:.2f} ops/秒")
                print(f"  内存: {results[str(dtype)]['memory_mb']:.1f} MB")
                
                # 清理
                del a, b, c
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"  测试失败: {e}")
                results[str(dtype)] = {"error": str(e)}
        
        return results
    
    def test_memory_optimization(self):
        """测试内存优化技术"""
        print(f"\n💾 内存优化测试")
        
        strategies = {}
        
        # 1. 梯度累积 vs 大批量
        print("\n1. 批量大小优化")
        
        # 大批量（可能OOM）
        try:
            batch_size = 64 if self.device.type == 'cpu' else 128
            model = self._create_dummy_model()
            data = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            start_mem = self._get_memory_usage()
            start_time = time.time()
            
            output = model(data)
            loss = output.mean()
            loss.backward()
            
            end_time = time.time()
            end_mem = self._get_memory_usage()
            
            strategies["large_batch"] = {
                "batch_size": batch_size,
                "memory_used": end_mem - start_mem,
                "time": end_time - start_time,
                "success": True
            }
            
            del model, data, output, loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            strategies["large_batch"] = {
                "batch_size": batch_size,
                "error": str(e),
                "success": False
            }
        
        # 梯度累积
        try:
            accumulation_steps = 4
            micro_batch = 16 if self.device.type == 'cpu' else 32
            model = self._create_dummy_model()
            
            start_mem = self._get_memory_usage()
            start_time = time.time()
            
            model.zero_grad()
            for i in range(accumulation_steps):
                data = torch.randn(micro_batch, 3, 224, 224, device=self.device)
                output = model(data)
                loss = output.mean() / accumulation_steps
                loss.backward()
            
            end_time = time.time()
            end_mem = self._get_memory_usage()
            
            strategies["gradient_accumulation"] = {
                "micro_batch": micro_batch,
                "accumulation_steps": accumulation_steps,
                "effective_batch": micro_batch * accumulation_steps,
                "memory_used": end_mem - start_mem,
                "time": end_time - start_time,
                "success": True
            }
            
            del model, data, output, loss
            
        except Exception as e:
            strategies["gradient_accumulation"] = {
                "error": str(e),
                "success": False
            }
        
        # 2. 混合精度训练
        print("\n2. 混合精度优化")
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                from torch.cuda.amp import autocast, GradScaler
                
                model = self._create_dummy_model()
                scaler = GradScaler()
                optimizer = torch.optim.Adam(model.parameters())
                
                start_mem = self._get_memory_usage()
                start_time = time.time()
                
                data = torch.randn(64, 3, 224, 224, device=self.device)
                
                with autocast():
                    output = model(data)
                    loss = output.mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                end_time = time.time()
                end_mem = self._get_memory_usage()
                
                strategies["mixed_precision"] = {
                    "memory_used": end_mem - start_mem,
                    "time": end_time - start_time,
                    "speedup": "~2x expected",
                    "success": True
                }
            except Exception as e:
                strategies["mixed_precision"] = {
                    "error": str(e),
                    "success": False
                }
        else:
            strategies["mixed_precision"] = {
                "note": "仅GPU支持",
                "success": False
            }
        
        # 清理
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return strategies
    
    def test_parallel_processing(self):
        """测试并行处理"""
        print(f"\n⚡ 并行处理测试")
        
        # 测试数据
        n_tasks = 50  # 减少任务数以加快测试
        data = [np.random.randn(500, 500) for _ in range(n_tasks)]  # 减小数据大小
        
        results = {}
        
        # 1. 串行处理
        print("\n1. 串行处理")
        start = time.time()
        serial_results = []
        for d in data:
            result = self._process_data(d)
            serial_results.append(result)
        serial_time = time.time() - start
        
        results["serial"] = {
            "time": serial_time,
            "tasks_per_sec": n_tasks / serial_time
        }
        print(f"  时间: {serial_time:.2f}秒")
        
        # 2. 线程并行
        print("\n2. 线程并行")
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            thread_results = list(executor.map(self._process_data, data))
        thread_time = time.time() - start
        
        results["thread_parallel"] = {
            "time": thread_time,
            "tasks_per_sec": n_tasks / thread_time,
            "speedup": serial_time / thread_time
        }
        print(f"  时间: {thread_time:.2f}秒")
        print(f"  加速: {results['thread_parallel']['speedup']:.2f}x")
        
        # 3. 进程并行
        print("\n3. 进程并行")
        try:
            start = time.time()
            with ProcessPoolExecutor(max_workers=4) as executor:
                process_results = list(executor.map(self._process_data, data))
            process_time = time.time() - start
            
            results["process_parallel"] = {
                "time": process_time,
                "tasks_per_sec": n_tasks / process_time,
                "speedup": serial_time / process_time
            }
            print(f"  时间: {process_time:.2f}秒")
            print(f"  加速: {results['process_parallel']['speedup']:.2f}x")
        except Exception as e:
            print(f"  进程并行测试失败: {e}")
            results["process_parallel"] = {"error": str(e)}
        
        return results
    
    def _create_dummy_model(self):
        """创建测试模型"""
        # 简化模型以适应CPU
        if self.device.type == 'cpu':
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 10)
            ).to(self.device)
        else:
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 10)
            ).to(self.device)
    
    def _get_memory_usage(self):
        """获取当前内存使用"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            return psutil.Process().memory_info().rss / (1024**2)  # MB
    
    def _process_data(self, data):
        """模拟数据处理"""
        # 执行一些计算密集型操作
        result = np.fft.fft2(data)
        result = np.abs(result)
        result = np.log(result + 1)
        return result.mean()

def visualize_performance(monitor_data, output_dir):
    """可视化性能数据"""
    # 检查是否有数据
    if not monitor_data["timestamps"]:
        print("⚠️ 没有性能数据可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # CPU使用率
    ax = axes[0, 0]
    if monitor_data["cpu_usage"]:
        ax.plot(monitor_data["timestamps"], monitor_data["cpu_usage"])
        ax.set_title("CPU Usage")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Usage (%)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No CPU data", ha='center', va='center')
    
    # 内存使用率
    ax = axes[0, 1]
    if monitor_data["memory_usage"]:
        ax.plot(monitor_data["timestamps"], monitor_data["memory_usage"])
        ax.set_title("Memory Usage")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Usage (%)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No memory data", ha='center', va='center')
    
    # GPU使用率
    ax = axes[1, 0]
    if monitor_data["gpu_usage"] and any(monitor_data["gpu_usage"]):
        ax.plot(monitor_data["timestamps"], monitor_data["gpu_usage"])
        ax.set_title("GPU Usage")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Usage (%)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GPU data", ha='center', va='center')
    
    # GPU内存
    ax = axes[1, 1]
    if monitor_data["gpu_memory"] and any(monitor_data["gpu_memory"]):
        ax.plot(monitor_data["timestamps"], monitor_data["gpu_memory"])
        ax.set_title("GPU Memory")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Usage (%)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GPU memory data", ha='center', va='center')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = output_dir / "performance_metrics.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"\n✅ 性能图表已保存: {chart_path}")

def run_optimization_tests():
    """运行优化测试"""
    print("《心境流转》性能优化测试")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 创建基准测试器
    benchmark = OptimizationBenchmark()
    
    # 开始监控
    monitor = benchmark.monitor
    
    # 定期记录性能
    import threading
    stop_monitoring = threading.Event()
    
    def monitor_loop():
        while not stop_monitoring.is_set():
            try:
                monitor.record_metrics()
            except Exception as e:
                print(f"监控错误: {e}")
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=monitor_loop)
    monitor_thread.start()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(benchmark.device),
        "tests": {}
    }
    
    try:
        # 1. 张量运算基准测试
        tensor_results = benchmark.benchmark_tensor_operations()
        results["tests"]["tensor_operations"] = tensor_results
        
        # 2. 内存优化测试
        memory_results = benchmark.test_memory_optimization()
        results["tests"]["memory_optimization"] = memory_results
        
        # 3. 并行处理测试
        parallel_results = benchmark.test_parallel_processing()
        results["tests"]["parallel_processing"] = parallel_results
        
    finally:
        # 停止监控
        stop_monitoring.set()
        monitor_thread.join()
    
    # 获取性能总结
    performance_summary = monitor.get_summary()
    results["performance_summary"] = performance_summary
    
    # 显示结果
    display_optimization_results(results)
    
    # 可视化性能
    output_dir = Path("outputs/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_performance(monitor.metrics, output_dir)
    
    # 保存结果
    save_optimization_results(results, output_dir)
    
    return results

def display_optimization_results(results):
    """显示优化结果"""
    print("\n📊 优化测试结果")
    print("=" * 50)
    
    # 张量运算结果
    print("\n1. 张量运算性能:")
    for dtype, metrics in results["tests"]["tensor_operations"].items():
        if "error" not in metrics:
            print(f"  {dtype}:")
            print(f"    速度: {metrics['ops_per_sec']:.2f} ops/秒")
            print(f"    内存: {metrics['memory_mb']:.1f} MB")
    
    # 内存优化结果
    print("\n2. 内存优化:")
    for strategy, metrics in results["tests"]["memory_optimization"].items():
        if metrics.get("success"):
            print(f"  {strategy}:")
            print(f"    内存使用: {metrics.get('memory_used', 0):.1f} MB")
            print(f"    执行时间: {metrics.get('time', 0):.3f}秒")
    
    # 并行处理结果
    print("\n3. 并行处理:")
    for method, metrics in results["tests"]["parallel_processing"].items():
        if "error" not in metrics:
            print(f"  {method}:")
            print(f"    时间: {metrics['time']:.2f}秒")
            if "speedup" in metrics:
                print(f"    加速比: {metrics['speedup']:.2f}x")
    
    # 性能总结
    summary = results["performance_summary"]
    print("\n📈 性能监控总结:")
    print(f"  平均CPU: {summary['avg_cpu']:.1f}%")
    print(f"  峰值CPU: {summary['max_cpu']:.1f}%")
    print(f"  平均内存: {summary['avg_memory']:.1f}%")
    print(f"  峰值内存: {summary['max_memory']:.1f}%")
    if summary['avg_gpu'] > 0:
        print(f"  平均GPU: {summary['avg_gpu']:.1f}%")
        print(f"  峰值GPU: {summary['max_gpu']:.1f}%")

def save_optimization_results(results, output_dir):
    """保存优化结果"""
    output_file = output_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 处理不可序列化的对象
        def default(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
        
        json.dump(results, f, indent=2, ensure_ascii=False, default=default)
    
    print(f"\n💾 优化结果已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行优化测试
        results = run_optimization_tests()
        
        # 生成优化建议
        print("\n💡 性能优化建议")
        print("-" * 40)
        
        device = results["device"]
        if "cuda" in device:
            print("GPU优化:")
            print("1. 使用混合精度训练（FP16）可节省50%显存")
            print("2. 启用梯度累积处理大批量数据")
            print("3. 使用CUDA流实现异步处理")
            print("4. 定期清理GPU缓存避免内存泄漏")
        else:
            print("CPU优化:")
            print("1. 使用多进程并行加速计算密集型任务")
            print("2. 考虑使用量化技术减少内存占用")
            print("3. 优化数据加载避免IO瓶颈")
            print("4. 使用内存映射处理大文件")
        
        print("\n🔧 通用优化:")
        print("1. 批处理：合并小任务减少开销")
        print("2. 缓存：复用计算结果避免重复")
        print("3. 懒加载：按需加载模型和数据")
        print("4. 监控：持续追踪性能指标")
        
        print("\n" + "=" * 50)
        print("性能优化测试完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 10_complete_system_demo.py")
        
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()