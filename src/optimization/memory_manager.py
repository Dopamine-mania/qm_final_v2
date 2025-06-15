"""
《心境流转》显存优化管理器
GPU Memory Optimization Manager for Mood Transitions System

专为JupyterHub环境设计的智能显存管理系统
- 动态显存分配和释放
- CPU-GPU混合计算
- 模型分片和流水线优化
- 实时性能监控
"""

import torch
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import time
from contextlib import contextmanager
import logging

@dataclass
class MemoryConfig:
    """显存配置"""
    max_gpu_memory_gb: float = 70.0  # 最大GPU显存使用(GB)
    cpu_offload_threshold: float = 0.8  # CPU分流阈值
    mixed_precision: bool = True  # 混合精度
    gradient_checkpointing: bool = True  # 梯度检查点
    model_parallel: bool = False  # 模型并行
    cache_size_mb: int = 2048  # 缓存大小(MB)

class GPUMemoryMonitor:
    """GPU显存监控器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_memory = self._get_total_memory()
        self.monitoring = False
        self.memory_log = []
        
    def _get_total_memory(self) -> float:
        """获取总显存"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return 0.0
    
    def get_current_usage(self) -> Dict[str, float]:
        """获取当前显存使用情况"""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "cached": 0.0, "free": 0.0, "total": 0.0}
            
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        free = self.total_memory - cached
        
        return {
            "allocated": allocated,
            "cached": cached, 
            "free": free,
            "total": self.total_memory,
            "utilization": cached / self.total_memory if self.total_memory > 0 else 0.0
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """开始显存监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.memory_log = []
        
        def monitor_loop():
            while self.monitoring:
                usage = self.get_current_usage()
                usage['timestamp'] = time.time()
                self.memory_log.append(usage)
                time.sleep(interval)
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict]:
        """停止监控并返回日志"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        return self.memory_log

class MemoryOptimizer:
    """显存优化器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitor = GPUMemoryMonitor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_models = {}  # CPU上的模型
        self.gpu_models = {}  # GPU上的模型
        self.model_metadata = {}  # 模型元数据
        
        # 设置混合精度
        if config.mixed_precision and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    @contextmanager
    def memory_management(self, model_name: str):
        """显存管理上下文"""
        initial_usage = self.monitor.get_current_usage()
        
        try:
            # 检查是否需要释放显存
            if initial_usage['utilization'] > self.config.cpu_offload_threshold:
                self._optimize_memory()
            
            yield
            
        finally:
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_optimized(self, model_class, model_name: str, 
                           model_config: Dict, force_cpu: bool = False) -> torch.nn.Module:
        """优化加载模型"""
        
        # 检查显存使用情况
        current_usage = self.monitor.get_current_usage()
        estimated_model_size = self._estimate_model_size(model_config)
        
        # 决定加载位置
        if (force_cpu or 
            current_usage['free'] < estimated_model_size or
            current_usage['utilization'] > self.config.cpu_offload_threshold):
            
            device = torch.device("cpu")
            print(f"🔄 将模型 {model_name} 加载到CPU (显存不足)")
        else:
            device = self.device
            print(f"⚡ 将模型 {model_name} 加载到GPU")
        
        # 加载模型
        with self.memory_management(model_name):
            if self.config.mixed_precision and device.type == "cuda":
                model = model_class(**model_config).half()  # FP16
            else:
                model = model_class(**model_config)
            
            model = model.to(device)
            model.eval()
            
            # 记录模型信息
            self.model_metadata[model_name] = {
                'device': device,
                'precision': 'fp16' if self.config.mixed_precision else 'fp32',
                'size_gb': estimated_model_size,
                'load_time': time.time()
            }
            
            if device.type == "cpu":
                self.cpu_models[model_name] = model
            else:
                self.gpu_models[model_name] = model
        
        return model
    
    def _estimate_model_size(self, model_config: Dict) -> float:
        """估算模型大小 (GB)"""
        # 基于模型配置估算大小
        if 'hidden_size' in model_config and 'num_layers' in model_config:
            hidden_size = model_config['hidden_size']
            num_layers = model_config['num_layers']
            # 简化估算：参数数量 * 4字节(fp32) 或 2字节(fp16)
            param_count = hidden_size * hidden_size * num_layers * 4  # 简化估算
            bytes_per_param = 2 if self.config.mixed_precision else 4
            return (param_count * bytes_per_param) / (1024**3)
        
        # 默认估算
        return 1.0  # 1GB default
    
    def _optimize_memory(self):
        """优化显存使用"""
        print("🧹 开始显存优化...")
        
        # 1. 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 2. 移动模型到CPU
        current_usage = self.monitor.get_current_usage()
        if current_usage['utilization'] > self.config.cpu_offload_threshold:
            self._offload_models_to_cpu()
        
        # 3. 再次清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        new_usage = self.monitor.get_current_usage()
        print(f"✅ 显存优化完成: {current_usage['utilization']:.1%} → {new_usage['utilization']:.1%}")
    
    def _offload_models_to_cpu(self):
        """将模型分流到CPU"""
        # 按加载时间排序，优先移动较早加载的模型
        models_by_time = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1]['load_time']
        )
        
        for model_name, metadata in models_by_time:
            if metadata['device'].type == "cuda" and model_name in self.gpu_models:
                print(f"📤 将模型 {model_name} 移动到CPU")
                
                model = self.gpu_models.pop(model_name)
                model = model.cpu()
                self.cpu_models[model_name] = model
                
                # 更新元数据
                self.model_metadata[model_name]['device'] = torch.device("cpu")
                
                # 检查是否已足够
                current_usage = self.monitor.get_current_usage()
                if current_usage['utilization'] < self.config.cpu_offload_threshold:
                    break
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """获取模型（自动处理设备位置）"""
        if model_name in self.gpu_models:
            return self.gpu_models[model_name]
        elif model_name in self.cpu_models:
            # 检查是否可以移回GPU
            current_usage = self.monitor.get_current_usage()
            model_size = self.model_metadata[model_name]['size_gb']
            
            if (current_usage['free'] > model_size and 
                current_usage['utilization'] < 0.6):  # 低于60%使用率时移回GPU
                
                print(f"📥 将模型 {model_name} 移回GPU")
                model = self.cpu_models.pop(model_name)
                model = model.to(self.device)
                self.gpu_models[model_name] = model
                
                # 更新元数据
                self.model_metadata[model_name]['device'] = self.device
                
                return model
            else:
                return self.cpu_models[model_name]
        
        return None
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        current_usage = self.monitor.get_current_usage()
        
        return {
            'memory_usage': current_usage,
            'models_on_gpu': len(self.gpu_models),
            'models_on_cpu': len(self.cpu_models),
            'total_models': len(self.model_metadata),
            'optimization_config': {
                'mixed_precision': self.config.mixed_precision,
                'cpu_offload_threshold': self.config.cpu_offload_threshold,
                'max_gpu_memory_gb': self.config.max_gpu_memory_gb
            },
            'model_distribution': {
                name: meta['device'].type 
                for name, meta in self.model_metadata.items()
            }
        }

class HardwareProfiler:
    """硬件性能分析器"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': self.gpu_available
        }
        
        if self.gpu_available:
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_info'] = {
                'name': gpu_props.name,
                'total_memory_gb': gpu_props.total_memory / (1024**3),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'multiprocessor_count': gpu_props.multi_processor_count
            }
        
        return info
    
    def benchmark_inference(self, model: torch.nn.Module, 
                          input_shape: Tuple, num_runs: int = 10) -> Dict[str, float]:
        """推理性能基准测试"""
        device = next(model.parameters()).device
        
        # 预热
        dummy_input = torch.randn(input_shape).to(device)
        for _ in range(3):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # 测试
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'device': str(device),
            'input_shape': input_shape
        }
    
    def get_optimal_config(self) -> MemoryConfig:
        """获取最优配置"""
        gpu_memory = self.device_info.get('gpu_info', {}).get('total_memory_gb', 0)
        cpu_memory = self.device_info['memory_total_gb']
        
        if gpu_memory >= 80:  # A100等高端GPU
            return MemoryConfig(
                max_gpu_memory_gb=70.0,
                cpu_offload_threshold=0.85,
                mixed_precision=True,
                model_parallel=True
            )
        elif gpu_memory >= 40:  # A40等中高端GPU
            return MemoryConfig(
                max_gpu_memory_gb=35.0,
                cpu_offload_threshold=0.8,
                mixed_precision=True,
                model_parallel=False
            )
        else:  # 低端GPU或CPU
            return MemoryConfig(
                max_gpu_memory_gb=min(gpu_memory * 0.8, 16.0),
                cpu_offload_threshold=0.6,
                mixed_precision=True,
                model_parallel=False
            )

# 全局优化器实例
_global_optimizer = None

def get_memory_optimizer(config: Optional[MemoryConfig] = None) -> MemoryOptimizer:
    """获取全局显存优化器"""
    global _global_optimizer
    
    if _global_optimizer is None:
        if config is None:
            profiler = HardwareProfiler()
            config = profiler.get_optimal_config()
        
        _global_optimizer = MemoryOptimizer(config)
        print(f"🚀 显存优化器已初始化: {config.max_gpu_memory_gb}GB GPU显存限制")
    
    return _global_optimizer

def optimize_for_jupyterhub():
    """JupyterHub环境优化"""
    print("🔧 正在为JupyterHub环境进行优化...")
    
    # 获取硬件信息
    profiler = HardwareProfiler()
    device_info = profiler.device_info
    
    print(f"💻 CPU核心数: {device_info['cpu_count']}")
    print(f"💾 系统内存: {device_info['memory_total_gb']:.1f}GB")
    
    if device_info['gpu_available']:
        gpu_info = device_info['gpu_info']
        print(f"🎮 GPU: {gpu_info['name']}")
        print(f"📊 GPU显存: {gpu_info['total_memory_gb']:.1f}GB")
        
        # 获取优化配置
        config = profiler.get_optimal_config()
        optimizer = get_memory_optimizer(config)
        
        print(f"⚙️  优化策略:")
        print(f"   • 最大GPU显存使用: {config.max_gpu_memory_gb}GB")
        print(f"   • CPU分流阈值: {config.cpu_offload_threshold:.0%}")
        print(f"   • 混合精度: {'✅' if config.mixed_precision else '❌'}")
        print(f"   • 模型并行: {'✅' if config.model_parallel else '❌'}")
        
        return optimizer
    else:
        print("⚠️  未检测到GPU，将使用CPU模式")
        return None

if __name__ == "__main__":
    # 测试显存优化器
    optimizer = optimize_for_jupyterhub()
    
    if optimizer:
        # 开始监控
        optimizer.monitor.start_monitoring()
        
        # 模拟加载模型
        time.sleep(2)
        
        # 停止监控并显示结果
        logs = optimizer.monitor.stop_monitoring()
        stats = optimizer.get_optimization_stats()
        
        print(f"\n📊 优化统计:")
        print(f"当前显存使用: {stats['memory_usage']['utilization']:.1%}")
        print(f"GPU模型数: {stats['models_on_gpu']}")
        print(f"CPU模型数: {stats['models_on_cpu']}")