"""
《心境流转》硬件适配器
Hardware Adapter for Mood Transitions System

智能硬件适配和性能优化系统
- 自动检测硬件配置
- 动态调整计算策略
- 多设备协同计算
- 实时性能调优
"""

import torch
import torch.nn as nn
import psutil
import platform
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

class DeviceType(Enum):
    """设备类型"""
    CPU = "cpu"
    CUDA_GPU = "cuda"
    MPS = "mps"  # Apple Silicon
    INTEL_GPU = "intel_gpu"

class ComputeStrategy(Enum):
    """计算策略"""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"

@dataclass
class HardwareSpec:
    """硬件规格"""
    device_type: DeviceType
    device_name: str
    compute_capability: Optional[str] = None
    memory_gb: float = 0.0
    core_count: int = 0
    frequency_ghz: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    power_watts: Optional[int] = None

@dataclass
class PerformanceProfile:
    """性能配置文件"""
    batch_size: int = 1
    sequence_length: int = 512
    precision: str = "fp32"  # fp32, fp16, int8
    compute_strategy: ComputeStrategy = ComputeStrategy.ADAPTIVE
    memory_optimization: bool = True
    parallel_workers: int = 1
    cache_enabled: bool = True

class HardwareDetector:
    """硬件检测器"""
    
    def __init__(self):
        self.detected_devices = []
        self.system_info = self._get_system_info()
        self._detect_all_devices()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def _detect_all_devices(self):
        """检测所有可用设备"""
        # 检测CPU
        self._detect_cpu()
        
        # 检测CUDA GPU
        if torch.cuda.is_available():
            self._detect_cuda_gpus()
        
        # 检测Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._detect_mps()
        
        # 检测Intel GPU（如果可用）
        self._detect_intel_gpu()
    
    def _detect_cpu(self):
        """检测CPU"""
        cpu_info = psutil.cpu_freq()
        
        spec = HardwareSpec(
            device_type=DeviceType.CPU,
            device_name=platform.processor() or "CPU",
            memory_gb=self.system_info['memory_total_gb'],
            core_count=self.system_info['cpu_count'],
            frequency_ghz=cpu_info.max / 1000 if cpu_info else None
        )
        
        self.detected_devices.append(spec)
    
    def _detect_cuda_gpus(self):
        """检测CUDA GPU"""
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            spec = HardwareSpec(
                device_type=DeviceType.CUDA_GPU,
                device_name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                memory_gb=props.total_memory / (1024**3),
                core_count=props.multi_processor_count,
                bandwidth_gbps=self._estimate_memory_bandwidth(props)
            )
            
            self.detected_devices.append(spec)
    
    def _detect_mps(self):
        """检测Apple Silicon MPS"""
        # MPS (Metal Performance Shaders) for Apple Silicon
        spec = HardwareSpec(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon GPU",
            memory_gb=self.system_info['memory_total_gb'],  # 共享内存
            core_count=self._get_apple_gpu_cores()
        )
        
        self.detected_devices.append(spec)
    
    def _detect_intel_gpu(self):
        """检测Intel GPU"""
        try:
            # 尝试检测Intel GPU（如果有相关库）
            pass
        except:
            pass
    
    def _estimate_memory_bandwidth(self, props) -> Optional[float]:
        """估算显存带宽"""
        # 基于GPU架构的简化估算
        memory_type_bandwidth = {
            "GDDR6": 672,  # GB/s
            "GDDR6X": 936,
            "HBM2": 900,
            "HBM3": 2400
        }
        
        # 简化估算，实际需要更详细的GPU信息
        if "A100" in props.name:
            return 1555  # A100 HBM2
        elif "V100" in props.name:
            return 900   # V100 HBM2
        elif "RTX" in props.name:
            return 672   # 大多数RTX使用GDDR6
        else:
            return 400   # 保守估算
    
    def _get_apple_gpu_cores(self) -> int:
        """获取Apple GPU核心数"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            # 解析输出获取GPU核心数（简化实现）
            return 8  # 默认值
        except:
            return 8
    
    def get_best_device(self) -> HardwareSpec:
        """获取最佳计算设备"""
        # 优先级：CUDA GPU > MPS > CPU
        cuda_devices = [d for d in self.detected_devices if d.device_type == DeviceType.CUDA_GPU]
        if cuda_devices:
            return max(cuda_devices, key=lambda x: x.memory_gb)
        
        mps_devices = [d for d in self.detected_devices if d.device_type == DeviceType.MPS]
        if mps_devices:
            return mps_devices[0]
        
        cpu_devices = [d for d in self.detected_devices if d.device_type == DeviceType.CPU]
        return cpu_devices[0] if cpu_devices else None

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, hardware_spec: HardwareSpec):
        self.hardware_spec = hardware_spec
        self.performance_history = []
        self.current_profile = self._generate_default_profile()
    
    def _generate_default_profile(self) -> PerformanceProfile:
        """生成默认性能配置"""
        if self.hardware_spec.device_type == DeviceType.CUDA_GPU:
            # GPU优化配置
            if self.hardware_spec.memory_gb >= 40:  # 高端GPU
                return PerformanceProfile(
                    batch_size=8,
                    sequence_length=1024,
                    precision="fp16",
                    compute_strategy=ComputeStrategy.GPU_ONLY,
                    parallel_workers=4
                )
            elif self.hardware_spec.memory_gb >= 16:  # 中端GPU
                return PerformanceProfile(
                    batch_size=4,
                    sequence_length=512,
                    precision="fp16",
                    compute_strategy=ComputeStrategy.HYBRID,
                    parallel_workers=2
                )
            else:  # 低端GPU
                return PerformanceProfile(
                    batch_size=2,
                    sequence_length=256,
                    precision="fp16",
                    compute_strategy=ComputeStrategy.HYBRID,
                    parallel_workers=1
                )
        
        elif self.hardware_spec.device_type == DeviceType.MPS:
            # Apple Silicon优化配置
            return PerformanceProfile(
                batch_size=4,
                sequence_length=512,
                precision="fp16",
                compute_strategy=ComputeStrategy.GPU_ONLY,
                parallel_workers=2
            )
        
        else:  # CPU
            return PerformanceProfile(
                batch_size=1,
                sequence_length=256,
                precision="fp32",
                compute_strategy=ComputeStrategy.CPU_ONLY,
                parallel_workers=min(4, self.hardware_spec.core_count)
            )
    
    def optimize_for_task(self, task_type: str, estimated_load: str) -> PerformanceProfile:
        """针对特定任务优化"""
        base_profile = self.current_profile
        
        # 根据任务类型调整
        if task_type == "emotion_recognition":
            # 情绪识别：低延迟优先
            base_profile.batch_size = 1
            base_profile.sequence_length = min(512, base_profile.sequence_length)
            
        elif task_type == "music_generation":
            # 音乐生成：中等延迟，较高吞吐量
            if estimated_load == "high":
                base_profile.batch_size = max(1, base_profile.batch_size // 2)
            
        elif task_type == "video_generation":
            # 视频生成：高计算需求
            if self.hardware_spec.memory_gb < 20:
                base_profile.batch_size = 1
                base_profile.compute_strategy = ComputeStrategy.HYBRID
        
        # 根据负载调整
        if estimated_load == "high":
            base_profile.precision = "fp16"
            base_profile.memory_optimization = True
        elif estimated_load == "low":
            base_profile.precision = "fp32"
        
        return base_profile
    
    def benchmark_configuration(self, model: nn.Module, 
                              input_shape: Tuple, 
                              profile: PerformanceProfile) -> Dict[str, float]:
        """基准测试配置"""
        device = torch.device(self.hardware_spec.device_type.value)
        model = model.to(device)
        
        # 设置精度
        if profile.precision == "fp16" and device.type in ["cuda", "mps"]:
            model = model.half()
        
        # 生成测试数据
        test_input = torch.randn(profile.batch_size, *input_shape[1:])
        test_input = test_input.to(device)
        
        if profile.precision == "fp16" and device.type in ["cuda", "mps"]:
            test_input = test_input.half()
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(test_input)
        
        # 计时测试
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(test_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = profile.batch_size / avg_time
        
        # 内存使用
        if device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            memory_used = 0.0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'memory_used_gb': memory_used,
            'device': str(device),
            'precision': profile.precision,
            'batch_size': profile.batch_size
        }
    
    def adaptive_optimization(self, performance_feedback: Dict[str, float]):
        """自适应优化"""
        self.performance_history.append(performance_feedback)
        
        # 分析性能趋势
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            avg_latency = sum(p.get('avg_inference_time_ms', 0) for p in recent_performance) / 3
            
            # 如果延迟过高，降低复杂度
            if avg_latency > 1000:  # 1秒
                self.current_profile.batch_size = max(1, self.current_profile.batch_size - 1)
                self.current_profile.sequence_length = max(128, self.current_profile.sequence_length - 128)
            
            # 如果延迟较低且内存充足，可以提高性能
            elif avg_latency < 200 and all(p.get('memory_used_gb', 0) < self.hardware_spec.memory_gb * 0.7 
                                         for p in recent_performance):
                self.current_profile.batch_size = min(8, self.current_profile.batch_size + 1)

class HardwareAdapter:
    """硬件适配器主类"""
    
    def __init__(self):
        self.detector = HardwareDetector()
        self.best_device = self.detector.get_best_device()
        self.optimizer = PerformanceOptimizer(self.best_device) if self.best_device else None
        self.device_pool = self._create_device_pool()
        
        print(f"🔧 硬件适配器初始化完成")
        print(f"🎮 最佳计算设备: {self.best_device.device_name}")
        print(f"📊 设备内存: {self.best_device.memory_gb:.1f}GB")
    
    def _create_device_pool(self) -> Dict[str, torch.device]:
        """创建设备池"""
        pool = {}
        
        for spec in self.detector.detected_devices:
            device_key = f"{spec.device_type.value}_{spec.device_name}"
            pool[device_key] = torch.device(spec.device_type.value)
        
        return pool
    
    def get_optimal_device(self, task_type: str = "general") -> torch.device:
        """获取任务最优设备"""
        if task_type == "emotion_recognition":
            # 情绪识别优先使用GPU加速
            cuda_devices = [d for d in self.detector.detected_devices 
                          if d.device_type == DeviceType.CUDA_GPU]
            if cuda_devices:
                return torch.device("cuda")
        
        return torch.device(self.best_device.device_type.value)
    
    def optimize_model_for_hardware(self, model: nn.Module, 
                                  task_type: str = "general",
                                  target_latency_ms: float = 500) -> Tuple[nn.Module, PerformanceProfile]:
        """为硬件优化模型"""
        if not self.optimizer:
            return model, PerformanceProfile()
        
        # 估算负载
        estimated_load = "high" if target_latency_ms < 200 else "medium"
        
        # 获取优化配置
        profile = self.optimizer.optimize_for_task(task_type, estimated_load)
        
        # 应用优化
        device = self.get_optimal_device(task_type)
        model = model.to(device)
        
        # 精度优化
        if profile.precision == "fp16" and device.type in ["cuda", "mps"]:
            model = model.half()
        
        # 编译优化（PyTorch 2.0+）
        if hasattr(torch, 'compile') and device.type == "cuda":
            try:
                model = torch.compile(model)
            except:
                pass  # 编译失败时继续使用原模型
        
        return model, profile
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """获取硬件摘要"""
        return {
            'system_info': self.detector.system_info,
            'detected_devices': [asdict(spec) for spec in self.detector.detected_devices],
            'best_device': asdict(self.best_device) if self.best_device else None,
            'current_profile': asdict(self.optimizer.current_profile) if self.optimizer else None,
            'device_pool': list(self.device_pool.keys())
        }
    
    def monitor_performance(self, interval: float = 5.0) -> threading.Thread:
        """启动性能监控"""
        def monitor_loop():
            while True:
                # 收集性能数据
                if torch.cuda.is_available():
                    gpu_util = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                else:
                    gpu_util = 0
                    gpu_memory = 0
                
                cpu_util = psutil.cpu_percent()
                memory_util = psutil.virtual_memory().percent
                
                performance_data = {
                    'timestamp': time.time(),
                    'gpu_utilization': gpu_util,
                    'gpu_memory_utilization': gpu_memory,
                    'cpu_utilization': cpu_util,
                    'memory_utilization': memory_util
                }
                
                # 自适应优化
                if self.optimizer:
                    self.optimizer.adaptive_optimization(performance_data)
                
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread

# 全局适配器实例
_global_adapter = None

def get_hardware_adapter() -> HardwareAdapter:
    """获取全局硬件适配器"""
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = HardwareAdapter()
    
    return _global_adapter

def auto_configure_for_jupyterhub() -> Dict[str, Any]:
    """JupyterHub环境自动配置"""
    print("🔧 正在为JupyterHub环境进行自动配置...")
    
    adapter = get_hardware_adapter()
    summary = adapter.get_hardware_summary()
    
    # 生成推荐配置
    recommendations = {
        'compute_strategy': 'hybrid' if summary['best_device']['device_type'] != 'cpu' else 'cpu_only',
        'batch_size': 4 if summary['best_device']['memory_gb'] > 16 else 2,
        'precision': 'fp16' if summary['best_device']['device_type'] in ['cuda', 'mps'] else 'fp32',
        'parallel_workers': min(4, summary['system_info']['cpu_count']),
        'memory_optimization': True
    }
    
    print(f"✅ 自动配置完成:")
    print(f"   🎮 最佳设备: {summary['best_device']['device_name']}")
    print(f"   📊 推荐批大小: {recommendations['batch_size']}")
    print(f"   🔢 推荐精度: {recommendations['precision']}")
    print(f"   🧵 并行工作线程: {recommendations['parallel_workers']}")
    
    return {
        'hardware_summary': summary,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # 测试硬件适配器
    config = auto_configure_for_jupyterhub()
    print(f"\n📋 配置摘要:")
    print(json.dumps(config, indent=2, ensure_ascii=False))