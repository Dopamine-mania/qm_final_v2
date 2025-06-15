"""
《心境流转》性能优化模块
Performance Optimization Module for Mood Transitions System

提供全面的硬件适配和性能优化解决方案
- 智能显存管理
- 硬件自动适配
- 性能实时监控
- JupyterHub环境优化
"""

from .memory_manager import (
    MemoryConfig,
    GPUMemoryMonitor,
    MemoryOptimizer,
    HardwareProfiler,
    get_memory_optimizer,
    optimize_for_jupyterhub
)

from .hardware_adapter import (
    DeviceType,
    ComputeStrategy,
    HardwareSpec,
    PerformanceProfile,
    HardwareDetector,
    PerformanceOptimizer,
    HardwareAdapter,
    get_hardware_adapter,
    auto_configure_for_jupyterhub
)

__all__ = [
    # Memory Management
    'MemoryConfig',
    'GPUMemoryMonitor', 
    'MemoryOptimizer',
    'HardwareProfiler',
    'get_memory_optimizer',
    'optimize_for_jupyterhub',
    
    # Hardware Adaptation
    'DeviceType',
    'ComputeStrategy', 
    'HardwareSpec',
    'PerformanceProfile',
    'HardwareDetector',
    'PerformanceOptimizer',
    'HardwareAdapter',
    'get_hardware_adapter',
    'auto_configure_for_jupyterhub'
]

# 版本信息
__version__ = "1.0.0"
__author__ = "Mood Transitions Team"
__description__ = "Performance optimization module for sleep-oriented audio-visual therapy system"

def get_optimization_info():
    """获取优化模块信息"""
    return {
        'version': __version__,
        'author': __author__, 
        'description': __description__,
        'features': [
            'GPU显存智能管理',
            'CPU-GPU混合计算',
            '硬件自动检测适配',
            '实时性能监控',
            'JupyterHub环境优化',
            '模型精度自动调整',
            '批处理大小优化',
            '并行计算调度'
        ]
    }

def quick_setup_for_jupyterhub():
    """JupyterHub环境快速设置"""
    print("🚀 《心境流转》性能优化模块 - JupyterHub快速设置")
    print("="*60)
    
    # 1. 硬件检测和适配
    adapter_config = auto_configure_for_jupyterhub()
    
    # 2. 显存优化
    memory_optimizer = optimize_for_jupyterhub()
    
    # 3. 返回配置信息
    return {
        'hardware_adapter': get_hardware_adapter(),
        'memory_optimizer': memory_optimizer,
        'configuration': adapter_config,
        'status': 'ready'
    }