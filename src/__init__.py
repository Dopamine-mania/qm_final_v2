"""
《心境流转》(Mood Transitions) - 睡前音画疗愈系统

一个基于科学证据的睡前情绪疗愈系统，采用ISO原则和多模态情绪识别技术。

核心特性:
- 基于ISO原则的三阶段疗愈：同频 → 引导 → 巩固
- 多模态情绪识别：文本 + 语音融合分析
- 个性化音乐生成：基于Valence-Arousal模型
- 音画同步生成：根据音乐特征驱动视觉内容
- 硬件友好设计：优化适配40-80GB GPU环境

作者: 陈万新
机构: 硕士论文项目
时间: 2025年
"""

__version__ = "1.0.0"
__author__ = "陈万新"
__email__ = "your.email@university.edu"  # 更换为实际邮箱
__description__ = "《心境流转》- 睡前音画疗愈系统"

# 核心模块导入
try:
    from .core import (
        EmotionEngine,
        TherapyOrchestrator, 
        ISOPlanner,
        SleepOptimizer
    )
except ImportError:
    # 如果模块尚未实现，则跳过
    pass

# 模型适配器
try:
    from .models import (
        ModelFactory,
        ModelRegistry
    )
except ImportError:
    pass

# 疗愈理论模块
try:
    from .therapy import (
        ISOStages,
        MusicPrescription,
        EmotionTrajectory,
        SleepInduction
    )
except ImportError:
    pass

# 生成器模块
try:
    from .generators import (
        MusicTherapist,
        VisualCompanion,
        NarrativeWeaver
    )
except ImportError:
    pass

# 工具函数
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    # 核心类 (如果可用)
    "EmotionEngine",
    "TherapyOrchestrator",
    "ISOPlanner", 
    "SleepOptimizer",
    # 模型类
    "ModelFactory",
    "ModelRegistry",
    # 疗愈类
    "ISOStages",
    "MusicPrescription",
    "EmotionTrajectory",
    "SleepInduction",
    # 生成器类
    "MusicTherapist",
    "VisualCompanion", 
    "NarrativeWeaver"
]
