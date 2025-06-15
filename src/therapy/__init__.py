"""
疗愈理论和情绪轨迹规划模块
整合科学理论与AI模型，实现个性化睡前疗愈

核心功能:
1. ISO三阶段疗愈编排
2. 个性化情绪轨迹规划
3. 多模态内容生成协调
4. 睡眠诱导策略优化
5. 疗愈效果评估与反馈
"""

from .core import (
    TherapySession,
    TherapyOrchestrator,
    EmotionTrajectoryPlanner,
    SleepInductionEngine
)

from .stages import (
    ISOStageManager,
    SynchronizationStage,
    GuidanceStage,
    ConsolidationStage
)

from .prescriptions import (
    TherapyPrescription,
    MusicPrescription,
    VisualPrescription,
    NarrativePrescription
)

from .evaluation import (
    TherapyEvaluator,
    EffectivenessMetrics,
    SleepQualityAssessment
)

from .personalization import (
    UserProfile,
    PersonalizationEngine,
    AdaptiveLearning
)

__all__ = [
    # 核心组件
    "TherapySession",
    "TherapyOrchestrator", 
    "EmotionTrajectoryPlanner",
    "SleepInductionEngine",
    
    # 阶段管理
    "ISOStageManager",
    "SynchronizationStage",
    "GuidanceStage", 
    "ConsolidationStage",
    
    # 处方系统
    "TherapyPrescription",
    "MusicPrescription",
    "VisualPrescription",
    "NarrativePrescription",
    
    # 评估系统
    "TherapyEvaluator",
    "EffectivenessMetrics",
    "SleepQualityAssessment",
    
    # 个性化
    "UserProfile",
    "PersonalizationEngine",
    "AdaptiveLearning"
]