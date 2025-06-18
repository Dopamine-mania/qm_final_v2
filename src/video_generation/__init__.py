"""
视频生成模块

提供治疗性视频生成功能，包括：
- 呼吸引导动画
- 渐变流动效果
- 波浪和星空效果
- 音乐节奏同步
- 三阶段治疗视觉
"""

from .therapeutic_video_generator import (
    TherapeuticVideoGenerator,
    VideoConfig,
    create_therapeutic_video_generator
)

from .video_adapter import (
    VideoGenerationAdapter,
    integrate_video_generation
)

__all__ = [
    'TherapeuticVideoGenerator',
    'VideoConfig',
    'create_therapeutic_video_generator',
    'VideoGenerationAdapter',
    'integrate_video_generation'
]