#!/usr/bin/env python3
"""
增强型ISO治疗规划器 - 基于ISO原则和Gross情绪调节模型

理论基础：
1. ISO原则 (Altshuler, 1948):
   - "Use of the Iso Principle as a Central Method in Mood Management" (2015)
   - https://doi.org/10.1093/mtp/24.2.94

2. Gross情绪调节过程模型 (2015):
   - Gross, J.J. (2015). "Emotion regulation: Current status and future prospects"
   - Psychological Inquiry, 26(1), 1-26

3. 音乐治疗中的ISO原则应用 (2024):
   - Starcke & von Georgi (2024): "Music listening according to the iso principle modulates affective state"
   - https://doi.org/10.1177/10298649231175029

4. 睡眠与情绪调节 (2024):
   - "Meta-narrative review: the impact of music therapy on sleep" 
   - Frontiers in Neurology, https://doi.org/10.3389/fneur.2024.1433592

作者：心境流转团队
日期：2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# 设置日志
logger = logging.getLogger(__name__)

class TherapyStage(Enum):
    """
    治疗阶段枚举
    
    基于ISO原则的三阶段模型
    参考：Heiderscheit & Madson (2015) - "Use of the ISO principle"
    """
    SYNCHRONIZATION = "同步化"  # 匹配当前情绪状态
    TRANSITION = "引导化"       # 逐渐过渡到目标状态
    STABILIZATION = "巩固化"    # 维持和深化目标状态

class GrossStage(Enum):
    """
    Gross情绪调节过程模型的五个阶段
    
    参考：Gross (2015) - "The Extended Process Model of Emotion Regulation"
    """
    SITUATION_SELECTION = "情境选择"      # 选择或避免特定情境
    SITUATION_MODIFICATION = "情境修改"   # 改变情境以调节情绪
    ATTENTIONAL_DEPLOYMENT = "注意分配"   # 转移注意力
    COGNITIVE_CHANGE = "认知改变"         # 重新评估（reappraisal）
    RESPONSE_MODULATION = "反应调节"      # 调节情绪表达

@dataclass
class TherapyStageConfig:
    """
    治疗阶段配置
    
    包含每个阶段的详细参数
    """
    stage: TherapyStage
    duration_ratio: float  # 占总时长的比例
    emotion_target: Tuple[float, float]  # 目标V-A值
    gross_strategies: List[GrossStage]  # 应用的Gross策略
    music_features: Dict[str, float]  # 音乐特征参数
    transition_curve: str  # 过渡曲线类型：'linear', 'exponential', 'sigmoid'

class EnhancedISOPlanner:
    """
    增强型ISO治疗规划器
    
    整合ISO原则、Gross模型和睡眠治疗特定需求
    
    创新点：
    1. 动态阶段时长分配（基于起始情绪强度）
    2. 多策略情绪调节路径
    3. 睡眠场景优化约束
    """
    
    # 基于2024年研究的最优阶段时长比例
    # 参考：Meta-analysis of music therapy for sleep (2024)
    DEFAULT_STAGE_RATIOS = {
        TherapyStage.SYNCHRONIZATION: 0.25,  # 25% - 建立信任和共鸣
        TherapyStage.TRANSITION: 0.50,       # 50% - 核心转换期
        TherapyStage.STABILIZATION: 0.25     # 25% - 巩固和深化
    }
    
    # 睡眠目标状态（基于睡眠研究）
    # 参考：睡眠EEG研究显示的理想入睡前状态
    SLEEP_TARGET_STATE = (0.3, -0.8)  # 轻度正面、极低唤醒
    
    def __init__(self, adaptive_planning: bool = True):
        """
        初始化规划器
        
        Args:
            adaptive_planning: 是否使用自适应规划（根据用户状态动态调整）
        """
        self.adaptive_planning = adaptive_planning
        
    def plan_therapy_path(self, 
                         start_emotion: Tuple[float, float],
                         target_emotion: Optional[Tuple[float, float]] = None,
                         total_duration: int = 20,
                         user_preferences: Optional[Dict] = None) -> List[TherapyStageConfig]:
        """
        规划完整的治疗路径
        
        基于ISO原则的核心逻辑：
        1. 从用户当前情绪状态开始（同步化）
        2. 逐步过渡到目标状态（引导化）
        3. 维持并深化目标状态（巩固化）
        
        Args:
            start_emotion: 起始情绪状态 (valence, arousal)
            target_emotion: 目标情绪状态，默认为睡眠适宜状态
            total_duration: 总时长（分钟）
            user_preferences: 用户偏好设置
            
        Returns:
            治疗阶段配置列表
        """
        # 使用默认睡眠目标状态
        if target_emotion is None:
            target_emotion = self.SLEEP_TARGET_STATE
        
        # 计算情绪距离和路径
        emotion_distance = self._calculate_emotion_distance(start_emotion, target_emotion)
        
        # 根据情绪距离动态调整阶段时长
        stage_ratios = self._adapt_stage_ratios(emotion_distance, start_emotion)
        
        # 规划每个阶段
        stages = []
        
        # 1. 同步化阶段
        sync_config = self._plan_synchronization_stage(
            start_emotion, 
            duration=total_duration * stage_ratios[TherapyStage.SYNCHRONIZATION]
        )
        stages.append(sync_config)
        
        # 2. 引导化阶段
        transition_config = self._plan_transition_stage(
            start_emotion,
            target_emotion,
            duration=total_duration * stage_ratios[TherapyStage.TRANSITION]
        )
        stages.append(transition_config)
        
        # 3. 巩固化阶段
        stabilization_config = self._plan_stabilization_stage(
            target_emotion,
            duration=total_duration * stage_ratios[TherapyStage.STABILIZATION]
        )
        stages.append(stabilization_config)
        
        # 应用用户偏好
        if user_preferences:
            stages = self._apply_user_preferences(stages, user_preferences)
        
        return stages
    
    def _calculate_emotion_distance(self, start: Tuple[float, float], 
                                  target: Tuple[float, float]) -> float:
        """
        计算情绪状态之间的欧氏距离
        
        用于评估治疗难度和调整策略
        """
        return np.sqrt((start[0] - target[0])**2 + (start[1] - target[1])**2)
    
    def _adapt_stage_ratios(self, emotion_distance: float, 
                           start_emotion: Tuple[float, float]) -> Dict[TherapyStage, float]:
        """
        根据情绪距离和起始状态动态调整阶段时长
        
        创新算法：
        - 高唤醒状态需要更长的同步化阶段
        - 情绪距离大需要更长的过渡阶段
        
        理论依据：
        - 高唤醒状态的个体需要更多时间建立信任（ISO原则）
        - 大幅度情绪转换需要渐进式过渡（Gross模型）
        """
        ratios = self.DEFAULT_STAGE_RATIOS.copy()
        
        if not self.adaptive_planning:
            return ratios
        
        # 根据起始唤醒度调整
        start_arousal = start_emotion[1]
        if start_arousal > 0.5:  # 高唤醒
            # 需要更长的同步化阶段来建立信任
            ratios[TherapyStage.SYNCHRONIZATION] = 0.30
            ratios[TherapyStage.TRANSITION] = 0.45
            ratios[TherapyStage.STABILIZATION] = 0.25
        
        # 根据情绪距离调整
        if emotion_distance > 1.5:  # 距离较大
            # 需要更长的过渡阶段
            ratios[TherapyStage.SYNCHRONIZATION] = 0.20
            ratios[TherapyStage.TRANSITION] = 0.60
            ratios[TherapyStage.STABILIZATION] = 0.20
        
        return ratios
    
    def _plan_synchronization_stage(self, start_emotion: Tuple[float, float], 
                                   duration: float) -> TherapyStageConfig:
        """
        规划同步化阶段
        
        ISO原则核心：音乐必须首先匹配用户当前状态
        参考：Starcke & von Georgi (2024) - 验证了ISO原则的有效性
        """
        # 应用的Gross策略
        gross_strategies = [
            GrossStage.SITUATION_SELECTION,     # 创建安全的治疗环境
            GrossStage.ATTENTIONAL_DEPLOYMENT   # 将注意力引向音乐
        ]
        
        # 音乐特征设计（匹配当前情绪）
        music_features = self._emotion_to_music_features(start_emotion, match_level=0.9)
        
        return TherapyStageConfig(
            stage=TherapyStage.SYNCHRONIZATION,
            duration_ratio=duration,
            emotion_target=start_emotion,  # 保持当前状态
            gross_strategies=gross_strategies,
            music_features=music_features,
            transition_curve='linear'  # 平稳维持
        )
    
    def _plan_transition_stage(self, start_emotion: Tuple[float, float],
                              target_emotion: Tuple[float, float],
                              duration: float) -> TherapyStageConfig:
        """
        规划引导化阶段
        
        核心：渐进式情绪转换
        参考：2024年研究建议使用>2首音乐进行渐进过渡
        """
        # 计算中间点（用于平滑过渡）
        mid_emotion = (
            (start_emotion[0] + target_emotion[0]) / 2,
            (start_emotion[1] + target_emotion[1]) / 2
        )
        
        # 应用的Gross策略
        gross_strategies = [
            GrossStage.COGNITIVE_CHANGE,       # 认知重评
            GrossStage.RESPONSE_MODULATION    # 生理反应调节
        ]
        
        # 音乐特征设计（渐变特征）
        music_features = self._emotion_to_music_features(mid_emotion, match_level=0.7)
        
        return TherapyStageConfig(
            stage=TherapyStage.TRANSITION,
            duration_ratio=duration,
            emotion_target=mid_emotion,
            gross_strategies=gross_strategies,
            music_features=music_features,
            transition_curve='sigmoid'  # S型曲线，更自然的过渡
        )
    
    def _plan_stabilization_stage(self, target_emotion: Tuple[float, float],
                                 duration: float) -> TherapyStageConfig:
        """
        规划巩固化阶段
        
        目标：维持并深化睡眠适宜状态
        参考：睡眠研究显示的理想入睡前脑电状态
        """
        # 应用的Gross策略
        gross_strategies = [
            GrossStage.SITUATION_MODIFICATION,  # 深化放松环境
            GrossStage.RESPONSE_MODULATION      # 维持低唤醒状态
        ]
        
        # 音乐特征设计（睡眠优化）
        music_features = {
            'tempo': 50.0,           # 极慢节奏（睡眠心率）
            'volume': 0.3,           # 低音量
            'frequency_range': 'low', # 低频为主
            'complexity': 0.2,       # 简单重复
            'harmonic_tension': 0.1  # 极低和声张力
        }
        
        return TherapyStageConfig(
            stage=TherapyStage.STABILIZATION,
            duration_ratio=duration,
            emotion_target=target_emotion,
            gross_strategies=gross_strategies,
            music_features=music_features,
            transition_curve='exponential'  # 指数衰减，逐渐淡出
        )
    
    def _emotion_to_music_features(self, emotion: Tuple[float, float], 
                                  match_level: float = 1.0) -> Dict[str, float]:
        """
        将情绪状态映射到音乐特征
        
        基于2024年bioRxiv研究：
        - Tempo与arousal相关性：0.88
        - Mode与valence相关性：0.74
        
        参考："Decoding Musical Valence And Arousal" (2024)
        """
        valence, arousal = emotion
        
        # 基于研究的映射公式
        features = {
            # Tempo: 60-180 BPM，与arousal线性相关
            'tempo': 60 + (arousal + 1) * 60 * match_level,
            
            # Volume: 与arousal正相关
            'volume': 0.3 + (arousal + 1) * 0.35 * match_level,
            
            # Mode: 大调/小调由valence决定
            'mode': 'major' if valence > 0 else 'minor',
            
            # Complexity: 高唤醒→复杂，低唤醒→简单
            'complexity': 0.2 + (arousal + 1) * 0.4 * match_level,
            
            # Harmonic tension: 负面情绪→高张力
            'harmonic_tension': 0.5 - valence * 0.3 * match_level
        }
        
        return features
    
    def _apply_user_preferences(self, stages: List[TherapyStageConfig], 
                               preferences: Dict) -> List[TherapyStageConfig]:
        """
        应用用户偏好设置
        
        例如：
        - 偏好的音乐风格
        - 对某些乐器的偏好/厌恶
        - 个人的放松模式
        """
        # 这里可以根据用户偏好调整音乐特征
        # 当前为简化实现
        return stages
    
    def adapt_to_realtime_feedback(self, current_stage: TherapyStageConfig,
                                  user_state: Tuple[float, float],
                                  progress: float) -> TherapyStageConfig:
        """
        根据实时反馈调整治疗计划
        
        创新功能：动态适应用户响应
        理论基础：反馈控制理论在音乐治疗中的应用
        
        Args:
            current_stage: 当前阶段配置
            user_state: 用户当前状态（可通过生理信号获得）
            progress: 当前阶段进度 (0-1)
            
        Returns:
            调整后的阶段配置
        """
        # 计算预期状态与实际状态的差异
        expected_progress = progress
        actual_valence, actual_arousal = user_state
        
        # 如果用户响应不如预期，调整音乐特征
        if current_stage.stage == TherapyStage.TRANSITION:
            # 在过渡阶段特别重要
            target_v, target_a = current_stage.emotion_target
            
            # 计算调整量
            v_diff = target_v - actual_valence
            a_diff = target_a - actual_arousal
            
            # 微调音乐特征以加强效果
            if abs(a_diff) > 0.2:  # 唤醒度差异较大
                current_stage.music_features['tempo'] *= (1 - a_diff * 0.1)
                current_stage.music_features['volume'] *= (1 - a_diff * 0.1)
        
        return current_stage


# 适配器类，用于与现有系统集成
class ISOPlannerAdapter:
    """
    适配器类，将增强型规划器适配到现有的ISOModel接口
    
    设计模式：适配器模式
    目的：保持向后兼容性
    """
    
    def __init__(self, enhanced_planner: Optional[EnhancedISOPlanner] = None):
        """
        初始化适配器
        
        Args:
            enhanced_planner: 增强型规划器实例，如果为None则创建默认实例
        """
        self.enhanced_planner = enhanced_planner or EnhancedISOPlanner()
    
    def plan_stages(self, current_emotion, target_emotion, duration):
        """
        适配到现有的plan_stages接口
        
        将增强型规划器的输出转换为现有系统期望的格式
        """
        # 将EmotionState转换为元组
        if hasattr(current_emotion, 'valence'):
            current_tuple = (current_emotion.valence, current_emotion.arousal)
        else:
            current_tuple = current_emotion
            
        if hasattr(target_emotion, 'valence'):
            target_tuple = (target_emotion.valence, target_emotion.arousal)
        else:
            target_tuple = target_emotion
        
        # 使用增强型规划器
        stage_configs = self.enhanced_planner.plan_therapy_path(
            current_tuple, target_tuple, duration
        )
        
        # 转换为现有格式
        stages = []
        for config in stage_configs:
            stage_dict = {
                'stage': config.stage,
                'duration': config.duration_ratio,
                'emotion': type('EmotionState', (), {
                    'valence': config.emotion_target[0],
                    'arousal': config.emotion_target[1]
                })()
            }
            stages.append(stage_dict)
        
        return stages


# 工厂函数
def create_iso_planner(enhanced: bool = True) -> object:
    """
    创建ISO规划器
    
    Args:
        enhanced: 是否使用增强版本
        
    Returns:
        规划器实例（增强版或适配器）
    """
    if enhanced:
        return ISOPlannerAdapter(EnhancedISOPlanner(adaptive_planning=True))
    else:
        # 返回适配器，使用默认设置
        return ISOPlannerAdapter(EnhancedISOPlanner(adaptive_planning=False))