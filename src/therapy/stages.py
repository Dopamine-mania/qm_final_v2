"""
ISO三阶段管理模块
实现ISO原则的三个治疗阶段的详细控制和优化
"""

import numpy as np
import time
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# 导入理论模块
from ..research.theory.iso_principle import ISOStage, ISOStageConfig, EmotionState
from ..research.theory.valence_arousal import ValenceArousalModel
from ..research.theory.sleep_physiology import SleepPhysiologyModel, PhysiologicalState
from ..research.theory.music_psychology import MusicPsychologyModel, MusicalCharacteristics

logger = logging.getLogger(__name__)

class StageTransitionMode(Enum):
    """阶段过渡模式"""
    SMOOTH = "smooth"           # 平滑过渡
    STEPPED = "stepped"         # 阶梯式过渡
    ADAPTIVE = "adaptive"       # 自适应过渡
    IMMEDIATE = "immediate"     # 立即过渡

@dataclass
class StagePerformanceMetrics:
    """阶段性能指标"""
    stage: ISOStage
    start_time: float
    end_time: Optional[float] = None
    target_emotion: Optional[EmotionState] = None
    achieved_emotion: Optional[EmotionState] = None
    effectiveness_score: float = 0.0
    user_feedback: Dict[str, Any] = field(default_factory=dict)
    physiological_changes: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def calculate_target_achievement(self) -> float:
        """计算目标达成度"""
        if self.target_emotion is None or self.achieved_emotion is None:
            return 0.0
        
        distance = self.achieved_emotion.distance_to(self.target_emotion)
        return max(0.0, 1.0 - distance / 2.0)  # 最大距离为2

@dataclass
class StageAdaptationConfig:
    """阶段自适应配置"""
    enable_realtime_adjustment: bool = True
    feedback_sensitivity: float = 0.5  # 0-1
    adaptation_speed: float = 0.3      # 0-1
    min_stage_duration: float = 180.0  # 最少3分钟
    max_stage_duration: float = 900.0  # 最多15分钟
    effectiveness_threshold: float = 0.6

class BaseStage(ABC):
    """
    基础阶段抽象类
    
    定义所有ISO阶段的通用接口和行为
    """
    
    def __init__(self, 
                 stage_config: ISOStageConfig,
                 adaptation_config: StageAdaptationConfig,
                 va_model: ValenceArousalModel,
                 sleep_model: SleepPhysiologyModel,
                 music_model: MusicPsychologyModel):
        self.stage_config = stage_config
        self.adaptation_config = adaptation_config
        self.va_model = va_model
        self.sleep_model = sleep_model
        self.music_model = music_model
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 阶段状态
        self.is_active = False
        self.start_time = None
        self.current_emotion = None
        self.performance_metrics = None
        
        # 实时调整参数
        self.adjustment_history = []
        self.feedback_buffer = []
        
        self.logger.info(f"{self.stage_config.stage.value}阶段初始化完成")
    
    @abstractmethod
    async def execute_stage(self, 
                          initial_emotion: EmotionState,
                          context: Dict[str, Any]) -> StagePerformanceMetrics:
        """执行阶段 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_stage_requirements(self) -> Dict[str, Any]:
        """获取阶段特定要求 - 子类必须实现"""
        pass
    
    @abstractmethod
    def adapt_to_feedback(self, feedback: Dict[str, Any]) -> None:
        """根据反馈调整阶段参数 - 子类必须实现"""
        pass
    
    async def start_stage(self, initial_emotion: EmotionState) -> None:
        """开始阶段"""
        self.is_active = True
        self.start_time = time.time()
        self.current_emotion = initial_emotion
        
        self.performance_metrics = StagePerformanceMetrics(
            stage=self.stage_config.stage,
            start_time=self.start_time,
            target_emotion=self.stage_config.target_state
        )
        
        self.logger.info(f"{self.stage_config.stage.value}阶段开始")
    
    async def end_stage(self, final_emotion: EmotionState) -> StagePerformanceMetrics:
        """结束阶段"""
        self.is_active = False
        
        if self.performance_metrics:
            self.performance_metrics.end_time = time.time()
            self.performance_metrics.achieved_emotion = final_emotion
            self.performance_metrics.effectiveness_score = self._calculate_effectiveness()
        
        self.logger.info(f"{self.stage_config.stage.value}阶段完成，效果评分: {self.performance_metrics.effectiveness_score:.3f}")
        
        return self.performance_metrics
    
    def _calculate_effectiveness(self) -> float:
        """计算阶段有效性"""
        if not self.performance_metrics:
            return 0.0
        
        # 目标达成度
        target_achievement = self.performance_metrics.calculate_target_achievement()
        
        # 时间效率（完成时间与预期时间的比较）
        expected_duration = self.stage_config.duration_ratio * 1200  # 假设总时长20分钟
        actual_duration = self.performance_metrics.duration
        time_efficiency = min(1.0, expected_duration / max(actual_duration, 1.0))
        
        # 综合效果评分
        effectiveness = target_achievement * 0.7 + time_efficiency * 0.3
        
        return effectiveness
    
    def update_current_emotion(self, emotion: EmotionState) -> None:
        """更新当前情绪"""
        self.current_emotion = emotion
        
        # 实时适应性调整
        if (self.adaptation_config.enable_realtime_adjustment and 
            len(self.feedback_buffer) >= 3):  # 至少需要3个反馈点
            self._perform_realtime_adjustment()
    
    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """添加用户反馈"""
        feedback_with_timestamp = {
            **feedback,
            "timestamp": time.time(),
            "stage_duration": time.time() - (self.start_time or time.time())
        }
        
        self.feedback_buffer.append(feedback_with_timestamp)
        
        # 保持缓冲区大小
        if len(self.feedback_buffer) > 10:
            self.feedback_buffer.pop(0)
        
        # 应用反馈调整
        if self.adaptation_config.enable_realtime_adjustment:
            self.adapt_to_feedback(feedback_with_timestamp)
    
    def _perform_realtime_adjustment(self) -> None:
        """执行实时调整"""
        if not self.current_emotion or not self.stage_config.target_state:
            return
        
        # 计算当前偏差
        distance_to_target = self.current_emotion.distance_to(self.stage_config.target_state)
        
        # 如果偏差过大，进行调整
        if distance_to_target > 0.3:  # 阈值
            adjustment_strength = min(
                self.adaptation_config.adaptation_speed,
                distance_to_target * self.adaptation_config.feedback_sensitivity
            )
            
            # 记录调整
            self.adjustment_history.append({
                "timestamp": time.time(),
                "distance_to_target": distance_to_target,
                "adjustment_strength": adjustment_strength,
                "emotion_before": self.current_emotion,
                "target_emotion": self.stage_config.target_state
            })
            
            self.logger.debug(f"实时调整: 距离目标{distance_to_target:.3f}, 调整强度{adjustment_strength:.3f}")
    
    def get_stage_progress(self) -> float:
        """获取阶段进度"""
        if not self.start_time:
            return 0.0
        
        elapsed = time.time() - self.start_time
        expected_duration = self.stage_config.duration_ratio * 1200  # 假设总时长
        
        return min(1.0, elapsed / expected_duration)
    
    def should_extend_stage(self) -> bool:
        """判断是否应该延长阶段"""
        if not self.performance_metrics:
            return False
        
        # 检查有效性
        current_effectiveness = self._calculate_effectiveness()
        
        # 检查时间限制
        current_duration = time.time() - self.start_time
        
        return (current_effectiveness < self.adaptation_config.effectiveness_threshold and
                current_duration < self.adaptation_config.max_stage_duration)
    
    def can_transition_to_next(self) -> bool:
        """判断是否可以过渡到下一阶段"""
        if not self.start_time:
            return False
        
        # 检查最少时间要求
        elapsed = time.time() - self.start_time
        if elapsed < self.adaptation_config.min_stage_duration:
            return False
        
        # 检查目标达成度
        if self.current_emotion and self.stage_config.target_state:
            distance = self.current_emotion.distance_to(self.stage_config.target_state)
            return distance < 0.3  # 足够接近目标
        
        return True


class SynchronizationStage(BaseStage):
    """
    同频阶段
    
    ISO原则第一阶段：与用户当前情绪状态同频共振
    目标是建立情绪连接和信任，稳定当前状态
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 同频阶段特定配置
        self.synchronization_config = {
            "matching_precision": 0.9,     # 情绪匹配精度
            "stability_window": 120.0,     # 稳定窗口时间（秒）
            "resonance_strength": 0.8,     # 共振强度
            "validation_threshold": 0.85   # 验证阈值
        }
    
    async def execute_stage(self, 
                          initial_emotion: EmotionState,
                          context: Dict[str, Any]) -> StagePerformanceMetrics:
        """执行同频阶段"""
        await self.start_stage(initial_emotion)
        
        try:
            # 1. 情绪状态确认和稳定化
            await self._establish_emotional_resonance(initial_emotion)
            
            # 2. 构建治疗联盟
            await self._build_therapeutic_alliance(context)
            
            # 3. 基线状态记录
            await self._record_baseline_state()
            
            # 4. 验证同频效果
            synchronization_quality = await self._validate_synchronization()
            
            # 更新性能指标
            if self.performance_metrics:
                self.performance_metrics.user_feedback["synchronization_quality"] = synchronization_quality
            
            return await self.end_stage(self.current_emotion)
            
        except Exception as e:
            self.logger.error(f"同频阶段执行失败: {e}")
            if self.performance_metrics:
                self.performance_metrics.effectiveness_score = 0.0
            return self.performance_metrics
    
    async def _establish_emotional_resonance(self, initial_emotion: EmotionState) -> None:
        """建立情绪共振"""
        self.logger.info("建立情绪共振")
        
        # 精确匹配用户情绪
        target_emotion = EmotionState(
            valence=initial_emotion.valence * self.synchronization_config["matching_precision"],
            arousal=initial_emotion.arousal * self.synchronization_config["matching_precision"],
            confidence=initial_emotion.confidence
        )
        
        # 生成匹配的音乐特征
        music_prescription = self.music_model.generate_musical_prescription(
            initial_emotion,
            target_emotion=target_emotion,
            sleep_context=False  # 同频阶段不是睡眠导向
        )
        
        # 模拟共振建立过程
        resonance_steps = 10
        for step in range(resonance_steps):
            progress = step / resonance_steps
            
            # 渐进调整到目标情绪
            current_valence = (
                initial_emotion.valence * (1 - progress) +
                target_emotion.valence * progress
            )
            current_arousal = (
                initial_emotion.arousal * (1 - progress) +
                target_emotion.arousal * progress
            )
            
            self.current_emotion = EmotionState(
                valence=current_valence,
                arousal=current_arousal,
                confidence=initial_emotion.confidence
            )
            
            await asyncio.sleep(self.synchronization_config["stability_window"] / resonance_steps)
        
        self.logger.info("情绪共振建立完成")
    
    async def _build_therapeutic_alliance(self, context: Dict[str, Any]) -> None:
        """构建治疗联盟"""
        self.logger.info("构建治疗联盟")
        
        # 分析用户偏好和历史
        user_preferences = context.get("user_preferences", {})
        
        # 适应用户的音乐偏好
        preferred_instruments = user_preferences.get("preferred_instruments", [])
        if preferred_instruments:
            # 调整音乐特征以匹配用户偏好
            pass
        
        # 建立信任感
        trust_building_duration = 60.0  # 1分钟
        await asyncio.sleep(trust_building_duration)
        
        self.logger.info("治疗联盟建立完成")
    
    async def _record_baseline_state(self) -> None:
        """记录基线状态"""
        self.logger.info("记录基线状态")
        
        if self.performance_metrics and self.current_emotion:
            # 记录生理基线
            baseline_physiology = self.sleep_model.analyze_current_state(self.current_emotion)
            
            self.performance_metrics.physiological_changes = {
                "baseline_heart_rate": baseline_physiology.heart_rate,
                "baseline_stress_level": baseline_physiology.stress_level,
                "baseline_drowsiness": baseline_physiology.drowsiness_level,
                "baseline_sleep_readiness": baseline_physiology.sleep_readiness
            }
        
        await asyncio.sleep(30.0)  # 基线记录时间
        
        self.logger.info("基线状态记录完成")
    
    async def _validate_synchronization(self) -> float:
        """验证同频效果"""
        self.logger.info("验证同频效果")
        
        if not self.current_emotion or not self.stage_config.target_state:
            return 0.0
        
        # 计算情绪匹配度
        distance = self.current_emotion.distance_to(self.stage_config.target_state)
        matching_score = max(0.0, 1.0 - distance / 2.0)
        
        # 检查稳定性
        stability_score = 1.0  # 简化实现，实际中需要分析情绪波动
        
        # 综合同频质量
        synchronization_quality = (
            matching_score * 0.7 +
            stability_score * 0.3
        )
        
        self.logger.info(f"同频效果验证完成，质量分数: {synchronization_quality:.3f}")
        
        return synchronization_quality
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """获取同频阶段要求"""
        return {
            "music_style": "emotional_matching",
            "tempo_adjustment": "minimal",
            "volume_consistency": "stable",
            "harmonic_complexity": "moderate",
            "emotional_intensity": "matching_user",
            "visual_elements": "supportive_not_dominant",
            "interaction_mode": "passive_receptive"
        }
    
    def adapt_to_feedback(self, feedback: Dict[str, Any]) -> None:
        """根据反馈调整同频参数"""
        comfort_level = feedback.get("comfort_level", 0.5)
        emotional_resonance = feedback.get("emotional_resonance", 0.5)
        
        # 调整匹配精度
        if comfort_level < 0.3:
            self.synchronization_config["matching_precision"] *= 0.9  # 降低匹配强度
        elif comfort_level > 0.8:
            self.synchronization_config["matching_precision"] = min(
                1.0, self.synchronization_config["matching_precision"] * 1.05
            )
        
        # 调整共振强度
        if emotional_resonance < 0.4:
            self.synchronization_config["resonance_strength"] *= 0.85
        elif emotional_resonance > 0.8:
            self.synchronization_config["resonance_strength"] = min(
                1.0, self.synchronization_config["resonance_strength"] * 1.1
            )
        
        self.logger.debug(f"同频参数调整: 精度={self.synchronization_config['matching_precision']:.3f}, 强度={self.synchronization_config['resonance_strength']:.3f}")


class GuidanceStage(BaseStage):
    """
    引导阶段
    
    ISO原则第二阶段：渐进式情绪引导和过渡
    目标是平滑地将用户从当前状态引导到睡前理想状态
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 引导阶段特定配置
        self.guidance_config = {
            "transition_speed": 0.5,        # 过渡速度（0-1）
            "gradient_smoothness": 0.8,     # 梯度平滑度
            "checkpoint_interval": 90.0,    # 检查点间隔（秒）
            "adaptation_sensitivity": 0.6,  # 适应敏感度
            "max_deviation_tolerance": 0.4  # 最大偏差容忍度
        }
        
        # 引导路径点
        self.guidance_waypoints = []
        self.current_waypoint_index = 0
    
    async def execute_stage(self, 
                          initial_emotion: EmotionState,
                          context: Dict[str, Any]) -> StagePerformanceMetrics:
        """执行引导阶段"""
        await self.start_stage(initial_emotion)
        
        try:
            # 1. 生成引导路径
            await self._generate_guidance_path(initial_emotion, context)
            
            # 2. 执行渐进引导
            await self._execute_progressive_guidance()
            
            # 3. 监控和调整
            await self._monitor_and_adjust_guidance()
            
            # 4. 验证引导效果
            guidance_effectiveness = await self._validate_guidance_effectiveness()
            
            # 更新性能指标
            if self.performance_metrics:
                self.performance_metrics.user_feedback["guidance_effectiveness"] = guidance_effectiveness
            
            return await self.end_stage(self.current_emotion)
            
        except Exception as e:
            self.logger.error(f"引导阶段执行失败: {e}")
            if self.performance_metrics:
                self.performance_metrics.effectiveness_score = 0.0
            return self.performance_metrics
    
    async def _generate_guidance_path(self, 
                                    initial_emotion: EmotionState,
                                    context: Dict[str, Any]) -> None:
        """生成引导路径"""
        self.logger.info("生成引导路径")
        
        target_emotion = self.stage_config.target_state
        
        # 计算中间路径点
        num_waypoints = max(3, int(self.stage_config.duration_ratio * 600 / self.guidance_config["checkpoint_interval"]))
        
        self.guidance_waypoints = []
        
        for i in range(num_waypoints + 1):
            progress = i / num_waypoints
            
            # 应用平滑过渡函数
            smooth_progress = self._apply_transition_smoothing(progress)
            
            # 计算中间情绪状态
            waypoint_valence = (
                initial_emotion.valence * (1 - smooth_progress) +
                target_emotion.valence * smooth_progress
            )
            waypoint_arousal = (
                initial_emotion.arousal * (1 - smooth_progress) +
                target_emotion.arousal * smooth_progress
            )
            
            waypoint = EmotionState(
                valence=waypoint_valence,
                arousal=waypoint_arousal,
                confidence=initial_emotion.confidence * 0.9 + 0.1  # 逐渐增强置信度
            )
            
            self.guidance_waypoints.append(waypoint)
        
        self.logger.info(f"引导路径生成完成，共{len(self.guidance_waypoints)}个路径点")
    
    def _apply_transition_smoothing(self, progress: float) -> float:
        """应用过渡平滑函数"""
        smoothness = self.guidance_config["gradient_smoothness"]
        
        if smoothness > 0.5:
            # S型曲线平滑
            return 1 / (1 + np.exp(-6 * (progress - 0.5)))
        else:
            # 线性过渡
            return progress
    
    async def _execute_progressive_guidance(self) -> None:
        """执行渐进引导"""
        self.logger.info("开始渐进引导")
        
        total_waypoints = len(self.guidance_waypoints)
        
        for i, waypoint in enumerate(self.guidance_waypoints[1:], 1):
            self.current_waypoint_index = i
            
            # 渐进过渡到下一个路径点
            await self._transition_to_waypoint(waypoint)
            
            # 生成对应的音乐特征
            music_prescription = self.music_model.generate_musical_prescription(
                self.current_emotion,
                target_emotion=waypoint,
                sleep_context=True if i > total_waypoints * 0.6 else False  # 后期开始睡眠导向
            )
            
            # 检查点监控
            await self._checkpoint_monitoring(waypoint)
            
            # 等待到下一个检查点
            await asyncio.sleep(self.guidance_config["checkpoint_interval"])
        
        self.logger.info("渐进引导完成")
    
    async def _transition_to_waypoint(self, target_waypoint: EmotionState) -> None:
        """过渡到指定路径点"""
        if not self.current_emotion:
            self.current_emotion = target_waypoint
            return
        
        transition_steps = 10
        transition_duration = self.guidance_config["checkpoint_interval"] * 0.8  # 80%时间用于过渡
        
        for step in range(transition_steps):
            progress = step / transition_steps
            
            # 线性插值
            current_valence = (
                self.current_emotion.valence * (1 - progress) +
                target_waypoint.valence * progress
            )
            current_arousal = (
                self.current_emotion.arousal * (1 - progress) +
                target_waypoint.arousal * progress
            )
            
            self.current_emotion = EmotionState(
                valence=current_valence,
                arousal=current_arousal,
                confidence=target_waypoint.confidence
            )
            
            await asyncio.sleep(transition_duration / transition_steps)
    
    async def _checkpoint_monitoring(self, expected_waypoint: EmotionState) -> None:
        """检查点监控"""
        if not self.current_emotion:
            return
        
        # 计算偏差
        deviation = self.current_emotion.distance_to(expected_waypoint)
        
        if deviation > self.guidance_config["max_deviation_tolerance"]:
            self.logger.warning(f"检查点偏差过大: {deviation:.3f}")
            
            # 执行纠正调整
            await self._apply_corrective_adjustment(expected_waypoint, deviation)
        
        # 记录检查点数据
        checkpoint_data = {
            "waypoint_index": self.current_waypoint_index,
            "expected_emotion": expected_waypoint,
            "actual_emotion": self.current_emotion,
            "deviation": deviation,
            "timestamp": time.time()
        }
        
        if self.performance_metrics:
            checkpoints = self.performance_metrics.user_feedback.get("checkpoints", [])
            checkpoints.append(checkpoint_data)
            self.performance_metrics.user_feedback["checkpoints"] = checkpoints
    
    async def _apply_corrective_adjustment(self, 
                                         target_waypoint: EmotionState,
                                         deviation: float) -> None:
        """应用纠正调整"""
        self.logger.info(f"应用纠正调整，偏差: {deviation:.3f}")
        
        # 调整过渡速度
        adjustment_strength = min(0.8, deviation * self.guidance_config["adaptation_sensitivity"])
        
        # 更快地向目标调整
        correction_steps = 5
        for step in range(correction_steps):
            progress = (step + 1) / correction_steps * adjustment_strength
            
            corrected_valence = (
                self.current_emotion.valence * (1 - progress) +
                target_waypoint.valence * progress
            )
            corrected_arousal = (
                self.current_emotion.arousal * (1 - progress) +
                target_waypoint.arousal * progress
            )
            
            self.current_emotion = EmotionState(
                valence=corrected_valence,
                arousal=corrected_arousal,
                confidence=self.current_emotion.confidence
            )
            
            await asyncio.sleep(2.0)  # 快速纠正
    
    async def _monitor_and_adjust_guidance(self) -> None:
        """监控和调整引导过程"""
        self.logger.info("开始引导监控")
        
        # 在实际实现中，这里会:
        # 1. 监控用户的生理反应
        # 2. 分析音乐和视觉的同步效果
        # 3. 根据实时反馈调整引导参数
        
        # 模拟监控过程
        monitoring_duration = 60.0
        await asyncio.sleep(monitoring_duration)
        
        self.logger.info("引导监控完成")
    
    async def _validate_guidance_effectiveness(self) -> float:
        """验证引导效果"""
        self.logger.info("验证引导效果")
        
        if not self.current_emotion or not self.stage_config.target_state:
            return 0.0
        
        # 计算目标达成度
        distance_to_target = self.current_emotion.distance_to(self.stage_config.target_state)
        target_achievement = max(0.0, 1.0 - distance_to_target / 2.0)
        
        # 计算路径平滑度
        path_smoothness = self._calculate_path_smoothness()
        
        # 计算适应性质量
        adaptation_quality = self._calculate_adaptation_quality()
        
        # 综合引导效果
        guidance_effectiveness = (
            target_achievement * 0.5 +
            path_smoothness * 0.3 +
            adaptation_quality * 0.2
        )
        
        self.logger.info(f"引导效果验证完成，效果分数: {guidance_effectiveness:.3f}")
        
        return guidance_effectiveness
    
    def _calculate_path_smoothness(self) -> float:
        """计算路径平滑度"""
        if len(self.guidance_waypoints) < 3:
            return 1.0
        
        # 计算路径的二阶导数变化
        valences = [wp.valence for wp in self.guidance_waypoints]
        arousals = [wp.arousal for wp in self.guidance_waypoints]
        
        valence_second_diff = np.diff(np.diff(valences))
        arousal_second_diff = np.diff(np.diff(arousals))
        
        valence_smoothness = 1.0 / (1.0 + np.var(valence_second_diff))
        arousal_smoothness = 1.0 / (1.0 + np.var(arousal_second_diff))
        
        return (valence_smoothness + arousal_smoothness) / 2.0
    
    def _calculate_adaptation_quality(self) -> float:
        """计算适应性质量"""
        # 基于调整历史计算适应性质量
        if not self.adjustment_history:
            return 0.8  # 默认值
        
        # 适应性质量 = 响应及时性 + 调整准确性
        response_timeliness = 0.8  # 简化实现
        adjustment_accuracy = 0.7   # 简化实现
        
        return (response_timeliness + adjustment_accuracy) / 2.0
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """获取引导阶段要求"""
        return {
            "music_style": "gradual_transition",
            "tempo_adjustment": "progressive_slowdown",
            "volume_consistency": "gentle_decrease",
            "harmonic_complexity": "gradual_simplification",
            "emotional_intensity": "controlled_reduction",
            "visual_elements": "synchronized_calming",
            "interaction_mode": "guided_passive"
        }
    
    def adapt_to_feedback(self, feedback: Dict[str, Any]) -> None:
        """根据反馈调整引导参数"""
        transition_comfort = feedback.get("transition_comfort", 0.5)
        guidance_pace = feedback.get("guidance_pace", 0.5)  # 0=太快, 1=太慢
        
        # 调整过渡速度
        if guidance_pace < 0.3:  # 太快
            self.guidance_config["transition_speed"] *= 0.8
        elif guidance_pace > 0.7:  # 太慢
            self.guidance_config["transition_speed"] = min(1.0, self.guidance_config["transition_speed"] * 1.2)
        
        # 调整平滑度
        if transition_comfort < 0.4:
            self.guidance_config["gradient_smoothness"] = min(1.0, self.guidance_config["gradient_smoothness"] * 1.1)
        
        self.logger.debug(f"引导参数调整: 速度={self.guidance_config['transition_speed']:.3f}, 平滑度={self.guidance_config['gradient_smoothness']:.3f}")


class ConsolidationStage(BaseStage):
    """
    巩固阶段
    
    ISO原则第三阶段：巩固睡前理想状态并诱导睡眠
    目标是稳定在睡前最佳状态，促进自然入睡
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 巩固阶段特定配置
        self.consolidation_config = {
            "stabilization_duration": 180.0,    # 稳定化时间
            "sleep_induction_strength": 0.9,    # 睡眠诱导强度
            "micro_adjustment_threshold": 0.1,  # 微调阈值
            "deep_relaxation_target": -0.8,    # 深度放松目标（arousal）
            "positive_valence_maintenance": 0.1  # 维持轻微积极情绪
        }
        
        # 睡眠诱导策略
        self.sleep_induction_strategy = None
    
    async def execute_stage(self, 
                          initial_emotion: EmotionState,
                          context: Dict[str, Any]) -> StagePerformanceMetrics:
        """执行巩固阶段"""
        await self.start_stage(initial_emotion)
        
        try:
            # 1. 情绪状态稳定化
            await self._stabilize_emotional_state()
            
            # 2. 深度放松诱导
            await self._induce_deep_relaxation()
            
            # 3. 睡眠准备优化
            await self._optimize_sleep_preparation(context)
            
            # 4. 睡眠诱导执行
            await self._execute_sleep_induction()
            
            # 5. 验证巩固效果
            consolidation_quality = await self._validate_consolidation_quality()
            
            # 更新性能指标
            if self.performance_metrics:
                self.performance_metrics.user_feedback["consolidation_quality"] = consolidation_quality
            
            return await self.end_stage(self.current_emotion)
            
        except Exception as e:
            self.logger.error(f"巩固阶段执行失败: {e}")
            if self.performance_metrics:
                self.performance_metrics.effectiveness_score = 0.0
            return self.performance_metrics
    
    async def _stabilize_emotional_state(self) -> None:
        """稳定化情绪状态"""
        self.logger.info("开始情绪状态稳定化")
        
        target_emotion = EmotionState(
            valence=self.consolidation_config["positive_valence_maintenance"],
            arousal=self.consolidation_config["deep_relaxation_target"] * 0.7,  # 渐进到深度目标
            confidence=0.95
        )
        
        # 稳定化过程
        stabilization_steps = 20
        step_duration = self.consolidation_config["stabilization_duration"] / stabilization_steps
        
        for step in range(stabilization_steps):
            if not self.current_emotion:
                self.current_emotion = target_emotion
                break
            
            # 微调向目标状态
            adjustment_factor = 0.05  # 每步5%的调整
            
            adjusted_valence = (
                self.current_emotion.valence * (1 - adjustment_factor) +
                target_emotion.valence * adjustment_factor
            )
            adjusted_arousal = (
                self.current_emotion.arousal * (1 - adjustment_factor) +
                target_emotion.arousal * adjustment_factor
            )
            
            self.current_emotion = EmotionState(
                valence=adjusted_valence,
                arousal=adjusted_arousal,
                confidence=self.current_emotion.confidence
            )
            
            await asyncio.sleep(step_duration)
        
        self.logger.info("情绪状态稳定化完成")
    
    async def _induce_deep_relaxation(self) -> None:
        """诱导深度放松"""
        self.logger.info("开始深度放松诱导")
        
        # 目标深度放松状态
        deep_relaxation_target = EmotionState(
            valence=self.consolidation_config["positive_valence_maintenance"],
            arousal=self.consolidation_config["deep_relaxation_target"],
            confidence=0.98
        )
        
        # 生理放松诱导
        current_physiology = self.sleep_model.analyze_current_state(self.current_emotion)
        target_physiology = self.sleep_model.analyze_current_state(deep_relaxation_target)
        
        # 计算放松音乐参数
        relaxation_music_params = self.sleep_model.calculate_music_parameters(target_physiology)
        
        # 深度放松过程
        relaxation_duration = 240.0  # 4分钟
        relaxation_steps = 30
        
        for step in range(relaxation_steps):
            progress = step / relaxation_steps
            
            # 非线性深度放松曲线
            deep_progress = self._calculate_deep_relaxation_curve(progress)
            
            current_valence = (
                self.current_emotion.valence * (1 - deep_progress) +
                deep_relaxation_target.valence * deep_progress
            )
            current_arousal = (
                self.current_emotion.arousal * (1 - deep_progress) +
                deep_relaxation_target.arousal * deep_progress
            )
            
            self.current_emotion = EmotionState(
                valence=current_valence,
                arousal=current_arousal,
                confidence=self.current_emotion.confidence
            )
            
            await asyncio.sleep(relaxation_duration / relaxation_steps)
        
        self.logger.info("深度放松诱导完成")
    
    def _calculate_deep_relaxation_curve(self, progress: float) -> float:
        """计算深度放松曲线"""
        # 使用指数衰减曲线，开始快速，后期缓慢
        return 1 - np.exp(-3 * progress)
    
    async def _optimize_sleep_preparation(self, context: Dict[str, Any]) -> None:
        """优化睡眠准备"""
        self.logger.info("开始睡眠准备优化")
        
        # 分析用户睡眠偏好
        sleep_preferences = context.get("user_preferences", {}).get("sleep_preferences", {})
        
        # 选择睡眠诱导策略
        preferred_strategy = sleep_preferences.get("induction_strategy", "gentle")
        
        # 创建睡眠诱导引擎实例并生成计划
        from .core import SleepInductionEngine
        sleep_engine = SleepInductionEngine(self.sleep_model, self.music_model)
        
        self.sleep_induction_strategy = sleep_engine.create_sleep_induction_plan(
            self.current_emotion,
            strategy=preferred_strategy
        )
        
        # 环境优化建议
        environment_recommendations = self.sleep_induction_strategy["recommended_environment"]
        
        # 记录优化参数
        if self.performance_metrics:
            self.performance_metrics.user_feedback["sleep_preparation"] = {
                "strategy": preferred_strategy,
                "environment_recommendations": environment_recommendations,
                "induction_plan": self.sleep_induction_strategy
            }
        
        await asyncio.sleep(60.0)  # 优化准备时间
        
        self.logger.info("睡眠准备优化完成")
    
    async def _execute_sleep_induction(self) -> None:
        """执行睡眠诱导"""
        self.logger.info("开始睡眠诱导")
        
        if not self.sleep_induction_strategy:
            self.logger.warning("睡眠诱导策略未初始化")
            return
        
        # 获取目标情绪
        target_emotion = self.sleep_induction_strategy["target_emotion"]
        transition_time = self.sleep_induction_strategy["transition_time"]
        
        # 睡眠诱导过程
        induction_steps = int(transition_time / 10)  # 每10秒一步
        
        for step in range(induction_steps):
            progress = step / induction_steps
            
            # 睡眠诱导曲线（更加渐进）
            sleep_progress = self._calculate_sleep_induction_curve(progress)
            
            current_valence = (
                self.current_emotion.valence * (1 - sleep_progress) +
                target_emotion.valence * sleep_progress
            )
            current_arousal = (
                self.current_emotion.arousal * (1 - sleep_progress) +
                target_emotion.arousal * sleep_progress
            )
            
            self.current_emotion = EmotionState(
                valence=current_valence,
                arousal=current_arousal,
                confidence=target_emotion.confidence
            )
            
            # 监控点检查
            if step % 3 == 0:  # 每30秒检查一次
                await self._monitor_sleep_readiness()
            
            await asyncio.sleep(10.0)
        
        self.logger.info("睡眠诱导完成")
    
    def _calculate_sleep_induction_curve(self, progress: float) -> float:
        """计算睡眠诱导曲线"""
        # 使用双指数曲线，模拟自然入睡过程
        return 1 - np.exp(-2 * progress) * np.exp(-0.5 * progress**2)
    
    async def _monitor_sleep_readiness(self) -> None:
        """监控睡眠准备度"""
        if not self.current_emotion:
            return
        
        # 使用睡眠诱导引擎评估准备度
        from .core import SleepInductionEngine
        sleep_engine = SleepInductionEngine(self.sleep_model, self.music_model)
        
        readiness_assessment = sleep_engine.evaluate_sleep_readiness(self.current_emotion)
        
        # 记录准备度数据
        if self.performance_metrics:
            readiness_history = self.performance_metrics.user_feedback.get("readiness_history", [])
            readiness_history.append({
                "timestamp": time.time(),
                "readiness": readiness_assessment,
                "emotion_state": self.current_emotion
            })
            self.performance_metrics.user_feedback["readiness_history"] = readiness_history
        
        self.logger.debug(f"睡眠准备度: {readiness_assessment['overall_readiness']:.3f}")
    
    async def _validate_consolidation_quality(self) -> float:
        """验证巩固质量"""
        self.logger.info("验证巩固质量")
        
        if not self.current_emotion or not self.stage_config.target_state:
            return 0.0
        
        # 1. 目标达成度
        distance_to_target = self.current_emotion.distance_to(self.stage_config.target_state)
        target_achievement = max(0.0, 1.0 - distance_to_target / 2.0)
        
        # 2. 睡眠准备度
        from .core import SleepInductionEngine
        sleep_engine = SleepInductionEngine(self.sleep_model, self.music_model)
        readiness_assessment = sleep_engine.evaluate_sleep_readiness(self.current_emotion)
        sleep_readiness = readiness_assessment["overall_readiness"]
        
        # 3. 状态稳定性
        stability_score = self._calculate_final_stability()
        
        # 4. 诱导效果
        induction_effectiveness = self._calculate_induction_effectiveness()
        
        # 综合巩固质量
        consolidation_quality = (
            target_achievement * 0.3 +
            sleep_readiness * 0.4 +
            stability_score * 0.2 +
            induction_effectiveness * 0.1
        )
        
        self.logger.info(f"巩固质量验证完成，质量分数: {consolidation_quality:.3f}")
        
        return consolidation_quality
    
    def _calculate_final_stability(self) -> float:
        """计算最终状态稳定性"""
        # 在实际实现中，这里会分析情绪变化的方差
        # 目前返回基于arousal水平的稳定性估计
        if self.current_emotion:
            arousal_stability = max(0.0, 1.0 + self.current_emotion.arousal)  # arousal越低越稳定
            return min(1.0, arousal_stability)
        return 0.5
    
    def _calculate_induction_effectiveness(self) -> float:
        """计算诱导效果"""
        # 基于睡眠诱导策略的执行情况
        if self.sleep_induction_strategy and self.current_emotion:
            target_arousal = self.sleep_induction_strategy["target_emotion"].arousal
            current_arousal = self.current_emotion.arousal
            
            arousal_achievement = max(0.0, 1.0 - abs(current_arousal - target_arousal) / 2.0)
            return arousal_achievement
        
        return 0.5
    
    def get_stage_requirements(self) -> Dict[str, Any]:
        """获取巩固阶段要求"""
        return {
            "music_style": "sleep_induction",
            "tempo_adjustment": "very_slow_consistent",
            "volume_consistency": "gradual_fade",
            "harmonic_complexity": "minimal_simple",
            "emotional_intensity": "deeply_calming",
            "visual_elements": "sleep_promoting",
            "interaction_mode": "completely_passive",
            "environmental_optimization": "sleep_conducive"
        }
    
    def adapt_to_feedback(self, feedback: Dict[str, Any]) -> None:
        """根据反馈调整巩固参数"""
        relaxation_depth = feedback.get("relaxation_depth", 0.5)
        sleep_readiness_feel = feedback.get("sleep_readiness", 0.5)
        
        # 调整深度放松目标
        if relaxation_depth < 0.4:
            self.consolidation_config["deep_relaxation_target"] = max(
                -1.0, self.consolidation_config["deep_relaxation_target"] * 1.1
            )
        elif relaxation_depth > 0.8:
            self.consolidation_config["deep_relaxation_target"] = min(
                -0.5, self.consolidation_config["deep_relaxation_target"] * 0.9
            )
        
        # 调整睡眠诱导强度
        if sleep_readiness_feel < 0.5:
            self.consolidation_config["sleep_induction_strength"] = min(
                1.0, self.consolidation_config["sleep_induction_strength"] * 1.1
            )
        
        self.logger.debug(f"巩固参数调整: 放松目标={self.consolidation_config['deep_relaxation_target']:.3f}, 诱导强度={self.consolidation_config['sleep_induction_strength']:.3f}")


class ISOStageManager:
    """
    ISO阶段管理器
    
    负责协调和管理ISO三阶段的执行、过渡和优化
    """
    
    def __init__(self, 
                 va_model: ValenceArousalModel,
                 sleep_model: SleepPhysiologyModel,
                 music_model: MusicPsychologyModel):
        self.va_model = va_model
        self.sleep_model = sleep_model
        self.music_model = music_model
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 阶段实例
        self.stages: Dict[ISOStage, BaseStage] = {}
        self.current_stage = None
        self.stage_history: List[StagePerformanceMetrics] = []
        
        # 过渡配置
        self.transition_config = {
            "mode": StageTransitionMode.ADAPTIVE,
            "overlap_duration": 30.0,  # 阶段重叠时间
            "transition_smoothness": 0.8,
            "auto_progression": True
        }
        
        # 全局适应配置
        self.global_adaptation_config = StageAdaptationConfig()
        
        self.logger.info("ISO阶段管理器初始化完成")
    
    def initialize_stages(self, stage_configs: List[ISOStageConfig]) -> None:
        """初始化所有阶段"""
        for stage_config in stage_configs:
            if stage_config.stage == ISOStage.SYNCHRONIZATION:
                self.stages[ISOStage.SYNCHRONIZATION] = SynchronizationStage(
                    stage_config, self.global_adaptation_config,
                    self.va_model, self.sleep_model, self.music_model
                )
            elif stage_config.stage == ISOStage.GUIDANCE:
                self.stages[ISOStage.GUIDANCE] = GuidanceStage(
                    stage_config, self.global_adaptation_config,
                    self.va_model, self.sleep_model, self.music_model
                )
            elif stage_config.stage == ISOStage.CONSOLIDATION:
                self.stages[ISOStage.CONSOLIDATION] = ConsolidationStage(
                    stage_config, self.global_adaptation_config,
                    self.va_model, self.sleep_model, self.music_model
                )
        
        self.logger.info(f"初始化了{len(self.stages)}个阶段")
    
    async def execute_iso_sequence(self, 
                                 initial_emotion: EmotionState,
                                 context: Dict[str, Any]) -> List[StagePerformanceMetrics]:
        """执行完整的ISO序列"""
        self.logger.info("开始执行ISO三阶段序列")
        
        try:
            # 执行同频阶段
            if ISOStage.SYNCHRONIZATION in self.stages:
                sync_metrics = await self._execute_stage_with_transition(
                    ISOStage.SYNCHRONIZATION, initial_emotion, context
                )
                self.stage_history.append(sync_metrics)
            
            # 执行引导阶段
            if ISOStage.GUIDANCE in self.stages:
                current_emotion = self._get_current_emotion()
                guidance_metrics = await self._execute_stage_with_transition(
                    ISOStage.GUIDANCE, current_emotion, context
                )
                self.stage_history.append(guidance_metrics)
            
            # 执行巩固阶段
            if ISOStage.CONSOLIDATION in self.stages:
                current_emotion = self._get_current_emotion()
                consolidation_metrics = await self._execute_stage_with_transition(
                    ISOStage.CONSOLIDATION, current_emotion, context
                )
                self.stage_history.append(consolidation_metrics)
            
            self.logger.info("ISO三阶段序列执行完成")
            return self.stage_history
            
        except Exception as e:
            self.logger.error(f"ISO序列执行失败: {e}")
            raise
    
    async def _execute_stage_with_transition(self, 
                                           stage_type: ISOStage,
                                           initial_emotion: EmotionState,
                                           context: Dict[str, Any]) -> StagePerformanceMetrics:
        """执行阶段并处理过渡"""
        stage = self.stages[stage_type]
        self.current_stage = stage
        
        self.logger.info(f"开始执行{stage_type.value}阶段")
        
        # 执行阶段
        metrics = await stage.execute_stage(initial_emotion, context)
        
        # 处理阶段过渡
        if self.transition_config["auto_progression"]:
            await self._handle_stage_transition(stage_type, metrics)
        
        return metrics
    
    async def _handle_stage_transition(self, 
                                     completed_stage: ISOStage,
                                     stage_metrics: StagePerformanceMetrics) -> None:
        """处理阶段过渡"""
        if self.transition_config["mode"] == StageTransitionMode.ADAPTIVE:
            # 自适应过渡：根据当前阶段效果决定下一步
            effectiveness = stage_metrics.effectiveness_score
            
            if effectiveness < 0.6:  # 效果不佳
                self.logger.warning(f"{completed_stage.value}阶段效果不佳({effectiveness:.3f})，可能需要延长或调整")
                
                # 可以选择延长当前阶段或调整下一阶段参数
                await self._adapt_next_stage_parameters(completed_stage, effectiveness)
        
        # 平滑过渡处理
        if self.transition_config["overlap_duration"] > 0:
            await self._execute_smooth_transition()
    
    async def _adapt_next_stage_parameters(self, 
                                         completed_stage: ISOStage,
                                         effectiveness: float) -> None:
        """根据前一阶段效果调整下一阶段参数"""
        next_stage_type = self._get_next_stage(completed_stage)
        
        if next_stage_type and next_stage_type in self.stages:
            next_stage = self.stages[next_stage_type]
            
            # 根据效果调整适应配置
            if effectiveness < 0.4:
                # 效果很差，增加下一阶段的适应性
                next_stage.adaptation_config.feedback_sensitivity *= 1.2
                next_stage.adaptation_config.adaptation_speed *= 1.1
            elif effectiveness < 0.6:
                # 效果一般，轻微增加适应性
                next_stage.adaptation_config.feedback_sensitivity *= 1.1
            
            self.logger.info(f"已调整{next_stage_type.value}阶段参数以补偿前阶段效果")
    
    def _get_next_stage(self, current_stage: ISOStage) -> Optional[ISOStage]:
        """获取下一阶段"""
        stage_order = [ISOStage.SYNCHRONIZATION, ISOStage.GUIDANCE, ISOStage.CONSOLIDATION]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def _execute_smooth_transition(self) -> None:
        """执行平滑过渡"""
        transition_duration = self.transition_config["overlap_duration"]
        
        # 在过渡期间，可以：
        # 1. 渐进调整音乐参数
        # 2. 平滑视觉过渡
        # 3. 监控用户反应
        
        self.logger.debug(f"执行平滑过渡，时长: {transition_duration}秒")
        await asyncio.sleep(transition_duration)
    
    def _get_current_emotion(self) -> EmotionState:
        """获取当前情绪状态"""
        if self.current_stage and self.current_stage.current_emotion:
            return self.current_stage.current_emotion
        
        # 默认返回中性状态
        return EmotionState(valence=0.0, arousal=0.0, confidence=0.5)
    
    def add_global_feedback(self, feedback: Dict[str, Any]) -> None:
        """添加全局反馈"""
        if self.current_stage:
            self.current_stage.add_feedback(feedback)
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """获取整体进度"""
        total_stages = len(self.stages)
        completed_stages = len(self.stage_history)
        
        # 计算当前阶段进度
        current_stage_progress = 0.0
        if self.current_stage:
            current_stage_progress = self.current_stage.get_stage_progress()
        
        # 整体进度
        overall_progress = (completed_stages + current_stage_progress) / total_stages
        
        return {
            "overall_progress": overall_progress,
            "completed_stages": completed_stages,
            "total_stages": total_stages,
            "current_stage": self.current_stage.stage_config.stage.value if self.current_stage else None,
            "current_stage_progress": current_stage_progress,
            "stage_history": [
                {
                    "stage": metrics.stage.value,
                    "effectiveness": metrics.effectiveness_score,
                    "duration": metrics.duration
                }
                for metrics in self.stage_history
            ]
        }
    
    def get_stage_requirements_summary(self) -> Dict[str, Any]:
        """获取各阶段要求汇总"""
        requirements_summary = {}
        
        for stage_type, stage in self.stages.items():
            requirements_summary[stage_type.value] = stage.get_stage_requirements()
        
        return requirements_summary
    
    def export_stage_analytics(self) -> Dict[str, Any]:
        """导出阶段分析数据"""
        analytics = {
            "execution_summary": {
                "total_stages_executed": len(self.stage_history),
                "average_effectiveness": np.mean([m.effectiveness_score for m in self.stage_history]) if self.stage_history else 0.0,
                "total_duration": sum(m.duration for m in self.stage_history),
                "transition_config": self.transition_config
            },
            "stage_performance": [
                {
                    "stage": metrics.stage.value,
                    "effectiveness_score": metrics.effectiveness_score,
                    "duration": metrics.duration,
                    "target_achievement": metrics.calculate_target_achievement(),
                    "user_feedback": metrics.user_feedback,
                    "physiological_changes": metrics.physiological_changes
                }
                for metrics in self.stage_history
            ],
            "adaptation_insights": {
                "global_config": self.global_adaptation_config.__dict__,
                "stage_specific_adaptations": {
                    stage_type.value: stage.adjustment_history
                    for stage_type, stage in self.stages.items()
                }
            }
        }
        
        return analytics


# 工具函数
def create_iso_stage_manager(va_model: ValenceArousalModel,
                           sleep_model: SleepPhysiologyModel, 
                           music_model: MusicPsychologyModel) -> ISOStageManager:
    """创建ISO阶段管理器"""
    return ISOStageManager(va_model, sleep_model, music_model)