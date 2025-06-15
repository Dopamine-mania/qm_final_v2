"""
疗愈系统核心模块
整合理论模型与AI生成，实现完整的睡前疗愈流程
"""

import numpy as np
import time
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta

# 导入理论模块
from ..research.theory.iso_principle import ISOPrinciple, EmotionState, ISOStage, ISOStageConfig
from ..research.theory.valence_arousal import ValenceArousalModel, MultimodalEmotion
from ..research.theory.sleep_physiology import SleepPhysiologyModel, PhysiologicalState, SleepStage
from ..research.theory.music_psychology import MusicPsychologyModel, MusicalCharacteristics

# 导入模型适配器
from ..models import ModelFactory, EmotionAnalyzer, MusicGenerator, VideoGenerator

logger = logging.getLogger(__name__)

class TherapySessionState(Enum):
    """疗愈会话状态"""
    INITIALIZING = "initializing"
    EMOTION_ANALYSIS = "emotion_analysis"
    TRAJECTORY_PLANNING = "trajectory_planning"
    CONTENT_GENERATION = "content_generation"
    THERAPY_ACTIVE = "therapy_active"
    SLEEP_INDUCTION = "sleep_induction"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class TherapyContext:
    """疗愈上下文"""
    user_id: str
    session_id: str
    initial_emotion: Optional[EmotionState] = None
    target_emotion: Optional[EmotionState] = None
    physiological_target: Optional[PhysiologicalState] = None
    duration_minutes: float = 20.0
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TherapyContent:
    """疗愈内容"""
    music_audio: Optional[np.ndarray] = None
    video_frames: Optional[np.ndarray] = None
    narrative_text: Optional[str] = None
    music_prescription: Optional[MusicalCharacteristics] = None
    visual_prescription: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TherapyProgress:
    """疗愈进度"""
    current_stage: ISOStage
    stage_progress: float  # 0.0-1.0
    overall_progress: float  # 0.0-1.0
    emotion_trajectory: List[EmotionState]
    physiological_trajectory: List[PhysiologicalState]
    content_generated: Dict[str, bool] = field(default_factory=dict)
    real_time_feedback: Dict[str, Any] = field(default_factory=dict)

class TherapySession:
    """
    疗愈会话
    
    代表一次完整的睡前疗愈流程，包含:
    - 情绪分析与识别
    - 个性化轨迹规划
    - 多模态内容生成
    - 实时疗愈监控
    """
    
    def __init__(self, 
                 context: TherapyContext,
                 orchestrator: 'TherapyOrchestrator'):
        self.context = context
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{context.session_id[:8]}")
        
        # 会话状态
        self.state = TherapySessionState.INITIALIZING
        self.start_time = time.time()
        self.end_time = None
        
        # 疗愈内容
        self.therapy_content = TherapyContent()
        self.progress = TherapyProgress(
            current_stage=ISOStage.SYNCHRONIZATION,
            stage_progress=0.0,
            overall_progress=0.0,
            emotion_trajectory=[],
            physiological_trajectory=[]
        )
        
        # ISO阶段配置
        self.iso_stages: List[ISOStageConfig] = []
        
        # 生成任务
        self._generation_tasks = {}
        
        self.logger.info(f"疗愈会话初始化: {context.user_id}")
    
    async def start_session(self) -> bool:
        """开始疗愈会话"""
        try:
            self.logger.info("开始疗愈会话")
            
            # 1. 情绪分析
            await self._analyze_emotions()
            
            # 2. 轨迹规划
            await self._plan_trajectory()
            
            # 3. 内容生成
            await self._generate_content()
            
            # 4. 开始疗愈
            await self._start_therapy()
            
            return True
            
        except Exception as e:
            self.logger.error(f"疗愈会话启动失败: {e}")
            self.state = TherapySessionState.ERROR
            return False
    
    async def _analyze_emotions(self) -> None:
        """分析用户情绪"""
        self.state = TherapySessionState.EMOTION_ANALYSIS
        self.logger.info("开始情绪分析")
        
        # 从上下文获取输入数据
        text_input = self.context.user_preferences.get("text_input")
        audio_input = self.context.user_preferences.get("audio_input")
        
        if text_input or audio_input:
            # 使用情绪分析器
            emotion_result = await self.orchestrator.analyze_emotion(text_input, audio_input)
            self.context.initial_emotion = emotion_result.fused_emotion
        else:
            # 默认情绪状态（中等焦虑）
            self.context.initial_emotion = EmotionState(
                valence=-0.2, arousal=0.4, confidence=0.7
            )
        
        self.logger.info(f"情绪分析完成: V={self.context.initial_emotion.valence:.2f}, A={self.context.initial_emotion.arousal:.2f}")
    
    async def _plan_trajectory(self) -> None:
        """规划情绪轨迹"""
        self.state = TherapySessionState.TRAJECTORY_PLANNING
        self.logger.info("开始轨迹规划")
        
        # 计算目标情绪
        self.context.target_emotion = self.orchestrator.va_model.get_sleep_conducive_target(
            self.context.initial_emotion,
            sleep_preference=self.context.user_preferences.get("sleep_preference", "gentle")
        )
        
        # 生成ISO阶段
        self.iso_stages = self.orchestrator.iso_planner.create_iso_stages(
            self.context.initial_emotion,
            total_duration=self.context.duration_minutes * 60
        )
        
        # 生成情绪轨迹
        emotion_trajectory = self.orchestrator.iso_planner.generate_emotion_trajectory(
            self.iso_stages,
            num_points=int(self.context.duration_minutes * 2)  # 每30秒一个点
        )
        self.progress.emotion_trajectory = emotion_trajectory
        
        # 生成生理轨迹
        initial_physiology = self.orchestrator.sleep_model.analyze_current_state(
            self.context.initial_emotion
        )
        physiological_trajectory = self.orchestrator.sleep_model.generate_therapy_progression(
            initial_physiology,
            target_sleep_stage=SleepStage.LIGHT_SLEEP,
            duration_minutes=self.context.duration_minutes
        )
        self.progress.physiological_trajectory = physiological_trajectory
        self.context.physiological_target = physiological_trajectory[-1]
        
        self.logger.info(f"轨迹规划完成: {len(emotion_trajectory)}个情绪点, {len(physiological_trajectory)}个生理点")
    
    async def _generate_content(self) -> None:
        """生成疗愈内容"""
        self.state = TherapySessionState.CONTENT_GENERATION
        self.logger.info("开始内容生成")
        
        # 并行生成音乐和视频
        tasks = []
        
        # 音乐生成任务
        if self.context.user_preferences.get("enable_music", True):
            music_task = asyncio.create_task(
                self._generate_music_content()
            )
            tasks.append(("music", music_task))
        
        # 视频生成任务
        if self.context.user_preferences.get("enable_video", True):
            video_task = asyncio.create_task(
                self._generate_video_content()
            )
            tasks.append(("video", video_task))
        
        # 等待生成完成
        for content_type, task in tasks:
            try:
                await task
                self.progress.content_generated[content_type] = True
                self.logger.info(f"{content_type}内容生成完成")
            except Exception as e:
                self.logger.error(f"{content_type}内容生成失败: {e}")
                self.progress.content_generated[content_type] = False
        
        self.logger.info("内容生成阶段完成")
    
    async def _generate_music_content(self) -> None:
        """生成音乐内容"""
        # 生成音乐处方
        music_prescription = self.orchestrator.music_model.generate_musical_prescription(
            self.context.initial_emotion,
            self.context.target_emotion,
            duration_minutes=self.context.duration_minutes,
            sleep_context=True
        )
        self.therapy_content.music_prescription = music_prescription
        
        # 创建治疗性提示词
        music_prompt = self.orchestrator._create_therapeutic_music_prompt(
            self.context.initial_emotion,
            music_prescription
        )
        
        # 生成音乐
        music_result = await self.orchestrator.generate_music(
            music_prompt,
            duration=self.context.duration_minutes * 60
        )
        
        if music_result and "audio" in music_result:
            self.therapy_content.music_audio = music_result["audio"]
            self.therapy_content.metadata["music"] = music_result
    
    async def _generate_video_content(self) -> None:
        """生成视频内容"""
        # 创建治疗性视频提示词
        video_prompt = self.orchestrator._create_therapeutic_video_prompt(
            self.context.initial_emotion,
            self.context.physiological_target
        )
        self.therapy_content.visual_prescription = video_prompt
        
        # 生成视频
        video_result = await self.orchestrator.generate_video(
            video_prompt,
            duration=min(self.context.duration_minutes * 60, 60)  # 最长60秒
        )
        
        if video_result and "frames" in video_result:
            self.therapy_content.video_frames = video_result["frames"]
            self.therapy_content.metadata["video"] = video_result
    
    async def _start_therapy(self) -> None:
        """开始疗愈过程"""
        self.state = TherapySessionState.THERAPY_ACTIVE
        self.logger.info("开始疗愈过程")
        
        # 模拟疗愈过程（在实际应用中，这里会控制音视频播放）
        total_duration = self.context.duration_minutes * 60
        update_interval = 5.0  # 每5秒更新一次
        
        for elapsed in np.arange(0, total_duration, update_interval):
            # 更新进度
            self.progress.overall_progress = elapsed / total_duration
            
            # 确定当前阶段
            self._update_current_stage(elapsed, total_duration)
            
            # 模拟实时反馈
            await self._collect_real_time_feedback()
            
            # 等待下一次更新
            await asyncio.sleep(update_interval)
        
        # 进入睡眠诱导阶段
        self.state = TherapySessionState.SLEEP_INDUCTION
        await self._finalize_sleep_induction()
        
        # 完成疗愈
        self.state = TherapySessionState.COMPLETED
        self.end_time = time.time()
        self.logger.info(f"疗愈会话完成，总时长: {self.end_time - self.start_time:.1f}秒")
    
    def _update_current_stage(self, elapsed: float, total_duration: float) -> None:
        """更新当前阶段"""
        # 计算阶段边界
        sync_end = total_duration * 0.25
        guidance_end = total_duration * 0.75
        
        if elapsed < sync_end:
            self.progress.current_stage = ISOStage.SYNCHRONIZATION
            self.progress.stage_progress = elapsed / sync_end
        elif elapsed < guidance_end:
            self.progress.current_stage = ISOStage.GUIDANCE
            self.progress.stage_progress = (elapsed - sync_end) / (guidance_end - sync_end)
        else:
            self.progress.current_stage = ISOStage.CONSOLIDATION
            self.progress.stage_progress = (elapsed - guidance_end) / (total_duration - guidance_end)
    
    async def _collect_real_time_feedback(self) -> None:
        """收集实时反馈"""
        # 在实际应用中，这里可以：
        # 1. 监测用户的生理指标（心率、脑波等）
        # 2. 分析用户的行为模式
        # 3. 动态调整疗愈参数
        
        current_time = time.time()
        self.progress.real_time_feedback = {
            "timestamp": current_time,
            "stage": self.progress.current_stage.value,
            "progress": self.progress.overall_progress,
            "estimated_arousal": max(-0.8, -0.2 - self.progress.overall_progress * 0.6),
            "estimated_valence": min(0.3, self.progress.overall_progress * 0.5)
        }
    
    async def _finalize_sleep_induction(self) -> None:
        """完成睡眠诱导"""
        self.logger.info("进入睡眠诱导阶段")
        
        # 生成最终的睡眠诱导状态
        final_emotion = EmotionState(
            valence=0.1,
            arousal=-0.8,
            confidence=0.9
        )
        
        self.progress.emotion_trajectory.append(final_emotion)
        await asyncio.sleep(2.0)  # 短暂延时以完成过渡
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "session_id": self.context.session_id,
            "state": self.state.value,
            "progress": {
                "overall": self.progress.overall_progress,
                "stage": self.progress.current_stage.value,
                "stage_progress": self.progress.stage_progress
            },
            "content_status": self.progress.content_generated,
            "duration_elapsed": time.time() - self.start_time,
            "real_time_feedback": self.progress.real_time_feedback
        }
    
    def export_session_data(self) -> Dict[str, Any]:
        """导出会话数据"""
        return {
            "context": {
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "duration_minutes": self.context.duration_minutes,
                "initial_emotion": self.context.initial_emotion.__dict__ if self.context.initial_emotion else None,
                "target_emotion": self.context.target_emotion.__dict__ if self.context.target_emotion else None,
                "user_preferences": self.context.user_preferences,
                "timestamp": self.context.timestamp.isoformat()
            },
            "progress": {
                "final_state": self.state.value,
                "emotion_trajectory_length": len(self.progress.emotion_trajectory),
                "physiological_trajectory_length": len(self.progress.physiological_trajectory),
                "content_generated": self.progress.content_generated
            },
            "performance": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": self.end_time - self.start_time if self.end_time else None
            },
            "therapy_content": {
                "has_music": self.therapy_content.music_audio is not None,
                "has_video": self.therapy_content.video_frames is not None,
                "music_prescription": self.therapy_content.music_prescription.__dict__ if self.therapy_content.music_prescription else None,
                "visual_prescription": self.therapy_content.visual_prescription,
                "metadata": self.therapy_content.metadata
            }
        }


class EmotionTrajectoryPlanner:
    """
    情绪轨迹规划器
    
    负责基于用户状态和目标生成个性化的情绪转换路径
    """
    
    def __init__(self, 
                 iso_planner: ISOPrinciple,
                 va_model: ValenceArousalModel,
                 sleep_model: SleepPhysiologyModel):
        self.iso_planner = iso_planner
        self.va_model = va_model
        self.sleep_model = sleep_model
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 轨迹优化参数
        self.optimization_params = {
            "smoothness_weight": 0.3,
            "naturalness_weight": 0.4,
            "effectiveness_weight": 0.3,
            "max_iterations": 10
        }
    
    def plan_personalized_trajectory(self, 
                                   initial_state: EmotionState,
                                   target_state: EmotionState,
                                   user_profile: Optional[Dict[str, Any]] = None,
                                   duration_minutes: float = 20.0) -> List[EmotionState]:
        """规划个性化情绪轨迹"""
        
        # 基础ISO轨迹
        iso_stages = self.iso_planner.create_iso_stages(
            initial_state,
            total_duration=duration_minutes * 60
        )
        
        base_trajectory = self.iso_planner.generate_emotion_trajectory(
            iso_stages,
            num_points=int(duration_minutes * 2)
        )
        
        # 个性化调整
        if user_profile:
            personalized_trajectory = self._apply_personalization(
                base_trajectory,
                user_profile,
                initial_state,
                target_state
            )
        else:
            personalized_trajectory = base_trajectory
        
        # 轨迹优化
        optimized_trajectory = self._optimize_trajectory(
            personalized_trajectory,
            target_state
        )
        
        self.logger.info(f"轨迹规划完成: {len(optimized_trajectory)}个点")
        return optimized_trajectory
    
    def _apply_personalization(self, 
                             base_trajectory: List[EmotionState],
                             user_profile: Dict[str, Any],
                             initial_state: EmotionState,
                             target_state: EmotionState) -> List[EmotionState]:
        """应用个性化调整"""
        
        # 用户偏好因子
        emotion_sensitivity = user_profile.get("emotion_sensitivity", 0.5)  # 0-1
        transition_preference = user_profile.get("transition_preference", "moderate")  # slow, moderate, fast
        
        adjusted_trajectory = []
        
        for i, state in enumerate(base_trajectory):
            # 调整过渡速度
            if transition_preference == "slow":
                # 更保守的变化
                progress = i / (len(base_trajectory) - 1)
                adjusted_progress = progress ** 1.5  # 减缓变化
            elif transition_preference == "fast":
                # 更激进的变化
                progress = i / (len(base_trajectory) - 1)
                adjusted_progress = progress ** 0.7  # 加速变化
            else:
                adjusted_progress = i / (len(base_trajectory) - 1)
            
            # 重新插值
            adjusted_valence = (
                initial_state.valence * (1 - adjusted_progress) +
                target_state.valence * adjusted_progress
            )
            adjusted_arousal = (
                initial_state.arousal * (1 - adjusted_progress) +
                target_state.arousal * adjusted_progress
            )
            
            # 应用敏感度调整
            valence_adjustment = (adjusted_valence - state.valence) * emotion_sensitivity
            arousal_adjustment = (adjusted_arousal - state.arousal) * emotion_sensitivity
            
            adjusted_state = EmotionState(
                valence=state.valence + valence_adjustment,
                arousal=state.arousal + arousal_adjustment,
                confidence=state.confidence,
                timestamp=state.timestamp
            )
            
            adjusted_trajectory.append(adjusted_state)
        
        return adjusted_trajectory
    
    def _optimize_trajectory(self, 
                           trajectory: List[EmotionState],
                           target_state: EmotionState) -> List[EmotionState]:
        """优化轨迹质量"""
        
        best_trajectory = trajectory.copy()
        best_score = self._evaluate_trajectory_quality(trajectory, target_state)
        
        for iteration in range(self.optimization_params["max_iterations"]):
            # 尝试微调
            candidate_trajectory = self._apply_trajectory_smoothing(best_trajectory)
            candidate_score = self._evaluate_trajectory_quality(candidate_trajectory, target_state)
            
            if candidate_score > best_score:
                best_trajectory = candidate_trajectory
                best_score = candidate_score
                self.logger.debug(f"轨迹优化迭代{iteration}: 分数提升到{best_score:.3f}")
            else:
                break  # 没有改善，停止优化
        
        return best_trajectory
    
    def _evaluate_trajectory_quality(self, 
                                   trajectory: List[EmotionState],
                                   target_state: EmotionState) -> float:
        """评估轨迹质量"""
        if len(trajectory) < 2:
            return 0.0
        
        # 平滑度评估
        smoothness_score = self._calculate_smoothness_score(trajectory)
        
        # 自然度评估
        naturalness_score = self._calculate_naturalness_score(trajectory)
        
        # 目标达成度评估
        final_state = trajectory[-1]
        distance_to_target = final_state.distance_to(target_state)
        effectiveness_score = max(0, 1.0 - distance_to_target / 2.0)
        
        # 加权综合分数
        total_score = (
            smoothness_score * self.optimization_params["smoothness_weight"] +
            naturalness_score * self.optimization_params["naturalness_weight"] +
            effectiveness_score * self.optimization_params["effectiveness_weight"]
        )
        
        return total_score
    
    def _calculate_smoothness_score(self, trajectory: List[EmotionState]) -> float:
        """计算平滑度分数"""
        if len(trajectory) < 3:
            return 1.0
        
        # 计算二阶差分的方差
        valences = [state.valence for state in trajectory]
        arousals = [state.arousal for state in trajectory]
        
        valence_second_diff = np.diff(np.diff(valences))
        arousal_second_diff = np.diff(np.diff(arousals))
        
        valence_smoothness = 1.0 / (1.0 + np.var(valence_second_diff))
        arousal_smoothness = 1.0 / (1.0 + np.var(arousal_second_diff))
        
        return (valence_smoothness + arousal_smoothness) / 2.0
    
    def _calculate_naturalness_score(self, trajectory: List[EmotionState]) -> float:
        """计算自然度分数"""
        if len(trajectory) < 2:
            return 1.0
        
        # 计算相邻点距离的方差
        distances = []
        for i in range(1, len(trajectory)):
            distance = trajectory[i].distance_to(trajectory[i-1])
            distances.append(distance)
        
        # 自然度 = 1 / (1 + 距离方差)
        naturalness = 1.0 / (1.0 + np.var(distances))
        
        return naturalness
    
    def _apply_trajectory_smoothing(self, trajectory: List[EmotionState]) -> List[EmotionState]:
        """应用轨迹平滑"""
        if len(trajectory) < 3:
            return trajectory
        
        smoothed_trajectory = [trajectory[0]]  # 保持起点
        
        # 应用移动平均
        for i in range(1, len(trajectory) - 1):
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            next_state = trajectory[i+1]
            
            # 三点平均
            smoothed_valence = (prev_state.valence + curr_state.valence + next_state.valence) / 3
            smoothed_arousal = (prev_state.arousal + curr_state.arousal + next_state.arousal) / 3
            
            smoothed_state = EmotionState(
                valence=smoothed_valence,
                arousal=smoothed_arousal,
                confidence=curr_state.confidence,
                timestamp=curr_state.timestamp
            )
            
            smoothed_trajectory.append(smoothed_state)
        
        smoothed_trajectory.append(trajectory[-1])  # 保持终点
        
        return smoothed_trajectory


class SleepInductionEngine:
    """
    睡眠诱导引擎
    
    专门负责睡眠过渡阶段的优化和控制
    """
    
    def __init__(self, 
                 sleep_model: SleepPhysiologyModel,
                 music_model: MusicPsychologyModel):
        self.sleep_model = sleep_model
        self.music_model = music_model
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 睡眠诱导策略
        self.induction_strategies = {
            "gentle": {
                "arousal_target": -0.7,
                "valence_target": 0.1,
                "transition_time": 300,  # 5分钟
                "music_tempo_range": [40, 55]
            },
            "deep": {
                "arousal_target": -0.9,
                "valence_target": 0.0,
                "transition_time": 600,  # 10分钟
                "music_tempo_range": [35, 45]
            },
            "natural": {
                "arousal_target": -0.6,
                "valence_target": 0.2,
                "transition_time": 240,  # 4分钟
                "music_tempo_range": [45, 60]
            }
        }
    
    def create_sleep_induction_plan(self, 
                                  current_state: EmotionState,
                                  strategy: str = "gentle") -> Dict[str, Any]:
        """创建睡眠诱导计划"""
        
        if strategy not in self.induction_strategies:
            strategy = "gentle"
        
        strategy_config = self.induction_strategies[strategy]
        
        # 目标状态
        target_emotion = EmotionState(
            valence=strategy_config["valence_target"],
            arousal=strategy_config["arousal_target"],
            confidence=0.9
        )
        
        # 生理目标
        target_physiology = self.sleep_model.analyze_current_state(target_emotion)
        
        # 音乐参数
        music_params = self.sleep_model.calculate_music_parameters(target_physiology)
        
        # 睡眠诱导计划
        plan = {
            "strategy": strategy,
            "target_emotion": target_emotion,
            "target_physiology": target_physiology,
            "transition_time": strategy_config["transition_time"],
            "music_parameters": music_params,
            "recommended_environment": self._get_environment_recommendations(strategy),
            "monitoring_points": self._create_monitoring_schedule(strategy_config["transition_time"])
        }
        
        self.logger.info(f"睡眠诱导计划已创建: {strategy}策略")
        return plan
    
    def _get_environment_recommendations(self, strategy: str) -> Dict[str, Any]:
        """获取环境建议"""
        base_recommendations = {
            "room_temperature": "18-21°C",
            "lighting": "dim warm light",
            "noise_level": "quiet",
            "air_quality": "well ventilated"
        }
        
        strategy_specific = {
            "gentle": {
                "lighting": "very dim amber light",
                "sound_level": "whisper quiet"
            },
            "deep": {
                "lighting": "complete darkness",
                "sound_level": "silent",
                "temperature": "16-19°C"
            },
            "natural": {
                "lighting": "natural sunset simulation",
                "sound_level": "nature sounds optional"
            }
        }
        
        recommendations = base_recommendations.copy()
        if strategy in strategy_specific:
            recommendations.update(strategy_specific[strategy])
        
        return recommendations
    
    def _create_monitoring_schedule(self, transition_time: int) -> List[Dict[str, Any]]:
        """创建监测时间表"""
        monitoring_points = []
        
        # 每分钟监测一次
        for minute in range(0, transition_time // 60 + 1):
            monitoring_points.append({
                "time_offset": minute * 60,
                "check_points": [
                    "heart_rate_trend",
                    "movement_reduction",
                    "breathing_pattern",
                    "response_latency"
                ],
                "adjustment_threshold": 0.1
            })
        
        return monitoring_points
    
    def evaluate_sleep_readiness(self, 
                                current_state: EmotionState,
                                physiological_indicators: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """评估睡眠准备度"""
        
        # 基于情绪状态的评估
        emotion_readiness = self._evaluate_emotion_readiness(current_state)
        
        # 基于生理指标的评估（如果可用）
        if physiological_indicators:
            physiology_readiness = self._evaluate_physiology_readiness(physiological_indicators)
        else:
            physiology_readiness = 0.5  # 默认值
        
        # 综合评估
        overall_readiness = (emotion_readiness * 0.6 + physiology_readiness * 0.4)
        
        # 睡眠准备度分级
        if overall_readiness >= 0.8:
            readiness_level = "excellent"
            recommendation = "可以开始入睡"
        elif overall_readiness >= 0.6:
            readiness_level = "good"
            recommendation = "继续放松5-10分钟"
        elif overall_readiness >= 0.4:
            readiness_level = "moderate"
            recommendation = "需要额外的放松时间"
        else:
            readiness_level = "low"
            recommendation = "建议延长疗愈时间或调整策略"
        
        return {
            "overall_readiness": overall_readiness,
            "emotion_readiness": emotion_readiness,
            "physiology_readiness": physiology_readiness,
            "readiness_level": readiness_level,
            "recommendation": recommendation,
            "optimal_sleep_window": self._calculate_optimal_sleep_window(overall_readiness)
        }
    
    def _evaluate_emotion_readiness(self, current_state: EmotionState) -> float:
        """评估情绪睡眠准备度"""
        
        # 理想睡前情绪区间
        ideal_valence_range = (0.0, 0.3)
        ideal_arousal_range = (-1.0, -0.5)
        
        # 计算与理想区间的匹配度
        valence_score = 1.0 if ideal_valence_range[0] <= current_state.valence <= ideal_valence_range[1] else \
                       max(0, 1.0 - abs(current_state.valence - np.mean(ideal_valence_range)) / 1.0)
        
        arousal_score = 1.0 if ideal_arousal_range[0] <= current_state.arousal <= ideal_arousal_range[1] else \
                       max(0, 1.0 - abs(current_state.arousal - np.mean(ideal_arousal_range)) / 1.0)
        
        # 置信度权重
        confidence_weight = current_state.confidence
        
        emotion_readiness = (valence_score * 0.3 + arousal_score * 0.7) * confidence_weight
        
        return emotion_readiness
    
    def _evaluate_physiology_readiness(self, indicators: Dict[str, float]) -> float:
        """评估生理睡眠准备度"""
        
        # 生理指标权重
        weights = {
            "heart_rate": 0.3,
            "breathing_rate": 0.2,
            "skin_temperature": 0.2,
            "movement_activity": 0.3
        }
        
        scores = []
        
        for indicator, value in indicators.items():
            if indicator == "heart_rate":
                # 心率：50-60 BPM为理想
                optimal_range = (50, 60)
                score = 1.0 if optimal_range[0] <= value <= optimal_range[1] else \
                       max(0, 1.0 - abs(value - np.mean(optimal_range)) / 20)
            
            elif indicator == "breathing_rate":
                # 呼吸率：12-16次/分钟为理想
                optimal_range = (12, 16)
                score = 1.0 if optimal_range[0] <= value <= optimal_range[1] else \
                       max(0, 1.0 - abs(value - np.mean(optimal_range)) / 8)
            
            elif indicator == "movement_activity":
                # 活动度：越低越好
                score = max(0, 1.0 - value)
            
            else:
                score = 0.5  # 未知指标的默认分数
            
            weight = weights.get(indicator, 0.1)
            scores.append(score * weight)
        
        return sum(scores) / sum(weights.values()) if scores else 0.5
    
    def _calculate_optimal_sleep_window(self, readiness: float) -> Dict[str, Any]:
        """计算最佳入睡时间窗口"""
        
        if readiness >= 0.8:
            window_start = 0  # 立即
            window_duration = 300  # 5分钟窗口
        elif readiness >= 0.6:
            window_start = 300  # 5分钟后
            window_duration = 600  # 10分钟窗口
        elif readiness >= 0.4:
            window_start = 600  # 10分钟后
            window_duration = 900  # 15分钟窗口
        else:
            window_start = 1200  # 20分钟后
            window_duration = 1800  # 30分钟窗口
        
        return {
            "window_start_seconds": window_start,
            "window_duration_seconds": window_duration,
            "optimal_time": window_start + window_duration // 2,
            "confidence": readiness
        }


class TherapyOrchestrator:
    """
    疗愈编排器
    
    系统的核心协调器，负责整合所有组件并管理疗愈流程
    """
    
    def __init__(self, 
                 model_factory: ModelFactory):
        self.model_factory = model_factory
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化理论模型
        self.iso_planner = ISOPrinciple()
        self.va_model = ValenceArousalModel()
        self.sleep_model = SleepPhysiologyModel()
        self.music_model = MusicPsychologyModel()
        
        # 初始化核心组件
        self.trajectory_planner = EmotionTrajectoryPlanner(
            self.iso_planner, self.va_model, self.sleep_model
        )
        self.sleep_induction_engine = SleepInductionEngine(
            self.sleep_model, self.music_model
        )
        
        # AI模型组件
        self.emotion_analyzer = None
        self.music_generator = None
        self.video_generator = None
        
        # 活跃会话
        self.active_sessions: Dict[str, TherapySession] = {}
        
        self.logger.info("疗愈编排器初始化完成")
    
    async def initialize_models(self) -> None:
        """初始化AI模型"""
        try:
            # 创建模型组件
            self.emotion_analyzer = self.model_factory.create_emotion_analyzer()
            self.music_generator = self.model_factory.create_music_generator()
            self.video_generator = self.model_factory.create_video_generator()
            
            self.logger.info("AI模型初始化完成")
        except Exception as e:
            self.logger.error(f"AI模型初始化失败: {e}")
            raise
    
    async def create_therapy_session(self, 
                                   user_id: str,
                                   user_input: Dict[str, Any],
                                   preferences: Optional[Dict[str, Any]] = None) -> str:
        """创建疗愈会话"""
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 创建疗愈上下文
        context = TherapyContext(
            user_id=user_id,
            session_id=session_id,
            duration_minutes=preferences.get("duration_minutes", 20.0) if preferences else 20.0,
            user_preferences={
                **(preferences or {}),
                **user_input
            }
        )
        
        # 创建疗愈会话
        session = TherapySession(context, self)
        self.active_sessions[session_id] = session
        
        self.logger.info(f"疗愈会话已创建: {session_id} (用户: {user_id})")
        
        return session_id
    
    async def start_therapy_session(self, session_id: str) -> bool:
        """启动疗愈会话"""
        if session_id not in self.active_sessions:
            self.logger.error(f"会话不存在: {session_id}")
            return False
        
        session = self.active_sessions[session_id]
        return await session.start_session()
    
    async def analyze_emotion(self, 
                            text_input: Optional[str] = None,
                            audio_input: Optional[Any] = None) -> MultimodalEmotion:
        """分析情绪"""
        if not self.emotion_analyzer:
            await self.initialize_models()
        
        text_emotion = None
        speech_emotion = None
        
        # 分析文本情绪
        if text_input:
            text_result = self.emotion_analyzer.analyze_text(text_input)
            text_emotion = text_result.get("emotion_state")
        
        # 分析语音情绪
        if audio_input:
            speech_result = self.emotion_analyzer.analyze_speech(audio_input)
            speech_emotion = speech_result.get("emotion_state")
        
        # 融合多模态情绪
        multimodal_emotion = self.va_model.fuse_multimodal_emotions(
            text_emotion, speech_emotion
        )
        
        return multimodal_emotion
    
    async def generate_music(self, 
                           prompt: str,
                           duration: float = 30.0) -> Optional[Dict[str, Any]]:
        """生成音乐"""
        if not self.music_generator:
            await self.initialize_models()
        
        try:
            result = self.music_generator.generate(
                prompt=prompt,
                duration=duration
            )
            return result
        except Exception as e:
            self.logger.error(f"音乐生成失败: {e}")
            return None
    
    async def generate_video(self, 
                           prompt: Union[str, Dict[str, str]],
                           duration: float = 10.0) -> Optional[Dict[str, Any]]:
        """生成视频"""
        if not self.video_generator:
            await self.initialize_models()
        
        try:
            result = self.video_generator.generate(
                prompt=prompt,
                duration=duration
            )
            return result
        except Exception as e:
            self.logger.error(f"视频生成失败: {e}")
            return None
    
    def _create_therapeutic_music_prompt(self, 
                                       emotion_state: EmotionState,
                                       music_prescription: MusicalCharacteristics) -> str:
        """创建治疗性音乐提示词"""
        # 使用音乐生成适配器的方法
        if hasattr(self.music_generator.model, 'create_therapeutic_prompt'):
            return self.music_generator.model.create_therapeutic_prompt(
                emotion_state, music_prescription
            )
        else:
            # 基础提示词生成
            return f"therapeutic sleep music, {music_prescription.tempo_bpm:.0f} BPM, {music_prescription.key.value}, peaceful, calming"
    
    def _create_therapeutic_video_prompt(self, 
                                       emotion_state: EmotionState,
                                       physiological_target: PhysiologicalState) -> Dict[str, str]:
        """创建治疗性视频提示词"""
        # 使用视频生成适配器的方法
        if hasattr(self.video_generator.model, 'create_therapeutic_video_prompt'):
            return self.video_generator.model.create_therapeutic_video_prompt(
                emotion_state, physiological_target
            )
        else:
            # 基础提示词生成
            return {
                "positive_prompt": "peaceful nature scene, gentle movement, calming colors, therapeutic",
                "negative_prompt": "no people, no text, no sudden movements, no bright lights"
            }
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return session.get_current_state()
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """列出活跃会话"""
        return [
            {
                "session_id": session_id,
                "user_id": session.context.user_id,
                "state": session.state.value,
                "duration_elapsed": time.time() - session.start_time
            }
            for session_id, session in self.active_sessions.items()
        ]
    
    async def cleanup_completed_sessions(self) -> None:
        """清理已完成的会话"""
        completed_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session.state in [TherapySessionState.COMPLETED, TherapySessionState.ERROR]:
                completed_sessions.append(session_id)
        
        for session_id in completed_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"清理已完成会话: {session_id}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "active_sessions": len(self.active_sessions),
            "model_factory_stats": self.model_factory.get_usage_statistics(),
            "session_states": {
                state.value: sum(
                    1 for session in self.active_sessions.values()
                    if session.state == state
                )
                for state in TherapySessionState
            }
        }


# 工具函数
async def create_therapy_orchestrator(model_factory: Optional[ModelFactory] = None) -> TherapyOrchestrator:
    """创建疗愈编排器"""
    if model_factory is None:
        from ..models import get_global_factory
        model_factory = get_global_factory()
    
    orchestrator = TherapyOrchestrator(model_factory)
    await orchestrator.initialize_models()
    
    return orchestrator