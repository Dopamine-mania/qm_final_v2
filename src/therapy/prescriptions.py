"""
疗愈处方系统
将理论分析转换为具体的多模态内容生成指令
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# 导入理论模块
from ..research.theory.iso_principle import EmotionState, ISOStage
from ..research.theory.valence_arousal import BasicEmotion, EmotionQuadrant
from ..research.theory.sleep_physiology import PhysiologicalState, SleepStage, BrainwaveType
from ..research.theory.music_psychology import MusicalCharacteristics, InstrumentFamily, MusicalKey

logger = logging.getLogger(__name__)

class PrescriptionType(Enum):
    """处方类型"""
    MUSIC = "music"
    VISUAL = "visual"
    NARRATIVE = "narrative"
    COMBINED = "combined"

class ContentComplexity(Enum):
    """内容复杂度"""
    MINIMAL = "minimal"
    SIMPLE = "simple"
    MODERATE = "moderate"
    RICH = "rich"
    COMPLEX = "complex"

class TherapeuticIntensity(Enum):
    """治疗强度"""
    GENTLE = "gentle"
    MODERATE = "moderate"
    INTENSIVE = "intensive"
    DEEP = "deep"

@dataclass
class PrescriptionMetadata:
    """处方元数据"""
    prescription_id: str
    created_at: datetime
    user_profile_id: Optional[str] = None
    session_context: Dict[str, Any] = field(default_factory=dict)
    theoretical_basis: Dict[str, Any] = field(default_factory=dict)
    expected_outcomes: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    adaptation_notes: str = ""

@dataclass
class TherapyPrescription:
    """
    治疗处方基类
    
    包含所有疗愈内容生成的核心指导信息
    """
    prescription_type: PrescriptionType
    target_emotion: EmotionState
    current_emotion: EmotionState
    therapeutic_goals: List[str]
    duration_minutes: float
    complexity: ContentComplexity
    intensity: TherapeuticIntensity
    metadata: PrescriptionMetadata
    
    # 个性化参数
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    
    # 质量控制
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    effectiveness_predictors: Dict[str, float] = field(default_factory=dict)
    
    def calculate_emotional_distance(self) -> float:
        """计算情绪距离"""
        return self.current_emotion.distance_to(self.target_emotion)
    
    def get_transition_direction(self) -> Dict[str, str]:
        """获取情绪转换方向"""
        valence_change = self.target_emotion.valence - self.current_emotion.valence
        arousal_change = self.target_emotion.arousal - self.current_emotion.arousal
        
        return {
            "valence_direction": "increase" if valence_change > 0.1 else "decrease" if valence_change < -0.1 else "maintain",
            "arousal_direction": "increase" if arousal_change > 0.1 else "decrease" if arousal_change < -0.1 else "maintain",
            "overall_intensity": "major" if abs(valence_change) + abs(arousal_change) > 1.0 else "moderate" if abs(valence_change) + abs(arousal_change) > 0.5 else "minor"
        }
    
    def estimate_completion_probability(self) -> float:
        """估计完成概率"""
        # 基于情绪距离、用户历史、内容适配性等因素
        distance_factor = max(0.1, 1.0 - self.calculate_emotional_distance() / 2.0)
        complexity_factor = {
            ContentComplexity.MINIMAL: 0.95,
            ContentComplexity.SIMPLE: 0.90,
            ContentComplexity.MODERATE: 0.85,
            ContentComplexity.RICH: 0.80,
            ContentComplexity.COMPLEX: 0.70
        }.get(self.complexity, 0.80)
        
        user_compatibility = self.user_preferences.get("compatibility_score", 0.7)
        
        return (distance_factor * 0.4 + complexity_factor * 0.3 + user_compatibility * 0.3)


@dataclass 
class MusicPrescription(TherapyPrescription):
    """
    音乐疗愈处方
    
    包含音乐生成的详细指导参数
    """
    musical_characteristics: MusicalCharacteristics
    prompt_template: str
    audio_specifications: Dict[str, Any] = field(default_factory=dict)
    progression_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.prescription_type != PrescriptionType.MUSIC:
            self.prescription_type = PrescriptionType.MUSIC
        
        # 默认音频规格
        if not self.audio_specifications:
            self.audio_specifications = {
                "sample_rate": 44100,
                "bit_depth": 16,
                "channels": 2,
                "format": "wav",
                "dynamic_range": "high",
                "frequency_response": "full_range"
            }
    
    def generate_therapeutic_prompt(self) -> str:
        """生成治疗性音乐提示词"""
        prompt_components = []
        
        # 基础治疗意图
        therapeutic_intent = self._get_therapeutic_intent()
        prompt_components.append(therapeutic_intent)
        
        # 音乐风格和流派
        style_description = self._get_style_description()
        prompt_components.append(style_description)
        
        # 技术特征
        technical_specs = self._get_technical_specifications()
        prompt_components.append(technical_specs)
        
        # 情绪特征
        emotional_qualities = self._get_emotional_qualities()
        prompt_components.append(emotional_qualities)
        
        # 睡眠优化特征
        sleep_optimization = self._get_sleep_optimization_features()
        prompt_components.append(sleep_optimization)
        
        # 质量和安全约束
        safety_features = self._get_safety_constraints()
        prompt_components.append(safety_features)
        
        # 组合成完整提示词
        full_prompt = ", ".join(filter(None, prompt_components))
        
        return full_prompt
    
    def _get_therapeutic_intent(self) -> str:
        """获取治疗意图描述"""
        distance = self.calculate_emotional_distance()
        transition = self.get_transition_direction()
        
        if distance > 1.5:
            return "deep therapeutic music for significant emotional transformation"
        elif distance > 0.8:
            return "therapeutic music for moderate emotional adjustment"
        elif transition["arousal_direction"] == "decrease":
            return "calming therapeutic music for relaxation and sleep preparation"
        else:
            return "gentle therapeutic music for emotional stabilization"
    
    def _get_style_description(self) -> str:
        """获取风格描述"""
        style_elements = []
        
        # 基于乐器选择
        primary_instruments = self.musical_characteristics.primary_instruments
        if InstrumentFamily.AMBIENT in primary_instruments:
            style_elements.append("ambient soundscape")
        if InstrumentFamily.STRINGS in primary_instruments:
            style_elements.append("string ensemble")
        if InstrumentFamily.KEYBOARD in primary_instruments:
            style_elements.append("piano meditation")
        
        # 基于复杂度
        if self.complexity == ContentComplexity.MINIMAL:
            style_elements.append("minimalist composition")
        elif self.complexity == ContentComplexity.RICH:
            style_elements.append("rich harmonic texture")
        
        return ", ".join(style_elements) if style_elements else "therapeutic ambient music"
    
    def _get_technical_specifications(self) -> str:
        """获取技术规格描述"""
        specs = []
        
        # 节奏
        tempo = self.musical_characteristics.tempo_bpm
        if tempo <= 60:
            specs.append("very slow tempo")
        elif tempo <= 80:
            specs.append("slow tempo")
        else:
            specs.append(f"{tempo:.0f} BPM")
        
        # 调性
        key_name = self.musical_characteristics.key.value.replace("_", " ")
        specs.append(f"{key_name}")
        
        # 拍号
        time_sig = self.musical_characteristics.time_signature
        specs.append(f"{time_sig[0]}/{time_sig[1]} time")
        
        # 力度
        dynamics = self.musical_characteristics.dynamics
        dynamics_desc = {
            "pp": "very soft dynamics",
            "p": "soft dynamics", 
            "mp": "medium soft dynamics",
            "mf": "medium dynamics"
        }.get(dynamics, "gentle dynamics")
        specs.append(dynamics_desc)
        
        return ", ".join(specs)
    
    def _get_emotional_qualities(self) -> str:
        """获取情绪质量描述"""
        qualities = []
        
        # 基于目标情绪
        if self.target_emotion.valence > 0.3:
            qualities.append("uplifting")
        elif self.target_emotion.valence < -0.3:
            qualities.append("melancholic")
        else:
            qualities.append("neutral")
        
        if self.target_emotion.arousal < -0.5:
            qualities.append("deeply calming")
        elif self.target_emotion.arousal < 0:
            qualities.append("relaxing")
        else:
            qualities.append("gentle")
        
        # 基于治疗强度
        if self.intensity == TherapeuticIntensity.DEEP:
            qualities.append("profoundly peaceful")
        elif self.intensity == TherapeuticIntensity.GENTLE:
            qualities.append("subtly soothing")
        
        return ", ".join(qualities)
    
    def _get_sleep_optimization_features(self) -> str:
        """获取睡眠优化特征"""
        features = []
        
        # 基于目标睡眠状态
        if self.target_emotion.arousal < -0.6:
            features.extend([
                "sleep inducing",
                "delta wave entrainment",
                "deep relaxation"
            ])
        elif self.target_emotion.arousal < -0.3:
            features.extend([
                "pre-sleep relaxation",
                "alpha wave promotion"
            ])
        
        # 睡眠友好特征
        features.extend([
            "no sudden changes",
            "consistent volume",
            "smooth transitions"
        ])
        
        return ", ".join(features)
    
    def _get_safety_constraints(self) -> str:
        """获取安全约束"""
        constraints = []
        
        # 基础安全特征
        constraints.extend([
            "no jarring sounds",
            "no loud passages",
            "no dissonant harmonies"
        ])
        
        # 特定约束
        if self.safety_constraints.get("no_binaural_beats", False):
            constraints.append("no binaural effects")
        
        if self.safety_constraints.get("volume_limited", True):
            constraints.append("volume controlled")
        
        return ", ".join(constraints)
    
    def create_progression_stages(self, num_stages: int = 4) -> List[Dict[str, Any]]:
        """创建渐进阶段"""
        stages = []
        
        for i in range(num_stages):
            progress = i / (num_stages - 1)
            
            # 情绪插值
            stage_valence = (
                self.current_emotion.valence * (1 - progress) +
                self.target_emotion.valence * progress
            )
            stage_arousal = (
                self.current_emotion.arousal * (1 - progress) +
                self.target_emotion.arousal * progress
            )
            
            # BPM调整
            initial_bpm = min(120, max(40, abs(self.current_emotion.arousal) * 80 + 60))
            target_bpm = self.musical_characteristics.tempo_bpm
            stage_bpm = initial_bpm * (1 - progress) + target_bpm * progress
            
            stage = {
                "stage_number": i + 1,
                "duration_ratio": 1.0 / num_stages,
                "emotion_target": {
                    "valence": stage_valence,
                    "arousal": stage_arousal
                },
                "tempo_bpm": stage_bpm,
                "musical_emphasis": self._get_stage_emphasis(progress),
                "prompt_modifier": self._get_stage_prompt_modifier(progress)
            }
            
            stages.append(stage)
        
        self.progression_stages = stages
        return stages
    
    def _get_stage_emphasis(self, progress: float) -> str:
        """获取阶段强调重点"""
        if progress < 0.25:
            return "emotional_synchronization"
        elif progress < 0.75:
            return "gradual_transition"  
        else:
            return "target_stabilization"
    
    def _get_stage_prompt_modifier(self, progress: float) -> str:
        """获取阶段提示词修饰符"""
        if progress < 0.25:
            return "matching current emotional state"
        elif progress < 0.5:
            return "beginning gentle transition"
        elif progress < 0.75:
            return "progressing toward target state"
        else:
            return "establishing final peaceful state"
    
    def estimate_generation_parameters(self) -> Dict[str, Any]:
        """估计生成参数"""
        return {
            "estimated_tokens": len(self.generate_therapeutic_prompt().split()) * 1.2,
            "complexity_score": {
                ContentComplexity.MINIMAL: 0.2,
                ContentComplexity.SIMPLE: 0.4,
                ContentComplexity.MODERATE: 0.6,
                ContentComplexity.RICH: 0.8,
                ContentComplexity.COMPLEX: 1.0
            }.get(self.complexity, 0.6),
            "estimated_inference_time": self.duration_minutes * 2.5,  # 秒数
            "memory_requirement": "4-8GB",
            "recommended_model": "musicgen_medium" if self.complexity in [ContentComplexity.RICH, ContentComplexity.COMPLEX] else "musicgen_small"
        }


@dataclass
class VisualPrescription(TherapyPrescription):
    """
    视觉疗愈处方
    
    包含视频/视觉内容生成的详细指导参数
    """
    visual_style: str
    color_palette: Dict[str, Any]
    movement_characteristics: Dict[str, Any]
    visual_elements: List[str]
    lighting_specifications: Dict[str, Any] = field(default_factory=dict)
    composition_rules: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.prescription_type != PrescriptionType.VISUAL:
            self.prescription_type = PrescriptionType.VISUAL
        
        # 默认视觉规格
        if not self.lighting_specifications:
            self.lighting_specifications = {
                "primary_lighting": "soft_diffused",
                "color_temperature": "warm",
                "brightness_level": "low_to_medium", 
                "contrast_level": "low",
                "lighting_direction": "natural"
            }
        
        if not self.composition_rules:
            self.composition_rules = {
                "focal_point": "center_soft",
                "visual_balance": "symmetric",
                "depth_of_field": "shallow",
                "visual_flow": "circular_gentle"
            }
    
    def generate_therapeutic_prompt(self) -> Dict[str, str]:
        """生成治疗性视觉提示词"""
        # 正面提示词
        positive_components = []
        
        # 视觉场景
        scene_description = self._get_scene_description()
        positive_components.append(scene_description)
        
        # 颜色和照明
        color_lighting = self._get_color_lighting_description()
        positive_components.append(color_lighting)
        
        # 运动特征
        movement_desc = self._get_movement_description()
        positive_components.append(movement_desc)
        
        # 情绪质量
        emotional_qualities = self._get_visual_emotional_qualities()
        positive_components.append(emotional_qualities)
        
        # 技术质量
        technical_quality = self._get_technical_quality_specs()
        positive_components.append(technical_quality)
        
        # 负面提示词（避免的元素）
        negative_components = self._get_negative_constraints()
        
        return {
            "positive_prompt": ", ".join(filter(None, positive_components)),
            "negative_prompt": ", ".join(negative_components)
        }
    
    def _get_scene_description(self) -> str:
        """获取场景描述"""
        base_scenes = {
            "nature_calm": "peaceful nature landscape, gentle rolling hills, calm water",
            "abstract_flow": "abstract flowing shapes, organic forms, fluid movement",
            "cosmic_peaceful": "serene cosmic space, distant stars, gentle nebula",
            "geometric_harmony": "harmonious geometric patterns, soft mathematical beauty",
            "forest_tranquil": "tranquil forest scene, dappled sunlight, peaceful atmosphere"
        }
        
        # 基于情绪目标选择场景
        if self.target_emotion.arousal < -0.6:
            scene_type = "cosmic_peaceful"  # 深度放松
        elif self.target_emotion.valence > 0.3:
            scene_type = "nature_calm"  # 积极平静
        elif self.complexity == ContentComplexity.MINIMAL:
            scene_type = "abstract_flow"  # 简约抽象
        else:
            scene_type = "forest_tranquil"  # 默认自然
        
        return base_scenes.get(scene_type, base_scenes["nature_calm"])
    
    def _get_color_lighting_description(self) -> str:
        """获取颜色和照明描述"""
        color_desc = []
        
        # 基于情绪的颜色选择
        if self.target_emotion.valence > 0.2:
            color_desc.append("warm golden tones")
        elif self.target_emotion.arousal < -0.5:
            color_desc.append("cool blue and purple hues")
        else:
            color_desc.append("soft neutral colors")
        
        # 照明描述
        lighting_desc = []
        lighting_spec = self.lighting_specifications
        
        if lighting_spec["primary_lighting"] == "soft_diffused":
            lighting_desc.append("soft diffused lighting")
        
        if lighting_spec["color_temperature"] == "warm":
            lighting_desc.append("warm color temperature")
        
        lighting_desc.append(f"{lighting_spec['brightness_level']} brightness")
        lighting_desc.append(f"{lighting_spec['contrast_level']} contrast")
        
        return ", ".join(color_desc + lighting_desc)
    
    def _get_movement_description(self) -> str:
        """获取运动描述"""
        movement_elements = []
        
        # 基于唤醒度决定运动特征
        if self.target_emotion.arousal < -0.6:
            movement_elements.extend([
                "very slow movement",
                "minimal motion",
                "gentle floating"
            ])
        elif self.target_emotion.arousal < -0.3:
            movement_elements.extend([
                "slow gentle movement",
                "smooth transitions",
                "peaceful flow"
            ])
        else:
            movement_elements.extend([
                "subtle movement",
                "calm rhythm"
            ])
        
        # 运动质量
        movement_char = self.movement_characteristics
        if movement_char.get("smoothness", "high") == "high":
            movement_elements.append("smooth motion")
        
        if movement_char.get("predictability", "high") == "high":
            movement_elements.append("predictable patterns")
        
        return ", ".join(movement_elements)
    
    def _get_visual_emotional_qualities(self) -> str:
        """获取视觉情绪质量"""
        qualities = []
        
        # 基于治疗强度
        if self.intensity == TherapeuticIntensity.DEEP:
            qualities.extend(["deeply peaceful", "profoundly calming"])
        elif self.intensity == TherapeuticIntensity.GENTLE:
            qualities.extend(["gently soothing", "subtly relaxing"])
        else:
            qualities.extend(["peaceful", "calming"])
        
        # 基于目标情绪
        if self.target_emotion.valence > 0.3:
            qualities.append("uplifting atmosphere")
        elif self.target_emotion.valence < -0.3:
            qualities.append("contemplative mood")
        
        # 睡眠导向特征
        if self.target_emotion.arousal < -0.5:
            qualities.extend(["sleep inducing", "deeply relaxing"])
        
        return ", ".join(qualities)
    
    def _get_technical_quality_specs(self) -> str:
        """获取技术质量规格"""
        specs = [
            "high quality",
            "cinematic",
            "professional lighting",
            "smooth animation"
        ]
        
        # 基于复杂度调整
        if self.complexity in [ContentComplexity.RICH, ContentComplexity.COMPLEX]:
            specs.extend(["detailed textures", "rich visual depth"])
        elif self.complexity == ContentComplexity.MINIMAL:
            specs.extend(["clean composition", "minimalist aesthetic"])
        
        return ", ".join(specs)
    
    def _get_negative_constraints(self) -> List[str]:
        """获取负面约束（避免的元素）"""
        negative_elements = [
            "no people",
            "no text",
            "no sudden movements",
            "no bright flashing lights",
            "no jarring transitions",
            "no disturbing content",
            "no sharp contrasts",
            "no loud colors"
        ]
        
        # 基于安全约束添加
        safety_constraints = self.safety_constraints
        
        if safety_constraints.get("no_faces", True):
            negative_elements.append("no human faces")
        
        if safety_constraints.get("no_animals", False):
            negative_elements.append("no animals")
        
        if safety_constraints.get("no_motion_sickness", True):
            negative_elements.extend(["no spinning", "no rapid camera movement"])
        
        return negative_elements
    
    def create_visual_progression(self, num_segments: int = 3) -> List[Dict[str, Any]]:
        """创建视觉渐进过程"""
        segments = []
        
        for i in range(num_segments):
            progress = i / (num_segments - 1)
            
            # 颜色渐进
            if progress < 0.5:
                color_emphasis = "maintaining_current_palette"
            else:
                color_emphasis = "transitioning_to_target_palette"
            
            # 运动渐进
            if progress < 0.33:
                movement_emphasis = "establishing_rhythm"
            elif progress < 0.67:
                movement_emphasis = "gradual_slowing"
            else:
                movement_emphasis = "minimal_final_movement"
            
            # 亮度渐进
            initial_brightness = 0.7
            target_brightness = 0.3
            segment_brightness = initial_brightness * (1 - progress) + target_brightness * progress
            
            segment = {
                "segment_number": i + 1,
                "duration_ratio": 1.0 / num_segments,
                "color_emphasis": color_emphasis,
                "movement_emphasis": movement_emphasis,
                "brightness_level": segment_brightness,
                "transition_style": "smooth_fade" if i > 0 else "gentle_intro",
                "focal_elements": self._get_segment_focal_elements(progress)
            }
            
            segments.append(segment)
        
        return segments
    
    def _get_segment_focal_elements(self, progress: float) -> List[str]:
        """获取片段焦点元素"""
        if progress < 0.33:
            return ["environmental_details", "texture_richness"]
        elif progress < 0.67:
            return ["motion_flow", "color_harmony"]
        else:
            return ["essential_shapes", "minimal_elements"]
    
    def estimate_video_parameters(self) -> Dict[str, Any]:
        """估计视频生成参数"""
        # 基于复杂度估计参数
        complexity_specs = {
            ContentComplexity.MINIMAL: {
                "resolution": "512x512",
                "fps": 24,
                "inference_steps": 30,
                "guidance_scale": 6.0
            },
            ContentComplexity.SIMPLE: {
                "resolution": "512x512", 
                "fps": 24,
                "inference_steps": 40,
                "guidance_scale": 7.0
            },
            ContentComplexity.MODERATE: {
                "resolution": "720x720",
                "fps": 24,
                "inference_steps": 50,
                "guidance_scale": 7.5
            },
            ContentComplexity.RICH: {
                "resolution": "720x720",
                "fps": 30,
                "inference_steps": 60,
                "guidance_scale": 8.0
            },
            ContentComplexity.COMPLEX: {
                "resolution": "1024x1024",
                "fps": 30,
                "inference_steps": 70,
                "guidance_scale": 8.5
            }
        }
        
        base_params = complexity_specs.get(self.complexity, complexity_specs[ContentComplexity.MODERATE])
        
        return {
            **base_params,
            "duration_seconds": min(60, self.duration_minutes * 60),  # 限制最长60秒
            "estimated_inference_time": self.duration_minutes * 4.0,  # 估计推理时间
            "memory_requirement": "8-16GB",
            "recommended_model": "hunyuan_video" if self.complexity in [ContentComplexity.RICH, ContentComplexity.COMPLEX] else "mochi_video"
        }


@dataclass
class NarrativePrescription(TherapyPrescription):
    """
    叙事疗愈处方
    
    包含引导性语言和叙事内容的生成指导
    """
    narrative_style: str
    voice_characteristics: Dict[str, Any]
    content_themes: List[str]
    language_patterns: Dict[str, Any]
    interaction_mode: str = "passive"  # passive, guided, interactive
    
    def __post_init__(self):
        if self.prescription_type != PrescriptionType.NARRATIVE:
            self.prescription_type = PrescriptionType.NARRATIVE
    
    def generate_narrative_framework(self) -> Dict[str, Any]:
        """生成叙事框架"""
        return {
            "opening": self._create_opening_framework(),
            "development": self._create_development_framework(),
            "resolution": self._create_resolution_framework(),
            "style_guidelines": self._get_style_guidelines(),
            "safety_guidelines": self._get_narrative_safety_guidelines()
        }
    
    def _create_opening_framework(self) -> Dict[str, str]:
        """创建开场框架"""
        return {
            "tone": "gentle_welcoming",
            "purpose": "establish_safety_and_comfort",
            "elements": "acknowledgment, validation, invitation_to_relax",
            "duration": "30-60_seconds"
        }
    
    def _create_development_framework(self) -> Dict[str, str]:
        """创建发展框架"""
        emotional_journey = self.get_transition_direction()
        
        if emotional_journey["overall_intensity"] == "major":
            development_style = "gradual_deep_transformation"
        elif emotional_journey["arousal_direction"] == "decrease":
            development_style = "progressive_relaxation"
        else:
            development_style = "gentle_stabilization"
        
        return {
            "structure": development_style,
            "pacing": "slow_and_mindful",
            "themes": ", ".join(self.content_themes),
            "techniques": "visualization, breathing_guidance, progressive_relaxation"
        }
    
    def _create_resolution_framework(self) -> Dict[str, str]:
        """创建结尾框架"""
        return {
            "goal": "peaceful_closure_and_sleep_preparation",
            "elements": "affirmation, gratitude, sleep_invitation",
            "fade_style": "gentle_fade_to_silence",
            "duration": "60-90_seconds"
        }
    
    def _get_style_guidelines(self) -> Dict[str, Any]:
        """获取风格指导"""
        return {
            "voice_pace": "slow_and_soothing",
            "language_complexity": "simple_and_clear",
            "metaphor_use": "nature_based_imagery",
            "repetition_strategy": "gentle_reinforcement",
            "pause_usage": "frequent_mindful_pauses"
        }
    
    def _get_narrative_safety_guidelines(self) -> List[str]:
        """获取叙事安全指导"""
        return [
            "avoid_negative_imagery",
            "no_sudden_loud_sounds",
            "inclusive_language_only",
            "culturally_neutral_content",
            "trauma_informed_approach",
            "respect_personal_boundaries"
        ]


class PrescriptionEngine:
    """
    处方生成引擎
    
    整合所有理论模型，生成个性化的多模态疗愈处方
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 处方模板库
        self.prescription_templates = self._load_prescription_templates()
        
        # 个性化规则
        self.personalization_rules = self._load_personalization_rules()
        
        self.logger.info("处方生成引擎初始化完成")
    
    def _load_prescription_templates(self) -> Dict[str, Any]:
        """加载处方模板"""
        return {
            "sleep_induction": {
                "music": {
                    "complexity": ContentComplexity.SIMPLE,
                    "intensity": TherapeuticIntensity.GENTLE,
                    "preferred_instruments": [InstrumentFamily.AMBIENT, InstrumentFamily.STRINGS],
                    "tempo_range": [40, 60]
                },
                "visual": {
                    "style": "nature_calm",
                    "movement": "minimal",
                    "colors": "cool_calming"
                }
            },
            "anxiety_relief": {
                "music": {
                    "complexity": ContentComplexity.MODERATE,
                    "intensity": TherapeuticIntensity.MODERATE,
                    "preferred_instruments": [InstrumentFamily.KEYBOARD, InstrumentFamily.STRINGS],
                    "tempo_range": [60, 80]
                },
                "visual": {
                    "style": "abstract_flow",
                    "movement": "gentle_rhythmic",
                    "colors": "warm_earth_tones"
                }
            },
            "mood_enhancement": {
                "music": {
                    "complexity": ContentComplexity.RICH,
                    "intensity": TherapeuticIntensity.MODERATE,
                    "preferred_instruments": [InstrumentFamily.STRINGS, InstrumentFamily.WINDS],
                    "tempo_range": [70, 90]
                },
                "visual": {
                    "style": "nature_uplifting",
                    "movement": "flowing_positive",
                    "colors": "bright_natural"
                }
            }
        }
    
    def _load_personalization_rules(self) -> Dict[str, Any]:
        """加载个性化规则"""
        return {
            "age_adaptations": {
                "young_adult": {"complexity_boost": 0.1, "tempo_boost": 5},
                "middle_aged": {"complexity_neutral": 0.0, "tempo_neutral": 0},
                "elderly": {"complexity_reduction": 0.2, "tempo_reduction": 10}
            },
            "cultural_adaptations": {
                "western": {"instrument_preference": ["keyboard", "strings"]},
                "eastern": {"instrument_preference": ["winds", "ambient"]},
                "universal": {"instrument_preference": ["ambient", "strings"]}
            },
            "medical_considerations": {
                "hearing_impaired": {"frequency_emphasis": "low_frequency"},
                "visual_impaired": {"audio_enhancement": True},
                "anxiety_disorder": {"gradual_approach": True, "safety_priority": True}
            }
        }
    
    def generate_comprehensive_prescription(self,
                                          current_emotion: EmotionState,
                                          target_emotion: EmotionState,
                                          user_profile: Dict[str, Any],
                                          session_context: Dict[str, Any]) -> Dict[str, TherapyPrescription]:
        """生成综合疗愈处方"""
        
        self.logger.info("开始生成综合疗愈处方")
        
        # 分析疗愈需求
        therapeutic_needs = self._analyze_therapeutic_needs(current_emotion, target_emotion)
        
        # 选择处方模板
        template_type = self._select_prescription_template(therapeutic_needs, user_profile)
        
        # 生成各模态处方
        prescriptions = {}
        
        # 音乐处方
        if session_context.get("enable_music", True):
            prescriptions["music"] = self._generate_music_prescription(
                current_emotion, target_emotion, user_profile, session_context, template_type
            )
        
        # 视觉处方
        if session_context.get("enable_video", True):
            prescriptions["visual"] = self._generate_visual_prescription(
                current_emotion, target_emotion, user_profile, session_context, template_type
            )
        
        # 叙事处方（可选）
        if session_context.get("enable_narrative", False):
            prescriptions["narrative"] = self._generate_narrative_prescription(
                current_emotion, target_emotion, user_profile, session_context, template_type
            )
        
        self.logger.info(f"综合处方生成完成，包含{len(prescriptions)}个模态")
        
        return prescriptions
    
    def _analyze_therapeutic_needs(self,
                                 current_emotion: EmotionState,
                                 target_emotion: EmotionState) -> Dict[str, Any]:
        """分析疗愈需求"""
        emotional_distance = current_emotion.distance_to(target_emotion)
        
        # 计算情绪变化向量
        valence_change = target_emotion.valence - current_emotion.valence
        arousal_change = target_emotion.arousal - current_emotion.arousal
        
        # 确定主要疗愈目标
        primary_goals = []
        
        if arousal_change < -0.5:
            primary_goals.append("relaxation")
        if arousal_change < -0.7:
            primary_goals.append("sleep_induction")
        if valence_change > 0.3:
            primary_goals.append("mood_enhancement")
        if valence_change < -0.3:
            primary_goals.append("emotional_support")
        if emotional_distance < 0.3:
            primary_goals.append("stabilization")
        else:
            primary_goals.append("transformation")
        
        # 确定治疗强度
        if emotional_distance > 1.5:
            intensity = TherapeuticIntensity.INTENSIVE
        elif emotional_distance > 1.0:
            intensity = TherapeuticIntensity.MODERATE
        else:
            intensity = TherapeuticIntensity.GENTLE
        
        # 确定复杂度需求
        if current_emotion.confidence < 0.6:
            complexity = ContentComplexity.SIMPLE  # 低置信度时使用简单内容
        elif emotional_distance > 1.2:
            complexity = ContentComplexity.RICH  # 大变化需要丰富内容
        else:
            complexity = ContentComplexity.MODERATE
        
        return {
            "primary_goals": primary_goals,
            "therapeutic_intensity": intensity,
            "content_complexity": complexity,
            "emotional_distance": emotional_distance,
            "valence_change": valence_change,
            "arousal_change": arousal_change,
            "transition_difficulty": self._assess_transition_difficulty(current_emotion, target_emotion)
        }
    
    def _assess_transition_difficulty(self,
                                   current_emotion: EmotionState,
                                   target_emotion: EmotionState) -> str:
        """评估过渡难度"""
        distance = current_emotion.distance_to(target_emotion)
        
        # 检查是否跨越情绪象限
        current_quadrant = self._get_emotion_quadrant(current_emotion)
        target_quadrant = self._get_emotion_quadrant(target_emotion)
        
        quadrant_change = current_quadrant != target_quadrant
        
        if distance > 1.5 and quadrant_change:
            return "very_difficult"
        elif distance > 1.0 or quadrant_change:
            return "moderate"
        elif distance > 0.5:
            return "easy"
        else:
            return "minimal"
    
    def _get_emotion_quadrant(self, emotion: EmotionState) -> str:
        """获取情绪象限"""
        if emotion.valence >= 0 and emotion.arousal >= 0:
            return "high_valence_high_arousal"
        elif emotion.valence >= 0 and emotion.arousal < 0:
            return "high_valence_low_arousal"
        elif emotion.valence < 0 and emotion.arousal >= 0:
            return "low_valence_high_arousal"
        else:
            return "low_valence_low_arousal"
    
    def _select_prescription_template(self,
                                    therapeutic_needs: Dict[str, Any],
                                    user_profile: Dict[str, Any]) -> str:
        """选择处方模板"""
        primary_goals = therapeutic_needs["primary_goals"]
        
        # 基于主要目标选择模板
        if "sleep_induction" in primary_goals:
            return "sleep_induction"
        elif "relaxation" in primary_goals and "mood_enhancement" not in primary_goals:
            return "anxiety_relief"
        elif "mood_enhancement" in primary_goals:
            return "mood_enhancement"
        else:
            return "sleep_induction"  # 默认模板
    
    def _generate_music_prescription(self,
                                   current_emotion: EmotionState,
                                   target_emotion: EmotionState,
                                   user_profile: Dict[str, Any],
                                   session_context: Dict[str, Any],
                                   template_type: str) -> MusicPrescription:
        """生成音乐处方"""
        
        # 获取模板
        template = self.prescription_templates[template_type]["music"]
        
        # 应用个性化
        complexity = self._personalize_complexity(template["complexity"], user_profile)
        intensity = template["intensity"]
        
        # 生成音乐特征
        musical_characteristics = self._create_musical_characteristics(
            current_emotion, target_emotion, template, user_profile
        )
        
        # 创建处方元数据
        metadata = PrescriptionMetadata(
            prescription_id=f"music_{int(time.time())}",
            created_at=datetime.now(),
            user_profile_id=user_profile.get("user_id"),
            session_context=session_context,
            theoretical_basis={
                "iso_principle": "three_stage_progression",
                "valence_arousal_model": "emotion_trajectory_mapping",
                "music_psychology": "therapeutic_harmony_selection"
            },
            expected_outcomes=["emotional_regulation", "relaxation_response", "sleep_preparation"]
        )
        
        prescription = MusicPrescription(
            prescription_type=PrescriptionType.MUSIC,
            target_emotion=target_emotion,
            current_emotion=current_emotion,
            therapeutic_goals=["emotional_transition", "relaxation", "sleep_preparation"],
            duration_minutes=session_context.get("duration_minutes", 20.0),
            complexity=complexity,
            intensity=intensity,
            metadata=metadata,
            musical_characteristics=musical_characteristics,
            prompt_template="",  # 将通过方法生成
            user_preferences=user_profile.get("music_preferences", {}),
            environmental_context=session_context.get("environment", {}),
            safety_constraints={"volume_limited": True, "no_binaural_beats": False}
        )
        
        # 生成渐进阶段
        prescription.create_progression_stages()
        
        return prescription
    
    def _generate_visual_prescription(self,
                                    current_emotion: EmotionState,
                                    target_emotion: EmotionState,
                                    user_profile: Dict[str, Any],
                                    session_context: Dict[str, Any],
                                    template_type: str) -> VisualPrescription:
        """生成视觉处方"""
        
        template = self.prescription_templates[template_type]["visual"]
        
        # 个性化调整
        complexity = self._personalize_complexity(ContentComplexity.MODERATE, user_profile)
        
        # 创建处方元数据
        metadata = PrescriptionMetadata(
            prescription_id=f"visual_{int(time.time())}",
            created_at=datetime.now(),
            user_profile_id=user_profile.get("user_id"),
            session_context=session_context,
            theoretical_basis={
                "sleep_physiology": "visual_sleep_induction",
                "color_psychology": "therapeutic_color_selection"
            },
            expected_outcomes=["visual_relaxation", "sleep_preparation", "stress_reduction"]
        )
        
        prescription = VisualPrescription(
            prescription_type=PrescriptionType.VISUAL,
            target_emotion=target_emotion,
            current_emotion=current_emotion,
            therapeutic_goals=["visual_relaxation", "sleep_induction"],
            duration_minutes=min(session_context.get("duration_minutes", 20.0), 10.0),  # 视频较短
            complexity=complexity,
            intensity=TherapeuticIntensity.GENTLE,
            metadata=metadata,
            visual_style=template["style"],
            color_palette=self._create_color_palette(target_emotion, template),
            movement_characteristics=self._create_movement_characteristics(target_emotion, template),
            visual_elements=self._select_visual_elements(target_emotion, user_profile),
            user_preferences=user_profile.get("visual_preferences", {}),
            environmental_context=session_context.get("environment", {}),
            safety_constraints={"no_motion_sickness": True, "no_faces": True}
        )
        
        return prescription
    
    def _generate_narrative_prescription(self,
                                       current_emotion: EmotionState,
                                       target_emotion: EmotionState,
                                       user_profile: Dict[str, Any],
                                       session_context: Dict[str, Any],
                                       template_type: str) -> NarrativePrescription:
        """生成叙事处方"""
        
        metadata = PrescriptionMetadata(
            prescription_id=f"narrative_{int(time.time())}",
            created_at=datetime.now(),
            user_profile_id=user_profile.get("user_id"),
            session_context=session_context
        )
        
        prescription = NarrativePrescription(
            prescription_type=PrescriptionType.NARRATIVE,
            target_emotion=target_emotion,
            current_emotion=current_emotion,
            therapeutic_goals=["guided_relaxation", "cognitive_reframing"],
            duration_minutes=session_context.get("duration_minutes", 20.0),
            complexity=ContentComplexity.SIMPLE,
            intensity=TherapeuticIntensity.GENTLE,
            metadata=metadata,
            narrative_style="guided_meditation",
            voice_characteristics={"pace": "slow", "tone": "warm", "volume": "soft"},
            content_themes=["nature", "safety", "peace", "rest"],
            language_patterns={"repetition": "gentle", "metaphors": "nature_based"}
        )
        
        return prescription
    
    def _personalize_complexity(self,
                              base_complexity: ContentComplexity,
                              user_profile: Dict[str, Any]) -> ContentComplexity:
        """个性化复杂度调整"""
        
        # 年龄调整
        age_group = user_profile.get("age_group", "middle_aged")
        age_rules = self.personalization_rules["age_adaptations"].get(age_group, {})
        
        complexity_levels = list(ContentComplexity)
        current_index = complexity_levels.index(base_complexity)
        
        if "complexity_boost" in age_rules:
            current_index = min(len(complexity_levels) - 1, current_index + 1)
        elif "complexity_reduction" in age_rules:
            current_index = max(0, current_index - 1)
        
        # 经验水平调整
        experience_level = user_profile.get("meditation_experience", "beginner")
        if experience_level == "beginner":
            current_index = max(0, current_index - 1)
        elif experience_level == "advanced":
            current_index = min(len(complexity_levels) - 1, current_index + 1)
        
        return complexity_levels[current_index]
    
    def _create_musical_characteristics(self,
                                      current_emotion: EmotionState,
                                      target_emotion: EmotionState,
                                      template: Dict[str, Any],
                                      user_profile: Dict[str, Any]) -> MusicalCharacteristics:
        """创建音乐特征"""
        
        # 导入音乐心理学模型
        from ..research.theory.music_psychology import MusicPsychologyModel
        music_model = MusicPsychologyModel()
        
        # 生成音乐处方
        musical_characteristics = music_model.generate_musical_prescription(
            current_emotion,
            target_emotion=target_emotion,
            duration_minutes=20.0,
            sleep_context=True
        )
        
        return musical_characteristics
    
    def _create_color_palette(self,
                            target_emotion: EmotionState,
                            template: Dict[str, Any]) -> Dict[str, Any]:
        """创建颜色调色板"""
        
        base_colors = template.get("colors", "cool_calming")
        
        if base_colors == "cool_calming":
            return {
                "primary": ["deep_blue", "soft_purple"],
                "secondary": ["pale_blue", "lavender"],
                "accent": ["silver", "white"],
                "temperature": "cool",
                "saturation": "low"
            }
        elif base_colors == "warm_earth_tones":
            return {
                "primary": ["warm_brown", "golden_yellow"],
                "secondary": ["soft_orange", "cream"],
                "accent": ["copper", "beige"],
                "temperature": "warm",
                "saturation": "medium"
            }
        else:  # bright_natural
            return {
                "primary": ["forest_green", "sky_blue"],
                "secondary": ["grass_green", "cloud_white"],
                "accent": ["sun_yellow", "earth_brown"],
                "temperature": "neutral",
                "saturation": "medium"
            }
    
    def _create_movement_characteristics(self,
                                       target_emotion: EmotionState,
                                       template: Dict[str, Any]) -> Dict[str, Any]:
        """创建运动特征"""
        
        movement_type = template.get("movement", "minimal")
        
        base_characteristics = {
            "speed": "very_slow" if target_emotion.arousal < -0.6 else "slow",
            "smoothness": "high",
            "predictability": "high",
            "direction": "flowing",
            "rhythm": "gentle"
        }
        
        if movement_type == "minimal":
            base_characteristics.update({
                "amplitude": "very_small",
                "frequency": "low",
                "pattern": "simple"
            })
        elif movement_type == "gentle_rhythmic":
            base_characteristics.update({
                "amplitude": "small",
                "frequency": "medium_low",
                "pattern": "rhythmic"
            })
        elif movement_type == "flowing_positive":
            base_characteristics.update({
                "amplitude": "medium",
                "frequency": "medium",
                "pattern": "organic_flow"
            })
        
        return base_characteristics
    
    def _select_visual_elements(self,
                              target_emotion: EmotionState,
                              user_profile: Dict[str, Any]) -> List[str]:
        """选择视觉元素"""
        
        base_elements = ["soft_lighting", "organic_shapes", "natural_textures"]
        
        # 基于目标情绪添加元素
        if target_emotion.arousal < -0.6:
            base_elements.extend(["floating_particles", "gentle_waves", "soft_gradients"])
        
        if target_emotion.valence > 0.2:
            base_elements.extend(["warm_highlights", "golden_accents"])
        
        # 基于用户偏好
        visual_prefs = user_profile.get("visual_preferences", {})
        if visual_prefs.get("nature_preference", True):
            base_elements.extend(["natural_landscapes", "organic_forms"])
        
        if visual_prefs.get("abstract_preference", False):
            base_elements.extend(["abstract_patterns", "geometric_harmony"])
        
        return base_elements
    
    def validate_prescription_safety(self, prescription: TherapyPrescription) -> Dict[str, Any]:
        """验证处方安全性"""
        
        safety_report = {
            "overall_safety": "safe",
            "warnings": [],
            "contraindications": [],
            "recommendations": []
        }
        
        # 检查情绪变化幅度
        emotional_distance = prescription.calculate_emotional_distance()
        if emotional_distance > 1.8:
            safety_report["warnings"].append("large_emotional_change_detected")
            safety_report["recommendations"].append("consider_gradual_approach")
        
        # 检查治疗强度
        if prescription.intensity == TherapeuticIntensity.INTENSIVE:
            safety_report["warnings"].append("intensive_therapy_approach")
            safety_report["recommendations"].append("monitor_user_response")
        
        # 检查用户安全约束
        user_constraints = prescription.user_preferences.get("medical_considerations", [])
        if "anxiety_disorder" in user_constraints and emotional_distance > 1.0:
            safety_report["contraindications"].append("rapid_change_with_anxiety_history")
            safety_report["overall_safety"] = "caution_required"
        
        return safety_report
    
    def optimize_prescription_effectiveness(self, prescription: TherapyPrescription) -> TherapyPrescription:
        """优化处方有效性"""
        
        # 计算预期有效性
        effectiveness_score = prescription.estimate_completion_probability()
        
        # 如果有效性较低，进行优化
        if effectiveness_score < 0.7:
            
            # 降低复杂度
            if prescription.complexity in [ContentComplexity.RICH, ContentComplexity.COMPLEX]:
                complexity_levels = list(ContentComplexity)
                current_index = complexity_levels.index(prescription.complexity)
                prescription.complexity = complexity_levels[max(0, current_index - 1)]
            
            # 调整治疗强度
            if prescription.intensity == TherapeuticIntensity.INTENSIVE:
                prescription.intensity = TherapeuticIntensity.MODERATE
            
            # 增加安全约束
            prescription.safety_constraints["gentle_approach"] = True
            prescription.safety_constraints["extended_duration"] = True
        
        return prescription


# 工具函数
def create_prescription_engine() -> PrescriptionEngine:
    """创建处方生成引擎"""
    return PrescriptionEngine()

import time  # 需要导入time模块