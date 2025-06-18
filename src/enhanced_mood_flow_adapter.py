#!/usr/bin/env python3
"""
增强型心境流转适配器 - 将新的理论驱动模块集成到现有系统

设计原则：
1. 保持向后兼容性
2. 渐进式增强
3. 可配置的模块选择
4. 优雅降级

作者：心境流转团队
日期：2024
"""

import os
import sys
import numpy as np
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import logging

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入增强模块
from src.emotion_recognition.enhanced_emotion_recognizer import (
    EnhancedEmotionRecognizer, DetailedEmotion, create_emotion_recognizer
)
from src.therapy_planning.enhanced_iso_planner import (
    EnhancedISOPlanner, create_iso_planner, TherapyStageConfig
)
from src.music_mapping.enhanced_music_mapper import (
    EnhancedMusicMapper, create_music_mapper, MusicProfile
)

# 导入SOTA模型适配器
try:
    from src.model_adapters.musicgen_adapter import (
        MusicGenAdapter, create_musicgen_adapter
    )
    from src.model_adapters.music_quality_evaluator import (
        MusicQualityEvaluator, create_music_quality_evaluator
    )
    SOTA_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SOTA模型适配器导入失败: {e}")
    SOTA_MODELS_AVAILABLE = False

# 设置日志
logger = logging.getLogger(__name__)

class EnhancedMoodFlowAdapter:
    """
    增强型适配器 - 无缝集成新模块到现有MoodFlowApp
    
    核心功能：
    1. 增强的情绪识别（9种细粒度情绪）
    2. 理论驱动的ISO治疗规划
    3. 精准的音乐特征映射
    4. 保持与原系统的完全兼容
    """
    
    def __init__(self, 
                 use_enhanced_emotion: bool = True,
                 use_enhanced_planning: bool = True,
                 use_enhanced_mapping: bool = True,
                 use_sota_music_generation: bool = False,
                 fallback_to_original: bool = True):
        """
        初始化适配器
        
        Args:
            use_enhanced_emotion: 是否使用增强情绪识别
            use_enhanced_planning: 是否使用增强治疗规划
            use_enhanced_mapping: 是否使用增强音乐映射
            use_sota_music_generation: 是否使用SOTA音乐生成模型（MusicGen）
            fallback_to_original: 出错时是否回退到原始实现
        """
        self.use_enhanced_emotion = use_enhanced_emotion
        self.use_enhanced_planning = use_enhanced_planning
        self.use_enhanced_mapping = use_enhanced_mapping
        self.use_sota_music_generation = use_sota_music_generation and SOTA_MODELS_AVAILABLE
        self.fallback_to_original = fallback_to_original
        
        # 初始化增强模块
        self._init_enhanced_modules()
        
    def _init_enhanced_modules(self):
        """初始化增强模块，带错误处理"""
        try:
            # 情绪识别器
            if self.use_enhanced_emotion:
                self.emotion_recognizer = create_emotion_recognizer(use_advanced=False)
                logger.info("✅ 增强情绪识别器初始化成功")
            else:
                self.emotion_recognizer = None
                
            # ISO规划器
            if self.use_enhanced_planning:
                self.iso_planner = create_iso_planner(enhanced=True)
                logger.info("✅ 增强ISO规划器初始化成功")
            else:
                self.iso_planner = None
                
            # 音乐映射器
            if self.use_enhanced_mapping:
                self.music_mapper = create_music_mapper(enhanced=True, sleep_optimized=True)
                logger.info("✅ 增强音乐映射器初始化成功")
            else:
                self.music_mapper = None
                
            # SOTA音乐生成器
            if self.use_sota_music_generation:
                try:
                    self.musicgen_adapter = create_musicgen_adapter(
                        model_size="auto",
                        use_melody_conditioning=True
                    )
                    self.music_quality_evaluator = create_music_quality_evaluator()
                    logger.info("✅ MusicGen音乐生成器初始化成功")
                except Exception as e:
                    logger.error(f"MusicGen初始化失败: {e}")
                    self.musicgen_adapter = None
                    self.music_quality_evaluator = None
                    if not self.fallback_to_original:
                        raise
            else:
                self.musicgen_adapter = None
                self.music_quality_evaluator = None
                
        except Exception as e:
            logger.error(f"❌ 增强模块初始化失败: {e}")
            if not self.fallback_to_original:
                raise
    
    def analyze_emotion_enhanced(self, text: str, original_method=None) -> Any:
        """
        增强的情绪分析
        
        Args:
            text: 用户输入文本
            original_method: 原始的情绪分析方法（用于回退）
            
        Returns:
            EmotionState对象（兼容原系统）
        """
        try:
            if self.use_enhanced_emotion and self.emotion_recognizer:
                # 使用增强识别器
                detailed_emotion = self.emotion_recognizer.recognize(text)
                
                # 转换为原系统格式
                # 创建兼容的EmotionState对象
                emotion_state = type('EmotionState', (), {
                    'valence': detailed_emotion.valence,
                    'arousal': detailed_emotion.arousal,
                    # 添加增强信息作为额外属性
                    '_detailed': detailed_emotion,
                    '_primary_emotion': detailed_emotion.primary_emotion,
                    '_confidence': detailed_emotion.confidence
                })()
                
                # 记录详细信息（增强版标识）
                print(f"\n{'='*60}")
                print(f"🧠 [增强情绪识别 v2.0] 细粒度分析结果:")
                print(f"{'='*60}")
                print(f"📝 输入文本: {text[:50]}...")
                print(f"🎯 主要情绪: {detailed_emotion.primary_emotion} ({self._emotion_to_chinese(detailed_emotion.primary_emotion)})")
                print(f"📊 V-A坐标: Valence={detailed_emotion.valence:.2f}, Arousal={detailed_emotion.arousal:.2f}")
                print(f"💯 置信度: {detailed_emotion.confidence:.1%}")
                print(f"💪 强度: {detailed_emotion.intensity:.1%}")
                if detailed_emotion.secondary_emotions:
                    print(f"🔄 次要情绪: {detailed_emotion.secondary_emotions}")
                print(f"{'='*60}\n")
                
                return emotion_state
                
        except Exception as e:
            logger.error(f"增强情绪识别失败: {e}")
            if self.fallback_to_original and original_method:
                logger.info("回退到原始情绪识别方法")
                return original_method(text)
            raise
        
        # 如果未启用增强或无原始方法，返回默认
        if original_method:
            return original_method(text)
        else:
            # 返回默认情绪状态
            return type('EmotionState', (), {'valence': -0.5, 'arousal': 0.5})()
    
    def plan_therapy_stages_enhanced(self, current_emotion, target_emotion, 
                                   duration, original_method=None) -> list:
        """
        增强的治疗阶段规划
        
        整合ISO原则和Gross模型
        """
        try:
            if self.use_enhanced_planning and self.iso_planner:
                # 使用增强规划器
                stages = self.iso_planner.plan_stages(current_emotion, target_emotion, duration)
                
                # 记录规划信息（增强版标识）
                print(f"\n{'='*60}")
                print(f"📋 [增强治疗规划 v2.0] ISO原则+Gross模型:")
                print(f"{'='*60}")
                for i, stage in enumerate(stages):
                    print(f"  阶段{i+1}: {stage['stage'].value}")
                    print(f"    - 时长: {stage['duration']:.1f}分钟")
                    print(f"    - 目标情绪: V={stage['emotion'].valence:.2f}, A={stage['emotion'].arousal:.2f}")
                    if hasattr(stage['stage'], 'value') and '同步化' in stage['stage'].value:
                        print(f"    - 策略: 匹配用户当前情绪，建立信任")
                    elif hasattr(stage['stage'], 'value') and '引导化' in stage['stage'].value:
                        print(f"    - 策略: 渐进式过渡，认知重评")
                    elif hasattr(stage['stage'], 'value') and '巩固化' in stage['stage'].value:
                        print(f"    - 策略: 维持低唤醒，深化放松")
                print(f"{'='*60}\n")
                
                return stages
                
        except Exception as e:
            logger.error(f"增强治疗规划失败: {e}")
            if self.fallback_to_original and original_method:
                logger.info("回退到原始规划方法")
                return original_method.plan_stages(current_emotion, target_emotion, duration)
            raise
        
        # 如果未启用增强，使用原始方法
        if original_method:
            return original_method.plan_stages(current_emotion, target_emotion, duration)
        else:
            # 返回简单的默认规划
            return self._create_default_stages(duration)
    
    def get_music_parameters_enhanced(self, emotion_state, stage=None, 
                                    original_method=None) -> Dict:
        """
        增强的音乐参数生成
        
        基于精准的情绪-音乐特征映射
        """
        try:
            if self.use_enhanced_mapping and self.music_mapper:
                # 提取V-A值
                valence = emotion_state.valence
                arousal = emotion_state.arousal
                
                # 获取阶段名称（如果有）
                stage_name = stage.get('stage').value if stage and 'stage' in stage else None
                
                # 使用增强映射器
                music_params = self.music_mapper.get_music_params(valence, arousal)
                
                # 添加阶段特定调整
                if stage_name:
                    music_params['stage'] = stage_name
                
                # 记录映射信息（增强版标识）
                print(f"\n{'='*60}")
                print(f"🎵 [增强音乐映射 v2.0] 精准特征生成:")
                print(f"{'='*60}")
                print(f"  情绪状态: V={valence:.2f}, A={arousal:.2f}")
                print(f"  BPM: {music_params.get('bpm', 'N/A')} (基于Arousal相关性0.88)")
                print(f"  调性: {music_params.get('key', 'N/A')} (基于Valence相关性0.74)")
                print(f"  乐器: {', '.join(music_params.get('instruments', [])[:3])}")
                print(f"  节奏复杂度: {music_params.get('rhythm_pattern_complexity', 0):.2f}")
                if 'binaural_frequency' in music_params:
                    print(f"  双耳节拍: {music_params['binaural_frequency']}Hz (诱导脑电波同步)")
                print(f"{'='*60}\n")
                
                return music_params
                
        except Exception as e:
            logger.error(f"增强音乐映射失败: {e}")
            if self.fallback_to_original and original_method:
                logger.info("回退到原始映射方法")
                # 原始方法可能只返回BPM
                bpm = original_method.calc_bpm(emotion_state.arousal)
                return {'bpm': bpm}
            raise
        
        # 如果未启用增强，使用简单映射
        if original_method:
            bpm = original_method.calc_bpm(emotion_state.arousal)
            return {'bpm': bpm}
        else:
            # 返回默认参数
            return {'bpm': 60}
    
    def _create_default_stages(self, duration: int) -> list:
        """创建默认的治疗阶段（用于回退）"""
        return [
            {
                'stage': type('Stage', (), {'value': '同步化'})(),
                'duration': duration * 0.25,
                'emotion': type('EmotionState', (), {'valence': -0.5, 'arousal': 0.5})()
            },
            {
                'stage': type('Stage', (), {'value': '引导化'})(),
                'duration': duration * 0.50,
                'emotion': type('EmotionState', (), {'valence': 0.0, 'arousal': 0.0})()
            },
            {
                'stage': type('Stage', (), {'value': '巩固化'})(),
                'duration': duration * 0.25,
                'emotion': type('EmotionState', (), {'valence': 0.3, 'arousal': -0.8})()
            }
        ]
    
    def get_enhancement_status(self) -> Dict[str, bool]:
        """获取增强模块的状态"""
        return {
            'emotion_recognition': self.use_enhanced_emotion and self.emotion_recognizer is not None,
            'therapy_planning': self.use_enhanced_planning and self.iso_planner is not None,
            'music_mapping': self.use_enhanced_mapping and self.music_mapper is not None,
            'sota_music_generation': self.use_sota_music_generation and self.musicgen_adapter is not None
        }
    
    def get_detailed_emotion_info(self, emotion_state) -> Optional[Dict]:
        """
        获取详细的情绪信息（如果使用了增强识别）
        
        用于在界面上显示更丰富的信息
        """
        if hasattr(emotion_state, '_detailed'):
            detailed = emotion_state._detailed
            return {
                'primary_emotion': detailed.primary_emotion,
                'primary_emotion_cn': self._emotion_to_chinese(detailed.primary_emotion),
                'confidence': detailed.confidence,
                'intensity': detailed.intensity,
                'secondary_emotions': detailed.secondary_emotions
            }
        return None
    
    def _emotion_to_chinese(self, emotion: str) -> str:
        """将英文情绪转换为中文"""
        mapping = {
            'anger': '愤怒',
            'fear': '恐惧/焦虑',
            'disgust': '厌恶',
            'sadness': '悲伤',
            'amusement': '愉悦',
            'joy': '喜悦',
            'inspiration': '灵感/激励',
            'tenderness': '温柔',
            'neutral': '中性'
        }
        return mapping.get(emotion, emotion)
    
    def generate_sota_music(self, 
                          emotion_state, 
                          stage_info: Dict, 
                          duration_seconds: float = 60,
                          original_method=None) -> Tuple[Optional[np.ndarray], Dict]:
        """
        使用SOTA模型生成高质量音乐
        
        Args:
            emotion_state: 情绪状态
            stage_info: 治疗阶段信息
            duration_seconds: 音乐时长（秒）
            original_method: 原始音乐生成方法（回退用）
            
        Returns:
            (audio_data, metadata): 音频数据和元数据
        """
        if not self.use_sota_music_generation or not self.musicgen_adapter:
            # 如果未启用或不可用，使用原始方法
            if original_method:
                return original_method(duration_seconds, stage_info), {}
            else:
                return None, {'error': 'SOTA music generation not available'}
        
        try:
            # 准备情绪状态信息
            emotion_dict = {
                'valence': emotion_state.valence,
                'arousal': emotion_state.arousal,
                'primary_emotion': getattr(emotion_state, '_primary_emotion', 'neutral')
            }
            
            # 计算目标BPM
            bpm_target = None
            if hasattr(self, 'music_mapper') and self.music_mapper:
                music_params = self.music_mapper.get_music_params(
                    emotion_state.valence, emotion_state.arousal
                )
                bpm_target = music_params.get('bpm')
            
            # 使用MusicGen生成音乐
            print(f"\n{'='*60}")
            print(f"🎼 [SOTA音乐生成 v1.0] MusicGen高质量生成:")
            print(f"{'='*60}")
            print(f"  情绪状态: V={emotion_dict['valence']:.2f}, A={emotion_dict['arousal']:.2f}")
            print(f"  主要情绪: {emotion_dict['primary_emotion']}")
            print(f"  治疗阶段: {stage_info.get('stage', 'unknown')}")
            print(f"  目标时长: {duration_seconds}秒")
            if bpm_target:
                print(f"  目标BPM: {bpm_target}")
            print(f"{'='*60}")
            
            audio_data, metadata = self.musicgen_adapter.generate_therapeutic_music(
                emotion_state=emotion_dict,
                stage_info=stage_info,
                duration_seconds=duration_seconds,
                bpm_target=bpm_target
            )
            
            # 质量评估
            if self.music_quality_evaluator and audio_data is not None:
                quality_metrics = self.music_quality_evaluator.evaluate_music_quality(
                    audio_data, metadata, therapy_context=stage_info
                )
                metadata['quality_metrics'] = quality_metrics
                
                print(f"\n🏆 质量评估结果:")
                print(f"  技术质量: {quality_metrics.technical_score:.2f}/1.0")
                print(f"  治疗效果: {quality_metrics.therapeutic_score:.2f}/1.0")
                print(f"  综合评分: {quality_metrics.overall_score:.2f}/1.0")
                
                if quality_metrics.warnings:
                    print(f"  ⚠️ 质量警告: {len(quality_metrics.warnings)}个")
                if quality_metrics.recommendations:
                    print(f"  💡 改进建议: {len(quality_metrics.recommendations)}个")
                print(f"{'='*60}")
            
            return audio_data, metadata
            
        except Exception as e:
            logger.error(f"SOTA音乐生成失败: {e}")
            if self.fallback_to_original and original_method:
                logger.info("回退到原始音乐生成方法")
                return original_method(duration_seconds, stage_info), {'fallback': True}
            else:
                return None, {'error': str(e)}


def integrate_enhanced_modules(mood_flow_app_instance, config: Optional[Dict] = None):
    """
    将增强模块集成到现有的MoodFlowApp实例
    
    这是一个便捷函数，用于修改现有实例以使用增强功能
    
    Args:
        mood_flow_app_instance: MoodFlowApp实例
        config: 配置选项
    """
    # 默认配置
    if config is None:
        config = {
            'use_enhanced_emotion': True,
            'use_enhanced_planning': True,
            'use_enhanced_mapping': True,
            'fallback_to_original': True
        }
    
    # 创建适配器
    adapter = EnhancedMoodFlowAdapter(**config)
    
    # 保存原始方法
    original_analyze = mood_flow_app_instance.analyze_emotion_from_text
    original_plan = mood_flow_app_instance.plan_therapy_stages
    original_music_model = mood_flow_app_instance.music_model
    
    # 替换方法（使用闭包保持原始方法引用）
    def enhanced_analyze(text):
        return adapter.analyze_emotion_enhanced(text, lambda t: original_analyze(t))
    
    def enhanced_plan(current_emotion, duration=20):
        # 目标情绪：平静入睡状态（与原始实现保持一致）
        target_emotion = type('EmotionState', (), {'valence': 0.3, 'arousal': -0.8})()
        return adapter.plan_therapy_stages_enhanced(
            current_emotion, target_emotion, duration, mood_flow_app_instance.iso_model
        )
    
    # 应用增强
    mood_flow_app_instance.analyze_emotion_from_text = enhanced_analyze
    mood_flow_app_instance.plan_therapy_stages = enhanced_plan
    
    # 添加增强状态查询方法
    mood_flow_app_instance.get_enhancement_status = adapter.get_enhancement_status
    mood_flow_app_instance.get_detailed_emotion_info = adapter.get_detailed_emotion_info
    
    # 添加音乐生成增强
    if hasattr(mood_flow_app_instance, '_generate_simple_music'):
        original_generate = mood_flow_app_instance._generate_simple_music
        
        def enhanced_generate(duration_seconds, bpm, key, stage_index):
            # 如果启用了SOTA音乐生成，使用MusicGen
            if adapter.use_sota_music_generation and adapter.musicgen_adapter:
                try:
                    # 获取当前情绪状态和阶段信息
                    if hasattr(mood_flow_app_instance, 'current_session') and mood_flow_app_instance.current_session:
                        emotion = mood_flow_app_instance.current_session.iso_stages[stage_index]['emotion']
                        stage_info = {
                            'stage_name': mood_flow_app_instance.current_session.iso_stages[stage_index]['stage'].value,
                            'stage_index': stage_index,
                            'therapy_goal': 'sleep_therapy'
                        }
                        
                        # 使用MusicGen生成音乐
                        audio_data, metadata = adapter.generate_sota_music(
                            emotion, stage_info, duration_seconds
                        )
                        
                        if audio_data is not None:
                            print(f"🎼 [SOTA生成] 阶段{stage_index+1}音乐生成成功: {len(audio_data)}样本")
                            return audio_data
                        else:
                            print(f"⚠️ [SOTA生成] 阶段{stage_index+1}生成失败，回退到基础方法")
                    
                except Exception as e:
                    print(f"⚠️ [SOTA生成] 出错，回退到基础方法: {e}")
            
            # 回退到增强的基础方法
            if hasattr(mood_flow_app_instance, 'current_session'):
                emotion = mood_flow_app_instance.current_session.iso_stages[stage_index]['emotion']
                params = adapter.get_music_parameters_enhanced(emotion, 
                                                              mood_flow_app_instance.current_session.iso_stages[stage_index],
                                                              original_music_model)
                # 使用增强参数
                bpm = params.get('bpm', bpm)
                key = params.get('key', key)
                if isinstance(key, str) and ' ' in key:
                    key = key.split()[0]  # 提取音符部分
            
            return original_generate(duration_seconds, bpm, key, stage_index)
        
        mood_flow_app_instance._generate_simple_music = enhanced_generate
    
    logger.info("✅ 增强模块集成完成")
    logger.info(f"增强状态: {adapter.get_enhancement_status()}")
    
    return adapter


# 配置预设
ENHANCEMENT_CONFIGS = {
    'full': {
        'use_enhanced_emotion': True,
        'use_enhanced_planning': True,
        'use_enhanced_mapping': True,
        'use_sota_music_generation': False,  # 默认关闭SOTA模型
        'fallback_to_original': True
    },
    'full_with_sota': {
        'use_enhanced_emotion': True,
        'use_enhanced_planning': True,
        'use_enhanced_mapping': True,
        'use_sota_music_generation': True,   # 启用SOTA音乐生成
        'fallback_to_original': True
    },
    'sota_only': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': False,
        'use_sota_music_generation': True,   # 仅SOTA音乐生成
        'fallback_to_original': True
    },
    'emotion_only': {
        'use_enhanced_emotion': True,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': False,
        'use_sota_music_generation': False,
        'fallback_to_original': True
    },
    'planning_only': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': True,
        'use_enhanced_mapping': False,
        'use_sota_music_generation': False,
        'fallback_to_original': True
    },
    'mapping_only': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': True,
        'use_sota_music_generation': False,
        'fallback_to_original': True
    },
    'disabled': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': False,
        'use_sota_music_generation': False,
        'fallback_to_original': True
    }
}