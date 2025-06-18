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
                 fallback_to_original: bool = True):
        """
        初始化适配器
        
        Args:
            use_enhanced_emotion: 是否使用增强情绪识别
            use_enhanced_planning: 是否使用增强治疗规划
            use_enhanced_mapping: 是否使用增强音乐映射
            fallback_to_original: 出错时是否回退到原始实现
        """
        self.use_enhanced_emotion = use_enhanced_emotion
        self.use_enhanced_planning = use_enhanced_planning
        self.use_enhanced_mapping = use_enhanced_mapping
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
                
                # 记录详细信息
                logger.info(f"🎯 识别到细粒度情绪: {detailed_emotion.primary_emotion} "
                          f"(V={detailed_emotion.valence:.2f}, A={detailed_emotion.arousal:.2f}, "
                          f"置信度={detailed_emotion.confidence:.2f})")
                
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
                
                # 记录规划信息
                logger.info(f"📋 生成增强治疗计划:")
                for i, stage in enumerate(stages):
                    logger.info(f"  阶段{i+1}: {stage['stage'].value} - "
                              f"{stage['duration']:.1f}分钟")
                
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
                
                # 记录映射信息
                logger.info(f"🎵 生成音乐参数: BPM={music_params.get('bpm', 'N/A')}, "
                          f"调性={music_params.get('key', 'N/A')}")
                
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
            'music_mapping': self.use_enhanced_mapping and self.music_mapper is not None
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
    
    def enhanced_plan(current, target, duration):
        return adapter.plan_therapy_stages_enhanced(
            current, target, duration, mood_flow_app_instance.iso_model
        )
    
    # 应用增强
    mood_flow_app_instance.analyze_emotion_from_text = enhanced_analyze
    mood_flow_app_instance.plan_therapy_stages = enhanced_plan
    
    # 添加增强状态查询方法
    mood_flow_app_instance.get_enhancement_status = adapter.get_enhancement_status
    mood_flow_app_instance.get_detailed_emotion_info = adapter.get_detailed_emotion_info
    
    # 添加音乐参数增强
    if hasattr(mood_flow_app_instance, '_generate_simple_music'):
        original_generate = mood_flow_app_instance._generate_simple_music
        
        def enhanced_generate(duration_seconds, bpm, key, stage_index):
            # 获取当前情绪状态
            if hasattr(mood_flow_app_instance, 'current_session'):
                emotion = mood_flow_app_instance.current_session.iso_stages[stage_index]['emotion']
                params = adapter.get_music_parameters_enhanced(emotion, 
                                                              mood_flow_app_instance.current_session.iso_stages[stage_index],
                                                              original_music_model)
                # 使用增强参数
                bpm = params.get('bpm', bpm)
                key = params.get('key', key).split()[0]  # 提取音符部分
            
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
        'fallback_to_original': True
    },
    'emotion_only': {
        'use_enhanced_emotion': True,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': False,
        'fallback_to_original': True
    },
    'planning_only': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': True,
        'use_enhanced_mapping': False,
        'fallback_to_original': True
    },
    'mapping_only': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': True,
        'fallback_to_original': True
    },
    'disabled': {
        'use_enhanced_emotion': False,
        'use_enhanced_planning': False,
        'use_enhanced_mapping': False,
        'fallback_to_original': True
    }
}