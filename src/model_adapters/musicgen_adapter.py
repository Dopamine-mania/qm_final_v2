#!/usr/bin/env python3
"""
MusicGen音乐生成适配器 - 集成Meta MusicGen模型进行高质量音乐生成

理论基础：
1. MusicGen (2023): "Simple and Controllable Music Generation"
   - Meta AI Research, https://arxiv.org/abs/2306.05284
   - 32kHz EnCodec tokenizer，4个codebook，50Hz采样

2. Language Models for Music Medicine Generation (ISMIR 2024)
   - 基于Iso原则的治疗音乐生成
   - 使用LoRA微调技术整合情绪标签

3. AudioCraft Toolkit Integration:
   - facebook/musicgen-melody: 1.5B参数，支持旋律引导
   - facebook/musicgen-large: 3.3B参数，最高质量
   - 支持长序列生成（窗口滑动技术）

作者：心境流转团队
日期：2024
"""

import os
import sys
import time
import warnings
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import logging

# 设置日志
logger = logging.getLogger(__name__)

class MusicGenAdapter:
    """
    MusicGen模型适配器 - 为睡眠治疗生成高质量音乐
    
    核心功能：
    1. 智能模型选择（根据可用GPU资源）
    2. 情绪到prompt的映射
    3. 治疗音乐生成（基于ISO原则）
    4. 长序列生成优化
    5. 质量评估和后处理
    """
    
    # 治疗音乐的BPM范围（基于2024年音乐治疗研究）
    THERAPEUTIC_BPM_RANGES = {
        'deep_sleep': (40, 60),      # 深度睡眠
        'relaxation': (60, 80),      # 放松
        'meditation': (70, 90),      # 冥想
        'stress_relief': (80, 100),  # 压力缓解
        'anxiety_reduction': (50, 70) # 焦虑缓解
    }
    
    # 基于FACED数据集的情绪到音乐特征映射
    EMOTION_TO_MUSICAL_FEATURES = {
        'anger': {
            'genre': 'rock, metal',
            'tempo': 'fast',
            'mood': 'aggressive, powerful',
            'instruments': 'electric guitar, drums',
            'key': 'minor'
        },
        'fear': {
            'genre': 'ambient, dark ambient',
            'tempo': 'slow',
            'mood': 'tense, mysterious',
            'instruments': 'strings, synthesizer',
            'key': 'minor'
        },
        'disgust': {
            'genre': 'experimental, industrial',
            'tempo': 'medium',
            'mood': 'harsh, dissonant',
            'instruments': 'synthesizer, noise',
            'key': 'atonal'
        },
        'sadness': {
            'genre': 'blues, ballad',
            'tempo': 'slow',
            'mood': 'melancholic, emotional',
            'instruments': 'piano, strings, acoustic guitar',
            'key': 'minor'
        },
        'amusement': {
            'genre': 'pop, jazz',
            'tempo': 'medium',
            'mood': 'playful, light',
            'instruments': 'piano, brass, percussion',
            'key': 'major'
        },
        'joy': {
            'genre': 'pop, dance',
            'tempo': 'fast',
            'mood': 'uplifting, energetic',
            'instruments': 'synthesizer, drums, bass',
            'key': 'major'
        },
        'inspiration': {
            'genre': 'orchestral, cinematic',
            'tempo': 'medium to fast',
            'mood': 'uplifting, motivational',
            'instruments': 'orchestra, piano, choir',
            'key': 'major'
        },
        'tenderness': {
            'genre': 'acoustic, folk',
            'tempo': 'slow',
            'mood': 'gentle, warm',
            'instruments': 'acoustic guitar, soft vocals, strings',
            'key': 'major'
        },
        'neutral': {
            'genre': 'ambient, instrumental',
            'tempo': 'medium',
            'mood': 'calm, balanced',
            'instruments': 'piano, soft pads',
            'key': 'major or minor'
        }
    }
    
    def __init__(self, 
                 model_size: str = "auto",
                 use_melody_conditioning: bool = True,
                 gpu_memory_gb: Optional[int] = None,
                 enable_long_generation: bool = True):
        """
        初始化MusicGen适配器
        
        Args:
            model_size: 模型大小 ("small", "medium", "large", "auto")
            use_melody_conditioning: 是否使用旋律条件生成
            gpu_memory_gb: GPU显存大小（GB），用于自动选择模型
            enable_long_generation: 是否启用长序列生成
        """
        self.model_size = model_size
        self.use_melody_conditioning = use_melody_conditioning
        self.gpu_memory_gb = gpu_memory_gb
        self.enable_long_generation = enable_long_generation
        
        # 模型和库加载状态
        self.model = None
        self.audiocraft_available = False
        self.sample_rate = 32000  # MusicGen标准采样率
        
        # 初始化模型
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化MusicGen模型"""
        try:
            # 尝试导入audiocraft
            import torch
            import torchaudio
            from audiocraft.models import MusicGen
            from audiocraft.data.audio import audio_write
            
            self.audiocraft_available = True
            self.torch = torch
            self.torchaudio = torchaudio
            self.MusicGen = MusicGen
            self.audio_write = audio_write
            
            # 检查GPU可用性
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"检测到 {gpu_count} 个GPU，显存: {gpu_memory:.1f}GB")
                
                if self.gpu_memory_gb is None:
                    self.gpu_memory_gb = gpu_memory
            else:
                logger.warning("未检测到GPU，将使用CPU模式（性能较低）")
                self.gpu_memory_gb = 0
            
            # 自动选择模型大小
            model_name = self._select_optimal_model()
            
            logger.info(f"正在加载MusicGen模型: {model_name}")
            self.model = self.MusicGen.get_pretrained(model_name)
            
            # 设置默认生成参数
            self._configure_generation_params()
            
            logger.info("✅ MusicGen模型初始化成功！")
            
        except ImportError as e:
            logger.error(f"AudioCraft库未安装: {e}")
            logger.error("请运行: pip install audiocraft")
            self.audiocraft_available = False
            
        except Exception as e:
            logger.error(f"MusicGen模型初始化失败: {e}")
            self.audiocraft_available = False
    
    def _select_optimal_model(self) -> str:
        """根据GPU资源自动选择最优模型"""
        
        if self.model_size != "auto":
            # 手动指定模型大小
            model_mapping = {
                "small": "facebook/musicgen-small",
                "medium": "facebook/musicgen-melody" if self.use_melody_conditioning else "facebook/musicgen-medium",
                "large": "facebook/musicgen-melody-large" if self.use_melody_conditioning else "facebook/musicgen-large"
            }
            return model_mapping.get(self.model_size, "facebook/musicgen-melody")
        
        # 自动选择（基于GPU显存）
        if self.gpu_memory_gb >= 40:
            # 40GB+ GPU：使用最大模型
            model_name = "facebook/musicgen-melody-large" if self.use_melody_conditioning else "facebook/musicgen-large"
            logger.info(f"🚀 使用大型模型 (3.3B参数): {model_name}")
            
        elif self.gpu_memory_gb >= 16:
            # 16-40GB GPU：使用中型模型
            model_name = "facebook/musicgen-melody" if self.use_melody_conditioning else "facebook/musicgen-medium"
            logger.info(f"⚡ 使用中型模型 (1.5B参数): {model_name}")
            
        elif self.gpu_memory_gb >= 8:
            # 8-16GB GPU：使用小型模型
            model_name = "facebook/musicgen-small"
            logger.info(f"💻 使用小型模型 (300M参数): {model_name}")
            
        else:
            # <8GB或CPU：使用最小模型
            model_name = "facebook/musicgen-small"
            logger.warning(f"⚠️ GPU显存不足，使用小型模型: {model_name}")
        
        return model_name
    
    def _configure_generation_params(self):
        """配置音乐生成参数"""
        # 针对治疗音乐优化的参数（v2.0 - 高质量版本）
        self.model.set_generation_params(
            duration=10,        # 默认10秒（可动态调整）
            temperature=0.9,    # 提高创造性和音乐质量 (0.8→0.9)
            top_k=200,         # 更严格的token选择，提高质量 (250→200)
            top_p=0.0,         # 禁用nucleus采样
            cfg_coef=5.0       # 大幅增强文本条件影响 (3.0→5.0)
        )
        logger.info("🎛️ MusicGen参数已优化: temp=0.9, top_k=200, cfg_coef=5.0")
    
    def generate_therapeutic_music(self, 
                                 emotion_state: Dict,
                                 stage_info: Dict,
                                 duration_seconds: float = 60,
                                 bpm_target: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        生成治疗音乐
        
        Args:
            emotion_state: 情绪状态信息 (valence, arousal, primary_emotion)
            stage_info: 治疗阶段信息 (stage_name, therapy_goal)
            duration_seconds: 音乐时长（秒）
            bpm_target: 目标BPM（可选）
            
        Returns:
            (audio_data, metadata): 音频数据和元数据
        """
        if not self.audiocraft_available:
            raise RuntimeError("MusicGen不可用，请检查audiocraft安装")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎵 [MusicGen v2.0] 生成治疗音乐:")
        logger.info(f"{'='*60}")
        
        # 1. 构建音乐prompt
        prompt = self._build_therapeutic_prompt(emotion_state, stage_info, bpm_target)
        logger.info(f"📝 生成Prompt: {prompt}")
        
        # 2. 设置生成参数
        self.model.set_generation_params(duration=min(duration_seconds, 30))  # MusicGen单次最大30秒
        
        try:
            # 3. 生成音乐
            start_time = time.time()
            
            if duration_seconds <= 30:
                # 短序列：直接生成
                wav = self.model.generate([prompt])
                
                # 检查生成结果
                if wav is None or len(wav) == 0:
                    raise RuntimeError("MusicGen生成失败：返回空结果")
                
                # 确保转换为numpy数组
                if hasattr(wav[0], 'cpu'):
                    audio_data = wav[0].cpu().numpy().flatten()
                else:
                    audio_data = wav[0].flatten()
                
                # 检查音频数据有效性
                if len(audio_data) == 0:
                    raise RuntimeError("MusicGen生成失败：音频数据为空")
                
            else:
                # 长序列：使用窗口滑动技术
                audio_data = self._generate_long_sequence(prompt, duration_seconds)
            
            generation_time = time.time() - start_time
            
            # 4. 后处理
            audio_data = self._post_process_audio(audio_data, stage_info)
            
            # 5. 生成元数据
            metadata = {
                'prompt': prompt,
                'emotion_state': emotion_state,
                'stage_info': stage_info,
                'duration': len(audio_data) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'generation_time': generation_time,
                'model_used': getattr(getattr(self.model, 'cfg', None), 'name', 'musicgen_model'),
                'bpm_target': bpm_target
            }
            
            logger.info(f"⏱️ 生成时间: {generation_time:.1f}秒")
            logger.info(f"🎼 音乐时长: {len(audio_data) / self.sample_rate:.1f}秒")
            logger.info(f"📊 采样率: {self.sample_rate}Hz")
            logger.info(f"{'='*60}")
            
            return audio_data, metadata
            
        except Exception as e:
            logger.error(f"音乐生成失败: {e}")
            # 返回空音频作为fallback
            fallback_audio = np.zeros(int(duration_seconds * self.sample_rate))
            return fallback_audio, {'error': str(e)}
    
    def _build_therapeutic_prompt(self, 
                                emotion_state: Dict, 
                                stage_info: Dict,
                                bpm_target: Optional[int] = None) -> str:
        """
        构建治疗音乐的prompt
        
        基于情绪状态、治疗阶段和音乐治疗理论
        """
        # 获取主要情绪
        primary_emotion = emotion_state.get('primary_emotion', 'neutral')
        valence = emotion_state.get('valence', 0.0)
        arousal = emotion_state.get('arousal', 0.0)
        
        # 获取治疗阶段
        stage_name = stage_info.get('stage_name', 'unknown')
        
        # 调试：显示stage_info内容
        logger.info(f"🔍 Prompt构建 - stage_info: {stage_info}")
        logger.info(f"🔍 Prompt构建 - stage_name: {stage_name}")
        
        # 基础音乐特征
        emotion_features = self.EMOTION_TO_MUSICAL_FEATURES.get(primary_emotion, 
                                                              self.EMOTION_TO_MUSICAL_FEATURES['neutral'])
        
        # 根据治疗阶段调整
        if '同步化' in stage_name or 'Sync' in stage_name:
            # 同步化：匹配当前情绪，但稍作缓和
            mood_adjustment = "slightly calmer"
            tempo_adjustment = "moderate"
            therapeutic_goal = "establishing emotional connection"
            
        elif '引导化' in stage_name or 'Guide' in stage_name:
            # 引导化：逐渐转向平静
            mood_adjustment = "gradually becoming peaceful"
            tempo_adjustment = "slowing down"
            therapeutic_goal = "transitioning to relaxation"
            
        elif '巩固化' in stage_name or 'Consolidate' in stage_name:
            # 巩固化：深度放松，准备睡眠
            mood_adjustment = "deeply relaxing and sleep-inducing"
            tempo_adjustment = "very slow"
            therapeutic_goal = "deep sleep preparation"
            
        else:
            mood_adjustment = "therapeutic and calming"
            tempo_adjustment = "relaxed"
            therapeutic_goal = "general relaxation"
        
        # 确定BPM
        if bpm_target:
            bpm_text = f"{bpm_target} BPM"
        else:
            # 根据arousal计算BPM（映射到治疗范围）
            if arousal > 0.5:
                bpm_range = self.THERAPEUTIC_BPM_RANGES['stress_relief']
            elif arousal > 0:
                bpm_range = self.THERAPEUTIC_BPM_RANGES['relaxation']
            else:
                bpm_range = self.THERAPEUTIC_BPM_RANGES['deep_sleep']
            
            bpm_target = int(bpm_range[0] + (bpm_range[1] - bpm_range[0]) * (arousal + 1) / 2)
            bpm_text = f"{bpm_target} BPM"
        
        # 构建prompt（基于2024年MusicGen最佳实践）
        prompt_parts = [
            # 核心治疗目标
            f"therapeutic {emotion_features['genre']} music for {therapeutic_goal}",
            
            # 情绪和氛围
            f"{mood_adjustment}, {emotion_features['mood']}",
            
            # 乐器和质感
            f"featuring {emotion_features['instruments']}",
            
            # 节奏和调性
            f"{bpm_text}, {emotion_features['key']} key",
            
            # 治疗特性
            f"{tempo_adjustment}, sleep therapy, ambient pads, gentle transitions",
            
            # 技术品质
            "high quality, professional, studio recording"
        ]
        
        prompt = ", ".join(prompt_parts)
        
        # 限制prompt长度（MusicGen最佳实践：<200字符）
        if len(prompt) > 200:
            prompt = prompt[:197] + "..."
        
        return prompt
    
    def _generate_long_sequence(self, prompt: str, duration_seconds: float) -> np.ndarray:
        """
        生成长序列音乐（使用窗口滑动技术）
        
        基于MusicGen论文中的长序列生成方法：
        - 30秒窗口，10秒滑动
        - 保持20秒上下文
        """
        if not self.enable_long_generation:
            # 如果禁用长序列生成，直接生成30秒
            wav = self.model.generate([prompt])
            return wav[0].cpu().numpy()
        
        logger.info(f"🔄 使用窗口滑动技术生成 {duration_seconds:.1f}秒音乐")
        
        # 参数设置
        window_duration = 30  # 30秒窗口
        slide_duration = 10   # 10秒滑动
        context_duration = 20 # 20秒上下文
        
        # 计算需要的窗口数
        num_windows = int(np.ceil((duration_seconds - window_duration) / slide_duration)) + 1
        
        # 存储完整音频
        full_audio = None
        context_audio = None
        
        for i in range(num_windows):
            logger.info(f"  生成窗口 {i+1}/{num_windows}")
            
            if context_audio is not None:
                # 使用上下文音频进行条件生成
                # 注意：这需要MusicGen的continuation功能
                # 当前简化实现：直接生成新片段
                pass
            
            # 生成当前窗口
            wav = self.model.generate([prompt])
            # 确保转换为numpy数组
            if hasattr(wav[0], 'cpu'):
                current_audio = wav[0].cpu().numpy().flatten()
            else:
                current_audio = wav[0].flatten()
            
            if full_audio is None:
                # 第一个窗口：完整使用
                full_audio = current_audio
            else:
                # 后续窗口：只使用滑动部分，并进行交叉淡化
                slide_samples = int(slide_duration * self.sample_rate)
                
                # 简单拼接（未来可改进为交叉淡化）
                full_audio = np.concatenate([full_audio, current_audio[:slide_samples]])
            
            # 更新上下文（保留最后20秒）
            context_samples = int(context_duration * self.sample_rate)
            if len(full_audio) > context_samples:
                context_audio = full_audio[-context_samples:]
            else:
                context_audio = full_audio
        
        # 截取到目标长度
        target_samples = int(duration_seconds * self.sample_rate)
        if len(full_audio) > target_samples:
            full_audio = full_audio[:target_samples]
        
        return full_audio
    
    def _post_process_audio(self, audio_data: np.ndarray, stage_info: Dict) -> np.ndarray:
        """
        音频后处理
        
        根据治疗阶段应用特定的音频处理
        """
        stage_name = stage_info.get('stage_name', '')
        
        # 应用阶段特定的后处理
        if '同步化' in stage_name:
            # 同步化阶段：保持原有动态
            pass
            
        elif '引导化' in stage_name:
            # 引导化阶段：逐渐降低音量
            fade_length = int(0.3 * len(audio_data))  # 最后30%淡出
            fade_curve = np.linspace(1.0, 0.8, fade_length)
            audio_data[-fade_length:] *= fade_curve
            
        elif '巩固化' in stage_name:
            # 巩固化阶段：轻微降低音量，保持音质
            audio_data *= 0.85  # 温和降低音量 (0.7→0.85)
            
            # 改进的低通滤波，减少音质损失
            audio_data = self._gentle_lowpass_filter(audio_data)
        
        # 改进的音量归一化，保持动态范围
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            # 使用更温和的归一化，保持音质
            target_level = 0.9 if max_amplitude > 0.9 else max_amplitude
            audio_data = audio_data / max_amplitude * target_level
        
        return audio_data
    
    def _simple_lowpass_filter(self, audio_data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """简单的低通滤波器（一阶IIR）"""
        filtered = np.zeros_like(audio_data)
        filtered[0] = audio_data[0]
        
        for i in range(1, len(audio_data)):
            filtered[i] = alpha * audio_data[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def _gentle_lowpass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """温和的低通滤波，保持音质"""
        # 使用更温和的滤波参数，减少音质损失
        alpha = 0.05  # 更小的alpha值，更温和的滤波
        
        # 应用双向滤波，减少相位失真
        forward_filtered = self._simple_lowpass_filter(audio_data, alpha)
        backward_filtered = self._simple_lowpass_filter(forward_filtered[::-1], alpha)
        result = backward_filtered[::-1]
        
        # 混合原始信号和滤波信号，保持清晰度
        mix_ratio = 0.7  # 70%滤波 + 30%原始
        return mix_ratio * result + (1 - mix_ratio) * audio_data
    
    def save_audio(self, audio_data: np.ndarray, file_path: str, metadata: Optional[Dict] = None):
        """
        保存音频文件
        
        Args:
            audio_data: 音频数据
            file_path: 保存路径
            metadata: 元数据（可选）
        """
        try:
            if self.audiocraft_available:
                # 使用audiocraft的保存功能
                audio_tensor = self.torch.from_numpy(audio_data).unsqueeze(0)
                self.audio_write(
                    Path(file_path).stem,
                    audio_tensor,
                    self.sample_rate,
                    strategy="loudness"  # 响度归一化
                )
                logger.info(f"✅ 音频已保存: {file_path}")
                
            else:
                # fallback到numpy保存
                # 这里需要其他音频库支持
                logger.warning("AudioCraft不可用，音频保存可能失败")
                
        except Exception as e:
            logger.error(f"音频保存失败: {e}")
    
    def is_available(self) -> bool:
        """检查MusicGen是否可用"""
        return self.audiocraft_available and self.model is not None
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.is_available():
            return {'status': 'unavailable'}
        
        return {
            'status': 'available',
            'model_name': getattr(self.model.cfg, 'name', 'unknown') if hasattr(self.model, 'cfg') else 'unknown',
            'sample_rate': self.sample_rate,
            'gpu_memory_gb': self.gpu_memory_gb,
            'use_melody_conditioning': self.use_melody_conditioning,
            'enable_long_generation': self.enable_long_generation
        }


# 工厂函数，便于集成
def create_musicgen_adapter(
    model_size: str = "auto",
    use_melody_conditioning: bool = True,
    gpu_memory_gb: Optional[int] = None
) -> MusicGenAdapter:
    """
    创建MusicGen适配器实例
    
    Args:
        model_size: 模型大小选择
        use_melody_conditioning: 是否使用旋律条件
        gpu_memory_gb: GPU显存大小（用于自动选择）
        
    Returns:
        MusicGenAdapter实例
    """
    return MusicGenAdapter(
        model_size=model_size,
        use_melody_conditioning=use_melody_conditioning,
        gpu_memory_gb=gpu_memory_gb
    )