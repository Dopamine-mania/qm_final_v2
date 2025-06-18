#!/usr/bin/env python3
"""
视频生成适配器 - 将治疗视频生成器集成到心境流转系统

设计模式：适配器模式
目的：无缝集成新的视频生成功能到现有系统

作者：心境流转团队
日期：2024
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.video_generation.therapeutic_video_generator import (
    TherapeuticVideoGenerator, VideoConfig, create_therapeutic_video_generator
)

# 设置日志
logger = logging.getLogger(__name__)

class VideoGenerationAdapter:
    """
    视频生成适配器
    
    将新的治疗视频生成器适配到现有的MoodFlowApp系统
    保持向后兼容性，同时提供增强功能
    """
    
    def __init__(self, 
                 use_therapeutic_generator: bool = True,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 quality: str = "medium"):
        """
        初始化适配器
        
        Args:
            use_therapeutic_generator: 是否使用新的治疗视频生成器
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            quality: 质量设置
        """
        self.use_therapeutic = use_therapeutic_generator
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        
        # 初始化生成器
        if self.use_therapeutic:
            try:
                self.generator = create_therapeutic_video_generator(
                    width=width, height=height, fps=fps, quality=quality
                )
                logger.info("✅ 治疗视频生成器初始化成功")
            except Exception as e:
                logger.error(f"治疗视频生成器初始化失败: {e}")
                self.generator = None
                self.use_therapeutic = False
        else:
            self.generator = None
            
        # 回退到原始生成器（如果需要）
        self.fallback_generator = None
    
    def generate_stage_videos_enhanced(self, 
                                     stages: List[Dict], 
                                     session_name: str,
                                     output_dir: Path,
                                     create_full_videos: bool = False,
                                     music_data: Optional[Dict] = None) -> List[str]:
        """
        增强的阶段视频生成
        
        Args:
            stages: 治疗阶段列表
            session_name: 会话名称
            output_dir: 输出目录
            create_full_videos: 是否生成完整视频
            music_data: 音乐同步数据
            
        Returns:
            生成的文件路径列表
        """
        if not self.use_therapeutic or not self.generator:
            # 回退到原始方法
            return self._fallback_generate(stages, session_name, output_dir, create_full_videos)
        
        logger.info(f"🎬 [治疗视频生成] 开始生成 {len(stages)} 个阶段的视频")
        
        video_files = []
        
        for i, stage in enumerate(stages):
            stage_name = stage['stage'].value if hasattr(stage['stage'], 'value') else str(stage['stage'])
            stage_duration = stage['duration'] * 60  # 分钟转秒
            
            # 从音乐数据提取BPM
            stage_bpm = None
            if music_data and 'bpms' in music_data:
                stage_bpm = music_data['bpms'].get(i, 60)
            
            logger.info(f"  阶段{i+1}: {stage_name}")
            logger.info(f"    时长: {stage_duration}秒")
            if stage_bpm:
                logger.info(f"    BPM: {stage_bpm}")
            
            # 创建阶段输出目录
            stage_dir = output_dir / f"{session_name}_stage_{i+1}"
            stage_dir.mkdir(exist_ok=True)
            
            if create_full_videos:
                # 生成完整视频
                video_path = stage_dir / f"stage_{i+1}_video.mp4"
                
                try:
                    self.generator.generate_therapy_video(
                        stage_name=stage_name,
                        duration_seconds=stage_duration,
                        music_bpm=stage_bpm,
                        output_path=str(video_path)
                    )
                    video_files.append(str(video_path))
                    logger.info(f"    ✅ 视频生成成功: {video_path.name}")
                    
                except Exception as e:
                    logger.error(f"    ❌ 视频生成失败: {e}")
                    # 生成预览图作为备份
                    preview_path = self._generate_preview_fallback(
                        stage_name, stage_dir, i
                    )
                    if preview_path:
                        video_files.append(preview_path)
            else:
                # 只生成预览图
                preview_path = self._generate_preview_enhanced(
                    stage_name, stage_duration, stage_bpm, stage_dir, i
                )
                if preview_path:
                    video_files.append(preview_path)
        
        return video_files
    
    def _generate_preview_enhanced(self, 
                                 stage_name: str,
                                 duration: float,
                                 bpm: Optional[int],
                                 output_dir: Path,
                                 stage_index: int) -> Optional[str]:
        """生成增强的预览图"""
        try:
            # 生成预览帧
            frames = self.generator.generate_therapy_video(
                stage_name=stage_name,
                duration_seconds=duration,
                music_bpm=bpm,
                output_path=None  # 返回帧列表
            )
            
            if frames and len(frames) > 0:
                # 保存第一帧作为预览
                preview_file = output_dir / "preview.png"
                
                # 选择中间帧（通常更有代表性）
                preview_frame = frames[len(frames)//2]
                
                # 使用生成器的保存方法（已处理cv2可用性）
                self.generator._save_frame_as_image(preview_frame, str(preview_file))
                
                logger.info(f"    ✅ 预览图生成成功: {preview_file.name}")
                return str(preview_file)
            
        except Exception as e:
            logger.error(f"预览生成失败: {e}")
        
        return None
    
    def _generate_preview_fallback(self, stage_name: str, output_dir: Path, stage_index: int) -> Optional[str]:
        """生成简单的备份预览图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            # 背景颜色（根据阶段）
            colors = {
                "同步化": "#4682B4",  # 钢蓝色
                "引导化": "#9370DB",  # 中紫色
                "巩固化": "#191970"   # 午夜蓝
            }
            bg_color = colors.get(stage_name, "#4169E1")
            
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            
            # 添加圆形元素
            circle = patches.Circle((0.5, 0.5), 0.3, 
                                  facecolor='white', 
                                  alpha=0.2,
                                  transform=ax.transAxes)
            ax.add_patch(circle)
            
            # 添加阶段文字
            ax.text(0.5, 0.5, stage_name, 
                   transform=ax.transAxes,
                   ha='center', va='center',
                   fontsize=24, color='white',
                   alpha=0.8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            preview_file = output_dir / "preview.png"
            plt.savefig(preview_file, dpi=150, bbox_inches='tight', 
                       facecolor=bg_color, edgecolor='none')
            plt.close()
            
            return str(preview_file)
            
        except Exception as e:
            logger.error(f"备份预览生成失败: {e}")
            return None
    
    def _fallback_generate(self, stages: List[Dict], session_name: str, 
                         output_dir: Path, create_full_videos: bool) -> List[str]:
        """回退到原始生成方法"""
        logger.warning("使用原始视频生成方法")
        
        video_files = []
        
        # 这里可以调用原始的SleepVideoGenerator
        # 或者生成简单的占位图片
        for i, stage in enumerate(stages):
            stage_dir = output_dir / f"{session_name}_stage_{i+1}"
            stage_dir.mkdir(exist_ok=True)
            
            preview_path = self._generate_preview_fallback(
                stage['stage'].value if hasattr(stage['stage'], 'value') else str(stage['stage']),
                stage_dir, i
            )
            
            if preview_path:
                video_files.append(preview_path)
        
        return video_files
    
    def generate_complete_therapy_video(self,
                                      stages: List[Dict],
                                      session_name: str,
                                      output_path: str,
                                      music_sync_data: Optional[Dict] = None) -> Optional[str]:
        """
        生成完整的治疗视频
        
        Args:
            stages: 阶段列表
            session_name: 会话名称
            output_path: 输出路径
            music_sync_data: 音乐同步数据
            
        Returns:
            视频文件路径或None
        """
        if not self.use_therapeutic or not self.generator:
            logger.warning("治疗视频生成器不可用")
            return None
        
        try:
            logger.info("🎬 生成完整治疗视频...")
            
            # 使用新的生成器创建完整视频
            video_path = self.generator.generate_complete_therapy_video(
                stages=stages,
                output_path=output_path,
                music_sync_data=music_sync_data
            )
            
            return video_path
            
        except Exception as e:
            logger.error(f"完整视频生成失败: {e}")
            return None
    
    def get_generator_info(self) -> Dict:
        """获取生成器信息"""
        if self.use_therapeutic and self.generator:
            return {
                "type": "therapeutic",
                "status": "active",
                "resolution": f"{self.width}x{self.height}",
                "fps": self.fps,
                "quality": self.quality,
                "features": [
                    "三阶段治疗视觉",
                    "音乐节奏同步",
                    "治疗色彩方案",
                    "呼吸引导动画",
                    "双耳视觉节拍"
                ]
            }
        else:
            return {
                "type": "fallback",
                "status": "limited",
                "features": ["基础预览图生成"]
            }


# 便捷集成函数
def integrate_video_generation(mood_flow_app_instance, config: Optional[Dict] = None):
    """
    将视频生成功能集成到MoodFlowApp
    
    Args:
        mood_flow_app_instance: MoodFlowApp实例
        config: 配置选项
    """
    # 默认配置
    if config is None:
        config = {
            'use_therapeutic_generator': True,
            'width': 1280,
            'height': 720,
            'fps': 30,
            'quality': 'medium'
        }
    
    # 创建适配器
    adapter = VideoGenerationAdapter(**config)
    
    # 保存原始方法
    original_generate_videos = mood_flow_app_instance.generate_stage_videos
    
    # 增强的视频生成方法
    def enhanced_generate_videos(stages, session_name, create_full_videos=False):
        # 收集音乐数据（如果有）
        music_data = None
        if hasattr(mood_flow_app_instance, 'current_session'):
            music_data = {
                'bpms': []
            }
            # 提取每个阶段的BPM
            for stage in stages:
                emotion = stage['emotion']
                bpm = mood_flow_app_instance.music_model.calc_bpm(emotion.arousal)
                music_data['bpms'].append(bpm)
        
        # 使用适配器生成视频
        return adapter.generate_stage_videos_enhanced(
            stages=stages,
            session_name=session_name,
            output_dir=mood_flow_app_instance.output_dir,
            create_full_videos=create_full_videos,
            music_data=music_data
        )
    
    # 替换方法
    mood_flow_app_instance.generate_stage_videos = enhanced_generate_videos
    
    # 添加完整视频生成功能
    mood_flow_app_instance.generate_complete_video = lambda stages, session_name, output_path, music_sync: \
        adapter.generate_complete_therapy_video(stages, session_name, output_path, music_sync)
    
    # 添加视频生成器信息查询
    mood_flow_app_instance.get_video_generator_info = adapter.get_generator_info
    
    logger.info("✅ 视频生成功能集成完成")
    
    return adapter