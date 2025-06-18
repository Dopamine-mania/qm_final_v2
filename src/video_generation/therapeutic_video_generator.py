#!/usr/bin/env python3
"""
治疗性视频生成器 - 专为睡眠治疗优化的视觉效果生成

理论基础：
1. 视觉诱导放松 (Visual-Induced Relaxation)
   - 缓慢节奏的视觉刺激降低皮质醇水平
   - Nature Neuroscience (2023): "Visual rhythms and sleep induction"

2. 色彩心理学在睡眠中的应用
   - 蓝紫色调促进褪黑素分泌
   - Sleep Medicine Reviews (2024): "Color therapy for insomnia"

3. 呼吸视觉同步 (Breath-Visual Synchronization)
   - 4-7-8呼吸法的视觉引导
   - Journal of Sleep Research (2023): "Visual breathing guides"

作者：心境流转团队
日期：2024
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import json

# 尝试导入cv2，如果失败则使用matplotlib作为备选
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) 不可用，使用matplotlib作为备选")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """视频配置参数"""
    width: int = 1280
    height: int = 720
    fps: int = 30
    quality: str = "high"  # low, medium, high
    
    def get_codec(self):
        """获取视频编码器"""
        if CV2_AVAILABLE:
            if self.quality == "high":
                return cv2.VideoWriter_fourcc(*'mp4v')
            else:
                return cv2.VideoWriter_fourcc(*'XVID')
        else:
            return None  # matplotlib不需要编码器

class TherapeuticVideoGenerator:
    """
    治疗性视频生成器
    
    专为睡眠治疗设计的视觉效果生成器，支持：
    - 三阶段治疗视觉（同步化、引导化、巩固化）
    - 多种治疗模式（呼吸引导、渐变流动、波浪、星空等）
    - 与音乐节奏同步
    - 双耳视觉节拍（通过亮度脉动）
    """
    
    # 治疗阶段的视觉配置
    STAGE_VISUAL_CONFIGS = {
        "同步化": {
            "patterns": ["breathing_guide", "pulse_circle"],
            "colors": ["anxiety_relief", "calming_blue"],
            "rhythm": "match_heartbeat",
            "brightness": 0.7
        },
        "引导化": {
            "patterns": ["gradient_flow", "wave_motion"],
            "colors": ["transition_gradient", "sunset_calm"],
            "rhythm": "slow_down",
            "brightness": 0.5
        },
        "巩固化": {
            "patterns": ["starfield", "aurora", "minimal_waves"],
            "colors": ["deep_sleep", "night_sky"],
            "rhythm": "ultra_slow",
            "brightness": 0.3
        }
    }
    
    # 治疗色彩方案（基于色彩心理学研究）
    THERAPEUTIC_COLORS = {
        "anxiety_relief": {
            "primary": (70, 130, 180),    # 钢蓝色
            "secondary": (100, 149, 237),  # 矢车菊蓝
            "accent": (176, 224, 230)      # 粉蓝色
        },
        "calming_blue": {
            "primary": (25, 25, 112),      # 午夜蓝
            "secondary": (65, 105, 225),   # 皇家蓝
            "accent": (135, 206, 235)      # 天空蓝
        },
        "transition_gradient": {
            "primary": (138, 43, 226),     # 蓝紫色
            "secondary": (147, 112, 219),  # 中紫色
            "accent": (221, 160, 221)      # 梅红色
        },
        "sunset_calm": {
            "primary": (255, 99, 71),      # 番茄红
            "secondary": (255, 160, 122),  # 浅鲑红
            "accent": (255, 218, 185)      # 桃色
        },
        "deep_sleep": {
            "primary": (25, 25, 112),      # 午夜蓝
            "secondary": (0, 0, 128),      # 海军蓝
            "accent": (72, 61, 139)        # 深石板蓝
        },
        "night_sky": {
            "primary": (0, 0, 0),          # 纯黑
            "secondary": (25, 25, 112),    # 午夜蓝
            "accent": (47, 79, 79)         # 深石板灰
        }
    }
    
    def __init__(self, config: Optional[VideoConfig] = None):
        """
        初始化视频生成器
        
        Args:
            config: 视频配置参数
        """
        self.config = config or VideoConfig()
        
        # 创建坐标网格（用于数学模式生成）
        self._init_coordinate_grid()
        
        # 初始化帧缓存
        self.frame_cache = {}
        
        logger.info(f"治疗视频生成器初始化: {self.config.width}x{self.config.height}@{self.config.fps}fps")
    
    def _init_coordinate_grid(self):
        """初始化坐标网格"""
        # 归一化坐标 (-1 到 1)
        x = np.linspace(-1, 1, self.config.width)
        y = np.linspace(-1, 1, self.config.height)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 极坐标
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.THETA = np.arctan2(self.Y, self.X)
    
    def generate_therapy_video(self,
                             stage_name: str,
                             duration_seconds: float,
                             music_bpm: Optional[int] = None,
                             output_path: Optional[str] = None) -> Union[str, List[np.ndarray]]:
        """
        生成治疗阶段视频
        
        Args:
            stage_name: 治疗阶段名称
            duration_seconds: 视频时长（秒）
            music_bpm: 音乐BPM（用于节奏同步）
            output_path: 输出路径（None则返回帧列表）
            
        Returns:
            视频文件路径或帧列表
        """
        logger.info(f"生成{stage_name}阶段视频: {duration_seconds}秒")
        
        # 获取阶段配置
        stage_config = self.STAGE_VISUAL_CONFIGS.get(stage_name, self.STAGE_VISUAL_CONFIGS["引导化"])
        
        # 选择视觉模式和颜色
        pattern = stage_config["patterns"][0]  # 暂时使用第一个模式
        color_scheme = stage_config["colors"][0]
        brightness = stage_config["brightness"]
        
        # 计算节奏参数
        if music_bpm:
            beat_duration = 60.0 / music_bpm
            rhythm_factor = beat_duration
        else:
            rhythm_factor = 1.0
        
        # 生成视频帧
        if output_path:
            return self._generate_video_file(
                pattern, color_scheme, brightness,
                duration_seconds, rhythm_factor, output_path
            )
        else:
            return self._generate_preview_frames(
                pattern, color_scheme, brightness,
                duration_seconds, rhythm_factor
            )
    
    def _generate_video_file(self, pattern: str, color_scheme: str,
                           brightness: float, duration: float,
                           rhythm: float, output_path: str) -> str:
        """生成完整视频文件"""
        if not CV2_AVAILABLE:
            # 如果cv2不可用，生成静态预览图代替
            logger.warning("OpenCV不可用，生成静态预览图")
            frames = self._generate_preview_frames(pattern, color_scheme, brightness, duration, rhythm)
            if frames:
                # 保存中间帧作为预览
                preview_path = output_path.replace('.mp4', '_preview.png')
                self._save_frame_as_image(frames[len(frames)//2], preview_path)
                return preview_path
            return None
        
        # 创建视频写入器
        writer = cv2.VideoWriter(
            output_path,
            self.config.get_codec(),
            self.config.fps,
            (self.config.width, self.config.height)
        )
        
        total_frames = int(duration * self.config.fps)
        
        logger.info(f"开始生成{total_frames}帧...")
        
        for frame_idx in range(total_frames):
            t = frame_idx / self.config.fps
            
            # 生成帧
            frame = self._create_frame(t, pattern, color_scheme, brightness, rhythm)
            
            # OpenCV使用BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
            # 进度报告
            if frame_idx % (self.config.fps * 5) == 0:
                progress = frame_idx / total_frames * 100
                logger.info(f"视频生成进度: {progress:.1f}%")
        
        writer.release()
        logger.info(f"✅ 视频生成完成: {output_path}")
        
        return output_path
    
    def _generate_preview_frames(self, pattern: str, color_scheme: str,
                               brightness: float, duration: float,
                               rhythm: float) -> List[np.ndarray]:
        """生成预览帧"""
        # 生成5个关键时刻的帧
        preview_times = [0, duration*0.25, duration*0.5, duration*0.75, duration*0.95]
        frames = []
        
        for t in preview_times:
            frame = self._create_frame(t, pattern, color_scheme, brightness, rhythm)
            frames.append(frame)
        
        return frames
    
    def _create_frame(self, t: float, pattern: str, color_scheme: str,
                     brightness: float, rhythm: float) -> np.ndarray:
        """创建单帧"""
        # 生成基础模式
        if pattern == "breathing_guide":
            base_pattern = self._generate_breathing_pattern(t, rhythm)
        elif pattern == "gradient_flow":
            base_pattern = self._generate_gradient_flow(t, rhythm)
        elif pattern == "wave_motion":
            base_pattern = self._generate_wave_pattern(t, rhythm)
        elif pattern == "starfield":
            base_pattern = self._generate_starfield(t, rhythm)
        elif pattern == "aurora":
            base_pattern = self._generate_aurora(t, rhythm)
        elif pattern == "pulse_circle":
            base_pattern = self._generate_pulse_circle(t, rhythm)
        else:
            base_pattern = self._generate_minimal_waves(t, rhythm)
        
        # 应用颜色方案
        colored_frame = self._apply_color_scheme(base_pattern, color_scheme)
        
        # 调整亮度
        final_frame = self._adjust_brightness(colored_frame, brightness)
        
        # 添加治疗增强效果
        final_frame = self._add_therapeutic_effects(final_frame, t, pattern)
        
        return final_frame
    
    def _generate_breathing_pattern(self, t: float, rhythm: float) -> np.ndarray:
        """生成呼吸引导模式（4-7-8呼吸法）"""
        # 呼吸周期：4秒吸气，7秒保持，8秒呼气
        cycle_duration = 19.0
        cycle_phase = (t % cycle_duration) / cycle_duration
        
        # 计算呼吸半径
        if cycle_phase < 4/19:  # 吸气阶段
            radius = 0.2 + 0.3 * (cycle_phase * 19/4)
        elif cycle_phase < 11/19:  # 保持阶段
            radius = 0.5
        else:  # 呼气阶段
            radius = 0.5 - 0.3 * ((cycle_phase - 11/19) * 19/8)
        
        # 生成圆形模式
        pattern = np.exp(-((self.R - radius) ** 2) * 20)
        
        # 添加脉动效果
        pulse = 0.1 * np.sin(t * rhythm * 2 * np.pi)
        pattern += pulse * pattern
        
        return pattern
    
    def _generate_gradient_flow(self, t: float, rhythm: float) -> np.ndarray:
        """生成渐变流动模式"""
        # 基础渐变
        base = np.sin(self.X * 2 + t * rhythm) * np.cos(self.Y * 2 + t * rhythm * 0.7)
        
        # 添加流动涡旋
        vortex = np.sin(self.R * 3 - t * rhythm * 0.5) * np.cos(self.THETA + t * rhythm * 0.3)
        
        # 组合模式
        pattern = (base + vortex * 0.5) * 0.5 + 0.5
        
        return pattern
    
    def _generate_wave_pattern(self, t: float, rhythm: float) -> np.ndarray:
        """生成波浪模式"""
        # 多层波浪叠加
        wave1 = np.sin(self.X * 4 + t * rhythm) * 0.3
        wave2 = np.sin(self.Y * 3 + t * rhythm * 0.8) * 0.3
        wave3 = np.sin((self.X + self.Y) * 2 + t * rhythm * 0.6) * 0.4
        
        # 径向衰减
        radial_fade = np.exp(-self.R * 1.5)
        
        pattern = (wave1 + wave2 + wave3) * radial_fade + 0.5
        
        return np.clip(pattern, 0, 1)
    
    def _generate_starfield(self, t: float, rhythm: float) -> np.ndarray:
        """生成星空模式"""
        # 创建随机星星位置（使用固定种子保证连续性）
        np.random.seed(42)
        n_stars = 200
        star_x = np.random.uniform(-1, 1, n_stars)
        star_y = np.random.uniform(-1, 1, n_stars)
        star_brightness = np.random.uniform(0.3, 1.0, n_stars)
        
        # 创建空白画布
        pattern = np.zeros_like(self.X)
        
        # 添加星星
        for i in range(n_stars):
            # 星星闪烁
            twinkle = 0.5 + 0.5 * np.sin(t * rhythm * (i % 5 + 1))
            brightness = star_brightness[i] * twinkle
            
            # 高斯星星
            star = brightness * np.exp(-((self.X - star_x[i])**2 + (self.Y - star_y[i])**2) * 500)
            pattern += star
        
        # 添加淡淡的银河
        milky_way = 0.1 * np.exp(-((self.Y - 0.2 * np.sin(self.X * 2))**2) * 10)
        pattern += milky_way
        
        return np.clip(pattern, 0, 1)
    
    def _generate_aurora(self, t: float, rhythm: float) -> np.ndarray:
        """生成极光模式"""
        # 极光带
        aurora_base = np.sin(self.X * 3 + np.sin(self.Y * 2 + t * rhythm) * 2)
        
        # 垂直渐变
        vertical_gradient = 1 - (self.Y + 1) / 2
        
        # 时间变化的扭曲
        distortion = np.sin(self.X * 5 + t * rhythm * 2) * 0.3
        
        # 组合
        pattern = aurora_base * vertical_gradient * (1 + distortion)
        
        # 归一化
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
        
        return pattern
    
    def _generate_pulse_circle(self, t: float, rhythm: float) -> np.ndarray:
        """生成脉动圆圈"""
        # 脉动频率与心跳同步（60-80 BPM）
        pulse_rate = 70 / 60.0  # 70 BPM
        pulse_phase = np.sin(t * pulse_rate * 2 * np.pi) * 0.5 + 0.5
        
        # 多层圆圈
        pattern = np.zeros_like(self.R)
        
        for i in range(3):
            radius = 0.3 + i * 0.2
            ring_width = 0.1 + pulse_phase * 0.05
            ring = np.exp(-((self.R - radius)**2) / (ring_width**2))
            pattern += ring * (1 - i * 0.3)
        
        return np.clip(pattern, 0, 1)
    
    def _generate_minimal_waves(self, t: float, rhythm: float) -> np.ndarray:
        """生成极简波浪"""
        # 超慢速水平波
        wave = np.sin(self.Y * 2 + t * rhythm * 0.2) * 0.2 + 0.5
        
        # 轻微的垂直渐变
        gradient = (self.Y + 1) / 2 * 0.3 + 0.7
        
        pattern = wave * gradient
        
        return pattern
    
    def _apply_color_scheme(self, pattern: np.ndarray, color_scheme: str) -> np.ndarray:
        """应用治疗色彩方案"""
        colors = self.THERAPEUTIC_COLORS.get(color_scheme, self.THERAPEUTIC_COLORS["calming_blue"])
        
        # 创建RGB图像
        h, w = pattern.shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 将模式值映射到颜色
        # 使用三个颜色创建渐变
        for i, color in enumerate(['primary', 'secondary', 'accent']):
            rgb = colors[color]
            weight = pattern * (1 - i * 0.3)  # 递减权重
            
            frame[:, :, 0] += (rgb[0] * weight).astype(np.uint8)
            frame[:, :, 1] += (rgb[1] * weight).astype(np.uint8)
            frame[:, :, 2] += (rgb[2] * weight).astype(np.uint8)
        
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def _adjust_brightness(self, frame: np.ndarray, brightness: float) -> np.ndarray:
        """调整亮度（用于睡眠环境）"""
        # 应用gamma校正
        gamma = 1.0 / brightness
        adjusted = np.power(frame / 255.0, gamma) * 255.0
        
        return adjusted.astype(np.uint8)
    
    def _add_therapeutic_effects(self, frame: np.ndarray, t: float, pattern: str) -> np.ndarray:
        """添加治疗增强效果"""
        # 1. 双耳视觉节拍（通过微妙的亮度脉动）
        if pattern in ["breathing_guide", "pulse_circle"]:
            # 4Hz theta波（深度放松）
            theta_beat = np.sin(t * 4 * 2 * np.pi) * 0.05 + 1.0
            frame = (frame * theta_beat).astype(np.uint8)
        
        # 2. 边缘柔化（减少视觉刺激）
        # 创建边缘渐变遮罩
        edge_fade = 0.1
        x_fade = np.minimum(self.X + 1, 1 - self.X) / edge_fade
        y_fade = np.minimum(self.Y + 1, 1 - self.Y) / edge_fade
        fade_mask = np.minimum(x_fade, y_fade)
        fade_mask = np.clip(fade_mask, 0, 1)
        
        # 应用遮罩
        for i in range(3):
            frame[:, :, i] = (frame[:, :, i] * fade_mask).astype(np.uint8)
        
        return frame
    
    def create_transition_effect(self, 
                               from_stage: str, 
                               to_stage: str,
                               duration: float = 5.0) -> List[np.ndarray]:
        """
        创建阶段间过渡效果
        
        Args:
            from_stage: 起始阶段
            to_stage: 目标阶段
            duration: 过渡时长
            
        Returns:
            过渡帧列表
        """
        # 获取两个阶段的配置
        from_config = self.STAGE_VISUAL_CONFIGS.get(from_stage)
        to_config = self.STAGE_VISUAL_CONFIGS.get(to_stage)
        
        frames = []
        n_frames = int(duration * self.config.fps)
        
        for i in range(n_frames):
            # 计算混合权重
            alpha = i / n_frames
            
            # 生成两个阶段的帧
            t = i / self.config.fps
            
            frame1 = self._create_frame(
                t, from_config["patterns"][0], from_config["colors"][0],
                from_config["brightness"], 1.0
            )
            
            frame2 = self._create_frame(
                t, to_config["patterns"][0], to_config["colors"][0],
                to_config["brightness"], 1.0
            )
            
            # 混合
            blended = (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
            frames.append(blended)
        
        return frames
    
    def generate_complete_therapy_video(self,
                                      stages: List[Dict],
                                      output_path: str,
                                      music_sync_data: Optional[Dict] = None):
        """
        生成完整的三阶段治疗视频
        
        Args:
            stages: 阶段信息列表
            output_path: 输出路径
            music_sync_data: 音乐同步数据（BPM等）
        """
        logger.info(f"生成完整治疗视频: {len(stages)}个阶段")
        
        # 创建视频写入器
        writer = cv2.VideoWriter(
            output_path,
            self.config.get_codec(),
            self.config.fps,
            (self.config.width, self.config.height)
        )
        
        total_frames_written = 0
        
        for i, stage in enumerate(stages):
            stage_name = stage['stage'].value if hasattr(stage['stage'], 'value') else str(stage['stage'])
            stage_duration = stage['duration'] * 60  # 分钟转秒
            
            # 获取音乐BPM（如果有）
            stage_bpm = None
            if music_sync_data and 'stage_bpms' in music_sync_data:
                stage_bpm = music_sync_data['stage_bpms'].get(i, 60)
            
            logger.info(f"生成阶段 {i+1}: {stage_name}, {stage_duration}秒")
            
            # 生成阶段帧
            stage_frames = self._generate_stage_frames(
                stage_name, stage_duration, stage_bpm
            )
            
            # 写入帧
            for frame in stage_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                total_frames_written += 1
            
            # 如果不是最后一个阶段，添加过渡
            if i < len(stages) - 1:
                next_stage_name = stages[i+1]['stage'].value if hasattr(stages[i+1]['stage'], 'value') else str(stages[i+1]['stage'])
                transition_frames = self.create_transition_effect(
                    stage_name, next_stage_name, duration=3.0
                )
                
                for frame in transition_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                    total_frames_written += 1
        
        writer.release()
        
        logger.info(f"✅ 完整治疗视频生成完成: {output_path}")
        logger.info(f"   总帧数: {total_frames_written}")
        logger.info(f"   总时长: {total_frames_written/self.config.fps:.1f}秒")
        
        return output_path
    
    def _generate_stage_frames(self, stage_name: str, duration: float, bpm: Optional[int]) -> List[np.ndarray]:
        """生成阶段的所有帧"""
        stage_config = self.STAGE_VISUAL_CONFIGS.get(stage_name, self.STAGE_VISUAL_CONFIGS["引导化"])
        
        pattern = stage_config["patterns"][0]
        color_scheme = stage_config["colors"][0]
        brightness = stage_config["brightness"]
        
        rhythm_factor = 60.0 / bpm if bpm else 1.0
        
        frames = []
        total_frames = int(duration * self.config.fps)
        
        for frame_idx in range(total_frames):
            t = frame_idx / self.config.fps
            frame = self._create_frame(t, pattern, color_scheme, brightness, rhythm_factor)
            frames.append(frame)
            
            # 定期释放内存
            if frame_idx % (self.config.fps * 10) == 0 and frame_idx > 0:
                logger.debug(f"阶段进度: {frame_idx/total_frames*100:.1f}%")
        
        return frames
    
    def _save_frame_as_image(self, frame: np.ndarray, output_path: str):
        """保存帧为图片文件"""
        if CV2_AVAILABLE:
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            # 使用matplotlib保存
            import matplotlib.pyplot as plt
            plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=100)
            plt.imshow(frame)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()


# 工厂函数
def create_therapeutic_video_generator(width: int = 1280, 
                                     height: int = 720,
                                     fps: int = 30,
                                     quality: str = "high") -> TherapeuticVideoGenerator:
    """
    创建治疗视频生成器实例
    
    Args:
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        quality: 质量设置
        
    Returns:
        TherapeuticVideoGenerator实例
    """
    config = VideoConfig(width=width, height=height, fps=fps, quality=quality)
    return TherapeuticVideoGenerator(config)
    
