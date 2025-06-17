#!/usr/bin/env python3
"""
《心境流转》睡眠治疗系统 - 交互式演示应用
用户输入文字/语音 → 情绪识别 → 生成三阶段音视频 → 引导入睡
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 添加scripts目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# 导入核心模块
import importlib.util
import sys

def import_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 动态导入scripts模块
theory_module = import_from_file("scripts/02_theory_models_demo.py", "theory_models")
music_module = import_from_file("scripts/06_music_generation_workshop.py", "music_workshop")
video_module = import_from_file("scripts/07_video_generation_workshop.py", "video_workshop")

ISOModel = theory_module.ISOModel
EmotionState = theory_module.EmotionState
MusicModel = theory_module.MusicModel
SleepMusicGenerator = music_module.SleepMusicGenerator
SleepVideoGenerator = video_module.SleepVideoGenerator

@dataclass
class TherapySession:
    """治疗会话数据"""
    user_input: str
    detected_emotion: EmotionState
    iso_stages: List[Dict]
    music_file: str
    video_files: List[str]
    start_time: datetime
    
class MoodFlowApp:
    """心境流转应用主类"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🌙 《心境流转》睡眠治疗系统 启动中...")
        print("="*60)
        
        # 初始化核心组件
        self.iso_model = ISOModel()
        self.music_model = MusicModel()
        self.music_generator = SleepMusicGenerator(sample_rate=44100)
        self.video_generator = SleepVideoGenerator(width=960, height=540, fps=24)
        
        # 创建输出目录
        self.output_dir = Path("outputs/demo_sessions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 情绪关键词映射
        self.emotion_keywords = {
            "焦虑": ["焦虑", "紧张", "担心", "害怕", "不安", "恐慌", "忧虑"],
            "压力": ["压力", "累", "疲惫", "工作", "忙", "烦", "头疼"],
            "抑郁": ["难过", "悲伤", "失望", "沮丧", "低落", "哭", "绝望"],
            "愤怒": ["生气", "愤怒", "恼火", "讨厌", "烦躁", "气", "恨"],
            "兴奋": ["兴奋", "激动", "开心", "高兴", "刺激"],
            "平静": ["平静", "放松", "舒适", "安静", "宁静"]
        }
        
        print("✅ 系统初始化完成！\n")
    
    def safe_progress_update(self, progress_callback, value, desc=""):
        """Safely update progress bar to avoid Gradio version compatibility issues"""
        try:
            if progress_callback is not None:
                progress_callback(value, desc=desc)
        except Exception as e:
            print(f"Progress update warning: {str(e)}")
            pass
    
    def analyze_emotion_from_text(self, text: str) -> EmotionState:
        """从文本分析情绪状态"""
        print("🔍 分析情绪状态...")
        
        # 计算各情绪得分
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                emotion_scores[emotion] = score
        
        # 如果没有检测到关键词，默认为轻度焦虑
        if not emotion_scores:
            emotion_scores["焦虑"] = 1
        
        # 找出主要情绪
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # 映射到V-A空间
        va_mapping = {
            "焦虑": (-0.6, 0.8),    # 负面高唤醒
            "压力": (-0.5, 0.6),    # 负面中高唤醒
            "抑郁": (-0.8, -0.3),   # 负面低唤醒
            "愤怒": (-0.7, 0.9),    # 负面高唤醒
            "兴奋": (0.6, 0.8),     # 正面高唤醒
            "平静": (0.5, -0.5)     # 正面低唤醒
        }
        
        valence, arousal = va_mapping.get(primary_emotion, (-0.5, 0.5))
        
        # 显示检测结果
        print(f"  检测到的主要情绪: {primary_emotion}")
        print(f"  情绪参数: Valence={valence:.2f}, Arousal={arousal:.2f}")
        
        return EmotionState(valence=valence, arousal=arousal)
    
    def plan_therapy_stages(self, current_emotion: EmotionState, duration: int = 20) -> List[Dict]:
        """规划三阶段治疗"""
        print("\n📋 规划治疗方案...")
        
        # 目标情绪：平静入睡状态
        target_emotion = EmotionState(valence=0.3, arousal=-0.8)
        
        # 使用ISO模型规划
        stages = self.iso_model.plan_stages(current_emotion, target_emotion, duration)
        
        # 验证阶段规划结果
        if not stages:
            raise ValueError("ISO模型未能生成有效的治疗阶段")
        
        print(f"  治疗总时长: {duration} 分钟")
        print(f"  生成了 {len(stages)} 个治疗阶段")
        for i, stage in enumerate(stages, 1):
            print(f"  第{i}阶段 - {stage['stage'].value}: {stage['duration']:.0f}分钟")
        
        return stages
    
    def generate_stage_music(self, stages: List[Dict], session_name: str) -> str:
        """为各阶段生成音乐"""
        print("\n🎵 生成治疗音乐...")
        
        if not stages:
            raise ValueError("无法为空的治疗阶段生成音乐")
        
        # 音频参数
        total_duration = sum(stage['duration'] for stage in stages)
        sample_rate = self.music_generator.sample_rate
        total_samples = int(total_duration * 60 * sample_rate)
        
        # 创建完整音轨
        full_track = np.zeros(total_samples)
        current_pos = 0
        
        for i, stage in enumerate(stages):
            stage_duration = stage['duration'] * 60  # 转换为秒
            stage_samples = int(stage_duration * sample_rate)
            
            # 获取阶段情绪
            emotion = stage['emotion']
            
            # 计算音乐参数
            bpm = self.music_model.calc_bpm(emotion.arousal)
            
            # 选择调性
            if emotion.valence > 0:
                key = 'C'
                mode = 'major'
            else:
                key = 'A'
                mode = 'minor'
            
            print(f"  第{i+1}阶段: BPM={bpm:.0f}, 调性={key} {mode}")
            
            # 生成该阶段的音乐
            stage_track = self._generate_simple_music(
                duration_seconds=stage_duration,
                bpm=bpm,
                key=key,
                stage_index=i
            )
            
            # 添加到完整音轨
            end_pos = min(current_pos + len(stage_track), total_samples)
            full_track[current_pos:end_pos] = stage_track[:end_pos-current_pos]
            current_pos = end_pos
        
        # 保存音频
        audio_file = self.output_dir / f"{session_name}_therapy_music.wav"
        self.music_generator.save_audio(full_track, str(audio_file))
        
        print(f"✅ 音乐生成完成: {audio_file.name}")
        
        return str(audio_file)
    
    def _generate_simple_music(self, duration_seconds: float, bpm: float, 
                              key: str, stage_index: int) -> np.ndarray:
        """生成简单的阶段音乐"""
        sample_rate = self.music_generator.sample_rate
        samples = int(duration_seconds * sample_rate)
        track = np.zeros(samples)
        
        # 基础频率（根据调性）
        base_frequencies = {
            'C': 261.63,  # C4
            'A': 440.00   # A4
        }
        base_freq = base_frequencies.get(key, 261.63)
        
        # 根据阶段调整音色
        if stage_index == 0:  # 同步化
            # 使用当前情绪的节奏，较为活跃
            frequencies = [base_freq, base_freq * 1.5, base_freq * 2]
            amplitudes = [0.5, 0.3, 0.2]
        elif stage_index == 1:  # 引导化
            # 逐渐放缓，添加和谐音
            frequencies = [base_freq * 0.5, base_freq, base_freq * 1.5]
            amplitudes = [0.4, 0.4, 0.2]
        else:  # 巩固化
            # 低频为主，营造睡眠氛围
            frequencies = [base_freq * 0.25, base_freq * 0.5, base_freq]
            amplitudes = [0.5, 0.3, 0.2]
        
        # 生成和弦
        beat_duration = 60.0 / bpm
        beat_samples = int(beat_duration * sample_rate)
        
        for beat_idx in range(int(duration_seconds / beat_duration)):
            start_idx = beat_idx * beat_samples
            end_idx = min(start_idx + beat_samples, samples)
            
            # 生成音符
            t = np.linspace(0, beat_duration, end_idx - start_idx)
            note = np.zeros_like(t)
            
            for freq, amp in zip(frequencies, amplitudes):
                # 添加轻微的频率变化，使音色更自然
                freq_mod = freq * (1 + 0.01 * np.sin(2 * np.pi * 0.1 * beat_idx))
                note += amp * np.sin(2 * np.pi * freq_mod * t)
            
            # 应用包络
            envelope = np.exp(-t * 2) * (1 - stage_index * 0.2)  # 逐阶段减弱
            note *= envelope
            
            # 添加到音轨
            track[start_idx:end_idx] += note
        
        # 应用整体淡入淡出
        fade_samples = int(5 * sample_rate)  # 5秒淡入淡出
        if fade_samples < len(track):
            track[:fade_samples] *= np.linspace(0, 1, fade_samples)
            track[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return track
    
    def generate_stage_videos(self, stages: List[Dict], session_name: str, create_full_videos: bool = False) -> List[str]:
        """为各阶段生成视频"""
        print("\n🎬 生成治疗视频...")
        
        video_files = []
        
        # 视觉模式映射
        stage_patterns = {
            0: ("breathing", "ocean"),      # 同步化：呼吸引导
            1: ("gradient", "sunset"),      # 引导化：渐变过渡
            2: ("waves", "lavender")        # 巩固化：柔和波浪
        }
        
        for i, stage in enumerate(stages):
            pattern, palette = stage_patterns.get(i, ("gradient", "ocean"))
            
            print(f"  第{i+1}阶段: {pattern} - {palette}")
            
            stage_dir = self.output_dir / f"{session_name}_stage_{i+1}"
            stage_dir.mkdir(exist_ok=True)
            
            if create_full_videos:
                # 生成完整视频
                video_file = stage_dir / f"stage_{i+1}_video.mp4"
                frames = self.video_generator.generate_video(
                    duration_seconds=stage['duration'] * 60,
                    pattern_type=pattern,
                    color_palette=palette,
                    output_path=str(video_file),
                    preview_only=False
                )
                video_files.append(str(video_file))
                print(f"  ✅ 保存视频: {video_file.name}")
            else:
                # 生成预览帧
                frames = self.video_generator.generate_video(
                    duration_seconds=stage['duration'] * 60,
                    pattern_type=pattern,
                    color_palette=palette,
                    output_path=None,
                    preview_only=True
                )
                
                # 保存第一帧作为预览
                if frames:
                    preview_file = stage_dir / "preview.png"
                    # 确保目录存在
                    preview_file.parent.mkdir(parents=True, exist_ok=True)
                    # 保存图片
                    plt.imsave(str(preview_file), frames[0])
                    # 验证文件是否成功保存
                    if preview_file.exists():
                        video_files.append(str(preview_file))
                        print(f"  ✅ 保存预览: {preview_file.name} (路径: {preview_file})")
                    else:
                        print(f"  ❌ 预览保存失败: {preview_file}")
        
        return video_files
    
    def create_visualization(self, session: TherapySession) -> str:
        """创建会话可视化"""
        print("\n📊 生成可视化报告...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 情绪轨迹
        ax = axes[0, 0]
        stages_data = []
        for stage in session.iso_stages:
            emotion = stage['emotion']
            stages_data.append({
                'name': stage['stage'].value,
                'valence': emotion.valence,
                'arousal': emotion.arousal,
                'duration': stage['duration']
            })
        
        # 绘制V-A空间轨迹
        valences = [session.detected_emotion.valence] + [s['valence'] for s in stages_data]
        arousals = [session.detected_emotion.arousal] + [s['arousal'] for s in stages_data]
        
        ax.plot(valences, arousals, 'o-', linewidth=2, markersize=8)
        ax.scatter(valences[0], arousals[0], c='red', s=100, label='Initial Emotion')
        ax.scatter(valences[-1], arousals[-1], c='green', s=100, label='Target Emotion')
        
        # 添加阶段标注
        for i, stage in enumerate(stages_data):
            ax.annotate(stage['name'], (valences[i+1], arousals[i+1]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        ax.set_title('Emotion Trajectory Planning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 阶段时间分配
        ax = axes[0, 1]
        stage_names = [s['name'] for s in stages_data]
        stage_durations = [s['duration'] for s in stages_data]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        # 将stage_names转换为英文
        stage_names_en = []
        for name in stage_names:
            if '同步化' in name:
                stage_names_en.append('Synchronization')
            elif '引导化' in name:
                stage_names_en.append('Guidance')
            elif '巩固化' in name:
                stage_names_en.append('Consolidation')
            else:
                stage_names_en.append(name)
        
        ax.pie(stage_durations, labels=stage_names_en, colors=colors, autopct='%1.0f%%')
        ax.set_title('Therapy Stage Duration Distribution')
        
        # 3. BPM变化曲线
        ax = axes[1, 0]
        time_points = []
        bpm_values = []
        current_time = 0
        
        for stage in session.iso_stages:
            emotion = stage['emotion']
            bpm = self.music_model.calc_bpm(emotion.arousal)
            time_points.extend([current_time, current_time + stage['duration']])
            bpm_values.extend([bpm, bpm])
            current_time += stage['duration']
        
        ax.plot(time_points, bpm_values, 'b-', linewidth=2)
        ax.fill_between(time_points, bpm_values, alpha=0.3)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('BPM')
        ax.set_title('Music Rhythm Changes')
        ax.grid(True, alpha=0.3)
        
        # 4. 治疗信息
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = f"""
用户输入: {session.user_input[:30]}...

检测情绪: V={session.detected_emotion.valence:.2f}, A={session.detected_emotion.arousal:.2f}

治疗方案:
• 总时长: {sum(s['duration'] for s in stages_data):.0f} 分钟
• 音乐文件: {Path(session.music_file).name}
• 视频预览: {len(session.video_files)} 个阶段

生成时间: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax.text(0.1, 0.9, "Therapy Session Info", fontsize=14, fontweight='bold', 
                transform=ax.transAxes)
        ax.text(0.1, 0.1, info_text, fontsize=10, transform=ax.transAxes,
                verticalalignment='bottom', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图表
        report_file = self.output_dir / f"{Path(session.music_file).stem}_report.png"
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(report_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 验证文件是否成功保存
        if report_file.exists():
            print(f"✅ 报告生成完成: {report_file.name} (路径: {report_file})")
        else:
            print(f"❌ 报告生成失败: {report_file}")
        
        return str(report_file)
    
    def run_therapy_session(self, user_input: str, duration: int = 20, create_full_videos: bool = False, progress_callback=None) -> TherapySession:
        """运行完整的治疗会话"""
        start_time = datetime.now()
        session_name = f"session_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"🌙 开始治疗会话: {session_name}")
        print(f"{'='*60}")
        print(f"参数: duration={duration}, create_full_videos={create_full_videos}")
        
        # 1. 情绪分析
        self.safe_progress_update(progress_callback, 0.2, "Analyzing emotions...")
        detected_emotion = self.analyze_emotion_from_text(user_input)
        
        # 2. 规划治疗阶段
        self.safe_progress_update(progress_callback, 0.3, "Planning therapy stages...")
        iso_stages = self.plan_therapy_stages(detected_emotion, duration)
        
        # 3. 生成音乐
        self.safe_progress_update(progress_callback, 0.4, "Generating therapy music...")
        music_file = self.generate_stage_music(iso_stages, session_name)
        
        # 4. 生成视频
        self.safe_progress_update(progress_callback, 0.7, "Creating visual guidance...")
        video_files = self.generate_stage_videos(iso_stages, session_name, create_full_videos)
        
        # 创建会话对象
        session = TherapySession(
            user_input=user_input,
            detected_emotion=detected_emotion,
            iso_stages=iso_stages,
            music_file=music_file,
            video_files=video_files,
            start_time=start_time
        )
        
        # 5. 生成可视化报告
        report_file = self.create_visualization(session)
        
        # 6. 保存会话数据
        session_data = {
            "session_name": session_name,
            "user_input": user_input,
            "detected_emotion": {
                "valence": detected_emotion.valence,
                "arousal": detected_emotion.arousal
            },
            "stages": [
                {
                    "name": stage["stage"].value,
                    "duration": stage["duration"],
                    "emotion": {
                        "valence": stage["emotion"].valence,
                        "arousal": stage["emotion"].arousal
                    }
                }
                for stage in iso_stages
            ],
            "outputs": {
                "music": music_file,
                "videos": video_files,
                "report": report_file
            },
            "timestamp": start_time.isoformat()
        }
        
        session_file = self.output_dir / f"{session_name}_data.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("✅ 治疗会话完成！")
        print(f"{'='*60}")
        print(f"\n📁 输出文件:")
        print(f"  • 音乐: {Path(music_file).name}")
        print(f"  • 视频: {len(video_files)} 个阶段预览")
        print(f"  • 报告: {Path(report_file).name}")
        print(f"  • 数据: {session_file.name}")
        
        return session

def main():
    """主程序"""
    # 创建应用实例
    app = MoodFlowApp()
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("欢迎使用《心境流转》睡眠治疗系统")
    print("="*60)
    print("\n请描述您现在的感受，系统将为您生成个性化的睡眠治疗方案。")
    print("(输入 'quit' 退出)\n")
    
    # 提供示例
    examples = [
        "今天工作压力很大，躺在床上翻来覆去睡不着，总是想着明天的会议",
        "最近总是感到焦虑，晚上很难入睡，即使睡着了也容易醒",
        "心情有些低落，感觉很疲惫但就是睡不着",
        "有点兴奋睡不着，脑子里想着很多事情"
    ]
    
    print("示例输入:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    while True:
        print("\n" + "-"*60)
        user_input = input("请输入您的感受 (或输入示例编号 1-4): ").strip()
        
        # 处理退出
        if user_input.lower() == 'quit':
            print("\n感谢使用，祝您好梦！晚安~ 🌙")
            break
        
        # 处理示例选择
        if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            user_input = examples[int(user_input) - 1]
            print(f"\n您选择了: {user_input}")
        
        # 检查输入
        if len(user_input) < 5:
            print("❌ 请输入更详细的描述（至少5个字）")
            continue
        
        try:
            # 运行治疗会话
            session = app.run_therapy_session(user_input)
            
            # 询问是否继续
            print("\n是否需要生成新的治疗方案？(y/n): ", end='')
            if input().strip().lower() != 'y':
                print("\n感谢使用，祝您好梦！晚安~ 🌙")
                break
                
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()