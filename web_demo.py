#!/usr/bin/env python3
"""
《心境流转》Web演示界面
使用Gradio创建简单的网页界面
"""

import gradio as gr
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import time
import os

# 导入核心功能
try:
    from mood_flow_app import MoodFlowApp, TherapySession
except ImportError as e:
    print(f"导入错误: {e}")
    print("正在尝试动态导入...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mood_flow_app", "mood_flow_app.py")
    mood_flow_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mood_flow_module)
    MoodFlowApp = mood_flow_module.MoodFlowApp
    TherapySession = mood_flow_module.TherapySession

class WebDemo:
    def __init__(self):
        self.app = MoodFlowApp()
        self.current_session = None
        # 设置matplotlib字体为英文，避免中文显示问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
    def process_voice_input(self, audio_file):
        """处理语音输入转换为文字"""
        if audio_file is None:
            return "", "请录制或上传音频文件"
        
        try:
            # 检查音频文件大小和格式
            file_path = Path(audio_file)
            if not file_path.exists() or file_path.stat().st_size == 0:
                return "", "❌ 音频文件无效或为空"
            
            # 简化版语音处理：目前使用示例文本
            # 在实际部署时可以集成专业的语音识别服务
            sample_texts = [
                "今天工作压力很大，躺在床上翻来覆去睡不着，总是想着明天的会议",
                "最近总是感到焦虑，晚上很难入睡，即使睡着了也容易醒",
                "心情有些低落，感觉很疲惫但就是睡不着，对什么都提不起兴趣",
                "有点兴奋睡不着，脑子里想着很多事情，越想越清醒",
                "身心俱疲，但躺下后大脑还是很活跃，总是胡思乱想"
            ]
            
            import random
            import hashlib
            
            # 基于文件内容生成一致的示例文本
            with open(audio_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # 使用文件哈希选择示例文本，确保同一文件返回相同结果
            text_index = int(file_hash[:8], 16) % len(sample_texts)
            selected_text = sample_texts[text_index]
            
            return selected_text, f"🎤 语音已处理 (演示模式): {selected_text}"
                
        except Exception as e:
            return "", f"❌ 语音处理出错: {str(e)}"
    
    def process_input(self, text_input, audio_input, emotion_type=None, demo_mode=True, playback_mode="audio_only", progress=gr.Progress()):
        """处理用户输入并生成治疗方案"""
        # 处理语音输入
        voice_text = ""
        voice_status = ""
        if audio_input is not None:
            voice_text, voice_status = self.process_voice_input(audio_input)
            if voice_text:
                text_input = voice_text if not text_input else f"{text_input} {voice_text}"
        
        if not text_input and emotion_type:
            # 如果没有文字输入，使用预设情绪
            emotion_templates = {
                "焦虑": "我感到很焦虑，心里总是不安，担心很多事情",
                "压力": "工作压力太大了，感觉喘不过气来",
                "失眠": "躺在床上翻来覆去就是睡不着",
                "抑郁": "心情很低落，什么都不想做",
                "疲惫": "身心俱疲，但就是无法入睡"
            }
            text_input = emotion_templates.get(emotion_type, "我睡不着")
        
        if not text_input:
            error_msg = "请通过以下任一方式提供输入:\n• 在文字框中描述您的感受\n• 录制语音描述\n• 选择预设情绪类型"
            if voice_status:
                error_msg = f"{voice_status}\n\n{error_msg}"
            return None, None, None, None, error_msg
        
        try:
            # 根据模式设置时长
            duration = 5 if demo_mode else 20
            progress(0.1, desc="Starting emotion analysis...")
            
            # 根据播放模式决定是否生成完整视频
            create_videos = (playback_mode == "🎵+🎬 音画结合")
            
            # 运行治疗会话
            session = self.app.run_therapy_session(text_input, duration=duration, create_full_videos=create_videos, progress_callback=progress)
            self.current_session = session
            
            progress(0.9, desc="Finalizing outputs...")
            
            # 读取生成的文件
            report_image = session.music_file.replace("_therapy_music.wav", "_report.png")
            
            # 根据播放模式处理输出
            if playback_mode == "🎵 仅音乐":
                video_output = None
                combined_output = session.music_file
            else:  # 🎵+🎬 音画结合
                # 创建音视频结合版本 (当前显示预览图)
                combined_output = session.music_file
                video_output = session.video_files[0] if session.video_files else None
            
            # 生成状态信息
            mode_text = "Demo (5 min)" if demo_mode else "Full (20 min)"
            playback_text = "Audio Only" if playback_mode == "🎵 仅音乐" else "Audio + Video"
            status_parts = [f"✅ Therapy plan generated! ({mode_text}, {playback_text})"]
            if voice_status:
                status_parts.append(f"\n🎤 {voice_status}")
            
            status = f"""
{status_parts[0]}{status_parts[1] if len(status_parts) > 1 else ''}

📊 Detected Emotion:
• Valence: {session.detected_emotion.valence:.2f}
• Arousal: {session.detected_emotion.arousal:.2f}

🎵 Music Therapy:
• Total Duration: {sum(s['duration'] for s in session.iso_stages)} minutes
• 3 Stages: {' → '.join(s['stage'].value for s in session.iso_stages)}

💡 Usage Guide:
1. Find a quiet and comfortable environment
2. Dim the lights and relax your body
3. Put on headphones to listen to the music
4. Follow the visual guidance to adjust breathing

📁 Generated Files:
• 🎵 Music: {Path(session.music_file).name}
• 🎬 Videos: {len(session.video_files)} stage previews
"""
            
            progress(1.0, desc="Complete!")
            return combined_output, video_output, report_image, self.create_simple_visualization(), status
            
        except Exception as e:
            error_msg = f"❌ 处理出错: {str(e)}"
            if voice_status:
                error_msg = f"{voice_status}\n\n{error_msg}"
            return None, None, None, None, error_msg
    
    def create_simple_visualization(self):
        """创建简单的可视化"""
        if not self.current_session:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制情绪轨迹
        stages = ["开始"] + [s['stage'].value for s in self.current_session.iso_stages]
        valences = [self.current_session.detected_emotion.valence] + \
                   [s['emotion'].valence for s in self.current_session.iso_stages]
        arousals = [self.current_session.detected_emotion.arousal] + \
                   [s['emotion'].arousal for s in self.current_session.iso_stages]
        
        # 创建时间轴
        times = [0]
        current_time = 0
        for stage in self.current_session.iso_stages:
            current_time += stage['duration']
            times.append(current_time)
        
        # 绘制曲线
        ax.plot(times, valences, 'b-o', linewidth=2, markersize=8, label='Valence')
        ax.plot(times, arousals, 'r-o', linewidth=2, markersize=8, label='Arousal')
        
        # 添加阶段标注
        for i, (t, stage) in enumerate(zip(times[1:], stages[1:])):
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            # 将中文阶段名转换为英文
            stage_en = stage
            if '同步化' in stage:
                stage_en = 'Sync'
            elif '引导化' in stage:
                stage_en = 'Guide'
            elif '巩固化' in stage:
                stage_en = 'Consolidate'
            ax.text(t-1, 0.9, stage_en, rotation=45, fontsize=10)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Emotion Value')
        ax.set_title('ISO 3-Stage Emotion Guidance Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        
        # 保存图片
        viz_path = Path("outputs/demo_sessions") / "current_visualization.png"
        plt.savefig(viz_path)
        plt.close()
        
        return str(viz_path)

def create_interface():
    """创建Gradio界面"""
    demo = WebDemo()
    
    with gr.Blocks(title="心境流转 - 睡眠治疗系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🌙 《心境流转》AI睡眠治疗系统
        
        基于ISO三阶段原则和情绪识别技术，为您生成个性化的音视频睡眠治疗方案。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入您的感受")
                
                # 选项卡：文字或语音输入
                with gr.Tabs():
                    with gr.TabItem("💬 文字输入"):
                        text_input = gr.Textbox(
                            label="描述您的感受",
                            placeholder="例如：今天工作压力很大，躺在床上翻来覆去睡不着...",
                            lines=4
                        )
                    
                    with gr.TabItem("🎤 语音输入"):
                        audio_input = gr.Audio(
                            label="录制或上传语音",
                            type="filepath",
                            format="wav"
                        )
                        gr.Markdown("💡 **提示**: 录制后请点击'生成治疗方案'按钮")
                
                gr.Markdown("### 🎯 或快速选择情绪:")
                emotion_buttons = gr.Radio(
                    choices=["焦虑", "压力", "失眠", "抑郁", "疲惫"],
                    label="预设情绪",
                    value=None
                )
                
                gr.Markdown("### ⚙️ 系统设置:")
                with gr.Row():
                    demo_mode_toggle = gr.Checkbox(
                        label="演示模式 (5分钟快速体验)",
                        value=True
                    )
                    
                playback_mode = gr.Radio(
                    choices=["🎵 仅音乐", "🎵+🎬 音画结合"],
                    label="播放模式",
                    value="🎵 仅音乐"
                )
                
                submit_btn = gr.Button("🚀 生成治疗方案", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 使用指南
                1. **输入方式**：选择文字或语音描述
                2. **生成方案**：点击按钮开始处理
                3. **聆听音乐**：播放个性化治疗音乐
                4. **观看视频**：查看配套的视觉引导
                5. **查看报告**：了解情绪分析和治疗轨迹
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 个性化治疗方案")
                
                status_output = gr.Textbox(
                    label="📋 处理状态",
                    lines=12,
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column():
                        audio_output = gr.Audio(
                            label="🎵 治疗音乐 (20分钟三阶段)",
                            type="filepath",
                            autoplay=False
                        )
                    
                    with gr.Column():
                        video_output = gr.Image(
                            label="🎬 视觉引导预览",
                            type="filepath"
                        )
                
                with gr.Row():
                    report_output = gr.Image(
                        label="📊 详细分析报告",
                        type="filepath"
                    )
                    
                    viz_output = gr.Image(
                        label="📈 情绪轨迹图",
                        type="filepath"
                    )
        
        # 示例
        gr.Markdown("### 🌟 示例场景")
        gr.Examples(
            examples=[
                ["最近总是失眠，一躺下就开始胡思乱想，越想越清醒"],
                ["工作压力太大了，每天都很焦虑，晚上根本睡不好"],
                ["心情很低落，对什么都提不起兴趣，晚上也睡不着"],
                ["今天发生了一些烦心事，现在很生气，完全没有睡意"],
                ["白天太累了，但躺在床上反而睡不着，大脑还是很活跃"]
            ],
            inputs=text_input
        )
        
        # 绑定事件
        submit_btn.click(
            fn=demo.process_input,
            inputs=[text_input, audio_input, emotion_buttons, demo_mode_toggle, playback_mode],
            outputs=[audio_output, video_output, report_output, viz_output, status_output]
        )
        
    return interface

def main():
    """主函数"""
    print("启动Web演示界面...")
    
    # 创建输出目录
    Path("outputs/demo_sessions").mkdir(parents=True, exist_ok=True)
    
    # 创建并启动界面
    interface = create_interface()
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=True,  # 创建公共链接
        inbrowser=True  # 自动打开浏览器
    )

if __name__ == "__main__":
    main()