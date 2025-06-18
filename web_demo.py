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
    def __init__(self, use_enhanced_modules: bool = False):
        """
        初始化Web演示界面
        
        Args:
            use_enhanced_modules: 是否使用理论驱动的增强模块
        """
        self.app = MoodFlowApp(use_enhanced_modules=use_enhanced_modules)
        self.current_session = None
        # 设置matplotlib使用系统默认字体，避免字体警告
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 选择可用的字体，优先使用常见字体
        preferred_fonts = ['DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
        selected_font = 'sans-serif'
        
        for font in preferred_fonts:
            if font in available_fonts or font == 'sans-serif':
                selected_font = font
                break
        
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['axes.unicode_minus'] = False
    
    def safe_progress_update(self, progress, value, desc=""):
        """安全地更新进度条，避免Gradio版本兼容性问题"""
        try:
            if progress is not None:
                progress(value, desc=desc)
        except Exception as e:
            # 忽略进度条更新错误，不影响主要功能
            print(f"进度条更新警告: {str(e)}")
            pass
        
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
    
    def process_input(self, text_input, audio_input, emotion_type=None, demo_mode=True, playback_mode="🎵 仅音乐", progress=gr.Progress()):
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
            return None, None, None, None, None, None, None, gr.Row(visible=False), error_msg
        
        try:
            # 根据模式设置时长
            duration = 5 if demo_mode else 20
            self.safe_progress_update(progress, 0.1, "Starting emotion analysis...")
            
            # 根据播放模式决定是否生成完整视频
            create_videos = (playback_mode == "🎵+🎬 音画结合")
            
            # 运行治疗会话
            session = self.app.run_therapy_session(text_input, duration=duration, create_full_videos=create_videos, progress_callback=progress)
            self.current_session = session
            
            self.safe_progress_update(progress, 0.9, "Finalizing outputs...")
            
            # 读取生成的文件
            report_image = session.music_file.replace("_therapy_music.wav", "_report.png")
            
            # 检查报告文件是否存在
            if not os.path.exists(report_image):
                print(f"⚠️ 报告文件不存在: {report_image}")
                # 手动触发报告生成
                report_image = self.app.create_visualization(session)
            
            # 视觉引导预览 - 查找PNG预览图而不是MP4视频
            video_output = None
            print(f"🔍 调试信息 - 视频文件列表: {session.video_files}")
            if session.video_files and len(session.video_files) > 0:
                # 寻找PNG预览图，而不是MP4视频
                for video_file in session.video_files:
                    if video_file.endswith('.png'):  # 只处理PNG图片
                        print(f"🔍 调试信息 - 检查图片预览路径: {video_file}")
                        if os.path.exists(video_file):
                            video_output = video_file
                            print(f"✅ 图片预览文件存在: {video_file}")
                            break
                        else:
                            print(f"⚠️ 图片预览文件不存在: {video_file}")
                
                # 如果找不到PNG文件，尝试从MP4同目录找preview.png
                if not video_output and session.video_files:
                    first_video_dir = os.path.dirname(session.video_files[0])
                    preview_path = os.path.join(first_video_dir, "preview.png")
                    print(f"🔍 尝试查找预览图: {preview_path}")
                    if os.path.exists(preview_path):
                        video_output = preview_path
                        print(f"✅ 找到预览图: {preview_path}")
                    else:
                        print(f"⚠️ 预览图不存在: {preview_path}")
                        # 列出目录内容
                        try:
                            if os.path.exists(first_video_dir):
                                files = os.listdir(first_video_dir)
                                print(f"🔍 目录 {first_video_dir} 内容: {files}")
                        except Exception as e:
                            print(f"⚠️ 无法列出目录内容: {e}")
            else:
                print("⚠️ 没有生成视频文件")
            
            # 检查报告文件
            print(f"🔍 调试信息 - 报告文件路径: {report_image}")
            if os.path.exists(report_image):
                print(f"✅ 报告文件存在: {report_image}")
            else:
                print(f"⚠️ 报告文件不存在: {report_image}")
            
            # 根据播放模式处理输出
            if playback_mode == "🎵 仅音乐":
                combined_output = session.music_file
            else:  # 🎵+🎬 音画结合
                # 音画结合模式，如果有合并视频则使用，否则使用音频
                if hasattr(session, 'combined_video') and session.combined_video:
                    combined_output = session.combined_video
                else:
                    combined_output = session.music_file
            
            # 生成状态信息
            mode_text = "Demo (5 min)" if demo_mode else "Full (20 min)"
            playback_text = "Audio Only" if playback_mode == "🎵 仅音乐" else "Audio + Video"
            status_parts = [f"✅ Therapy plan generated! ({mode_text}, {playback_text})"]
            if voice_status:
                status_parts.append(f"\n🎤 {voice_status}")
            
            # 检查是否有音画结合文件
            has_combined_video = hasattr(session, 'combined_video') and session.combined_video
            output_file_info = ""
            if has_combined_video:
                output_file_info = f"• 🎬+🎵 Combined Video: {Path(session.combined_video).name}\n• 🎵 Audio Only: {Path(session.music_file).name}"
            else:
                output_file_info = f"• 🎵 Music: {Path(session.music_file).name}"
            
            # 检查是否有详细情绪信息（增强模块）
            detailed_emotion_info = ""
            if hasattr(self.app, 'get_detailed_emotion_info'):
                detailed_info = self.app.get_detailed_emotion_info(session.detected_emotion)
                if detailed_info:
                    detailed_emotion_info = f"""

🧠 Fine-grained Emotion Analysis:
• Primary: {detailed_info['primary_emotion_cn']} ({detailed_info['primary_emotion']})
• Confidence: {detailed_info['confidence']:.1%}
• Intensity: {detailed_info['intensity']:.1%}"""
            
            status = f"""
{status_parts[0]}{status_parts[1] if len(status_parts) > 1 else ''}

📊 Detected Emotion:
• Valence: {session.detected_emotion.valence:.2f}
• Arousal: {session.detected_emotion.arousal:.2f}{detailed_emotion_info}

🎵 Music Therapy:
• Total Duration: {sum(s['duration'] for s in session.iso_stages)} minutes
• 3 Stages: {' → '.join(s['stage'].value for s in session.iso_stages)}

💡 Usage Guide:
1. Find a quiet and comfortable environment
2. Dim the lights and relax your body
3. Put on headphones to listen to the music
4. Follow the visual guidance to adjust breathing

📁 Generated Files:
{output_file_info}
• 🎬 Video Previews: {len(session.video_files)} stage previews
"""
            
            # 根据播放模式准备输出
            if playback_mode == "🎵 仅音乐":
                # 仅音乐模式
                audio_out = session.music_file
                audio_download_out = session.music_file
                video_player_out = None
                video_download_out = None
                video_row_visible = gr.Row(visible=False)
            else:  # 🎵+🎬 音画结合
                # 音画结合模式
                if hasattr(session, 'combined_video') and session.combined_video:
                    audio_out = None  # 不显示音频播放器
                    audio_download_out = session.music_file  # 提供音频下载
                    video_player_out = session.combined_video  # 显示视频播放器
                    video_download_out = session.combined_video
                    video_row_visible = gr.Row(visible=True)
                else:
                    # 如果没有生成合并视频，回退到音频模式
                    audio_out = session.music_file
                    audio_download_out = session.music_file
                    video_player_out = None
                    video_download_out = None
                    video_row_visible = gr.Row(visible=False)
            
            self.safe_progress_update(progress, 1.0, "Complete!")
            return audio_out, audio_download_out, video_output, video_player_out, video_download_out, report_image, self.create_simple_visualization(), video_row_visible, status
            
        except Exception as e:
            import traceback
            # 打印详细错误信息到后端终端
            print(f"\n{'='*60}")
            print("🚨 Web界面处理出错:")
            print(f"{'='*60}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"输入参数:")
            print(f"  - text_input: {text_input}")
            print(f"  - audio_input: {audio_input}")
            print(f"  - emotion_type: {emotion_type}")
            print(f"  - demo_mode: {demo_mode}")
            print(f"  - playback_mode: {playback_mode}")
            print(f"\n完整错误堆栈:")
            traceback.print_exc()
            print(f"{'='*60}")
            
            # 生成前端错误信息
            error_msg = f"❌ 处理出错: {str(e)}\n\n错误类型: {type(e).__name__}"
            if voice_status:
                error_msg = f"{voice_status}\n\n{error_msg}"
            
            return None, None, None, None, None, None, None, gr.Row(visible=False), error_msg
    
    def create_simple_visualization(self):
        """创建简单的可视化"""
        try:
            if not self.current_session:
                print("⚠️ 警告: current_session为空，无法创建可视化")
                return None
            
            if not self.current_session.iso_stages:
                print("⚠️ 警告: iso_stages为空，无法创建可视化")
                return None
            
            fig, ax = plt.subplots(figsize=(8, 6))
        
            # 安全地构建数据
            stages = ["Start"]
            valences = [self.current_session.detected_emotion.valence]
            arousals = [self.current_session.detected_emotion.arousal]
            times = [0]
            
            current_time = 0
            for stage in self.current_session.iso_stages:
                # 转换阶段名为英文
                stage_name = stage['stage'].value
                if '同步化' in stage_name:
                    stage_en = 'Sync'
                elif '引导化' in stage_name:
                    stage_en = 'Guide'
                elif '巩固化' in stage_name:
                    stage_en = 'Consolidate'
                else:
                    stage_en = stage_name
                
                stages.append(stage_en)
                valences.append(stage['emotion'].valence)
                arousals.append(stage['emotion'].arousal)
                current_time += stage['duration']
                times.append(current_time)
            
            # 验证数据一致性
            if not (len(times) == len(valences) == len(arousals) == len(stages)):
                print(f"⚠️ 数据长度不一致: times={len(times)}, valences={len(valences)}, arousals={len(arousals)}, stages={len(stages)}")
                return None
            
            # 绘制曲线
            ax.plot(times, valences, 'b-o', linewidth=2, markersize=8, label='Valence')
            ax.plot(times, arousals, 'r-o', linewidth=2, markersize=8, label='Arousal')
            
            # 添加阶段标注
            for i in range(1, len(times)):
                if i < len(stages):
                    ax.axvline(x=times[i], color='gray', linestyle='--', alpha=0.3)
                    ax.text(times[i]-0.5, 0.9, stages[i], rotation=45, fontsize=10)
            
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Emotion Value')
            ax.set_title('ISO 3-Stage Emotion Guidance Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            plt.tight_layout()
            
            # 保存图片
            viz_path = Path("outputs/demo_sessions") / "current_visualization.png"
            # 确保目录存在
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path)
            plt.close()
            
            # 验证文件是否成功保存
            if viz_path.exists():
                print(f"✅ 可视化图表保存成功: {viz_path}")
                return str(viz_path)
            else:
                print(f"❌ 可视化图表保存失败: {viz_path}")
                return None
        
        except Exception as e:
            import traceback
            print(f"⚠️ 可视化创建出错: {str(e)}")
            traceback.print_exc()
            return None

def create_interface(use_enhanced_modules: bool = False):
    """
    创建Gradio界面
    
    Args:
        use_enhanced_modules: 是否使用理论驱动的增强模块
    """
    demo = WebDemo(use_enhanced_modules=use_enhanced_modules)
    
    with gr.Blocks(title="心境流转 - 睡眠治疗系统", theme=gr.themes.Soft()) as interface:
        # 显示当前运行模式
        if demo.app.use_enhanced and hasattr(demo.app, 'enhancement_adapter'):
            enhancement_status = demo.app.get_enhancement_status() if hasattr(demo.app, 'get_enhancement_status') else {}
            status_text = "✅ **增强模式** (理论驱动优化)"
            
            # 显示各模块状态
            module_status = []
            if enhancement_status.get('emotion_recognition', False):
                module_status.append("🧠 细粒度情绪识别")
            if enhancement_status.get('therapy_planning', False):
                module_status.append("📋 ISO治疗规划")
            if enhancement_status.get('music_mapping', False):
                module_status.append("🎵 精准音乐映射")
            
            if module_status:
                status_text += f"\n\n已启用模块：{' | '.join(module_status)}"
        else:
            status_text = "🔧 **基础模式**"
        
        gr.Markdown(f"""
        # 🌙 《心境流转》AI睡眠治疗系统
        
        {status_text}
        
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
                            label="🎵 治疗音乐",
                            type="filepath",
                            autoplay=False
                        )
                        audio_download = gr.File(
                            label="📥 音频下载",
                            type="filepath",
                            visible=False
                        )
                    
                    with gr.Column():
                        video_output = gr.Image(
                            label="🎬 视觉引导预览",
                            type="filepath"
                        )
                
                # 音画结合视频播放器（条件显示）
                with gr.Row(visible=False) as video_player_row:
                    with gr.Column():
                        combined_video_player = gr.Video(
                            label="🎬+🎵 音画结合治疗视频",
                            autoplay=False
                        )
                        combined_video_download = gr.File(
                            label="📥 完整视频下载",
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
            outputs=[audio_output, audio_download, video_output, combined_video_player, combined_video_download, report_output, viz_output, video_player_row, status_output]
        )
        
    return interface

def find_free_port(start_port=7860, max_port=7900):
    """查找可用端口"""
    import socket
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='心境流转睡眠治疗系统Web界面')
    parser.add_argument('--enhanced', action='store_true', 
                       help='启用理论驱动的增强模块（细粒度情绪识别、精准音乐映射等）')
    parser.add_argument('--port', type=int, default=None,
                       help='指定端口号（默认自动查找7860-7900）')
    parser.add_argument('--share', action='store_true', default=True,
                       help='创建公共分享链接（默认开启）')
    parser.add_argument('--no-browser', action='store_true',
                       help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    print("启动Web演示界面...")
    if args.enhanced:
        print("📚 使用理论驱动的增强模块")
        print("  - 细粒度情绪识别（9种情绪分类）")
        print("  - ISO原则治疗路径规划")
        print("  - 精准音乐特征映射")
    
    # 创建输出目录
    Path("outputs/demo_sessions").mkdir(parents=True, exist_ok=True)
    
    # 查找可用端口
    if args.port:
        port = args.port
        print(f"🚀 使用指定端口: {port}")
    else:
        port = find_free_port()
        if port is None:
            print("❌ 无法找到可用端口 (7860-7900)")
            print("请手动终止占用端口的进程或指定其他端口")
            return
        print(f"🚀 使用端口: {port}")
    
    # 创建并启动界面
    interface = create_interface(use_enhanced_modules=args.enhanced)
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=port,
        share=args.share,  # 创建公共链接
        inbrowser=not args.no_browser,  # 自动打开浏览器
        show_error=True
    )

if __name__ == "__main__":
    main()