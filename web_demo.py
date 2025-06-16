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

# 导入核心功能
from mood_flow_app import MoodFlowApp, TherapySession

class WebDemo:
    def __init__(self):
        self.app = MoodFlowApp()
        self.current_session = None
        
    def process_input(self, text_input, emotion_type=None):
        """处理用户输入并生成治疗方案"""
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
            return None, None, None, "请输入您的感受或选择一种情绪状态"
        
        try:
            # 运行治疗会话
            session = self.app.run_therapy_session(text_input)
            self.current_session = session
            
            # 读取生成的文件
            report_image = session.music_file.replace("_therapy_music.wav", "_report.png")
            
            # 生成状态信息
            status = f"""
✅ 治疗方案生成完成！

📊 检测到的情绪:
• 效价 (Valence): {session.detected_emotion.valence:.2f}
• 唤醒 (Arousal): {session.detected_emotion.arousal:.2f}

🎵 音乐治疗:
• 总时长: {sum(s['duration'] for s in session.iso_stages)} 分钟
• 三阶段: {' → '.join(s['stage'].value for s in session.iso_stages)}

💡 使用建议:
1. 找一个安静舒适的环境
2. 调暗灯光，放松身体
3. 戴上耳机聆听音乐
4. 跟随视觉引导调整呼吸
"""
            
            return session.music_file, report_image, self.create_simple_visualization(), status
            
        except Exception as e:
            return None, None, None, f"❌ 处理出错: {str(e)}"
    
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
        ax.plot(times, valences, 'b-o', linewidth=2, markersize=8, label='效价 (Valence)')
        ax.plot(times, arousals, 'r-o', linewidth=2, markersize=8, label='唤醒 (Arousal)')
        
        # 添加阶段标注
        for i, (t, stage) in enumerate(zip(times[1:], stages[1:])):
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            ax.text(t-1, 0.9, stage, rotation=45, fontsize=10)
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('情绪值')
        ax.set_title('ISO三阶段情绪引导轨迹')
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
                gr.Markdown("### 📝 描述您的感受")
                
                text_input = gr.Textbox(
                    label="文字输入",
                    placeholder="例如：今天工作压力很大，躺在床上翻来覆去睡不着...",
                    lines=3
                )
                
                gr.Markdown("或选择情绪类型:")
                emotion_buttons = gr.Radio(
                    choices=["焦虑", "压力", "失眠", "抑郁", "疲惫"],
                    label="快速选择",
                    value=None
                )
                
                submit_btn = gr.Button("🚀 生成治疗方案", variant="primary")
                
                gr.Markdown("""
                ### 💡 使用说明
                1. 描述您当前的情绪感受
                2. 点击生成治疗方案
                3. 聆听生成的音乐
                4. 查看情绪引导轨迹
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 治疗方案")
                
                status_output = gr.Textbox(
                    label="状态信息",
                    lines=10,
                    interactive=False
                )
                
                audio_output = gr.Audio(
                    label="🎵 治疗音乐",
                    type="filepath"
                )
                
                with gr.Row():
                    report_output = gr.Image(
                        label="📊 详细报告",
                        type="filepath"
                    )
                    
                    viz_output = gr.Image(
                        label="📈 情绪轨迹",
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
            inputs=[text_input, emotion_buttons],
            outputs=[audio_output, report_output, viz_output, status_output]
        )
        
        emotion_buttons.change(
            fn=lambda x: x,
            inputs=[emotion_buttons],
            outputs=[]
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