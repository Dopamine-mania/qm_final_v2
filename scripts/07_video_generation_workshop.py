#!/usr/bin/env python3
"""
07 - 视频生成工作坊
展示AI视觉内容生成在睡眠治疗中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 尝试导入cv2，如果失败则使用标志
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV (cv2) 不可用，某些视频生成功能将受限")

class VisualPatternGenerator:
    """视觉模式生成器"""
    
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # 创建坐标网格
        self.x = np.linspace(-1, 1, width)
        self.y = np.linspace(-1, 1, height)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def generate_gradient_flow(self, t, color_palette):
        """生成渐变流动效果"""
        # 基础渐变
        base = np.sin(self.X * 2 + t) * np.cos(self.Y * 2 + t * 0.7)
        
        # 添加流动效果
        flow = np.sin(np.sqrt(self.X**2 + self.Y**2) * 3 - t * 0.5)
        
        # 组合
        pattern = (base + flow) / 2
        
        # 归一化到0-1
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        return pattern
    
    def generate_breathing_circle(self, t, base_radius=0.3):
        """生成呼吸圆圈效果"""
        # 呼吸节奏（4-7-8呼吸法）
        breath_cycle = 19  # 4+7+8
        phase = (t % breath_cycle) / breath_cycle
        
        if phase < 4/19:  # 吸气
            scale = 1 + 0.3 * (phase * 19 / 4)
        elif phase < 11/19:  # 屏息
            scale = 1.3
        else:  # 呼气
            scale = 1.3 - 0.3 * ((phase - 11/19) * 19 / 8)
        
        # 创建圆形mask
        radius = base_radius * scale
        mask = np.sqrt(self.X**2 + self.Y**2) < radius
        
        # 添加柔和边缘
        distance = np.sqrt(self.X**2 + self.Y**2)
        soft_mask = np.exp(-10 * np.maximum(0, distance - radius)**2)
        
        return soft_mask
    
    def generate_wave_pattern(self, t, frequency=2, amplitude=0.3):
        """生成波浪图案"""
        # 多层波浪叠加
        wave1 = amplitude * np.sin(self.X * frequency + t)
        wave2 = amplitude * 0.5 * np.sin(self.Y * frequency * 1.5 + t * 1.2)
        wave3 = amplitude * 0.3 * np.sin((self.X + self.Y) * frequency * 0.7 + t * 0.8)
        
        # 组合波浪
        waves = wave1 + wave2 + wave3
        
        # 创建渐变效果
        gradient = 1 - np.sqrt(self.X**2 + self.Y**2) * 0.5
        
        pattern = waves * gradient
        
        # 归一化
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        return pattern
    
    def generate_mandala_pattern(self, t, n_fold=8):
        """生成曼陀罗图案"""
        # 极坐标
        r = np.sqrt(self.X**2 + self.Y**2)
        theta = np.arctan2(self.Y, self.X)
        
        # 旋转动画
        theta_rotated = theta + t * 0.1
        
        # 创建对称图案
        pattern = 0
        for i in range(n_fold):
            angle = theta_rotated + i * 2 * np.pi / n_fold
            petal = np.exp(-r * 2) * np.cos(angle * 3) * np.sin(r * 10 - t)
            pattern += petal
        
        # 添加中心装饰
        center = np.exp(-r * 10) * np.cos(t * 2)
        pattern += center * 2
        
        # 归一化
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        return pattern
    
    def apply_color_palette(self, pattern, palette_name="ocean"):
        """应用颜色调色板"""
        palettes = {
            "ocean": [
                [0.05, 0.1, 0.3],    # 深蓝
                [0.1, 0.3, 0.5],     # 海蓝
                [0.2, 0.5, 0.7],     # 浅蓝
                [0.3, 0.7, 0.9]      # 天蓝
            ],
            "sunset": [
                [0.4, 0.1, 0.1],     # 深红
                [0.6, 0.3, 0.1],     # 橙红
                [0.8, 0.5, 0.2],     # 橙色
                [1.0, 0.7, 0.4]      # 浅橙
            ],
            "forest": [
                [0.1, 0.2, 0.1],     # 深绿
                [0.2, 0.4, 0.2],     # 森林绿
                [0.3, 0.6, 0.3],     # 草绿
                [0.4, 0.8, 0.4]      # 浅绿
            ],
            "lavender": [
                [0.3, 0.2, 0.4],     # 深紫
                [0.5, 0.4, 0.6],     # 紫色
                [0.7, 0.6, 0.8],     # 淡紫
                [0.9, 0.8, 1.0]      # 浅紫
            ]
        }
        
        palette = palettes.get(palette_name, palettes["ocean"])
        
        # 创建RGB图像
        height, width = pattern.shape
        image = np.zeros((height, width, 3))
        
        # 应用颜色映射
        for i in range(3):  # RGB通道
            colors = [p[i] for p in palette]
            # 线性插值
            image[:, :, i] = np.interp(pattern, 
                                      np.linspace(0, 1, len(colors)), 
                                      colors)
        
        return image

class SleepVideoGenerator:
    """睡眠视频生成器"""
    
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pattern_gen = VisualPatternGenerator(width, height, fps)
    
    def create_frame(self, t, pattern_type, color_palette):
        """创建单帧"""
        if pattern_type == "gradient":
            pattern = self.pattern_gen.generate_gradient_flow(t, color_palette)
        elif pattern_type == "breathing":
            pattern = self.pattern_gen.generate_breathing_circle(t)
        elif pattern_type == "waves":
            pattern = self.pattern_gen.generate_wave_pattern(t)
        elif pattern_type == "mandala":
            pattern = self.pattern_gen.generate_mandala_pattern(t)
        else:
            pattern = np.random.rand(self.height, self.width)
        
        # 应用颜色
        frame = self.pattern_gen.apply_color_palette(pattern, color_palette)
        
        # 应用亮度调整（睡眠场景需要低亮度）
        # 为预览图片使用稍高的亮度，实际使用时应该更暗
        frame = frame * 0.6  # 适中亮度便于预览
        
        # 转换为uint8
        frame = (frame * 255).astype(np.uint8)
        
        return frame
    
    def generate_video(self, duration_seconds, pattern_type, color_palette, 
                      output_path, preview_only=False):
        """生成视频"""
        if preview_only:
            # 只生成预览帧
            frames = []
            preview_times = [0, duration_seconds/4, duration_seconds/2, 
                           3*duration_seconds/4, duration_seconds-0.1]
            
            for t in preview_times:
                frame = self.create_frame(t, pattern_type, color_palette)
                frames.append(frame)
            
            return frames
        
        # 完整视频生成
        if not CV2_AVAILABLE:
            print("⚠️ OpenCV不可用，生成预览帧代替完整视频")
            frames = []
            preview_times = [0, duration_seconds/2, duration_seconds-0.1]
            for t in preview_times:
                frame = self.create_frame(t, pattern_type, color_palette)
                frames.append(frame)
            # 保存第一帧作为预览
            preview_path = str(output_path).replace('.mp4', '_preview.png')
            plt.imsave(preview_path, frames[1])
            print(f"✅ 预览图已保存: {preview_path}")
            return frames
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, 
                            (self.width, self.height))
        
        total_frames = int(duration_seconds * self.fps)
        
        print(f"生成{total_frames}帧...")
        for i in range(total_frames):
            t = i / self.fps
            frame = self.create_frame(t, pattern_type, color_palette)
            
            # OpenCV使用BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if i % (self.fps * 5) == 0:  # 每5秒报告一次
                print(f"  进度: {i/total_frames*100:.1f}%")
        
        out.release()
        print(f"✅ 视频已保存: {output_path}")
    
    def create_thumbnail(self, pattern_type, color_palette, output_path):
        """创建缩略图"""
        # 生成中间时刻的帧
        frame = self.create_frame(5.0, pattern_type, color_palette)
        
        # 添加文字标签
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(frame)
        ax.set_title(f"{pattern_type.title()} - {color_palette.title()}", 
                    fontsize=16, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def run_video_workshop():
    """运行视频生成工作坊"""
    print("《心境流转》视频生成工作坊")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    print("\n📌 注意：当前为预览模式")
    print("  - 只生成5帧预览图片，不生成完整视频")
    print("  - 亮度已调高便于查看（实际使用应更暗）")
    print("  - 完整视频生成需要修改 preview_only=False")
    
    # 创建生成器（使用较小分辨率以加快速度）
    generator = SleepVideoGenerator(width=960, height=540, fps=24)
    
    # 创建输出目录
    output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试配置
    test_configs = [
        {
            "name": "海洋呼吸",
            "pattern": "breathing",
            "palette": "ocean",
            "duration": 30,  # 30秒
            "description": "呼吸引导圆圈，海洋色调"
        },
        {
            "name": "晚霞渐变",
            "pattern": "gradient",
            "palette": "sunset",
            "duration": 30,
            "description": "流动渐变，晚霞色调"
        },
        {
            "name": "森林波浪",
            "pattern": "waves",
            "palette": "forest",
            "duration": 30,
            "description": "柔和波浪，森林色调"
        },
        {
            "name": "薰衣草曼陀罗",
            "pattern": "mandala",
            "palette": "lavender",
            "duration": 30,
            "description": "旋转曼陀罗，薰衣草色调"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*40}")
        print(f"🎬 生成: {config['name']}")
        print(f"说明: {config['description']}")
        
        # 生成缩略图
        thumbnail_path = output_dir / f"{config['name'].replace(' ', '_')}_thumbnail.png"
        generator.create_thumbnail(
            config['pattern'], 
            config['palette'], 
            thumbnail_path
        )
        print(f"✅ 缩略图已保存: {thumbnail_path}")
        
        # 生成预览帧（不生成完整视频以节省时间）
        print(f"  生成预览帧...")
        preview_frames = generator.generate_video(
            config['duration'],
            config['pattern'],
            config['palette'],
            None,
            preview_only=True
        )
        print(f"  ✅ 生成了 {len(preview_frames)} 个预览帧")
        
        # 保存预览帧
        preview_dir = output_dir / f"{config['name'].replace(' ', '_')}_preview"
        preview_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(preview_frames):
            frame_path = preview_dir / f"frame_{i:02d}.png"
            if CV2_AVAILABLE:
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                plt.imsave(str(frame_path), frame)
        
        print(f"✅ 预览帧已保存: {preview_dir}")
        
        # 分析视觉特征
        features = analyze_visual_features(preview_frames)
        
        # 记录结果
        results.append({
            "name": config['name'],
            "config": config,
            "thumbnail": str(thumbnail_path),
            "preview_dir": str(preview_dir),
            "features": features
        })
    
    # 显示分析结果
    print_visual_analysis(results)
    
    # 创建对比图
    create_comparison_chart(results, output_dir)
    
    # 保存工作坊结果
    save_workshop_results(results)
    
    return results

def analyze_visual_features(frames):
    """分析视觉特征"""
    features = {
        "brightness": [],
        "contrast": [],
        "color_variance": [],
        "motion": []
    }
    
    for i, frame in enumerate(frames):
        # 亮度
        brightness = np.mean(frame)
        features["brightness"].append(brightness)
        
        # 对比度
        contrast = np.std(frame)
        features["contrast"].append(contrast)
        
        # 颜色变化
        color_var = np.mean([np.std(frame[:,:,c]) for c in range(3)])
        features["color_variance"].append(color_var)
        
        # 运动（相邻帧差异）
        if i > 0:
            motion = np.mean(np.abs(frame.astype(float) - frames[i-1].astype(float)))
            features["motion"].append(motion)
    
    # 计算平均值
    return {
        "avg_brightness": np.mean(features["brightness"]),
        "avg_contrast": np.mean(features["contrast"]),
        "avg_color_variance": np.mean(features["color_variance"]),
        "avg_motion": np.mean(features["motion"]) if features["motion"] else 0,
        "brightness_range": [min(features["brightness"]), max(features["brightness"])]
    }

def print_visual_analysis(results):
    """打印视觉分析结果"""
    print("\n📊 视觉特征分析")
    print("=" * 50)
    
    for result in results:
        print(f"\n{result['name']}:")
        features = result['features']
        print(f"  平均亮度: {features['avg_brightness']:.1f}")
        print(f"  平均对比度: {features['avg_contrast']:.1f}")
        print(f"  颜色变化: {features['avg_color_variance']:.1f}")
        print(f"  运动强度: {features['avg_motion']:.1f}")
        print(f"  亮度范围: {features['brightness_range'][0]:.1f}-{features['brightness_range'][1]:.1f}")

def create_comparison_chart(results, output_dir):
    """创建对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = [r['name'] for r in results]
    
    # 亮度对比
    ax = axes[0, 0]
    brightness = [r['features']['avg_brightness'] for r in results]
    bars = ax.bar(names, brightness)
    ax.set_title('Average Brightness')
    ax.set_ylabel('Brightness')
    ax.set_ylim(0, 100)
    
    # 添加颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 对比度对比
    ax = axes[0, 1]
    contrast = [r['features']['avg_contrast'] for r in results]
    bars = ax.bar(names, contrast)
    ax.set_title('Average Contrast')
    ax.set_ylabel('Contrast')
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 颜色变化对比
    ax = axes[1, 0]
    color_var = [r['features']['avg_color_variance'] for r in results]
    bars = ax.bar(names, color_var)
    ax.set_title('Color Variance')
    ax.set_ylabel('Variance')
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 运动强度对比
    ax = axes[1, 1]
    motion = [r['features']['avg_motion'] for r in results]
    bars = ax.bar(names, motion)
    ax.set_title('Motion Intensity')
    ax.set_ylabel('Motion')
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 调整布局
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = output_dir / "visual_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 对比图表已保存: {chart_path}")

def save_workshop_results(results):
    """保存工作坊结果"""
    output_file = Path("outputs/videos/workshop_results.json")
    
    # 准备数据
    workshop_data = {
        "timestamp": datetime.now().isoformat(),
        "total_patterns": len(results),
        "patterns": results,
        "technical_specs": {
            "resolution": "960x540",
            "fps": 24,
            "format": "preview frames (PNG)",
            "color_depth": "8-bit RGB"
        },
        "features_analyzed": [
            "Brightness",
            "Contrast", 
            "Color Variance",
            "Motion Intensity"
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workshop_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 工作坊结果已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行工作坊
        results = run_video_workshop()
        
        # 生成建议
        print("\n💡 视觉治疗建议")
        print("-" * 40)
        print("1. 呼吸圆圈适合引导放松和呼吸练习")
        print("2. 渐变流动营造平静氛围")
        print("3. 波浪图案模拟自然节奏")
        print("4. 曼陀罗图案用于冥想专注")
        
        print("\n🎬 技术优化建议")
        print("-" * 40)
        print("1. 使用GPU加速视频渲染")
        print("2. 实现实时参数调整")
        print("3. 添加粒子效果增强视觉体验")
        print("4. 支持4K分辨率输出")
        
        print("\n⚠️ 注意事项")
        print("-" * 40)
        print("1. 睡眠场景需要低亮度、低对比度")
        print("2. 避免快速运动和闪烁")
        print("3. 使用柔和的颜色过渡")
        print("4. 考虑用户的视觉敏感性")
        
        print("\n" + "=" * 50)
        print("视频生成工作坊完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 08_multimodal_therapy_test.py")
        
    except Exception as e:
        print(f"\n❌ 工作坊出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()