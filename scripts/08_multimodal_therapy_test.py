#!/usr/bin/env python3
"""
08 - 多模态治疗测试
测试音视频融合的睡眠治疗效果
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MultimodalSynchronizer:
    """多模态同步器"""
    
    def __init__(self):
        self.audio_fps = 44100  # 音频采样率
        self.video_fps = 30     # 视频帧率
        self.sync_tolerance = 0.033  # 33ms容差
    
    def calculate_sync_points(self, duration_seconds):
        """计算同步点"""
        # 关键同步时刻
        sync_points = []
        
        # 开始点
        sync_points.append({
            "time": 0.0,
            "event": "start",
            "audio": "fade_in",
            "video": "fade_in"
        })
        
        # 呼吸引导点（每20秒）
        for i in range(1, int(duration_seconds / 20) + 1):
            sync_points.append({
                "time": i * 20.0,
                "event": "breathing_cue",
                "audio": "breathing_sound",
                "video": "breathing_circle"
            })
        
        # 中间过渡点
        if duration_seconds > 60:
            sync_points.append({
                "time": duration_seconds / 2,
                "event": "transition",
                "audio": "key_change",
                "video": "pattern_change"
            })
        
        # 结束点
        sync_points.append({
            "time": duration_seconds - 10,
            "event": "ending",
            "audio": "fade_out",
            "video": "fade_out"
        })
        
        return sync_points
    
    def generate_sync_map(self, audio_events, video_events, duration):
        """生成同步映射"""
        sync_map = []
        
        # 对齐音视频事件
        for audio_event in audio_events:
            best_match = None
            min_diff = float('inf')
            
            for video_event in video_events:
                time_diff = abs(audio_event['time'] - video_event['time'])
                if time_diff < min_diff and time_diff < self.sync_tolerance:
                    min_diff = time_diff
                    best_match = video_event
            
            if best_match:
                sync_map.append({
                    "time": audio_event['time'],
                    "audio_event": audio_event['type'],
                    "video_event": best_match['type'],
                    "sync_quality": 1.0 - (min_diff / self.sync_tolerance)
                })
        
        return sync_map

class TherapyEffectAnalyzer:
    """治疗效果分析器"""
    
    def __init__(self):
        self.metrics = {
            "relaxation_score": 0.0,
            "coherence_score": 0.0,
            "engagement_score": 0.0,
            "effectiveness_score": 0.0
        }
    
    def analyze_multimodal_effect(self, audio_features, video_features, sync_quality):
        """分析多模态治疗效果"""
        # 放松度评分
        audio_relaxation = self._calculate_audio_relaxation(audio_features)
        video_relaxation = self._calculate_video_relaxation(video_features)
        self.metrics["relaxation_score"] = (audio_relaxation + video_relaxation) / 2
        
        # 协调性评分
        self.metrics["coherence_score"] = sync_quality * 0.8 + 0.2
        
        # 参与度评分
        variation_score = self._calculate_variation_score(audio_features, video_features)
        self.metrics["engagement_score"] = variation_score
        
        # 整体效果评分
        self.metrics["effectiveness_score"] = (
            self.metrics["relaxation_score"] * 0.4 +
            self.metrics["coherence_score"] * 0.3 +
            self.metrics["engagement_score"] * 0.3
        )
        
        return self.metrics
    
    def _calculate_audio_relaxation(self, features):
        """计算音频放松度"""
        # 基于BPM、音量变化等
        bpm_score = 1.0 - min(features.get("avg_bpm", 80) / 120, 1.0)
        volume_score = 1.0 - features.get("volume_variance", 0.5)
        
        return (bpm_score + volume_score) / 2
    
    def _calculate_video_relaxation(self, features):
        """计算视频放松度"""
        # 基于亮度、运动等
        brightness_score = 1.0 - min(features.get("avg_brightness", 50) / 100, 1.0)
        motion_score = 1.0 - min(features.get("motion_intensity", 0.5), 1.0)
        
        return (brightness_score + motion_score) / 2
    
    def _calculate_variation_score(self, audio_features, video_features):
        """计算变化度评分"""
        # 适度的变化保持注意力
        audio_var = audio_features.get("pattern_changes", 3)
        video_var = video_features.get("scene_changes", 2)
        
        optimal_changes = 4
        score = 1.0 - abs((audio_var + video_var) / 2 - optimal_changes) / optimal_changes
        
        return max(0, min(1, score))

class MultimodalTherapySession:
    """多模态治疗会话"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.synchronizer = MultimodalSynchronizer()
        self.analyzer = TherapyEffectAnalyzer()
        self.timeline = []
        
    def design_session(self, user_profile, duration_minutes=30):
        """设计治疗会话"""
        print(f"\n🎯 设计多模态治疗方案")
        print(f"时长: {duration_minutes}分钟")
        
        # 生成同步点
        sync_points = self.synchronizer.calculate_sync_points(duration_minutes * 60)
        
        # 设计音频轨道
        audio_track = self._design_audio_track(user_profile, sync_points)
        
        # 设计视频轨道
        video_track = self._design_video_track(user_profile, sync_points)
        
        # 创建时间轴
        self.timeline = self._create_timeline(audio_track, video_track, sync_points)
        
        return self.timeline
    
    def _design_audio_track(self, user_profile, sync_points):
        """设计音频轨道"""
        track = {
            "type": "audio",
            "segments": []
        }
        
        # 根据用户问题选择音乐风格
        if "入睡困难" in user_profile.get("issues", []):
            base_style = "ambient_slow"
            base_bpm = 50
        else:
            base_style = "nature_sounds"
            base_bpm = 60
        
        # 创建音频段
        for i in range(len(sync_points) - 1):
            start = sync_points[i]
            end = sync_points[i + 1]
            
            segment = {
                "start_time": start["time"],
                "end_time": end["time"],
                "style": base_style,
                "bpm": base_bpm - i * 5,  # 逐渐减慢
                "volume": 0.8 - i * 0.1,   # 逐渐减弱
                "instruments": ["piano", "strings", "nature"],
                "effects": ["reverb", "gentle_eq"]
            }
            
            track["segments"].append(segment)
        
        return track
    
    def _design_video_track(self, user_profile, sync_points):
        """设计视频轨道"""
        track = {
            "type": "video",
            "segments": []
        }
        
        # 根据用户偏好选择视觉风格
        if user_profile.get("preferences", {}).get("visual") == "abstract":
            base_pattern = "mandala"
        else:
            base_pattern = "nature_scene"
        
        # 创建视频段
        patterns = ["breathing_circle", "gradient_flow", "wave_pattern", "mandala"]
        colors = ["ocean", "sunset", "forest", "lavender"]
        
        for i in range(len(sync_points) - 1):
            start = sync_points[i]
            end = sync_points[i + 1]
            
            segment = {
                "start_time": start["time"],
                "end_time": end["time"],
                "pattern": patterns[i % len(patterns)],
                "color_palette": colors[i % len(colors)],
                "brightness": 0.5 - i * 0.05,  # 逐渐变暗
                "motion_speed": "slow",
                "transitions": ["smooth_fade"]
            }
            
            track["segments"].append(segment)
        
        return track
    
    def _create_timeline(self, audio_track, video_track, sync_points):
        """创建统一时间轴"""
        timeline = {
            "total_duration": sync_points[-1]["time"],
            "sync_points": sync_points,
            "audio_track": audio_track,
            "video_track": video_track,
            "coordination": []
        }
        
        # 添加协调事件
        for point in sync_points:
            if point["event"] == "breathing_cue":
                timeline["coordination"].append({
                    "time": point["time"],
                    "type": "synchronized_breathing",
                    "audio_action": "breathing_rhythm",
                    "video_action": "breathing_visual"
                })
        
        return timeline
    
    def simulate_playback(self):
        """模拟播放过程"""
        print("\n▶️ 模拟多模态播放")
        print("-" * 40)
        
        # 模拟特征提取
        audio_features = {
            "avg_bpm": 55,
            "volume_variance": 0.2,
            "pattern_changes": 3
        }
        
        video_features = {
            "avg_brightness": 30,
            "motion_intensity": 0.2,
            "scene_changes": 4
        }
        
        # 模拟同步质量
        sync_quality = 0.95
        
        # 分析效果
        effects = self.analyzer.analyze_multimodal_effect(
            audio_features, video_features, sync_quality
        )
        
        return effects

def run_multimodal_test():
    """运行多模态治疗测试"""
    print("《心境流转》多模态治疗测试")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 测试用户配置
    test_users = [
        {
            "id": "user_001",
            "name": "轻度失眠用户",
            "issues": ["入睡困难"],
            "preferences": {"visual": "nature", "audio": "ambient"}
        },
        {
            "id": "user_002", 
            "name": "焦虑失眠用户",
            "issues": ["入睡困难", "频繁醒来"],
            "preferences": {"visual": "abstract", "audio": "music"}
        },
        {
            "id": "user_003",
            "name": "重度失眠用户",
            "issues": ["失眠", "早醒", "噩梦"],
            "preferences": {"visual": "mixed", "audio": "mixed"}
        }
    ]
    
    results = []
    
    for user in test_users:
        print(f"\n{'='*40}")
        print(f"👤 测试用户: {user['name']}")
        print(f"问题: {', '.join(user['issues'])}")
        
        # 创建会话
        session = MultimodalTherapySession(f"session_{user['id']}")
        
        # 设计方案
        timeline = session.design_session(user, duration_minutes=20)
        
        # 显示方案概览
        display_session_plan(timeline)
        
        # 模拟播放
        effects = session.simulate_playback()
        
        # 显示效果评估
        display_effectiveness(effects)
        
        # 记录结果
        results.append({
            "user": user,
            "timeline": timeline,
            "effectiveness": effects
        })
    
    # 生成对比报告
    generate_comparison_report(results)
    
    # 保存测试结果
    save_test_results(results)
    
    return results

def display_session_plan(timeline):
    """显示会话计划"""
    print(f"\n📋 治疗方案概览")
    print(f"总时长: {timeline['total_duration']/60:.1f}分钟")
    print(f"同步点: {len(timeline['sync_points'])}个")
    
    print("\n音频轨道:")
    for i, seg in enumerate(timeline['audio_track']['segments'][:3]):
        print(f"  段{i+1}: {seg['style']} @ {seg['bpm']}BPM")
    
    print("\n视频轨道:")
    for i, seg in enumerate(timeline['video_track']['segments'][:3]):
        print(f"  段{i+1}: {seg['pattern']} - {seg['color_palette']}")

def display_effectiveness(effects):
    """显示效果评估"""
    print(f"\n📊 治疗效果评估")
    print(f"放松度: {effects['relaxation_score']:.1%}")
    print(f"协调性: {effects['coherence_score']:.1%}")
    print(f"参与度: {effects['engagement_score']:.1%}")
    print(f"整体效果: {effects['effectiveness_score']:.1%}")

def generate_comparison_report(results):
    """生成对比报告"""
    print(f"\n{'='*50}")
    print("📊 多模态治疗效果对比")
    print("=" * 50)
    
    # 创建对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    users = [r['user']['name'] for r in results]
    metrics = ['relaxation_score', 'coherence_score', 'engagement_score', 'effectiveness_score']
    metric_names = ['放松度', '协调性', '参与度', '整体效果']
    
    x = np.arange(len(users))
    width = 0.2
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [r['effectiveness'][metric] for r in results]
        ax.bar(x + i * width, values, width, label=name)
    
    ax.set_xlabel('用户类型')
    ax.set_ylabel('评分')
    ax.set_title('多模态治疗效果对比')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(users)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 保存图表
    output_dir = Path("outputs/multimodal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chart_path = output_dir / "effectiveness_comparison.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"\n✅ 对比图表已保存: {chart_path}")

def save_test_results(results):
    """保存测试结果"""
    output_dir = Path("outputs/multimodal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "multimodal_therapy",
        "total_users": len(results),
        "results": []
    }
    
    for r in results:
        test_data["results"].append({
            "user": r["user"],
            "effectiveness": r["effectiveness"],
            "timeline_summary": {
                "duration": r["timeline"]["total_duration"],
                "sync_points": len(r["timeline"]["sync_points"]),
                "audio_segments": len(r["timeline"]["audio_track"]["segments"]),
                "video_segments": len(r["timeline"]["video_track"]["segments"])
            }
        })
    
    # 保存文件
    output_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行测试
        results = run_multimodal_test()
        
        # 生成建议
        print("\n💡 多模态治疗优化建议")
        print("-" * 40)
        print("1. 精确同步：确保音视频在关键时刻完美配合")
        print("2. 个性化调整：根据实时反馈动态调整参数")
        print("3. 渐进式设计：从活跃到平静的平滑过渡")
        print("4. 感官平衡：避免某一模态过于突出")
        
        print("\n🔬 技术实现要点")
        print("-" * 40)
        print("1. 使用时间码确保精确同步")
        print("2. 实现跨模态特征提取和分析")
        print("3. 建立用户反馈机制")
        print("4. 优化资源使用避免延迟")
        
        print("\n" + "=" * 50)
        print("多模态治疗测试完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 09_performance_optimization.py")
        
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()