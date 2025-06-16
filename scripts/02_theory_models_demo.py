#!/usr/bin/env python3
"""
02 - 理论模型演示
演示《心境流转》系统的核心理论:
- ISO三阶段治疗原则
- Valence-Arousal情绪模型
- 音乐治疗参数推荐
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class EmotionState:
    """情绪状态数据类"""
    valence: float  # -1到1
    arousal: float  # -1到1
    confidence: float = 0.8
    
    def distance_to(self, other):
        """计算到另一个情绪状态的欧氏距离"""
        return ((self.valence - other.valence)**2 + (self.arousal - other.arousal)**2)**0.5

class ISOStage(Enum):
    """ISO治疗阶段枚举"""
    SYNC = "同步化"
    GUIDE = "引导化" 
    CONSOLIDATE = "巩固化"

class ISOModel:
    """ISO三阶段治疗模型"""
    
    def __init__(self):
        print("🎵 ISO三阶段治疗模型初始化")
    
    def plan_stages(self, current, target, duration):
        """规划三个治疗阶段"""
        return [
            {'stage': ISOStage.SYNC, 'duration': duration * 0.25, 'emotion': current},
            {'stage': ISOStage.GUIDE, 'duration': duration * 0.50, 
             'emotion': EmotionState((current.valence + target.valence)/2, 
                                   (current.arousal + target.arousal)/2)},
            {'stage': ISOStage.CONSOLIDATE, 'duration': duration * 0.25, 'emotion': target}
        ]
    
    def generate_trajectory(self, current, target, duration, points=50):
        """生成情绪变化轨迹"""
        trajectory = []
        for i in range(points):
            progress = i / (points - 1)
            # S型平滑曲线
            smooth = 3 * progress**2 - 2 * progress**3
            
            valence = current.valence + (target.valence - current.valence) * smooth
            arousal = current.arousal + (target.arousal - current.arousal) * smooth
            
            if progress < 0.25:
                stage = ISOStage.SYNC
            elif progress < 0.75:
                stage = ISOStage.GUIDE
            else:
                stage = ISOStage.CONSOLIDATE
            
            trajectory.append({
                'time': progress * duration,
                'emotion': EmotionState(valence, arousal),
                'stage': stage
            })
        return trajectory

class MusicModel:
    """音乐治疗参数模型"""
    
    def __init__(self):
        print("🎼 音乐治疗模型初始化")
    
    def calc_bpm(self, arousal):
        """根据唤醒度计算BPM"""
        # 唤醒度映射到BPM: -1→40, 0→80, 1→120
        return 80 + (arousal * 40)
    
    def recommend_music(self, emotion, stage):
        """推荐音乐参数"""
        bpm = self.calc_bpm(emotion.arousal)
        
        if emotion.valence > 0.2:
            key_type = "大调"
        elif emotion.valence < -0.2:
            key_type = "小调"
        else:
            key_type = "中性调"
        
        if stage == ISOStage.SYNC:
            instruments = ["小提琴", "钢琴"]
        elif stage == ISOStage.GUIDE:
            instruments = ["长笛", "弦乐"]
        else:
            instruments = ["大提琴", "竖琴"]
        
        return {
            'bpm': round(bpm),
            'key': key_type,
            'instruments': instruments,
            'volume': 'soft' if emotion.arousal < 0 else 'moderate'
        }

def plot_emotion_trajectory(trajectory, current_emotion, target_emotion, 
                          output_dir="outputs/figures"):
    """绘制情绪轨迹图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    times = [p['time'] for p in trajectory]
    valences = [p['emotion'].valence for p in trajectory]
    arousals = [p['emotion'].arousal for p in trajectory]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 时间序列图
    ax1.plot(times, valences, 'b-', linewidth=3, label='Valence')
    ax1.plot(times, arousals, 'r-', linewidth=3, label='Arousal')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 阶段分界
    ax1.axvspan(0, 5, alpha=0.2, color='lightblue', label='Sync')
    ax1.axvspan(5, 15, alpha=0.2, color='lightgreen', label='Guide')
    ax1.axvspan(15, 20, alpha=0.2, color='lightcoral', label='Consolidate')
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Emotion Value')
    ax1.set_title('ISO Three-Stage Emotion Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # VA空间图
    ax2.plot(valences, arousals, 'purple', linewidth=3)
    ax2.scatter(current_emotion.valence, current_emotion.arousal, 
                c='red', s=100, label='Start')
    ax2.scatter(target_emotion.valence, target_emotion.arousal, 
                c='green', s=100, label='Target')
    
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=0, color='black', linewidth=1)
    
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('Valence')
    ax2.set_ylabel('Arousal')
    ax2.set_title('Emotion Trajectory in V-A Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "emotion_trajectory.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ 情绪轨迹图已保存: {output_path}")

def plot_music_parameters(trajectory, music_model, output_dir="outputs/figures"):
    """绘制音乐参数变化图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    times = [p['time'] for p in trajectory]
    bpms = [music_model.calc_bpm(p['emotion'].arousal) for p in trajectory]
    volumes = [50 + (p['emotion'].arousal * 25) for p in trajectory]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # BPM变化
    ax1.plot(times, bpms, 'orange', linewidth=3)
    ax1.fill_between(times, bpms, alpha=0.3, color='orange')
    ax1.set_ylabel('BPM')
    ax1.set_title('Music Tempo Changes')
    ax1.grid(True, alpha=0.3)
    
    # 音量变化
    ax2.plot(times, volumes, 'green', linewidth=3)
    ax2.fill_between(times, volumes, alpha=0.3, color='green')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Volume')
    ax2.set_title('Volume Changes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "music_parameters.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ 音乐参数图已保存: {output_path}")
    
    return bpms, volumes

def main():
    """主函数"""
    print("《心境流转》理论模型演示")
    print("=" * 40)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. 初始化模型
    iso_model = ISOModel()
    music_model = MusicModel()
    print("✅ 模型初始化完成")
    
    # 2. 定义情绪状态
    # 当前：焦虑状态（负效价，高唤醒）
    current_emotion = EmotionState(valence=-0.6, arousal=0.8)
    # 目标：平静状态（正效价，低唤醒）  
    target_emotion = EmotionState(valence=0.5, arousal=-0.7)
    
    distance = current_emotion.distance_to(target_emotion)
    
    print(f"\n当前情绪: V={current_emotion.valence:.2f}, A={current_emotion.arousal:.2f} (焦虑)")
    print(f"目标情绪: V={target_emotion.valence:.2f}, A={target_emotion.arousal:.2f} (平静)")
    print(f"情绪距离: {distance:.3f}")
    
    # 3. ISO三阶段规划
    duration = 20.0  # 20分钟
    stages = iso_model.plan_stages(current_emotion, target_emotion, duration)
    
    print(f"\nISO三阶段治疗规划 (总时长: {duration}分钟)")
    print("=" * 45)
    
    for i, stage in enumerate(stages, 1):
        emotion = stage['emotion']
        print(f"第{i}阶段: {stage['stage'].value}")
        print(f"  时长: {stage['duration']:.1f}分钟")
        print(f"  目标: V={emotion.valence:.2f}, A={emotion.arousal:.2f}")
        print()
    
    # 4. 生成轨迹
    trajectory = iso_model.generate_trajectory(current_emotion, target_emotion, duration)
    print(f"轨迹生成完成: {len(trajectory)}个时间点")
    
    # 5. 可视化
    plot_emotion_trajectory(trajectory, current_emotion, target_emotion)
    bpms, volumes = plot_music_parameters(trajectory, music_model)
    
    # 6. 音乐推荐
    print("\n音乐治疗方案:")
    print("=" * 30)
    
    for stage in stages:
        music = music_model.recommend_music(stage['emotion'], stage['stage'])
        print(f"{stage['stage'].value}:")
        print(f"  BPM: {music['bpm']}")
        print(f"  调性: {music['key']}")
        print(f"  乐器: {', '.join(music['instruments'])}")
        print(f"  音量: {music['volume']}")
        print()
    
    # 7. 验证结果
    print("\n🔬 理论模型验证:")
    print("=" * 25)
    
    checks = [
        ("情绪变化合理", distance < 2.5),
        ("引导阶段最长", stages[1]['duration'] >= stages[0]['duration']),
        ("BPM递减", bpms[0] > bpms[-1]),
        ("音量递减", volumes[0] > volumes[-1]),
        ("达到目标", trajectory[-1]['emotion'].distance_to(target_emotion) < 0.1)
    ]
    
    passed = 0
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    score = passed / len(checks)
    print(f"\n验证得分: {score:.1%} ({passed}/{len(checks)})")
    
    # 8. 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'scenario': {
            'current': {'valence': current_emotion.valence, 'arousal': current_emotion.arousal},
            'target': {'valence': target_emotion.valence, 'arousal': target_emotion.arousal},
            'distance': distance
        },
        'iso_planning': {
            'duration': duration,
            'stages': len(stages),
            'trajectory_points': len(trajectory)
        },
        'music_params': {
            'bpm_start': bpms[0],
            'bpm_end': bpms[-1],
            'bpm_reduction': bpms[0] - bpms[-1]
        },
        'validation': {
            'score': score,
            'checks_passed': passed,
            'total_checks': len(checks)
        }
    }
    
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / 'theory_demo_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存: {result_file}")
    
    # 总结
    print("\n" + "=" * 50)
    print("《心境流转》理论模型演示完成")
    print("=" * 50)
    print(f"✅ ISO三阶段规划: {len(stages)}阶段")
    print(f"✅ 情绪轨迹: {len(trajectory)}个时间点")
    print(f"✅ 音乐参数: BPM {bpms[0]:.0f}→{bpms[-1]:.0f}")
    print(f"✅ 验证得分: {score:.1%}")
    print(f"\n🚀 下一步: 运行 03_model_adapters_test.py")
    print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()