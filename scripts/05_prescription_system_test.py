#!/usr/bin/env python3
"""
05 - 处方系统测试
测试个性化睡眠治疗处方的生成和管理
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import warnings
warnings.filterwarnings('ignore')

class SleepIssueType(Enum):
    """睡眠问题类型"""
    DIFFICULTY_FALLING_ASLEEP = "入睡困难"
    FREQUENT_AWAKENING = "频繁醒来"
    EARLY_AWAKENING = "早醒"
    LIGHT_SLEEP = "浅睡眠"
    NIGHTMARES = "噩梦"
    INSOMNIA = "失眠"

class TherapyModality(Enum):
    """治疗模态"""
    MUSIC = "音乐疗法"
    VIDEO = "视觉疗法"
    BREATHING = "呼吸练习"
    MEDITATION = "冥想引导"
    MULTIMODAL = "多模态融合"

@dataclass
class MusicTherapyParams:
    """音乐治疗参数"""
    genre: str
    tempo_range: Tuple[int, int]  # BPM范围
    key_type: str  # 大调/小调
    instruments: List[str]
    duration_minutes: int
    volume_profile: str  # 音量变化模式

@dataclass
class VideoTherapyParams:
    """视觉治疗参数"""
    content_type: str  # 自然风景/抽象图案/渐变色彩
    color_palette: List[str]
    movement_speed: str  # 慢速/中速/静态
    brightness_profile: str  # 亮度变化模式
    duration_minutes: int

@dataclass
class BreathingExerciseParams:
    """呼吸练习参数"""
    pattern: str  # 4-7-8, 4-4-4-4等
    cycles: int
    guidance_voice: str
    background_sound: Optional[str]

@dataclass
class TherapyPrescription:
    """治疗处方"""
    prescription_id: str
    user_id: str
    created_at: str
    sleep_issues: List[SleepIssueType]
    primary_modality: TherapyModality
    secondary_modalities: List[TherapyModality]
    music_params: Optional[MusicTherapyParams]
    video_params: Optional[VideoTherapyParams]
    breathing_params: Optional[BreathingExerciseParams]
    duration_total: int
    schedule: Dict[str, str]
    expected_efficacy: float
    notes: str

class PrescriptionEngine:
    """处方生成引擎"""
    
    def __init__(self):
        self.issue_modality_map = {
            SleepIssueType.DIFFICULTY_FALLING_ASLEEP: [
                TherapyModality.BREATHING,
                TherapyModality.MUSIC
            ],
            SleepIssueType.FREQUENT_AWAKENING: [
                TherapyModality.MUSIC,
                TherapyModality.MEDITATION
            ],
            SleepIssueType.EARLY_AWAKENING: [
                TherapyModality.VIDEO,
                TherapyModality.MULTIMODAL
            ],
            SleepIssueType.LIGHT_SLEEP: [
                TherapyModality.MUSIC,
                TherapyModality.VIDEO
            ],
            SleepIssueType.NIGHTMARES: [
                TherapyModality.MEDITATION,
                TherapyModality.VIDEO
            ],
            SleepIssueType.INSOMNIA: [
                TherapyModality.MULTIMODAL,
                TherapyModality.BREATHING
            ]
        }
    
    def analyze_user_profile(self, user_data: Dict) -> Tuple[List[SleepIssueType], Dict]:
        """分析用户档案"""
        # 解析睡眠问题
        issues = []
        issue_mapping = {
            "入睡困难": SleepIssueType.DIFFICULTY_FALLING_ASLEEP,
            "频繁醒来": SleepIssueType.FREQUENT_AWAKENING,
            "早醒": SleepIssueType.EARLY_AWAKENING,
            "浅睡眠": SleepIssueType.LIGHT_SLEEP,
            "噩梦": SleepIssueType.NIGHTMARES,
            "失眠": SleepIssueType.INSOMNIA
        }
        
        for issue_str in user_data.get("sleep_issues", []):
            if issue_str in issue_mapping:
                issues.append(issue_mapping[issue_str])
        
        # 分析偏好
        preferences = {
            "prefers_music": "music" in user_data.get("preferences", {}).get("therapy_types", ""),
            "prefers_visual": "visual" in user_data.get("preferences", {}).get("therapy_types", ""),
            "session_duration": user_data.get("preferences", {}).get("session_duration", 30),
            "sensitivity": user_data.get("sensitivity", "normal")
        }
        
        return issues, preferences
    
    def select_modalities(self, issues: List[SleepIssueType], 
                         preferences: Dict) -> Tuple[TherapyModality, List[TherapyModality]]:
        """选择治疗模态"""
        # 统计推荐的模态
        modality_scores = {}
        for issue in issues:
            recommended = self.issue_modality_map.get(issue, [])
            for modality in recommended:
                modality_scores[modality] = modality_scores.get(modality, 0) + 1
        
        # 根据偏好调整
        if preferences["prefers_music"]:
            modality_scores[TherapyModality.MUSIC] = modality_scores.get(TherapyModality.MUSIC, 0) + 2
        if preferences["prefers_visual"]:
            modality_scores[TherapyModality.VIDEO] = modality_scores.get(TherapyModality.VIDEO, 0) + 2
        
        # 排序选择
        sorted_modalities = sorted(modality_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_modalities:
            primary = TherapyModality.MUSIC
            secondary = [TherapyModality.BREATHING]
        else:
            primary = sorted_modalities[0][0]
            secondary = [m[0] for m in sorted_modalities[1:3] if m[0] != primary]
        
        return primary, secondary
    
    def generate_music_params(self, issues: List[SleepIssueType], 
                            duration: int) -> MusicTherapyParams:
        """生成音乐治疗参数"""
        # 根据问题类型确定音乐风格
        if SleepIssueType.DIFFICULTY_FALLING_ASLEEP in issues:
            tempo_range = (50, 70)
            instruments = ["钢琴", "弦乐", "竖琴"]
            volume_profile = "渐弱"
        elif SleepIssueType.NIGHTMARES in issues:
            tempo_range = (60, 80)
            instruments = ["长笛", "钢琴", "自然音"]
            volume_profile = "稳定柔和"
        else:
            tempo_range = (55, 75)
            instruments = ["大提琴", "钢琴", "合成器"]
            volume_profile = "波浪式"
        
        return MusicTherapyParams(
            genre="睡眠音乐",
            tempo_range=tempo_range,
            key_type="大调" if random.random() > 0.3 else "小调",
            instruments=instruments,
            duration_minutes=duration,
            volume_profile=volume_profile
        )
    
    def generate_video_params(self, issues: List[SleepIssueType], 
                            duration: int) -> VideoTherapyParams:
        """生成视觉治疗参数"""
        # 根据问题类型确定视觉内容
        if SleepIssueType.NIGHTMARES in issues:
            content_type = "温暖光晕"
            color_palette = ["#FFE4B5", "#FFA07A", "#FFB6C1"]
            brightness_profile = "恒定柔和"
        elif SleepIssueType.EARLY_AWAKENING in issues:
            content_type = "星空渐变"
            color_palette = ["#191970", "#000080", "#4B0082"]
            brightness_profile = "缓慢降低"
        else:
            content_type = "自然风景"
            color_palette = ["#87CEEB", "#98FB98", "#F0E68C"]
            brightness_profile = "自然变化"
        
        return VideoTherapyParams(
            content_type=content_type,
            color_palette=color_palette,
            movement_speed="慢速",
            brightness_profile=brightness_profile,
            duration_minutes=duration
        )
    
    def generate_breathing_params(self) -> BreathingExerciseParams:
        """生成呼吸练习参数"""
        patterns = ["4-7-8", "4-4-4-4", "5-5-5-5", "3-4-5"]
        voices = ["女声温柔", "男声低沉", "中性柔和"]
        
        return BreathingExerciseParams(
            pattern=random.choice(patterns),
            cycles=random.randint(6, 10),
            guidance_voice=random.choice(voices),
            background_sound="白噪音" if random.random() > 0.5 else None
        )
    
    def calculate_efficacy(self, prescription: TherapyPrescription) -> float:
        """计算预期疗效"""
        base_efficacy = 0.6
        
        # 模态匹配度
        if prescription.primary_modality == TherapyModality.MULTIMODAL:
            base_efficacy += 0.15
        
        # 时长合理性
        if 20 <= prescription.duration_total <= 40:
            base_efficacy += 0.1
        
        # 问题针对性
        if len(prescription.sleep_issues) <= 2:
            base_efficacy += 0.1
        
        # 添加随机因素
        base_efficacy += random.uniform(-0.05, 0.1)
        
        return min(0.95, max(0.5, base_efficacy))
    
    def generate_prescription(self, user_data: Dict) -> TherapyPrescription:
        """生成个性化处方"""
        # 分析用户
        issues, preferences = self.analyze_user_profile(user_data)
        
        # 选择治疗模态
        primary, secondary = self.select_modalities(issues, preferences)
        
        # 确定时长
        total_duration = preferences.get("session_duration", 30)
        
        # 生成各模态参数
        music_params = None
        video_params = None
        breathing_params = None
        
        if primary == TherapyModality.MUSIC or TherapyModality.MUSIC in secondary:
            music_params = self.generate_music_params(issues, total_duration // 2)
        
        if primary == TherapyModality.VIDEO or TherapyModality.VIDEO in secondary:
            video_params = self.generate_video_params(issues, total_duration // 2)
        
        if primary == TherapyModality.BREATHING or TherapyModality.BREATHING in secondary:
            breathing_params = self.generate_breathing_params()
        
        if primary == TherapyModality.MULTIMODAL:
            music_params = self.generate_music_params(issues, total_duration // 2)
            video_params = self.generate_video_params(issues, total_duration)
        
        # 创建处方
        prescription = TherapyPrescription(
            prescription_id=f"RX_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_data.get("user_id", "unknown"),
            created_at=datetime.now().isoformat(),
            sleep_issues=issues,
            primary_modality=primary,
            secondary_modalities=secondary,
            music_params=music_params,
            video_params=video_params,
            breathing_params=breathing_params,
            duration_total=total_duration,
            schedule={
                "frequency": "每晚睡前",
                "duration_weeks": 4,
                "adjustment": "每周评估"
            },
            expected_efficacy=0.0,  # 稍后计算
            notes="建议在安静环境中使用，保持规律作息"
        )
        
        # 计算疗效
        prescription.expected_efficacy = self.calculate_efficacy(prescription)
        
        return prescription

def run_prescription_test():
    """运行处方系统测试"""
    print("《心境流转》处方系统测试")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 创建处方引擎
    engine = PrescriptionEngine()
    
    # 测试用户数据
    test_users = [
        {
            "user_id": "test_001",
            "age": 25,
            "gender": "male",
            "sleep_issues": ["入睡困难", "浅睡眠"],
            "preferences": {
                "therapy_types": "music",
                "session_duration": 30
            },
            "sensitivity": "normal"
        },
        {
            "user_id": "test_002",
            "age": 35,
            "gender": "female",
            "sleep_issues": ["频繁醒来", "噩梦"],
            "preferences": {
                "therapy_types": "visual,music",
                "session_duration": 25
            },
            "sensitivity": "high"
        },
        {
            "user_id": "test_003",
            "age": 42,
            "gender": "male",
            "sleep_issues": ["失眠", "早醒", "入睡困难"],
            "preferences": {
                "therapy_types": "all",
                "session_duration": 40
            },
            "sensitivity": "low"
        }
    ]
    
    prescriptions = []
    
    # 为每个测试用户生成处方
    for user_data in test_users:
        print(f"\n{'='*40}")
        print(f"👤 用户: {user_data['user_id']}")
        print(f"睡眠问题: {', '.join(user_data['sleep_issues'])}")
        
        # 生成处方
        prescription = engine.generate_prescription(user_data)
        prescriptions.append(prescription)
        
        # 显示处方
        display_prescription(prescription)
    
    # 统计分析
    print_statistics(prescriptions)
    
    # 保存结果
    save_prescriptions(prescriptions)
    
    return prescriptions

def display_prescription(prescription: TherapyPrescription):
    """显示处方详情"""
    print(f"\n📋 处方ID: {prescription.prescription_id}")
    print(f"主要治疗: {prescription.primary_modality.value}")
    
    if prescription.secondary_modalities:
        secondary_str = ", ".join([m.value for m in prescription.secondary_modalities])
        print(f"辅助治疗: {secondary_str}")
    
    print(f"总时长: {prescription.duration_total}分钟")
    print(f"预期疗效: {prescription.expected_efficacy:.1%}")
    
    # 显示具体参数
    if prescription.music_params:
        print(f"\n🎵 音乐参数:")
        print(f"  - 节奏: {prescription.music_params.tempo_range[0]}-{prescription.music_params.tempo_range[1]} BPM")
        print(f"  - 乐器: {', '.join(prescription.music_params.instruments)}")
        print(f"  - 音量: {prescription.music_params.volume_profile}")
    
    if prescription.video_params:
        print(f"\n🎬 视觉参数:")
        print(f"  - 内容: {prescription.video_params.content_type}")
        print(f"  - 色彩: {', '.join(prescription.video_params.color_palette[:3])}")
        print(f"  - 亮度: {prescription.video_params.brightness_profile}")
    
    if prescription.breathing_params:
        print(f"\n🫁 呼吸参数:")
        print(f"  - 模式: {prescription.breathing_params.pattern}")
        print(f"  - 循环: {prescription.breathing_params.cycles}次")
        print(f"  - 引导: {prescription.breathing_params.guidance_voice}")
    
    print(f"\n📅 治疗计划: {prescription.schedule['frequency']}, 持续{prescription.schedule['duration_weeks']}周")

def print_statistics(prescriptions: List[TherapyPrescription]):
    """打印统计信息"""
    print(f"\n{'='*50}")
    print("📊 处方统计")
    print("=" * 50)
    
    print(f"生成处方数: {len(prescriptions)}")
    
    # 模态统计
    modality_count = {}
    for p in prescriptions:
        modality_count[p.primary_modality.value] = modality_count.get(p.primary_modality.value, 0) + 1
    
    print("\n主要治疗模态分布:")
    for modality, count in modality_count.items():
        print(f"  - {modality}: {count}次 ({count/len(prescriptions)*100:.0f}%)")
    
    # 疗效统计
    efficacies = [p.expected_efficacy for p in prescriptions]
    avg_efficacy = sum(efficacies) / len(efficacies)
    max_efficacy = max(efficacies)
    min_efficacy = min(efficacies)
    
    print(f"\n预期疗效:")
    print(f"  - 平均: {avg_efficacy:.1%}")
    print(f"  - 最高: {max_efficacy:.1%}")
    print(f"  - 最低: {min_efficacy:.1%}")
    
    # 时长统计
    durations = [p.duration_total for p in prescriptions]
    avg_duration = sum(durations) / len(durations)
    
    print(f"\n治疗时长:")
    print(f"  - 平均: {avg_duration:.0f}分钟")
    print(f"  - 范围: {min(durations)}-{max(durations)}分钟")

def save_prescriptions(prescriptions: List[TherapyPrescription]):
    """保存处方数据"""
    output_dir = Path("outputs/prescriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化格式
    prescriptions_data = []
    for p in prescriptions:
        data = {
            "prescription_id": p.prescription_id,
            "user_id": p.user_id,
            "created_at": p.created_at,
            "sleep_issues": [issue.value for issue in p.sleep_issues],
            "primary_modality": p.primary_modality.value,
            "secondary_modalities": [m.value for m in p.secondary_modalities],
            "duration_total": p.duration_total,
            "schedule": p.schedule,
            "expected_efficacy": p.expected_efficacy,
            "notes": p.notes
        }
        
        # 添加各模态参数
        if p.music_params:
            data["music_params"] = asdict(p.music_params)
        if p.video_params:
            data["video_params"] = asdict(p.video_params)
        if p.breathing_params:
            data["breathing_params"] = asdict(p.breathing_params)
        
        prescriptions_data.append(data)
    
    # 保存文件
    output_file = output_dir / f"prescriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prescriptions_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 处方数据已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行测试
        prescriptions = run_prescription_test()
        
        # 生成建议
        print("\n💡 系统建议")
        print("-" * 40)
        print("1. 定期评估处方效果，根据反馈调整")
        print("2. 结合用户睡眠数据优化参数")
        print("3. 建立处方效果追踪机制")
        print("4. 考虑季节和环境因素的影响")
        
        print("\n" + "=" * 50)
        print("处方系统测试完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 06_music_generation_workshop.py")
        
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()