#!/usr/bin/env python3
"""
04 - 治疗会话演示
演示完整的睡眠治疗会话流程
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import random
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    age: int
    gender: str
    sleep_issues: List[str]
    preferences: Dict[str, str]
    history: List[Dict] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []

@dataclass
class EmotionAssessment:
    """情绪评估结果"""
    timestamp: str
    text_emotion: Dict[str, float]
    voice_emotion: Optional[Dict[str, float]] = None
    combined_emotion: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    
    def get_primary_emotion(self) -> str:
        """获取主要情绪"""
        emotions = self.combined_emotion or self.text_emotion
        return max(emotions.items(), key=lambda x: x[1])[0]

class TherapySession:
    """治疗会话管理器"""
    
    def __init__(self, user_profile: UserProfile):
        self.user = user_profile
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.conversation_history = []
        self.emotion_trajectory = []
        self.stage = "initial"
        
    def assess_emotion_from_text(self, text: str) -> Dict[str, float]:
        """从文本评估情绪（模拟）"""
        # 简单的关键词匹配（实际应使用深度学习模型）
        keywords = {
            "焦虑": ["担心", "紧张", "害怕", "焦虑", "不安"],
            "抑郁": ["难过", "沮丧", "失望", "悲伤", "绝望"],
            "愤怒": ["生气", "愤怒", "恼火", "讨厌", "烦躁"],
            "平静": ["平静", "放松", "舒适", "安心", "宁静"],
            "快乐": ["开心", "高兴", "愉快", "满意", "幸福"]
        }
        
        scores = {}
        for emotion, words in keywords.items():
            score = sum(1 for word in words if word in text) * 0.2
            scores[emotion] = min(score, 1.0)
        
        # 归一化
        total = sum(scores.values()) or 1
        return {k: v/total for k, v in scores.items()}
    
    def simulate_voice_emotion(self, text_emotion: Dict[str, float]) -> Dict[str, float]:
        """模拟语音情绪分析"""
        # 添加一些随机变化来模拟语音特征
        voice_emotion = {}
        for emotion, score in text_emotion.items():
            variation = random.uniform(-0.1, 0.1)
            voice_emotion[emotion] = max(0, min(1, score + variation))
        
        # 归一化
        total = sum(voice_emotion.values()) or 1
        return {k: v/total for k, v in voice_emotion.items()}
    
    def combine_emotions(self, text_emotion: Dict[str, float], 
                        voice_emotion: Dict[str, float]) -> Dict[str, float]:
        """融合文本和语音情绪"""
        # 加权平均（文本60%，语音40%）
        combined = {}
        for emotion in text_emotion:
            combined[emotion] = text_emotion[emotion] * 0.6 + voice_emotion.get(emotion, 0) * 0.4
        
        # 归一化
        total = sum(combined.values()) or 1
        return {k: v/total for k, v in combined.items()}
    
    def process_user_input(self, text: str, has_voice: bool = False) -> EmotionAssessment:
        """处理用户输入"""
        print(f"\n👤 用户: {text}")
        
        # 文本情绪分析
        text_emotion = self.assess_emotion_from_text(text)
        
        # 语音情绪分析（如果有）
        voice_emotion = None
        combined_emotion = text_emotion
        
        if has_voice:
            voice_emotion = self.simulate_voice_emotion(text_emotion)
            combined_emotion = self.combine_emotions(text_emotion, voice_emotion)
        
        # 创建评估结果
        assessment = EmotionAssessment(
            timestamp=datetime.now().isoformat(),
            text_emotion=text_emotion,
            voice_emotion=voice_emotion,
            combined_emotion=combined_emotion,
            confidence=0.85 if has_voice else 0.7
        )
        
        # 记录到历史
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "emotion": assessment.get_primary_emotion(),
            "timestamp": assessment.timestamp
        })
        
        self.emotion_trajectory.append(assessment)
        
        return assessment
    
    def generate_response(self, emotion: str) -> str:
        """生成治疗师响应"""
        responses = {
            "焦虑": [
                "我理解您现在感到焦虑。让我们一起通过深呼吸来缓解这种感觉。",
                "焦虑是很常见的情绪。我会帮助您找到适合的放松方法。",
                "让我们先关注当下，慢慢地将注意力转移到呼吸上。"
            ],
            "抑郁": [
                "我能感受到您的情绪低落。请记住，这种感觉是暂时的。",
                "让我们一起寻找一些积极的事物，哪怕是很小的快乐。",
                "您并不孤单，我会陪伴您度过这个困难时期。"
            ],
            "愤怒": [
                "我理解您的愤怒。让我们先冷静下来，再来解决问题。",
                "愤怒往往掩盖着其他情绪。您愿意和我分享更多吗？",
                "深呼吸，让身体放松。愤怒会过去的。"
            ],
            "平静": [
                "很高兴看到您现在比较平静。让我们保持这种状态。",
                "您做得很好！继续保持这种放松的感觉。",
                "平静的心态对睡眠很有帮助。我们可以进一步加深这种感觉。"
            ],
            "快乐": [
                "看到您心情愉快真是太好了！",
                "积极的情绪对睡眠质量有很大帮助。",
                "让我们记住这种美好的感觉，它会帮助您更好地入睡。"
            ]
        }
        
        response = random.choice(responses.get(emotion, ["我在倾听，请继续。"]))
        
        # 记录到历史
        self.conversation_history.append({
            "role": "therapist",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🤖 治疗师: {response}")
        return response
    
    def update_stage(self):
        """更新会话阶段"""
        num_interactions = len(self.conversation_history) // 2
        
        if num_interactions < 3:
            self.stage = "initial"
        elif num_interactions < 6:
            self.stage = "exploration"
        elif num_interactions < 9:
            self.stage = "intervention"
        else:
            self.stage = "closure"
    
    def generate_session_summary(self) -> Dict:
        """生成会话总结"""
        # 分析情绪变化
        emotions = [e.get_primary_emotion() for e in self.emotion_trajectory]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 主导情绪
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # 情绪改善度（模拟）
        initial_negative = sum(1 for e in emotions[:3] if e in ["焦虑", "抑郁", "愤怒"])
        final_negative = sum(1 for e in emotions[-3:] if e in ["焦虑", "抑郁", "愤怒"])
        improvement = max(0, (initial_negative - final_negative) / 3 * 100)
        
        return {
            "session_id": self.session_id,
            "duration": (datetime.now() - self.start_time).total_seconds() / 60,
            "total_interactions": len(self.conversation_history) // 2,
            "emotion_distribution": emotion_counts,
            "dominant_emotion": dominant_emotion,
            "improvement_rate": improvement,
            "final_stage": self.stage
        }

def run_demo_session():
    """运行演示会话"""
    print("《心境流转》治疗会话演示")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 创建用户档案
    user = UserProfile(
        user_id="demo_user_001",
        age=28,
        gender="female",
        sleep_issues=["入睡困难", "多梦", "早醒"],
        preferences={
            "music_style": "轻音乐",
            "video_type": "自然风景",
            "session_duration": "20分钟"
        }
    )
    
    print("\n👤 用户档案")
    print("-" * 40)
    print(f"ID: {user.user_id}")
    print(f"年龄: {user.age}")
    print(f"睡眠问题: {', '.join(user.sleep_issues)}")
    
    # 创建会话
    session = TherapySession(user)
    
    # 模拟对话
    demo_conversations = [
        ("最近总是睡不着，躺在床上脑子里想很多事情，越想越焦虑。", True),
        ("工作压力很大，明天还有重要的会议，我担心睡不好会影响表现。", True),
        ("我试过数羊，但是没用，反而更清醒了。", False),
        ("有时候会突然惊醒，然后就很难再入睡。", True),
        ("我想要放松下来，但不知道该怎么做。", False),
        ("听您这么说，我觉得稍微好一些了。", True),
        ("深呼吸确实让我感觉平静了一些。", False),
        ("我愿意尝试您推荐的音乐疗法。", True),
        ("现在感觉比刚才放松多了，谢谢您。", True)
    ]
    
    print("\n💬 会话过程")
    print("-" * 40)
    
    for text, has_voice in demo_conversations:
        # 处理用户输入
        assessment = session.process_user_input(text, has_voice)
        
        # 显示情绪分析
        primary_emotion = assessment.get_primary_emotion()
        confidence = assessment.confidence
        print(f"   📊 情绪: {primary_emotion} (置信度: {confidence:.1%})")
        
        # 生成响应
        time.sleep(1)  # 模拟思考时间
        session.generate_response(primary_emotion)
        
        # 更新阶段
        session.update_stage()
        
        time.sleep(0.5)  # 模拟对话节奏
    
    # 生成总结
    summary = session.generate_session_summary()
    
    print("\n📋 会话总结")
    print("-" * 40)
    print(f"会话ID: {summary['session_id']}")
    print(f"时长: {summary['duration']:.1f}分钟")
    print(f"交互次数: {summary['total_interactions']}")
    print(f"主导情绪: {summary['dominant_emotion']}")
    print(f"改善率: {summary['improvement_rate']:.1f}%")
    
    print("\n📊 情绪分布:")
    for emotion, count in summary['emotion_distribution'].items():
        percentage = count / summary['total_interactions'] * 100
        print(f"  {emotion}: {count}次 ({percentage:.1f}%)")
    
    # 保存结果
    save_session_results(session, summary)
    
    return session, summary

def save_session_results(session: TherapySession, summary: Dict):
    """保存会话结果"""
    output_dir = Path("outputs/sessions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    session_data = {
        "session_info": {
            "session_id": session.session_id,
            "user_id": session.user.user_id,
            "start_time": session.start_time.isoformat(),
            "user_profile": asdict(session.user)
        },
        "conversation_history": session.conversation_history,
        "emotion_trajectory": [
            {
                "timestamp": e.timestamp,
                "primary_emotion": e.get_primary_emotion(),
                "confidence": e.confidence,
                "details": {
                    "text": e.text_emotion,
                    "voice": e.voice_emotion,
                    "combined": e.combined_emotion
                }
            }
            for e in session.emotion_trajectory
        ],
        "summary": summary
    }
    
    # 保存文件
    output_file = output_dir / f"{session.session_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 会话记录已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行演示
        session, summary = run_demo_session()
        
        # 生成治疗建议
        print("\n💊 治疗建议")
        print("-" * 40)
        
        if summary['improvement_rate'] >= 50:
            print("✅ 会话效果良好，建议：")
            print("  1. 继续使用渐进式放松技术")
            print("  2. 配合轻柔的音乐引导")
            print("  3. 保持规律的睡前准备")
        else:
            print("⚠️ 需要更多支持，建议：")
            print("  1. 增加会话频率")
            print("  2. 尝试多模态干预（音乐+视觉）")
            print("  3. 考虑专业医师评估")
        
        print("\n" + "=" * 50)
        print("治疗会话演示完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 05_prescription_system_test.py")
        
    except Exception as e:
        print(f"\n❌ 演示出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()