#!/usr/bin/env python3
"""
测试增强模块功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.emotion_recognition.enhanced_emotion_recognizer import create_emotion_recognizer
from src.therapy_planning.enhanced_iso_planner import create_iso_planner
from src.music_mapping.enhanced_music_mapper import create_music_mapper

def test_emotion_recognition():
    """测试情绪识别模块"""
    print("\n=== 测试情绪识别模块 ===")
    
    recognizer = create_emotion_recognizer(use_advanced=False)
    
    test_texts = [
        "最近压力很大，晚上总是睡不着",
        "今天心情特别好，感觉充满希望",
        "有点累，但是心情还不错",
        "很生气，什么都不顺心"
    ]
    
    for text in test_texts:
        emotion = recognizer.recognize(text)
        print(f"\n输入: {text}")
        print(f"识别结果: {emotion.primary_emotion} (V={emotion.valence:.2f}, A={emotion.arousal:.2f})")
        print(f"置信度: {emotion.confidence:.2%}, 强度: {emotion.intensity:.2%}")

def test_therapy_planning():
    """测试治疗规划模块"""
    print("\n\n=== 测试治疗规划模块 ===")
    
    planner = create_iso_planner(enhanced=True)
    
    # 测试不同的起始情绪
    test_cases = [
        ("高唤醒负面（焦虑）", (-0.6, 0.8)),
        ("低唤醒负面（抑郁）", (-0.8, -0.5)),
        ("高唤醒正面（兴奋）", (0.7, 0.8))
    ]
    
    for name, start_emotion in test_cases:
        print(f"\n{name}: V={start_emotion[0]}, A={start_emotion[1]}")
        
        # 创建情绪对象
        emotion_obj = type('EmotionState', (), {
            'valence': start_emotion[0],
            'arousal': start_emotion[1]
        })()
        
        target_obj = type('EmotionState', (), {
            'valence': 0.3,
            'arousal': -0.8
        })()
        
        stages = planner.plan_stages(emotion_obj, target_obj, 20)
        
        for i, stage in enumerate(stages):
            print(f"  阶段{i+1}: {stage['stage'].value} - {stage['duration']:.1f}分钟 "
                  f"(V={stage['emotion'].valence:.2f}, A={stage['emotion'].arousal:.2f})")

def test_music_mapping():
    """测试音乐映射模块"""
    print("\n\n=== 测试音乐映射模块 ===")
    
    mapper = create_music_mapper(enhanced=True, sleep_optimized=True)
    
    test_emotions = [
        ("焦虑状态", -0.6, 0.8),
        ("平静状态", 0.2, -0.3),
        ("悲伤状态", -0.8, -0.5)
    ]
    
    for name, valence, arousal in test_emotions:
        print(f"\n{name}: V={valence}, A={arousal}")
        params = mapper.get_music_params(valence, arousal)
        
        print(f"  BPM: {params.get('bpm', 'N/A')}")
        print(f"  调性: {params.get('key', 'N/A')}")
        print(f"  乐器: {params.get('instruments', [])[:3]}")  # 显示前3个

def test_integration():
    """测试模块集成"""
    print("\n\n=== 测试模块集成 ===")
    
    try:
        from mood_flow_app import MoodFlowApp
        
        # 创建启用增强模块的应用
        app = MoodFlowApp(use_enhanced_modules=True)
        
        if hasattr(app, 'get_enhancement_status'):
            status = app.get_enhancement_status()
            print("\n增强模块状态:")
            for module, enabled in status.items():
                print(f"  {module}: {'✅ 已启用' if enabled else '❌ 未启用'}")
        else:
            print("⚠️ 增强模块未正确集成")
            
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")

def main():
    """运行所有测试"""
    print("🧪 心境流转增强模块测试")
    print("=" * 50)
    
    try:
        test_emotion_recognition()
        test_therapy_planning()
        test_music_mapping()
        test_integration()
        
        print("\n\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"\n\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()