#!/usr/bin/env python3
"""
对比基础模式和增强模式的差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mood_flow_app import MoodFlowApp

def test_emotion_recognition(text):
    """对比情绪识别结果"""
    print(f"\n🔍 测试文本: '{text}'")
    print("-" * 60)
    
    # 基础版
    print("\n📌 基础版结果:")
    app_basic = MoodFlowApp(use_enhanced_modules=False)
    emotion_basic = app_basic.analyze_emotion_from_text(text)
    print(f"  V-A值: V={emotion_basic.valence:.2f}, A={emotion_basic.arousal:.2f}")
    
    # 增强版
    print("\n📌 增强版结果:")
    app_enhanced = MoodFlowApp(use_enhanced_modules=True)
    emotion_enhanced = app_enhanced.analyze_emotion_from_text(text)
    print(f"  V-A值: V={emotion_enhanced.valence:.2f}, A={emotion_enhanced.arousal:.2f}")
    
    # 获取详细信息（如果有）
    if hasattr(app_enhanced, 'get_detailed_emotion_info'):
        detailed = app_enhanced.get_detailed_emotion_info(emotion_enhanced)
        if detailed:
            print(f"\n  🎯 增强版额外信息:")
            print(f"    - 细粒度情绪: {detailed['primary_emotion_cn']} ({detailed['primary_emotion']})")
            print(f"    - 置信度: {detailed['confidence']:.1%}")
            print(f"    - 强度: {detailed['intensity']:.1%}")

def test_therapy_planning():
    """对比治疗规划"""
    print("\n\n🎯 治疗规划对比")
    print("=" * 60)
    
    # 创建测试情绪状态
    test_emotion = type('EmotionState', (), {'valence': -0.6, 'arousal': 0.8})()
    
    # 基础版
    print("\n📌 基础版规划:")
    app_basic = MoodFlowApp(use_enhanced_modules=False)
    stages_basic = app_basic.plan_therapy_stages(test_emotion, 20)
    for i, stage in enumerate(stages_basic):
        print(f"  阶段{i+1}: {stage['stage'].value} - {stage['duration']:.1f}分钟")
    
    # 增强版
    print("\n📌 增强版规划:")
    app_enhanced = MoodFlowApp(use_enhanced_modules=True)
    stages_enhanced = app_enhanced.plan_therapy_stages(test_emotion, 20)
    # 增强版会在后台打印详细信息

def main():
    print("🔬 心境流转系统 - 基础版 vs 增强版对比")
    print("=" * 80)
    
    # 测试不同的输入
    test_texts = [
        "最近压力很大，晚上总是睡不着",
        "心情很低落，什么都不想做",
        "有点兴奋睡不着，脑子里想着很多事情"
    ]
    
    for text in test_texts:
        test_emotion_recognition(text)
    
    test_therapy_planning()
    
    print("\n\n📊 主要差异总结:")
    print("-" * 60)
    print("1. 情绪识别:")
    print("   - 基础版: 简单关键词匹配，固定V-A映射")
    print("   - 增强版: 9种细粒度分类，包含置信度和强度")
    print("\n2. 治疗规划:")
    print("   - 基础版: 固定25%-50%-25%时长分配")
    print("   - 增强版: 根据情绪状态动态调整，整合Gross模型")
    print("\n3. 音乐映射:")
    print("   - 基础版: 简单线性映射")
    print("   - 增强版: 多维度精准映射，睡眠场景优化")

if __name__ == "__main__":
    main()