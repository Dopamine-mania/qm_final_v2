#!/usr/bin/env python3
"""
测试情绪识别和SOTA音乐生成修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_emotion_recognition():
    """测试情绪识别修复"""
    print("🧠 测试情绪识别修复...")
    
    try:
        from src.emotion_recognition.enhanced_emotion_recognizer import create_emotion_recognizer
        
        # 创建增强情绪识别器
        recognizer = create_emotion_recognizer(use_advanced=False)
        
        # 测试用户报告的问题文本
        test_text = "身心俱疲，但躺下后大脑还是很活跃，总是胡思乱想"
        
        print(f"📝 测试文本: {test_text}")
        
        # 识别情绪
        emotion = recognizer.recognize(test_text)
        
        print(f"🎯 识别结果:")
        print(f"  - 主要情绪: {emotion.primary_emotion}")
        print(f"  - V-A坐标: V={emotion.valence:.2f}, A={emotion.arousal:.2f}")
        print(f"  - 置信度: {emotion.confidence:.2f}")
        
        # 检查是否正确识别为焦虑相关情绪
        if emotion.primary_emotion in ['fear', 'sadness']:
            print("✅ 情绪识别修复成功！正确识别为焦虑/疲惫相关情绪")
            return True
        else:
            print(f"❌ 情绪识别仍有问题，识别为: {emotion.primary_emotion}")
            return False
            
    except Exception as e:
        print(f"❌ 情绪识别测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_musicgen_availability():
    """测试MusicGen可用性"""
    print("\n🎼 测试MusicGen可用性...")
    
    try:
        from src.model_adapters.musicgen_adapter import create_musicgen_adapter
        
        # 创建MusicGen适配器
        adapter = create_musicgen_adapter(model_size="small")  # 使用小模型测试
        
        if adapter.is_available():
            print("✅ MusicGen模型可用")
            model_info = adapter.get_model_info()
            print(f"  - 模型状态: {model_info['status']}")
            print(f"  - 采样率: {model_info['sample_rate']}Hz")
            print(f"  - GPU显存: {model_info['gpu_memory_gb']:.1f}GB")
            return True
        else:
            print("❌ MusicGen模型不可用")
            return False
            
    except Exception as e:
        print(f"❌ MusicGen测试失败: {e}")
        return False

def test_enhanced_adapter():
    """测试增强适配器的SOTA集成"""
    print("\n🔧 测试增强适配器SOTA集成...")
    
    try:
        from src.enhanced_mood_flow_adapter import EnhancedMoodFlowAdapter
        
        # 创建带SOTA功能的适配器
        adapter = EnhancedMoodFlowAdapter(
            use_enhanced_emotion=True,
            use_enhanced_planning=True,
            use_enhanced_mapping=True,
            use_sota_music_generation=True,
            fallback_to_original=True
        )
        
        status = adapter.get_enhancement_status()
        print(f"📊 增强状态: {status}")
        
        if status.get('sota_music_generation', False):
            print("✅ SOTA音乐生成已启用")
            return True
        else:
            print("⚠️ SOTA音乐生成未启用（可能是依赖问题）")
            return False
            
    except Exception as e:
        print(f"❌ 增强适配器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🔍 开始修复验证测试")
    print("="*60)
    
    results = []
    
    # 测试1: 情绪识别修复
    results.append(test_emotion_recognition())
    
    # 测试2: MusicGen可用性
    results.append(test_musicgen_availability())
    
    # 测试3: 增强适配器
    results.append(test_enhanced_adapter())
    
    print("\n" + "="*60)
    print("📋 测试结果总结:")
    print("="*60)
    
    test_names = [
        "情绪识别修复",
        "MusicGen可用性", 
        "增强适配器SOTA集成"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\n🎯 总体结果: {total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("🎉 所有修复验证成功！")
        return True
    else:
        print("⚠️ 仍有问题需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)