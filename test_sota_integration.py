#!/usr/bin/env python3
"""
测试SOTA模式的完整集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complete_flow():
    """测试完整的SOTA音乐生成流程"""
    print("🎼 测试SOTA模式完整流程...")
    
    try:
        from mood_flow_app import MoodFlowApp
        
        # 创建应用实例（SOTA模式）
        print("📦 初始化应用（SOTA模式）...")
        app = MoodFlowApp(use_enhanced_modules=True, enhancement_config='full_with_sota')
        
        # 检查SOTA状态
        if hasattr(app, 'get_enhancement_status'):
            status = app.get_enhancement_status()
            print(f"✅ 增强状态: {status}")
            
            if not status.get('sota_music_generation', False):
                print("⚠️ SOTA音乐生成未启用")
                return False
        
        # 运行简短的治疗会话
        print("🎵 运行测试治疗会话...")
        test_input = "感到有点焦虑，需要放松"
        
        session = app.run_therapy_session(
            user_input=test_input,
            duration=1,  # 1分钟测试
            create_full_videos=False
        )
        
        print(f"✅ 会话完成！")
        print(f"  - 音乐文件: {session.music_file}")
        print(f"  - 检测情绪: V={session.detected_emotion.valence:.2f}, A={session.detected_emotion.arousal:.2f}")
        
        # 检查音频文件
        import os
        if os.path.exists(session.music_file):
            file_size = os.path.getsize(session.music_file)
            print(f"  - 文件大小: {file_size/1024:.1f}KB")
            return True
        else:
            print("❌ 音频文件未生成")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_complete_flow():
        print("\n🎉 SOTA集成测试成功！")
        print("重新启动Web界面应该能听到真正的AI生成音乐了！")
    else:
        print("\n❌ SOTA集成有问题，需要修复")