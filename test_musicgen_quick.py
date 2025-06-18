#!/usr/bin/env python3
"""
MusicGen快速测试脚本
验证安装是否成功以及基本功能
"""

import sys
import time

def test_imports():
    """测试基础导入"""
    print("🔍 测试基础导入...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import torchaudio
        print(f"✅ torchaudio: {torchaudio.__version__}")
        
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")
        return False

def test_gpu():
    """测试GPU状态"""
    print("\n🔍 测试GPU状态...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 个GPU")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            # 测试GPU操作
            x = torch.rand(100, 100).cuda()
            y = torch.mm(x, x)
            print("✅ GPU计算测试通过")
            
            return True
        else:
            print("⚠️ 未检测到CUDA GPU，将使用CPU模式")
            return True
            
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def test_audiocraft():
    """测试AudioCraft导入"""
    print("\n🔍 测试AudioCraft...")
    
    try:
        import audiocraft
        print(f"✅ AudioCraft版本: {audiocraft.__version__}")
        
        from audiocraft.models import MusicGen
        print("✅ MusicGen导入成功")
        
        return True
    except Exception as e:
        print(f"❌ AudioCraft测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载（small模型）"""
    print("\n🔍 测试模型加载...")
    
    try:
        from audiocraft.models import MusicGen
        
        print("📦 加载MusicGen-small模型（首次使用会下载模型）...")
        start_time = time.time()
        
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        load_time = time.time() - start_time
        
        print(f"✅ 模型加载成功! 耗时: {load_time:.1f}秒")
        
        # 检查模型设备
        device = next(model.parameters()).device
        print(f"📍 模型设备: {device}")
        
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_generation(model):
    """测试音乐生成"""
    print("\n🔍 测试音乐生成...")
    
    try:
        # 设置生成参数（短时间测试）
        model.set_generation_params(duration=3)  # 3秒测试
        
        # 测试prompt
        descriptions = ['peaceful ambient music for relaxation']
        
        print("🎵 生成3秒测试音乐...")
        start_time = time.time()
        
        wav = model.generate(descriptions)
        generation_time = time.time() - start_time
        
        print(f"✅ 音乐生成成功!")
        print(f"  - 生成时间: {generation_time:.1f}秒")
        print(f"  - 音频形状: {wav.shape}")
        print(f"  - 采样率: {model.sample_rate}Hz")
        
        return True
    except Exception as e:
        print(f"❌ 音乐生成失败: {e}")
        return False

def main():
    """主测试流程"""
    print("🎼 MusicGen安装验证测试")
    print("=" * 50)
    
    # 1. 基础导入测试
    if not test_imports():
        print("\n❌ 基础导入失败，请检查PyTorch安装")
        return False
    
    # 2. GPU测试
    if not test_gpu():
        print("\n❌ GPU测试失败")
        return False
    
    # 3. AudioCraft测试
    if not test_audiocraft():
        print("\n❌ AudioCraft测试失败，请检查audiocraft安装")
        return False
    
    # 4. 模型加载测试
    model = test_model_loading()
    if model is None:
        print("\n❌ 模型加载失败")
        return False
    
    # 5. 生成测试
    if not test_generation(model):
        print("\n❌ 音乐生成测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！MusicGen安装成功！")
    print("\n📋 下一步:")
    print("1. 运行: python web_demo.py --sota")
    print("2. 或者: python web_demo.py --enhanced --sota")
    print("\n💡 首次使用时会下载更大的模型，请耐心等待")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n🛠️ 故障排除建议:")
        print("1. 确保所有依赖都已安装: pip install torchvision==0.16.0")
        print("2. 重启Python环境")
        print("3. 检查网络连接（模型下载需要网络）")
        print("4. 查看详细安装指南: INSTALL_MANUAL.md")
    
    sys.exit(0 if success else 1)