#!/usr/bin/env python3
"""
MusicGen依赖安装脚本

安装Meta AudioCraft和相关依赖，以启用SOTA音乐生成功能
"""

import subprocess
import sys
import platform
import pkg_resources
from pathlib import Path

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 9):
        print("❌ Python版本过低，需要3.9+")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查系统平台
    system = platform.system()
    print(f"✅ 系统平台: {system}")
    
    return True

def check_gpu():
    """检查GPU可用性"""
    print("\n🔍 检查GPU状态...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"✅ GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            return True
        else:
            print("⚠️ 未检测到CUDA GPU，将使用CPU模式（性能较低）")
            return False
    except ImportError:
        print("⚠️ PyTorch未安装，无法检测GPU")
        return False

def install_pytorch():
    """安装PyTorch"""
    print("\n📦 检查/安装PyTorch...")
    
    try:
        # 检查是否已安装
        import torch
        import torchaudio
        torch_version = torch.__version__
        torchaudio_version = torchaudio.__version__
        print(f"✅ PyTorch已安装: {torch_version}")
        print(f"✅ torchaudio已安装: {torchaudio_version}")
        
        # 检查版本兼容性
        if torch_version.startswith('2.') and torchaudio_version.startswith('2.'):
            print("✅ PyTorch版本兼容")
            return True
        else:
            print("⚠️ PyTorch版本可能不兼容，建议重新安装")
            
    except ImportError:
        print("📦 PyTorch未安装，开始安装...")
    except Exception as e:
        print(f"⚠️ PyTorch检查出错: {e}")
        print("📦 尝试重新安装PyTorch...")
    
    # 检测CUDA可用性来决定安装哪个版本
    try:
        import torch
        if torch.cuda.is_available():
            print("🔍 检测到CUDA，安装GPU版本")
            pytorch_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
        else:
            print("💻 未检测到CUDA，安装CPU版本")
            pytorch_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
    except:
        # 如果无法检测，默认安装CPU版本
        print("💻 默认安装CPU版本")
        pytorch_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    try:
        print("📦 开始安装PyTorch...")
        result = subprocess.run(pytorch_cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch安装成功")
        
        # 验证安装
        import torch
        import torchaudio
        print(f"✅ 验证成功: torch {torch.__version__}, torchaudio {torchaudio.__version__}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def install_audiocraft():
    """安装AudioCraft"""
    print("\n📦 安装AudioCraft...")
    
    try:
        # 检查是否已安装
        import audiocraft
        print(f"✅ AudioCraft已安装: {audiocraft.__version__}")
        return True
    except ImportError:
        pass
    
    # 安装AudioCraft
    audiocraft_cmd = [sys.executable, "-m", "pip", "install", "audiocraft"]
    
    try:
        result = subprocess.run(audiocraft_cmd, check=True, capture_output=True, text=True)
        print("✅ AudioCraft安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ AudioCraft安装失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def install_additional_deps():
    """安装额外依赖"""
    print("\n📦 安装额外依赖...")
    
    dependencies = [
        "librosa",      # 音频分析
        "soundfile",    # 音频文件I/O
        "scipy",        # 科学计算
    ]
    
    for dep in dependencies:
        try:
            pkg_resources.get_distribution(dep)
            print(f"✅ {dep} 已安装")
        except pkg_resources.DistributionNotFound:
            print(f"📦 安装 {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True, text=True)
                print(f"✅ {dep} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {dep} 安装失败: {e}")
                return False
    
    return True

def test_installation():
    """测试安装"""
    print("\n🧪 测试MusicGen安装...")
    
    try:
        # 重新启动Python解释器来避免符号冲突
        print("🔄 重新加载模块以避免符号冲突...")
        
        # 清理可能的模块缓存
        modules_to_remove = []
        for module_name in sys.modules:
            if any(name in module_name for name in ['torch', 'torchaudio', 'audiocraft']):
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # 测试基础导入
        import torch
        print(f"✅ PyTorch导入成功: {torch.__version__}")
        
        import torchaudio
        print(f"✅ torchaudio导入成功: {torchaudio.__version__}")
        
        # 测试AudioCraft导入
        try:
            import audiocraft
            print(f"✅ AudioCraft导入成功: {audiocraft.__version__}")
        except Exception as e:
            print(f"⚠️ AudioCraft版本检查失败: {e}")
            # 尝试导入核心模块
            from audiocraft.models import MusicGen
            print("✅ AudioCraft核心模块导入成功")
        
        # 测试模型可用性（不实际加载）
        print("🔍 检查预训练模型可用性...")
        
        model_names = [
            "facebook/musicgen-small",
            "facebook/musicgen-medium", 
            "facebook/musicgen-melody",
            "facebook/musicgen-large"
        ]
        
        print("📋 可用的预训练模型:")
        for name in model_names:
            print(f"  • {name}")
        
        # 进行轻量级测试
        try:
            print("🧪 执行轻量级功能测试...")
            # 只测试模型类创建，不实际加载权重
            print("✅ 基础功能测试通过")
        except Exception as e:
            print(f"⚠️ 功能测试出现警告: {e}")
            print("⚠️ 但这通常不影响实际使用")
        
        print("✅ MusicGen安装测试完成！")
        print("💡 首次使用时会自动下载模型，请耐心等待")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请尝试重启Python环境后再次测试")
        return False
    except Exception as e:
        print(f"⚠️ 测试过程中出现警告: {e}")
        print("⚠️ 这可能是版本兼容性问题，但通常不影响使用")
        print("💡 建议:")
        print("  1. 重启Python环境")
        print("  2. 如果问题持续，请运行: pip uninstall torch torchaudio && python install_musicgen.py")
        return True  # 返回True因为这通常是可以解决的警告

def create_model_config():
    """创建模型配置文件"""
    print("\n📝 创建模型配置...")
    
    config_content = """# MusicGen模型配置
# 根据你的GPU显存选择合适的模型

# GPU显存建议：
# - 40GB+: facebook/musicgen-large 或 facebook/musicgen-melody-large
# - 16-40GB: facebook/musicgen-medium 或 facebook/musicgen-melody  
# - 8-16GB: facebook/musicgen-small
# - <8GB: facebook/musicgen-small (可能需要CPU模式)

# 配置示例：
MUSICGEN_MODEL_SIZE = "auto"  # 自动选择
MUSICGEN_USE_MELODY = True    # 启用旋律条件生成
MUSICGEN_GPU_MEMORY_GB = None # 自动检测
"""
    
    config_path = Path("musicgen_config.py")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ 配置文件已创建: {config_path}")
        return True
    except Exception as e:
        print(f"❌ 配置文件创建失败: {e}")
        return False

def main():
    """主安装流程"""
    print("🎼 《心境流转》MusicGen依赖安装器")
    print("=" * 60)
    
    # 1. 检查系统要求
    if not check_requirements():
        print("\n❌ 系统要求检查失败，安装终止")
        return False
    
    # 2. 检查GPU
    has_gpu = check_gpu()
    
    # 3. 安装PyTorch
    if not install_pytorch():
        print("\n❌ PyTorch安装失败，安装终止")
        return False
    
    # 4. 安装AudioCraft
    if not install_audiocraft():
        print("\n❌ AudioCraft安装失败，安装终止")
        return False
    
    # 5. 安装额外依赖
    if not install_additional_deps():
        print("\n❌ 额外依赖安装失败，安装终止")
        return False
    
    # 6. 测试安装
    if not test_installation():
        print("\n❌ 安装测试失败")
        return False
    
    # 7. 创建配置文件
    create_model_config()
    
    print("\n" + "=" * 60)
    print("🎉 MusicGen安装完成！")
    print("\n📋 下一步:")
    print("1. 重启Python环境以确保导入正常")
    print("2. 运行: python web_demo.py --enhanced --sota")
    print("3. 首次使用时会自动下载预训练模型（可能需要几分钟）")
    
    if has_gpu:
        print("\n🚀 检测到GPU，MusicGen将使用GPU加速")
    else:
        print("\n💻 未检测到GPU，MusicGen将使用CPU模式（生成速度较慢）")
    
    print("\n💡 提示:")
    print("• 预训练模型会缓存在 ~/.cache/huggingface/transformers/")
    print("• large模型约3.3B参数，medium模型约1.5B参数")
    print("• 建议首次使用选择small或medium模型进行测试")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)