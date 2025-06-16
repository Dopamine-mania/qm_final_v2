#!/usr/bin/env python3
"""
01 - 系统初始化检查
检查《心境流转》系统的运行环境和依赖
"""

import sys
import platform
from datetime import datetime
import subprocess
import warnings
warnings.filterwarnings('ignore')

def print_header():
    """打印头部信息"""
    print("《心境流转》系统初始化检查")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_python():
    """检查Python环境"""
    print("1. Python环境检查")
    print("-" * 30)
    
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    print(f"Python路径: {sys.executable}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python版本符合要求 (>= 3.8)")
    else:
        print("❌ Python版本过低，需要 >= 3.8")
    
    print()

def check_platform():
    """检查系统平台"""
    print("2. 系统平台检查")
    print("-" * 30)
    
    print(f"操作系统: {platform.system()}")
    print(f"系统版本: {platform.version()}")
    print(f"机器类型: {platform.machine()}")
    print(f"处理器: {platform.processor()}")
    
    print()

def check_gpu():
    """检查GPU环境"""
    print("3. GPU环境检查")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            
    except ImportError:
        print("❌ PyTorch未安装")
    
    print()

def check_dependencies():
    """检查主要依赖"""
    print("4. 核心依赖检查")
    print("-" * 30)
    
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'transformers': 'Transformers',
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'torchvision': 'TorchVision',
        'diffusers': 'Diffusers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'safetensors': 'SafeTensors',
        'scipy': 'SciPy',
        'librosa': 'Librosa',
        'moviepy': 'MoviePy',
        'opencv-cv2': 'OpenCV',
        'gradio': 'Gradio',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn'
    }
    
    installed = []
    missing = []
    
    for package, name in dependencies.items():
        try:
            if package == 'opencv-cv2':
                __import__('cv2')
            else:
                __import__(package)
            installed.append(name)
        except ImportError:
            missing.append(name)
    
    print(f"✅ 已安装: {len(installed)}个")
    for pkg in installed[:5]:  # 只显示前5个
        print(f"  - {pkg}")
    if len(installed) > 5:
        print(f"  ... 还有{len(installed)-5}个")
    
    if missing:
        print(f"\n❌ 缺失: {len(missing)}个")
        for pkg in missing:
            print(f"  - {pkg}")
    
    print()

def check_models():
    """检查模型文件"""
    print("5. 模型文件检查")
    print("-" * 30)
    
    import os
    from pathlib import Path
    
    # 检查预训练模型目录
    model_dir = Path("../data/pretrained_models")
    
    if model_dir.exists():
        print(f"✅ 模型目录存在: {model_dir}")
        
        # 列出模型文件
        model_files = list(model_dir.glob("**/*.safetensors")) + \
                      list(model_dir.glob("**/*.bin")) + \
                      list(model_dir.glob("**/*.pt"))
        
        if model_files:
            print(f"找到 {len(model_files)} 个模型文件")
            for f in model_files[:3]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
            if len(model_files) > 3:
                print(f"  ... 还有{len(model_files)-3}个")
        else:
            print("⚠️ 未找到模型文件")
    else:
        print("⚠️ 模型目录不存在")
        print("  建议创建: data/pretrained_models/")
    
    print()

def check_outputs():
    """检查输出目录"""
    print("6. 输出目录检查")
    print("-" * 30)
    
    from pathlib import Path
    
    output_dirs = [
        "outputs",
        "outputs/validation",
        "outputs/generation",
        "outputs/evaluation"
    ]
    
    for dir_path in output_dirs:
        dir_obj = Path(dir_path)
        if not dir_obj.exists():
            dir_obj.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {dir_path}")
        else:
            print(f"✅ 目录存在: {dir_path}")
    
    print()

def generate_summary():
    """生成总结报告"""
    print("=" * 50)
    print("初始化检查完成！")
    print("=" * 50)
    
    import json
    from pathlib import Path
    
    # 保存检查结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "status": "ready"
    }
    
    output_file = Path("outputs/system_check_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 检查结果已保存: {output_file}")
    print(f"\n🚀 下一步: 运行 02_theory_models_demo.py")
    print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """主函数"""
    print_header()
    check_python()
    check_platform()
    check_gpu()
    check_dependencies()
    check_models()
    check_outputs()
    generate_summary()

if __name__ == "__main__":
    main()