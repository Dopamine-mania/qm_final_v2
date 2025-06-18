# 🛠️ MusicGen手动安装指南

如果自动安装脚本遇到问题，请按照此手动安装指南进行。

## 🚨 常见问题解决

### 问题1: torchaudio符号冲突
```
undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorE...
```

**解决方案：**

1. **完全卸载现有PyTorch**：
```bash
pip uninstall torch torchaudio torchvision
pip cache purge
```

2. **重新安装兼容版本**：
```bash
# 对于CUDA 11.8 (推荐)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 或者对于CUDA 12.1
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 或者CPU版本（如果GPU有问题）
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

3. **安装AudioCraft**：
```bash
pip install audiocraft
```

4. **验证安装**：
```python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
python -c "from audiocraft.models import MusicGen; print('AudioCraft: OK')"
```

## 📋 完整安装流程

### Step 1: 环境准备
```bash
# 确保Python 3.9+
python --version

# 创建新的conda环境（推荐）
conda create -n musicgen python=3.11
conda activate musicgen
```

### Step 2: 安装PyTorch
```bash
# 检查CUDA版本
nvidia-smi

# 根据CUDA版本安装（选择一个）：

# CUDA 11.8
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: 安装依赖
```bash
# 安装AudioCraft
pip install audiocraft

# 安装额外依赖
pip install librosa soundfile scipy
```

### Step 4: 验证安装
```bash
python -c "
import torch
import torchaudio
from audiocraft.models import MusicGen
print('✅ 所有组件安装成功!')
print(f'PyTorch: {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
"
```

## 🎯 测试MusicGen

创建测试文件 `test_musicgen.py`：

```python
#!/usr/bin/env python3
import torch
from audiocraft.models import MusicGen

print("🎼 测试MusicGen...")

# 检查GPU
if torch.cuda.is_available():
    print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"📊 GPU显存: {memory_gb:.1f}GB")
else:
    print("💻 使用CPU模式")

# 测试模型加载（small模型，快速测试）
try:
    print("📦 加载MusicGen-small模型...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    print("✅ 模型加载成功!")
    
    # 简单生成测试
    model.set_generation_params(duration=5)
    descriptions = ['relaxing ambient music for sleep']
    
    print("🎵 生成测试音乐...")
    wav = model.generate(descriptions)
    print(f"✅ 音乐生成成功! 形状: {wav.shape}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("💡 这可能需要重启Python环境")
```

运行测试：
```bash
python test_musicgen.py
```

## 🔧 故障排除

### 问题2: 网络连接问题
如果下载模型失败：

```bash
# 设置镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ audiocraft

# 或手动下载模型到缓存
export HF_HOME=~/.cache/huggingface
huggingface-cli download facebook/musicgen-small
```

### 问题3: 内存不足
如果GPU内存不足：

```python
# 在Python中设置
import torch
torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存
```

### 问题4: 模型下载慢
首次使用会下载模型，大小：
- small: ~1.2GB  
- medium: ~6GB
- large: ~13GB

可以预先下载：
```bash
python -c "
from audiocraft.models import MusicGen
print('下载small模型...')
MusicGen.get_pretrained('facebook/musicgen-small')
print('✅ 下载完成!')
"
```

## 📱 验证心境流转系统

安装完成后，测试完整系统：

```bash
# 测试基础模式
python web_demo.py

# 测试SOTA模式  
python web_demo.py --sota

# 测试完整模式
python web_demo.py --enhanced --sota
```

## 💡 性能优化建议

### GPU优化
- 确保CUDA驱动最新
- 关闭其他GPU占用程序
- 使用nvidia-smi监控显存使用

### 系统优化  
- 增加虚拟内存
- 确保足够磁盘空间（模型缓存）
- 使用SSD存储模型缓存

## 📞 获取帮助

如果仍有问题：

1. **检查环境**：
```bash
python -c "
import sys, torch, torchaudio
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')  
print(f'torchaudio: {torchaudio.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

2. **查看详细错误**：
```bash
python -c "from audiocraft.models import MusicGen" 2>&1 | tee error.log
```

3. **重新开始**：
```bash
# 完全清理环境
conda deactivate
conda remove -n musicgen --all
# 然后重新按照Step 1开始
```

希望这个指南能帮你成功安装MusicGen！🎉