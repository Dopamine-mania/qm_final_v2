# 📚 心境流转系统使用指南

## 🚀 快速启动

### 1. 基础模式（原版系统）
```bash
# 终端运行
python mood_flow_app.py

# Web界面
python web_demo.py
```

### 2. 增强模式（理论驱动优化版）
```bash
# 所有增强功能，但不使用SOTA音乐生成
python web_demo.py --enhancement_config=full

# 启用SOTA音乐生成（需要GPU和PyTorch）
python web_demo.py --enhancement_config=full_with_sota
```

### 3. 特定功能测试
```bash
# 仅测试情绪识别增强
python web_demo.py --enhancement_config=emotion_only

# 仅测试治疗规划增强  
python web_demo.py --enhancement_config=planning_only

# 仅测试音乐映射增强
python web_demo.py --enhancement_config=mapping_only

# 仅测试SOTA音乐生成
python web_demo.py --enhancement_config=sota_only
```

## 🎛️ 配置选项详解

| 配置名称 | 情绪识别 | 治疗规划 | 音乐映射 | SOTA生成 | 适用场景 |
|---------|---------|---------|---------|---------|---------|
| `disabled` | ❌ | ❌ | ❌ | ❌ | 基础演示 |
| `emotion_only` | ✅ | ❌ | ❌ | ❌ | 测试情绪识别 |
| `planning_only` | ❌ | ✅ | ❌ | ❌ | 测试治疗规划 |
| `mapping_only` | ❌ | ❌ | ✅ | ❌ | 测试音乐映射 |
| `full` | ✅ | ✅ | ✅ | ❌ | 完整增强（不含SOTA） |
| `full_with_sota` | ✅ | ✅ | ✅ | ✅ | 最强配置（需GPU） |
| `sota_only` | ❌ | ❌ | ❌ | ✅ | 仅测试SOTA生成 |

## 🎤 语音输入使用方法

### Web界面语音输入
1. **启动Web界面**
   ```bash
   python web_demo.py --enhancement_config=full_with_sota
   ```

2. **在Web界面中**
   - 点击"🎤 录音"按钮开始录制
   - 说出你的感受，比如："身心俱疲，但躺下后大脑还是很活跃，总是胡思乱想"
   - 点击停止录音
   - 系统会处理语音并转换为文本

3. **当前语音处理状态**
   - 🎭 **演示模式**: 根据音频文件哈希选择预设的示例文本
   - 🔮 **未来增强**: 可集成专业语音识别服务（如Whisper、百度语音等）

### 代码中启用真实语音识别
```python
# 在web_demo.py的process_voice_input方法中
# 可以替换为真实的语音识别实现：

def process_voice_input(self, audio_file):
    """处理语音输入转换为文字"""
    try:
        # 方案1: 使用OpenAI Whisper
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"], "🎤 语音识别成功"
        
        # 方案2: 使用百度AI语音识别
        # from aip import AipSpeech
        # client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        # ...
        
    except Exception as e:
        return "", f"❌ 语音识别失败: {str(e)}"
```

## 🎼 SOTA音乐生成使用

### 1. 系统要求
- **GPU**: 建议16GB+显存
- **PyTorch**: 已安装且支持CUDA
- **AudioCraft**: Meta的音乐生成库

### 2. 安装依赖
```bash
# 自动安装脚本
python install_musicgen.py

# 手动安装（如果自动失败）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install audiocraft
```

### 3. 测试SOTA功能
```bash
# 快速测试MusicGen可用性
python test_musicgen_quick.py

# 完整集成测试
python test_sota_integration.py
```

### 4. 使用SOTA模式
```python
from mood_flow_app import MoodFlowApp

# 创建SOTA模式应用
app = MoodFlowApp(
    use_enhanced_modules=True,
    enhancement_config='full_with_sota'
)

# 运行治疗会话
session = app.run_therapy_session(
    user_input="身心俱疲，但躺下后大脑还是很活跃，总是胡思乱想",
    duration=5  # 5分钟测试
)
```

## 🧠 增强情绪识别功能

### 支持的细粒度情绪
- **anger** (愤怒): 生气、恼火、讨厌、烦躁
- **fear** (恐惧/焦虑): 害怕、担心、紧张、焦虑、睡不着、胡思乱想
- **disgust** (厌恶): 厌恶、反感、恶心、讨厌
- **sadness** (悲伤): 难过、沮丧、疲惫、身心俱疲、精神不振
- **amusement** (愉悦): 好玩、有趣、搞笑、幽默
- **joy** (喜悦): 开心、快乐、愉快、欢乐
- **inspiration** (灵感): 激励、鼓舞、振奋、充满希望
- **tenderness** (温柔): 温柔、温暖、感动、温馨
- **neutral** (中性): 一般、还好、普通、平常

### 测试情绪识别
```python
from src.emotion_recognition.enhanced_emotion_recognizer import create_emotion_recognizer

recognizer = create_emotion_recognizer()
emotion = recognizer.recognize("身心俱疲，但躺下后大脑还是很活跃，总是胡思乱想")

print(f"主要情绪: {emotion.primary_emotion}")  # fear
print(f"V-A坐标: V={emotion.valence:.2f}, A={emotion.arousal:.2f}")  # V=-0.60, A=0.70
print(f"置信度: {emotion.confidence:.2f}")  # 0.50
```

## 🎵 音乐治疗阶段

### 三阶段ISO原则
1. **同步化阶段** (25%时长)
   - 匹配用户当前情绪
   - 建立情感连接和信任
   - 音乐特征：保持原有节奏和情绪色彩

2. **引导化阶段** (50%时长)  
   - 渐进式情绪过渡
   - 认知重评和放松引导
   - 音乐特征：BPM逐渐降低，和声更柔和

3. **巩固化阶段** (25%时长)
   - 维持低唤醒状态
   - 深化放松，准备入睡
   - 音乐特征：低频为主，极低BPM，睡眠诱导

### 音乐参数映射
- **BPM**: 基于Arousal相关性0.88计算
- **调性**: 基于Valence相关性0.74选择大小调
- **乐器**: 根据情绪类型和治疗阶段选择
- **双耳节拍**: 诱导特定脑电波状态

## 🔧 故障排除

### 常见问题

**Q: 情绪识别不准确？**
A: 使用`enhancement_config=emotion_only`测试，确保关键词匹配正常

**Q: SOTA音乐生成失败？**  
A: 
1. 检查GPU可用性：`nvidia-smi`
2. 检查PyTorch安装：`python -c "import torch; print(torch.cuda.is_available())"`
3. 运行诊断：`python test_musicgen_quick.py`

**Q: 语音输入无反应？**
A: 
1. 检查音频文件权限
2. 确认浏览器麦克风权限
3. 查看控制台错误信息

**Q: Web界面启动失败？**
A: 
1. 检查端口占用：`python check_port.py`
2. 更新Gradio：`pip install --upgrade gradio`
3. 使用简化模式：`python web_demo.py --enhancement_config=disabled`

### 日志调试
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看详细的增强模块状态
app = MoodFlowApp(enhancement_config='full_with_sota')
print(app.get_enhancement_status())
```

## 📊 性能优化建议

1. **GPU内存优化**
   - 小显存：使用`model_size=small`
   - 中等显存：使用`model_size=medium` 
   - 大显存：使用`model_size=large`

2. **CPU模式回退**
   - 自动检测GPU不可用时回退到基础音乐生成
   - 保持完整功能，仅音乐质量略低

3. **缓存机制**
   - MusicGen模型自动缓存，第二次启动更快
   - 预生成音乐可存储复用

## 📞 技术支持

遇到问题可以：
1. 查看项目GitHub Issues
2. 运行测试脚本确认状态
3. 检查系统兼容性和依赖安装