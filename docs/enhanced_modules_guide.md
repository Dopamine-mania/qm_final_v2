# 理论驱动的增强模块使用指南

## 概述

心境流转系统现在支持理论驱动的增强模块，这些模块基于2024年最新的情感计算、音乐治疗和睡眠医学研究成果。增强模块提供了更精准的情绪识别、更科学的治疗路径规划和更细致的音乐特征映射。

## 主要增强功能

### 1. 细粒度情绪识别
- **9种情绪分类**：基于FACED数据集（Nature Scientific Data, 2023）
  - 负面情绪：愤怒(anger)、恐惧(fear)、厌恶(disgust)、悲伤(sadness)
  - 正面情绪：愉悦(amusement)、喜悦(joy)、灵感(inspiration)、温柔(tenderness)
  - 中性(neutral)
- **多模态融合**：支持文本和语音的融合识别（准备中）
- **置信度评估**：每个识别结果都包含置信度和强度信息

### 2. ISO原则治疗规划
- **基于ISO原则**（Altshuler, 1948）的三阶段治疗：
  - 同步化(25%)：匹配用户当前情绪状态
  - 引导化(50%)：渐进式过渡到目标状态
  - 巩固化(25%)：维持并深化睡眠适宜状态
- **整合Gross情绪调节模型**（2015）的五阶段策略
- **自适应规划**：根据情绪距离动态调整阶段时长

### 3. 精准音乐特征映射
- **基于2024年研究**的高精度映射：
  - Tempo与arousal相关性：0.88
  - Mode与valence相关性：0.74
- **多维度音乐特征**：
  - 基础参数：节奏、调性、拍号
  - 音色参数：乐器选择、明亮度
  - 和声参数：和弦进行、不协和度
  - 特殊效果：混响、颤音、双耳节拍
- **睡眠场景优化**：
  - 最大节奏不超过70 BPM
  - 频率范围20-1000 Hz
  - 双耳节拍诱导脑电波同步

## 使用方法

### 命令行启动

```bash
# 使用基础版本（默认）
python web_demo.py

# 启用增强模块
python web_demo.py --enhanced

# 指定端口并启用增强模块
python web_demo.py --enhanced --port 8080

# 不自动打开浏览器
python web_demo.py --enhanced --no-browser
```

### 编程接口

```python
from mood_flow_app import MoodFlowApp

# 创建使用增强模块的实例
app = MoodFlowApp(use_enhanced_modules=True)

# 运行治疗会话
session = app.run_therapy_session(
    user_input="最近压力很大，晚上总是睡不着",
    duration=20  # 分钟
)
```

### Web界面使用

启用增强模块后，界面会显示更丰富的信息：

1. **情绪分析结果**：
   - 基础V-A值（兼容原系统）
   - 细粒度情绪类别（如"恐惧/焦虑"）
   - 识别置信度百分比
   - 情绪强度评分

2. **治疗方案**：
   - 三阶段时长可能根据您的情绪状态动态调整
   - 音乐参数更加精准匹配您的情绪需求

## 理论基础与文献

### 情绪识别
1. **FACED Dataset** (2023): "A Large Finer-grained Affective Computing EEG Dataset"
   - Nature Scientific Data, https://doi.org/10.1038/s41597-023-02650-w

2. **Multimodal Emotion Recognition** (2024): 
   - WIREs Data Mining and Knowledge Discovery, https://doi.org/10.1002/widm.1563

### 音乐治疗
1. **ISO Principle** (2024): 
   - Starcke & von Georgi: "Music listening according to the iso principle modulates affective state"
   - https://doi.org/10.1177/10298649231175029

2. **Music Therapy for Sleep** (2024):
   - "Meta-narrative review: the impact of music therapy on sleep"
   - Frontiers in Neurology, https://doi.org/10.3389/fneur.2024.1433592

### 音乐特征映射
1. **Musical Valence and Arousal** (2024):
   - "Decoding Musical Valence And Arousal"
   - bioRxiv, https://doi.org/10.1101/2024.02.27.582309

2. **Emotion Regulation** (2015):
   - Gross, J.J.: "Emotion regulation: Current status and future prospects"
   - Psychological Inquiry, 26(1), 1-26

## 性能考虑

- 增强模块会略微增加计算开销
- 在40GB GPU上运行流畅
- 如遇到性能问题，可以分别启用/禁用特定增强功能

## 后续开发计划

1. **第二阶段**：集成SOTA音乐生成模型（MusicGen）
2. **第三阶段**：集成SOTA视频生成模型（SkyReels）
3. **第四阶段**：实现音视频精准同步
4. **第五阶段**：添加实时生理信号反馈

## 故障排除

如果增强模块加载失败：
1. 检查Python路径是否包含src目录
2. 确保所有依赖已安装
3. 查看终端输出的详细错误信息
4. 系统会自动回退到基础版本

## 贡献指南

欢迎贡献新的理论模型和优化算法！请确保：
1. 每个新功能都有充分的理论依据
2. 代码中包含详细的文献引用
3. 保持与现有系统的兼容性
4. 提供单元测试和文档

---

*心境流转团队 - 让科技守护您的睡眠*