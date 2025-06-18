# 心境流转系统 - 理论驱动增强版

## 🚀 快速开始

### 使用增强版（推荐）
```bash
# 方式1：使用便捷脚本
./run_enhanced.sh

# 方式2：直接运行
python web_demo.py --enhanced

# 方式3：指定端口
python web_demo.py --enhanced --port 8080
```

### 使用基础版
```bash
python web_demo.py
```

## 📚 增强功能概览

### 1. 细粒度情绪识别（基于FACED数据集）

**9种情绪分类**取代原有的简单关键词匹配：

| 情绪类别 | 英文 | V-A坐标 | 典型场景 |
|---------|------|---------|---------|
| 愤怒 | anger | (-0.7, 0.9) | 高唤醒负面 |
| 恐惧/焦虑 | fear | (-0.6, 0.7) | 高唤醒负面 |
| 厌恶 | disgust | (-0.5, 0.3) | 中唤醒负面 |
| 悲伤 | sadness | (-0.8, -0.5) | 低唤醒负面 |
| 愉悦 | amusement | (0.7, 0.8) | 高唤醒正面 |
| 喜悦 | joy | (0.8, 0.6) | 高唤醒正面 |
| 灵感 | inspiration | (0.6, 0.7) | 高唤醒正面 |
| 温柔 | tenderness | (0.5, 0.2) | 低唤醒正面 |
| 中性 | neutral | (0.0, 0.0) | 中性状态 |

### 2. ISO原则治疗规划

基于音乐治疗ISO原则（Altshuler, 1948）和Gross情绪调节模型（2015）：

**三阶段自适应规划**：
- **同步化（25%）**：匹配用户当前情绪，建立信任
- **引导化（50%）**：渐进式过渡到睡眠适宜状态
- **巩固化（25%）**：维持并深化放松状态

**动态时长调整**：
- 高唤醒状态 → 延长同步化阶段（30%）
- 情绪距离大 → 延长引导化阶段（60%）

### 3. 精准音乐特征映射

基于2024年最新研究（bioRxiv）的高精度映射：

**核心映射关系**：
- Tempo ← Arousal（相关性0.88）
- Mode ← Valence（相关性0.74）
- Timbre ← Emotion complexity
- Dynamics ← Energy level

**睡眠优化约束**：
- 最大节奏：70 BPM（静息心率）
- 频率范围：20-1000 Hz
- 双耳节拍：Delta波（2Hz）诱导深度睡眠

## 🧪 测试增强功能

```bash
# 运行测试套件
python test_enhanced_modules.py
```

## 📊 性能对比

| 功能 | 基础版 | 增强版 |
|-----|-------|-------|
| 情绪识别 | 6种关键词匹配 | 9种细粒度分类 + 置信度 |
| 治疗规划 | 固定时长分配 | 自适应动态调整 |
| 音乐映射 | 简单线性映射 | 多维非线性映射 |
| 理论支撑 | 基础ISO原则 | ISO + Gross + 2024研究 |

## 🔬 学术引用

使用本系统进行研究时，请引用以下关键文献：

```bibtex
@article{faced2023,
  title={A Large Finer-grained Affective Computing EEG Dataset},
  journal={Nature Scientific Data},
  year={2023},
  doi={10.1038/s41597-023-02650-w}
}

@article{starcke2024,
  title={Music listening according to the iso principle modulates affective state},
  author={Starcke, K. and von Georgi, R.},
  journal={Psychology of Music},
  year={2024},
  doi={10.1177/10298649231175029}
}

@article{musical2024,
  title={Decoding Musical Valence And Arousal},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.02.27.582309}
}
```

## 🛠️ 技术架构

```
增强模块架构：
├── emotion_recognition/     # 细粒度情绪识别
│   └── enhanced_emotion_recognizer.py
├── therapy_planning/       # ISO治疗规划
│   └── enhanced_iso_planner.py
├── music_mapping/         # 音乐特征映射
│   └── enhanced_music_mapper.py
└── enhanced_mood_flow_adapter.py  # 适配器集成
```

## 🔄 后续计划

- [ ] 第二阶段：集成MusicGen（Meta AudioCraft）
- [ ] 第三阶段：集成SkyReels视频生成
- [ ] 第四阶段：实现音视频精准同步
- [ ] 第五阶段：添加生理信号实时反馈

## 💡 使用建议

1. **首次使用**：建议先尝试基础版，了解系统流程
2. **研究用途**：使用增强版获得更精准的数据
3. **临床应用**：请在专业指导下使用
4. **性能考虑**：增强版需要更多计算资源

## 🤝 贡献指南

欢迎贡献新的理论模型和优化！请确保：
- 每个功能都有充分的学术依据
- 代码包含详细的文献引用
- 保持向后兼容性
- 提供测试用例

---

*让科技与理论结合，守护每一个安眠之夜* 🌙