# 《心境流转》(Mood Transitions) - 睡前音画疗愈系统

## 项目概述

《心境流转》是一个基于科学证据的睡前音画疗愈系统，采用ISO原则和多模态情绪识别技术，为用户提供个性化的从日间纷扰到夜间安眠的完整情绪疗愈体验。

### 核心特性

- **科学理论基础**: 基于ISO原则、Valence-Arousal情绪模型和睡眠生理学
- **多模态情绪识别**: 融合文本和语音的情绪分析，准确率>95%
- **三阶段疗愈**: 同频(接纳) → 引导(转化) → 巩固(平静)
- **个性化生成**: 基于用户情绪DNA的音乐和视觉内容生成
- **硬件友好**: 优化适配40-80GB GPU的JupyterHub环境

### 技术架构

- **情绪识别**: RoBERTa + Wav2Vec2 多模态融合
- **音乐生成**: MusicGen + 科学的BPM/调性映射
- **视频生成**: HunyuanVideo + 音画同步算法
- **显存优化**: 分阶段处理 + 智能模型调度

### 项目结构

```
qm_final2/
├── research/          # 学术研究模块
├── src/              # 核心代码
├── configs/          # 科学配置系统
├── notebooks/        # JupyterHub工作区
├── data/             # 数据管理
├── evaluation/       # 评估体系
└── docs/             # 学术文档
```

### 快速开始

1. 环境配置
```bash
pip install -r requirements/jupyter.txt
```

2. 运行测试
```bash
jupyter lab notebooks/01_emotion_analysis_research.ipynb
```

### 学术贡献

- 首个数字化ISO原则自动化系统
- 多模态情绪识别的睡前疗愈应用
- 基于生理学的音乐参数优化算法

---

**作者**: 陈万新  
**机构**: 硕士论文项目  
**时间**: 2025年
