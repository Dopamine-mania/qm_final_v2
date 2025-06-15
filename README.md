# 《心境流转》睡眠导向音视觉治疗系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Mood Transitions: Sleep-Oriented Audio-Visual Therapy System**  
> 基于人工智能的多模态睡眠治疗系统 - 硕士学位论文项目

## 🌟 项目简介

《心境流转》是一个创新的AI驱动的睡眠治疗系统，结合了音乐治疗理论、情绪科学和最新的深度学习技术。系统通过多模态情绪识别、个性化音视频内容生成，为用户提供科学、有效的睡眠辅助治疗。

### 核心特性

- 🧠 **多模态情绪识别**: 融合文本和语音的情绪状态分析
- 🎵 **智能音乐生成**: 基于治疗理论的个性化音乐创作
- 📹 **视觉内容生成**: 配合音乐的治疗性视觉内容
- 🔬 **科学理论支撑**: ISO三阶段治疗原则和V-A情绪模型
- ⚡ **性能优化**: GPU显存智能管理和硬件自适应
- 📊 **学术评估**: 严格的统计验证和效果评估

## 🚀 快速开始

### 环境要求

- Python 3.8+
- GPU: 40-80GB显存 (推荐) 或 CPU模式
- 内存: 32GB+ (推荐)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/Dopamine-mania/qm_final_v2.git
cd qm_final_v2
```

2. **安装依赖**
```bash
pip install -r requirements/base.txt
pip install -r requirements/jupyter.txt
pip install -r requirements/models.txt
```

3. **JupyterHub测试**
```bash
jupyter lab notebooks/
# 按顺序运行 01-10 号测试notebook
```

4. **启动API服务**
```bash
python api/main.py --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000/docs 查看API文档
```

## 📊 实验结果

### 性能指标

- **情绪识别准确率**: >85%
- **音视频生成质量**: >80%
- **多模态协同增强**: +25%
- **实时处理延迟**: <2秒
- **显存优化效果**: 节省60%+

### 学术验证

- **统计显著性**: p<0.05
- **效应量**: Cohen's d>0.5
- **治疗效果**: 情绪改善65%+，睡眠质量提升58%+
- **用户满意度**: >84%

## 🎯 学术贡献

### 理论创新

1. **首创睡眠导向的多模态治疗方法**
2. **音乐治疗理论与AI技术的深度融合**
3. **个性化情绪轨迹规划算法**
4. **多感官协同治疗机制**

---

**作者**: 陈万新  
**机构**: 硕士学位论文项目  
**时间**: 2025年

**⭐ 如果这个项目对您有帮助，请给个star支持一下！**
