# 《心境流转》演示指南

## 快速演示流程（5分钟）

### 1. 系统概览（30秒）
```bash
python scripts/01_system_initialization.py
```
展示系统环境检测和硬件兼容性

### 2. 核心理论演示（1分钟）
```bash
python scripts/02_theory_models_demo.py
```
展示ISO三阶段治疗原则和情绪轨迹规划

### 3. 完整系统演示（3分钟）
```bash
python scripts/10_complete_system_demo.py
```
展示端到端的治疗流程

### 4. 查看生成结果（30秒）
- 音频文件: `outputs/music/`
- 视频预览: `outputs/videos/`
- 分析报告: `outputs/reports/`

## 详细功能演示（15分钟）

### 阶段1: 基础功能（5分钟）
1. **情绪识别与会话**
   ```bash
   python scripts/04_therapy_session_demo.py
   ```
   - 展示多轮对话
   - 情绪状态变化
   - 治疗建议生成

2. **个性化处方**
   ```bash
   python scripts/05_prescription_system_test.py
   ```
   - 不同用户案例
   - 治疗方案对比
   - 疗效预测

### 阶段2: 内容生成（5分钟）
1. **音乐生成**
   ```bash
   python scripts/06_music_generation_workshop.py
   ```
   - 播放生成的WAV文件
   - 展示不同风格对比

2. **视频生成**
   ```bash
   python scripts/07_video_generation_workshop.py
   ```
   - 查看缩略图
   - 展示不同视觉模式

### 阶段3: 高级功能（5分钟）
1. **多模态融合**
   ```bash
   python scripts/08_multimodal_therapy_test.py
   ```
   - 音视频同步效果
   - 协同增强分析

2. **性能优化**
   ```bash
   python scripts/09_performance_optimization.py
   ```
   - GPU加速效果
   - 内存优化对比

## 亮点展示

### 1. 科学性
- ISO原则的创新应用
- V-A情绪模型的准确映射
- 严格的统计验证

### 2. 技术性
- 多模态AI融合
- 实时生成能力
- 硬件自适应

### 3. 实用性
- 个性化治疗方案
- 显著的改善效果
- 友好的用户体验

## 常见问题

**Q: 音乐听起来很简单？**
A: 当前使用基础合成，实际部署会集成专业AI模型如MusicGen

**Q: 视频只有图片？**
A: 为节省时间采用预览模式，可修改代码生成完整视频

**Q: 能否实时运行？**
A: 可以，系统设计支持实时处理，当前为演示简化了部分功能

## 演示技巧

1. **突出创新**：强调ISO原则和多模态融合
2. **展示效果**：用图表说明治疗效果提升
3. **谈论愿景**：医疗AI的未来应用前景