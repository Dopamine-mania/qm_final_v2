# 《心境流转》JupyterHub 测试 Notebooks

本目录包含了完整的 Jupyter Notebook 测试套件，用于在 JupyterHub 环境中验证和演示"心境流转"睡前音画疗愈系统。

## 📚 Notebook 概览

### 🧪 **01_system_initialization.ipynb**
- 系统环境检测和初始化
- 硬件资源评估 (40-80GB GPU)
- 依赖库安装和配置验证
- 模型下载和缓存设置

### 🧠 **02_theory_models_demo.ipynb**
- ISO原则理论演示
- Valence-Arousal情绪模型测试
- 睡眠生理学模型验证
- 音乐心理学映射实验

### 🤖 **03_model_adapters_test.ipynb**
- 情绪识别模型测试 (RoBERTa + Wav2Vec2)
- 音乐生成模型验证 (MusicGen + AudioLDM)
- 视频生成模型测试 (HunyuanVideo + Mochi)
- 硬件优化和内存管理验证

### 🎭 **04_therapy_session_demo.ipynb**
- 完整疗愈会话演示
- ISO三阶段执行流程
- 实时情绪轨迹跟踪
- 多模态内容生成协调

### 📋 **05_prescription_system_test.ipynb**
- 疗愈处方生成系统
- 个性化参数调整
- 安全性验证
- 效果预测评估

### 🎵 **06_music_generation_workshop.ipynb**
- 治疗性音乐生成深度测试
- 不同情绪状态的音乐处方
- 音乐质量评估
- 用户反馈集成

### 🎬 **07_video_generation_workshop.ipynb**
- 睡眠导向视频生成
- 视觉疗愈内容优化
- 颜色心理学应用
- 动态效果调整

### 📊 **08_performance_analysis.ipynb**
- 系统性能基准测试
- 内存使用模式分析
- 推理时间优化
- 资源利用率监控

### 🎯 **09_academic_validation.ipynb**
- 学术评估指标计算
- 理论模型验证
- 效果统计分析
- 科学报告生成

### 🔧 **10_troubleshooting_guide.ipynb**
- 常见问题诊断
- 模型加载故障排除
- 内存不足解决方案
- 性能调优建议

## 🚀 使用指南

### 推荐执行顺序
1. **01_system_initialization** - 首次运行必须
2. **02_theory_models_demo** - 理解理论基础
3. **03_model_adapters_test** - 验证模型功能
4. **04_therapy_session_demo** - 体验完整流程
5. **05-09** - 根据具体需求选择性运行
6. **10_troubleshooting_guide** - 遇到问题时参考

### 硬件要求
- **最低配置**: 40GB GPU + 32GB RAM
- **推荐配置**: 80GB GPU + 64GB RAM
- **JupyterHub环境**: Python 3.8+ with CUDA support

### 预期运行时间
- **快速测试** (Notebooks 1-3): ~30分钟
- **完整演示** (Notebooks 1-5): ~90分钟
- **深度测试** (所有Notebooks): ~3小时

### 输出内容
- 理论模型验证结果
- 生成的音乐和视频样本
- 性能指标报告
- 学术评估数据
- 可视化图表和分析

## 📝 注意事项

- 确保在运行前执行系统初始化
- 某些模型首次下载可能需要较长时间
- 大型模型测试需要足够的GPU内存
- 建议在非高峰时段运行以获得最佳性能
- 所有生成的内容和日志都会保存在 `outputs/` 目录

## 🆘 获取帮助

如果在使用过程中遇到问题:
1. 先查看 `10_troubleshooting_guide.ipynb`
2. 检查系统资源使用情况
3. 查看 JupyterHub 日志
4. 联系技术支持