# 🎼 MusicGen SOTA音乐生成集成

《心境流转》睡眠治疗系统现已集成Meta MusicGen，实现基于AI的高质量治疗音乐生成。

## 🌟 功能特性

### Phase 2: SOTA音乐生成
- **Meta MusicGen集成**: 使用最先进的文本到音乐生成模型
- **情绪驱动生成**: 基于9种细粒度情绪自动构建音乐prompt
- **治疗场景优化**: 专门针对睡眠治疗场景的参数调优
- **质量实时评估**: 综合技术质量和治疗效果的多维度评估
- **长序列生成**: 支持1-5分钟的长音乐片段生成

### 核心优势
1. **理论支撑**: 基于ISO原则和音乐治疗研究的prompt工程
2. **自适应模型**: 根据GPU资源自动选择最优模型大小
3. **质量保证**: 实时评估生成音乐的治疗适用性
4. **无缝回退**: 模型不可用时自动回退到基础生成方法

## 🚀 快速开始

### 1. 安装依赖

```bash
# 运行自动安装脚本
python install_musicgen.py

# 或手动安装
pip install torch==2.1.0 torchaudio==2.1.0
pip install audiocraft
pip install librosa soundfile scipy
```

### 2. 启动系统

```bash
# 基础增强模式（推荐新手）
python web_demo.py --enhanced

# 完整SOTA模式（需要较好的GPU）
python web_demo.py --enhanced --sota

# 仅SOTA音乐生成
python web_demo.py --sota
```

### 3. 使用说明

启动后，系统会根据您的GPU自动选择合适的MusicGen模型：
- **40GB+ GPU**: `musicgen-large` (3.3B参数)
- **16-40GB GPU**: `musicgen-melody` (1.5B参数) 
- **8-16GB GPU**: `musicgen-small` (300M参数)
- **<8GB或CPU**: `musicgen-small` (CPU模式)

## 📊 技术架构

### MusicGen适配器 (`src/model_adapters/musicgen_adapter.py`)
- **智能prompt构建**: 将情绪状态转换为MusicGen理解的文本描述
- **长序列技术**: 使用窗口滑动生成1-5分钟音乐
- **后处理优化**: 根据治疗阶段应用音频后处理

### 质量评估器 (`src/model_adapters/music_quality_evaluator.py`)
- **技术质量**: 信噪比、动态范围、频谱分析
- **治疗效果**: 节奏稳定性、情绪一致性、睡眠适用性
- **实时反馈**: 生成改进建议和质量警告

### 集成适配器 (`src/enhanced_mood_flow_adapter.py`)
- **配置管理**: 支持多种增强组合模式
- **优雅降级**: SOTA模型失败时自动回退
- **状态监控**: 实时显示各模块运行状态

## 🎯 使用模式

| 模式 | 命令 | 特性 |
|------|------|------|
| 基础模式 | `python web_demo.py` | 简单音乐生成，快速响应 |
| 增强模式 | `python web_demo.py --enhanced` | 理论驱动优化，精准映射 |
| SOTA模式 | `python web_demo.py --sota` | 高质量AI音乐生成 |
| 完整模式 | `python web_demo.py --enhanced --sota` | 所有功能，最佳体验 |

## 📈 性能优化

### GPU要求
- **推荐**: 16GB+ VRAM for medium models
- **最佳**: 40GB+ VRAM for large models  
- **最低**: 8GB VRAM for small models
- **CPU模式**: 可用但生成较慢

### 生成时间参考
- **Small模型**: ~5-10秒/分钟音乐
- **Medium模型**: ~10-20秒/分钟音乐  
- **Large模型**: ~15-30秒/分钟音乐

### 首次使用
首次启动会自动下载预训练模型：
- Small: ~1.2GB
- Medium: ~6GB
- Large: ~13GB

模型缓存位置: `~/.cache/huggingface/transformers/`

## 🔬 质量评估

生成的音乐会实时评估以下指标：

### 技术质量 (0-1评分)
- **信噪比**: 音频清晰度
- **动态范围**: 音量变化幅度
- **频谱分析**: 频率分布合理性

### 治疗效果 (0-1评分)
- **节奏稳定性**: BPM一致性
- **情绪一致性**: 与目标情绪匹配度
- **治疗适用性**: 睡眠场景适配度

### 综合评分
技术质量30% + 治疗效果70% = 综合评分

## 🛠️ 故障排除

### 常见问题

**Q: MusicGen模型加载失败？**
A: 检查网络连接，确保能访问Hugging Face。首次下载需要时间。

**Q: GPU内存不足？**
A: 降低模型大小或使用CPU模式。检查其他程序的显存占用。

**Q: 生成的音乐质量不佳？**
A: 查看质量评估报告，根据建议调整参数或升级模型。

**Q: 生成速度太慢？**
A: 使用更小的模型或检查GPU利用率。确保CUDA正确安装。

### 日志检查

系统运行时会输出详细日志：
```
🎼 [SOTA音乐生成 v1.0] MusicGen高质量生成:
📊 [音乐质量评估 v1.0] 评估完成:
```

查看这些日志可以了解生成过程和质量评估结果。

## 🔄 版本更新

### v2.0 (当前)
- ✅ MusicGen集成
- ✅ 质量评估系统
- ✅ 长序列生成
- ✅ 自适应模型选择

### v2.1 (计划中)
- 🔄 视频生成集成
- 🔄 音画同步技术
- 🔄 用户偏好学习

## 📞 支持

如遇问题，请：
1. 查看终端日志输出
2. 检查系统要求
3. 尝试重新安装依赖
4. 降级到基础模式测试

更多技术细节请参考源码注释和研究论文引用。