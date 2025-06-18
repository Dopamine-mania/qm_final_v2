#!/bin/bash
# 启动增强版心境流转系统

echo "🌙 心境流转睡眠治疗系统 - 增强版"
echo "=================================="
echo ""
echo "📚 增强功能："
echo "  • 细粒度情绪识别（9种情绪分类）"
echo "  • ISO原则治疗路径规划"
echo "  • 精准音乐特征映射"
echo "  • 基于最新研究的理论支撑"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到Python3"
    exit 1
fi

# 运行增强版
python3 web_demo.py --enhanced "$@"