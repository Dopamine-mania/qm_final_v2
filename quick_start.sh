#!/bin/bash
# 《心境流转》快速启动脚本

echo "======================================"
echo "🌙 心境流转 - 睡眠治疗系统"
echo "======================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到Python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 安装依赖（如果需要）
echo ""
echo "检查依赖..."
pip install gradio numpy matplotlib opencv-python 2>/dev/null

# 选择启动模式
echo ""
echo "请选择启动模式:"
echo "1. 命令行交互模式"
echo "2. Web界面模式（推荐）"
echo "3. 运行完整测试"
read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "启动命令行模式..."
        python3 mood_flow_app.py
        ;;
    2)
        echo "启动Web界面..."
        echo "浏览器将自动打开，如未打开请访问: http://localhost:7860"
        python3 web_demo.py
        ;;
    3)
        echo "运行完整测试..."
        python3 scripts/run_all_tests.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac