#!/bin/bash
# Linux/Mac虚拟环境设置脚本
# 用途：自动创建Python虚拟环境并安装依赖

set -e

echo "========================================"
echo "股票分析系统 - 虚拟环境设置"
echo "========================================"
echo ""

# 检查Python是否安装（需 Python 3.8+，推荐 3.10）
echo "[1/4] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.x（推荐 3.10）"
    echo "Ubuntu/Debian: sudo apt-get install python3.10 python3.10-venv"
    echo "macOS: brew install python@3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "[错误] 需要 Python 3.8 及以上版本"
    exit 1
fi
echo "[提示] 推荐使用 Python 3.10 以获得与 ta/akshare 的最佳兼容性。"

# 创建虚拟环境
echo "[2/4] 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[错误] 创建虚拟环境失败"
        exit 1
    fi
    echo "虚拟环境创建成功"
fi

# 激活虚拟环境并升级pip
echo "[3/4] 激活虚拟环境并升级pip..."
source venv/bin/activate
python -m pip install --upgrade pip > /dev/null 2>&1

# 安装依赖
echo "[4/4] 安装Python依赖..."
if [ ! -f "requirements.txt" ]; then
    echo "[错误] 未找到requirements.txt文件"
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[错误] 依赖安装失败"
    exit 1
fi

echo ""
echo "========================================"
echo "虚拟环境设置完成！"
echo "========================================"
echo ""
echo "使用说明："
echo "1. 激活虚拟环境："
echo "   source venv/bin/activate"
echo ""
echo "2. 启动后端服务："
echo "   python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload"
echo "   或使用: npm run api"
echo ""
echo "3. 退出虚拟环境："
echo "   deactivate"
echo ""
