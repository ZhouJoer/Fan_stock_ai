#!/bin/bash
# Linux/Mac Docker部署脚本
# 用途：一键部署股票分析系统（Docker Compose）

set -e

echo "========================================"
echo "股票分析系统 - Docker部署"
echo "========================================"
echo ""

# 检查Docker是否安装
echo "[1/5] 检查Docker环境..."
if ! command -v docker &> /dev/null; then
    echo "[错误] 未找到Docker，请先安装Docker"
    echo "安装指南: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "[错误] 未找到docker-compose，请先安装docker-compose"
    echo "Docker Compose安装指南: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Docker版本: $(docker --version)"
if command -v docker-compose &> /dev/null; then
    echo "Docker Compose版本: $(docker-compose --version)"
    COMPOSE_CMD="docker-compose"
else
    echo "Docker Compose版本: $(docker compose version)"
    COMPOSE_CMD="docker compose"
fi

# 检查.env文件
echo "[2/5] 检查环境变量配置..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "未找到.env文件，从.env.example创建..."
        cp .env.example .env
        echo "[警告] 请编辑.env文件，设置DEEPSEEK_API_KEY"
        echo "编辑命令: nano .env 或 vi .env"
        read -p "按Enter继续（确保已配置API密钥）..."
    else
        echo "[错误] 未找到.env或.env.example文件"
        exit 1
    fi
else
    echo ".env文件已存在"
fi

# 创建必要的目录
echo "[3/5] 创建数据目录..."
mkdir -p data db pools data/etf_sim_accounts
echo "数据目录创建完成"

# 构建镜像
echo "[4/5] 构建Docker镜像..."
$COMPOSE_CMD build
if [ $? -ne 0 ]; then
    echo "[错误] Docker镜像构建失败"
    exit 1
fi

# 启动服务
echo "[5/5] 启动服务..."
$COMPOSE_CMD up -d
if [ $? -ne 0 ]; then
    echo "[错误] 服务启动失败"
    exit 1
fi

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "服务访问地址："
echo "  前端: http://localhost:5173"
echo "  后端API: http://localhost:8000"
echo "  健康检查: http://localhost:8000/api/health"
echo ""
echo "常用命令："
echo "  查看日志: $COMPOSE_CMD logs -f"
echo "  停止服务: $COMPOSE_CMD down"
echo "  重启服务: $COMPOSE_CMD restart"
echo "  查看状态: $COMPOSE_CMD ps"
echo ""
