#!/bin/bash
# 绿联 NAS (arm64/rk3588) Docker 部署脚本
# 用途：在 arm64 NAS 上一键部署股票分析系统

set -e

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.nas.yml"

echo "========================================"
echo "股票分析系统 - 绿联 NAS 部署 (arm64)"
echo "========================================"
echo ""

# 检查Docker
echo "[1/5] 检查Docker环境..."
if ! command -v docker &> /dev/null; then
    echo "[错误] 未找到Docker，请先在 NAS 中启用 Docker/容器功能"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi
$COMPOSE_CMD version > /dev/null 2>&1 || { echo "[错误] docker compose 不可用"; exit 1; }

echo "Docker: $(docker --version)"

# 检查.env
echo "[2/5] 检查环境变量..."
if [ ! -f ".env" ]; then
    [ -f ".env.example" ] && cp .env.example .env
    echo "[警告] 请编辑 .env 设置 DEEPSEEK_API_KEY: nano .env"
    read -p "按 Enter 继续..."
fi

# 创建目录
echo "[3/5] 创建数据目录..."
mkdir -p data db pools data/etf_sim_accounts

# 构建（arm64）
echo "[4/5] 构建 Docker 镜像 (linux/arm64)..."
echo "     首次构建可能需 10-20 分钟，请耐心等待..."
$COMPOSE_CMD $COMPOSE_FILES build --no-cache
[ $? -ne 0 ] && { echo "[错误] 构建失败"; exit 1; }

# 启动
echo "[5/5] 启动服务..."
$COMPOSE_CMD $COMPOSE_FILES up -d
[ $? -ne 0 ] && { echo "[错误] 启动失败"; exit 1; }

# 获取本机 IP
NAS_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "本机IP")

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "访问地址（局域网内设备均可访问）："
echo "  前端:     http://${NAS_IP}:5173"
echo "  健康检查: http://${NAS_IP}:8000/api/health"
echo ""
echo "常用命令："
echo "  查看日志: $COMPOSE_CMD $COMPOSE_FILES logs -f"
echo "  停止服务: $COMPOSE_CMD $COMPOSE_FILES down"
echo "  重启:     $COMPOSE_CMD $COMPOSE_FILES restart"
echo ""
