#!/bin/bash

# Perovskite Multi-AI Agent Web Server 启动脚本

echo "🚀 启动 Perovskite Multi-AI Agent Web 服务器..."
echo ""

# 检查是否在正确的目录
if [ ! -d "AI_agents" ]; then
    echo "❌ 错误: 请在项目根目录 (AI_agents_0616) 运行此脚本"
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

# 检查必要的包
echo "📦 检查依赖..."
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖，正在安装..."
    pip install fastapi uvicorn websockets
fi

# 读取 .env 文件（如果存在）
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: 未设置 OPENAI_API_KEY"
    echo "请复制 SAM_AGENT/.env.example 为 SAM_AGENT/.env，或在当前终端设置环境变量。"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  提示: 未设置 TAVILY_API_KEY，Web search 工具可能不可用。"
fi

if [ -z "$DeepSeek_API_KEY" ]; then
    echo "⚠️  提示: 未设置 DeepSeek_API_KEY，DeepSeek 模型可能不可用。"
fi

echo "✅ 环境检查完成"
echo ""
echo "📡 启动 Web 服务器..."
echo "   地址: http://localhost:8000"
echo "   前端: http://localhost:8000"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动服务器
cd "$(dirname "$0")/.." || exit
python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000 --reload

