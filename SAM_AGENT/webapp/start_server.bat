@echo off
REM Perovskite Multi-AI Agent Web Server 启动脚本 (Windows)

echo 🚀 启动 Perovskite Multi-AI Agent Web 服务器...
echo.

REM 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 读取 .env 文件（如果存在）
if exist "%~dp0\..\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0\..\.env") do (
        if not "%%A"=="" if not "%%A:~0,1"=="#" set "%%A=%%B"
    )
)

REM 检查环境变量
if not defined OPENAI_API_KEY (
    echo ⚠️ 警告: 未设置 OPENAI_API_KEY
    echo 请复制 SAM_AGENT\.env.example 为 SAM_AGENT\.env，或在当前终端设置环境变量。
)

if not defined TAVILY_API_KEY (
    echo ⚠️ 提示: 未设置 TAVILY_API_KEY，Web search 工具可能不可用。
)

if not defined DeepSeek_API_KEY (
    echo ⚠️ 提示: 未设置 DeepSeek_API_KEY，DeepSeek 模型可能不可用。
)

echo ✅ 环境检查完成
echo.
echo 📡 启动 Web 服务器...
echo    地址: http://localhost:8000
echo    前端: http://localhost:8000
echo.
echo 按 Ctrl+C 停止服务器
echo.

REM 启动服务器
cd /d "%~dp0\.."
python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000 --reload
pause

