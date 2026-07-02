#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动 Perovskite Multi-AI Agent Web 应用
"""

import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PROJECT_ROOT = Path(__file__).resolve().parent
load_env_file(PROJECT_ROOT / ".env")

required_keys = ["OPENAI_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]
optional_keys = ["TAVILY_API_KEY", "DeepSeek_API_KEY"]
missing_optional_keys = [key for key in optional_keys if not os.environ.get(key)]

if missing_keys:
    print("[!] Missing required API keys: " + ", ".join(missing_keys))
    print("[!] Please copy .env.example to .env and fill in your keys before starting.")

if missing_optional_keys:
    print("[!] Optional API keys not set: " + ", ".join(missing_optional_keys))

print("=" * 80)
print("[*] Multi-AI Agent Web Application")
print("=" * 80)
print()
print("[*] 正在启动服务器...")
print()
print("[*] 服务地址:")
print("   主界面: http://localhost:8000")
print("   API 文档: http://localhost:8000/docs")
print("   健康检查: http://localhost:8000/health")
print()
print("[!] 提示:")
print("   - 按 Ctrl+C 停止服务器")
print("   - 首次启动可能需要加载模型，请耐心等待")
print()
print("=" * 80)
print()

# 启动 uvicorn 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webapp.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

