#!/usr/bin/env bash
# TRAE-llamaCPP-Bridge Linux 启动脚本（前台运行，Ctrl+C 退出）
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 未找到 python3，请先安装 Python 3.10+"
    exit 1
fi

if ! python3 -c "import requests" >/dev/null 2>&1; then
    echo "[首次运行] 正在安装依赖 requests ..."
    python3 -m pip install -r requirements.txt --disable-pip-version-check -q || \
        pip3 install -r requirements.txt -q
fi

echo "启动代理服务（Ctrl+C 退出并停止模型）..."
exec python3 cli_app.py serve "$@"
