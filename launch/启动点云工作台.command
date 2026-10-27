#!/bin/zsh
# 位置无关：从脚本所在目录推仓库根，换机器/换目录都不用改
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT" || exit 1

PY="${PCD_PYTHON:-/usr/bin/python3}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
exec "$PY" tools/bag_studio.py
