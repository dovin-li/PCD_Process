#!/bin/zsh
set -u
PROJECT="/Users/dovin_lee/Documents/项目资料/Projects/HR-RD-004/sftware/pointcloud-tools"
cd "$PROJECT" || exit 1
exec /usr/bin/python3 tools/bag_studio.py
