#!/bin/sh
# Clone yolo-mlx at the commit this port was written against, apply the patch, install both.
set -e
UPSTREAM_COMMIT=abbcb79df7f48235ea13e1218b37045f133e1ac7
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ -d "$HERE/yolo-mlx" ] || git clone https://github.com/thewebAI/yolo-mlx "$HERE/yolo-mlx"
cd "$HERE/yolo-mlx"
git checkout -q "$UPSTREAM_COMMIT"
git apply --check "$HERE/patches/yolo-mlx-yolo12.patch" 2>/dev/null && git apply "$HERE/patches/yolo-mlx-yolo12.patch" || echo "patch already applied (or does not apply cleanly)"
pip install -e .
pip install -e "$HERE"
