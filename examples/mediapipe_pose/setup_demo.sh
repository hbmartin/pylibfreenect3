#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$demo_dir/../.." && pwd)"
python_bin="${MEDIAPIPE_PYTHON:-python3.12}"

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "MediaPipe demo requires Python 3.12"'
"$python_bin" -m venv "$demo_dir/.venv"
"$demo_dir/.venv/bin/python" -m pip install --upgrade pip
"$demo_dir/.venv/bin/python" -m pip install "$project_dir"
"$demo_dir/.venv/bin/python" -m pip install -r "$demo_dir/requirements.txt"
"$demo_dir/.venv/bin/python" "$demo_dir/download_model.py"

echo "MediaPipe environment ready: $demo_dir/.venv"
