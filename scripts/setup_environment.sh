#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
if ! command -v uv >/dev/null 2>&1; then
  echo "请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  uv venv --python 3.12 --seed "$project_dir/.venv"
fi
python_bin="$project_dir/.venv/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "CPython 3.12 required"'
"$python_bin" scripts/fetch_release.py --group runtime
"$python_bin" scripts/restore_runtime.py
requirements_tmp="$(mktemp)"
trap 'rm -f -- "$requirements_tmp"' EXIT
"$python_bin" - "$requirements_tmp" <<'PY'
from pathlib import Path
import sys
lines = Path('environment/python-packages.lock').read_text().splitlines()
Path(sys.argv[1]).write_text('\n'.join(line for line in lines if not line.lower().startswith('torch==')) + '\n')
PY
# The lock is the complete recorded distribution set. Avoid an implicit Torch
# replacement while installing those exact wheels; verify dependencies below.
uv pip install --python "$python_bin" --no-deps -r "$requirements_tmp"
uv pip install --python "$python_bin" --no-deps --reinstall \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$python_bin" --no-deps -e ./genesis-world
uv pip check --python "$python_bin"
"$python_bin" scripts/check_environment.py
echo "环境已恢复。运行：./run.sh --headless --task full --output my_full_run"
