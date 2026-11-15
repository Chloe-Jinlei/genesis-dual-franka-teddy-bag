#!/usr/bin/env bash
set -euo pipefail

scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
genesis_root="$(dirname -- "$scene_dir")"
python_bin="$genesis_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "缺少本地 Python 环境：$python_bin" >&2
  exit 1
fi
if [[ ! -d "$scene_dir/vendor/uipc" ]]; then
  echo "缺少此场景隔离安装的 IPC 依赖：$scene_dir/vendor/uipc" >&2
  exit 1
fi

# The existing Genesis installation remains unchanged. Only this process gets
# the scene-local pyuipc package and separate runtime/cache output directory.
export PYTHONPATH="$scene_dir/vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export GS_CACHE_FILE_PATH="$scene_dir/.cache/genesis"
export XDG_CACHE_HOME="$scene_dir/.cache"
export PYTHONUNBUFFERED=1

show_viewer=1
forward_args=()
for argument in "$@"; do
  case "$argument" in
    --no-viewer|--headless) show_viewer=0 ;;
    --viewer) show_viewer=1 ;;
    *) forward_args+=("$argument") ;;
  esac
done
if (( show_viewer )); then
  forward_args+=(--viewer)
fi
cd -- "$scene_dir"
exec "$python_bin" "$scene_dir/run.py" "${forward_args[@]}"
