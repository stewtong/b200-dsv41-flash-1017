#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  printf '%s\n' 'usage: install_downloader.sh LOCK_FILE NEW_VENV_PATH' >&2
  exit 2
fi
lock_file=$1
venv_path=$2
case "$venv_path" in /*) ;; *) printf '%s\n' 'venv path must be absolute' >&2; exit 2;; esac
if [ -e "$venv_path" ] || [ -L "$venv_path" ]; then
  printf '%s\n' 'refusing to overwrite downloader environment' >&2
  exit 1
fi
python3 -m venv "$venv_path"
"$venv_path/bin/python" -m pip --isolated install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r "$lock_file"
"$venv_path/bin/python" -m pip check
"$venv_path/bin/python" -m pip freeze --all > "$venv_path/installed-packages.txt"
"$venv_path/bin/python" -c 'import huggingface_hub; assert huggingface_hub.__version__ == "1.33.0"'
printf '%s\n' 'downloader environment created; staging still requires preflight and watchdog'
