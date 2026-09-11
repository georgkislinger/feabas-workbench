#!/usr/bin/env bash
# Start the FEABAS Workbench GUI (Linux/macOS). Counterpart of start_gui.bat.
#
#     ./start_gui.sh                        open the last project
#     ./start_gui.sh --project /data/proj   open a specific project folder
#     ./start_gui.sh --page 3               jump straight to a window (0..6)
#
# Looks for a .venv in this folder or a conda environment that has PySide6 installed.
# To force one:  FW_PYTHON=/path/to/env/bin/python ./start_gui.sh
cd "$(dirname "$0")"

PY=""
if [ -n "${FW_PYTHON:-}" ] && [ -x "$FW_PYTHON" ]; then
  PY="$FW_PYTHON"
fi

# a plain venv in the repository root
if [ -z "$PY" ] && [ -x ".venv/bin/python" ] && ".venv/bin/python" -c "import PySide6" >/dev/null 2>&1; then
  PY="$(pwd)/.venv/bin/python"
fi

# conda environments: <conda root>/envs or ~/.conda/envs
if [ -z "$PY" ]; then
  roots="$HOME/.conda/envs"
  for base in "$HOME" /opt; do
    for dist in miniforge3 mambaforge miniconda3 anaconda3; do
      [ -d "$base/$dist/envs" ] && roots="$roots $base/$dist/envs"
    done
  done
  for r in $roots; do
    for e in feabas-gui feabas-workbench feabas_env feabas; do
      if [ -z "$PY" ] && [ -x "$r/$e/bin/python" ] && "$r/$e/bin/python" -c "import PySide6" >/dev/null 2>&1; then
        PY="$r/$e/bin/python"
      fi
    done
  done
fi

if [ -z "$PY" ]; then
  echo
  echo "Could not find a Python environment with PySide6 installed."
  echo "Run tools/install.sh once to create the 'feabas-workbench' conda environment,"
  echo "or create a virtual environment here:"
  echo "    python3 -m venv .venv && .venv/bin/python -m pip install -e ."
  echo "or point this script at an existing one:"
  echo "    FW_PYTHON=/path/to/env/bin/python ./start_gui.sh"
  echo
  exit 1
fi

echo "Starting FEABAS Workbench with $PY"
exec "$PY" -m feabas_workbench "$@"
