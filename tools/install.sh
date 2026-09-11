#!/usr/bin/env bash
# FEABAS Workbench installer (Linux/macOS). Uses conda/mamba/micromamba from PATH or a usual install root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENVNAME=feabas-workbench
CONDA=$(command -v mamba || command -v conda || command -v micromamba || true)
if [ -z "$CONDA" ]; then
  # not on PATH (e.g. conda init not run for this shell): same roots core/envs.py looks in
  for base in "$HOME" /opt; do
    for dist in miniforge3 mambaforge miniconda3 anaconda3; do
      if [ -z "$CONDA" ] && [ -x "$base/$dist/bin/conda" ]; then CONDA="$base/$dist/bin/conda"; fi
    done
  done
fi
if [ -z "$CONDA" ]; then
  echo "No conda/mamba/micromamba found. Install Miniforge: https://conda-forge.org/download/"; exit 1
fi
echo "Using $CONDA"
if ! "$CONDA" env list | grep -q "^$ENVNAME "; then
  "$CONDA" create -y -n "$ENVNAME" --override-channels -c conda-forge python=3.11 pip
fi
PY=$("$CONDA" run -n "$ENVNAME" python -c "import sys; print(sys.executable)")
"$PY" -m pip install --index-url https://pypi.org/simple -e "$HERE"
echo "Start with:  $PY -m feabas_workbench"
"$PY" -m feabas_workbench
