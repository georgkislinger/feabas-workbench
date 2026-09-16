#!/usr/bin/env bash
# FEABAS Workbench installer (Linux/macOS). Counterpart of install.bat.
#
# Creates the GUI environment 'feabas-workbench' with micromamba / mamba / conda, installs the
# workbench into it, records that interpreter in start_gui.local.sh (so ./start_gui.sh is
# hard-wired to this installation and only falls back to its own search if the environment
# disappears) and starts the app.
#
#     FW_CONDA=/path/to/micromamba bash tools/install.sh    use this package manager
#     FW_DEBUG=1                                            show every place searched
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENVNAME=feabas-workbench
CONDA=""

try() { [ -n "$CONDA" ] && return 0; [ -n "${FW_DEBUG:-}" ] && echo "[debug] try $1"; [ -x "$1" ] && CONDA="$1"; return 0; }

# 1. explicit override
[ -n "${FW_CONDA:-}" ] && try "$FW_CONDA"
# 2. on PATH; micromamba/mamba first (faster, no Anaconda channel terms)
for n in micromamba mamba conda; do
  if [ -z "$CONDA" ]; then
    p="$(command -v "$n" 2>/dev/null || true)"
    [ -n "$p" ] && try "$p"
  fi
done
# 3. what 'conda init' / 'micromamba shell init' left in the environment
[ -n "${MAMBA_EXE:-}" ] && try "$MAMBA_EXE"
[ -n "${CONDA_EXE:-}" ] && try "$CONDA_EXE"
[ -n "${MAMBA_ROOT_PREFIX:-}" ] && try "$MAMBA_ROOT_PREFIX/bin/micromamba"
# 4. the usual install folders
for p in "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba" "$HOME/bin/micromamba" \
         "$HOME/Library/Application Support/FeabasWorkbench/micromamba/bin/micromamba" \
         "$HOME/.local/share/FeabasWorkbench/micromamba/bin/micromamba" /usr/local/bin/micromamba; do
  try "$p"
done
for base in "$HOME" /opt /usr/local; do
  for dist in miniforge3 mambaforge miniconda3 anaconda3 miniconda anaconda conda micromamba; do
    try "$base/$dist/bin/micromamba"; try "$base/$dist/bin/mamba"; try "$base/$dist/bin/conda"
    try "$base/$dist/condabin/conda"
  done
done

if [ -z "$CONDA" ]; then
  echo "No conda, mamba or micromamba found: looked on PATH, in CONDA_EXE / MAMBA_EXE and in the usual"
  echo "install folders (FW_DEBUG=1 shows every path tried). Install Miniforge from"
  echo "https://conda-forge.org/download/ or micromamba, or: FW_CONDA=/path/to/micromamba bash tools/install.sh"
  exit 1
fi
echo "Using $CONDA"

# Does the environment exist? Ask it for its interpreter (that also tells us where it lives).
env_python() { "$CONDA" run -n "$ENVNAME" python -c "import sys; print(sys.executable)" 2>/dev/null || true; }
PY="$(env_python)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "Creating environment $ENVNAME ..."
  "$CONDA" create -y -n "$ENVNAME" --override-channels -c conda-forge python=3.11 pip
  PY="$(env_python)"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "Could not determine the python of environment $ENVNAME."; exit 1
fi
echo "Installing the workbench from $HERE into $PY"
"$PY" -m pip install --index-url https://pypi.org/simple -e "$HERE"

# Hard-wire the launcher to this installation: start_gui.sh sources start_gui.local.sh before it
# searches for environments; if this interpreter ever disappears, its own search takes over.
LOCAL="$HERE/start_gui.local.sh"
{
  echo "# Written by tools/install.sh on $(date '+%Y-%m-%d %H:%M'): the environment the workbench was installed into."
  echo "# start_gui.sh reads this first. Delete the file to go back to automatic discovery."
  printf 'FW_PYTHON=%q\n' "$PY"
} > "$LOCAL"
echo "Recorded the interpreter in $LOCAL"
echo "Done. Starting the workbench (later: ./start_gui.sh)."
"$PY" -m feabas_workbench
