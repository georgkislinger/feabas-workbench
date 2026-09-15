#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start the FEABAS Workbench GUI (Linux/macOS). Counterpart of start_gui.bat.
#
#     ./start_gui.sh                        open the last project
#     ./start_gui.sh --project /data/proj   open a specific project folder
#     ./start_gui.sh --page 3               jump straight to a window (0..6)
#     ./start_gui.sh --envs                 list every environment that would work
#
# Search order:
#     1  $FW_PYTHON
#     2  the environment already active in this shell (conda or venv)
#     3  a venv next to this script (.venv, .venv-*, venv, env, ../.venv*)
#     4  conda/mamba/micromamba envs whose name contains "feabas"
#     5  any other env under those roots that has PySide6
#     6  "conda env list --json" and PATH (covers custom envs_dirs)
#
# Knobs:
#     FW_PYTHON=/path/to/env/bin/python     force one interpreter
#     FW_ENV_DIRS="/opt/envs /share/envs"   extra folders full of envs
#     FW_DEBUG=1                            show every candidate tried
# ---------------------------------------------------------------------------
cd "$(dirname "$0")" || exit 1

PY=""
LISTONLY=""
if [ "${1:-}" = "--envs" ]; then LISTONLY=1; fi
if [ -n "$LISTONLY" ]; then echo "Environments that can run the workbench:"; fi

debug() { [ -n "${FW_DEBUG:-}" ] && echo "[debug] $*"; return 0; }

# Test one interpreter. Pass "force" to skip the cheap site-packages pre-filter.
check() {
  [ -n "$PY" ] && return 0
  local cand="$1"
  [ -x "$cand" ] || return 0
  if [ "${2:-}" != "force" ]; then
    local sp="" d
    for d in "$(dirname "$cand")"/../lib/python*/site-packages/PySide6 \
             "$(dirname "$cand")"/../lib64/python*/site-packages/PySide6; do
      [ -d "$d" ] && sp=1 && break
    done
    if [ -z "$sp" ]; then
      debug "skip $cand  (no PySide6 in site-packages)"
      return 0
    fi
  fi
  debug "test $cand"
  "$cand" -c "import PySide6" >/dev/null 2>&1 || return 0
  if [ -n "$LISTONLY" ]; then
    echo "    $cand"
    return 0
  fi
  PY="$cand"
}

# Every subfolder of one env root that matches $PATTERN.
scan_root() {
  [ -n "$PY" ] && return 0
  [ -d "$1" ] || return 0
  debug "root $1"
  local e
  for e in "$1"/$PATTERN; do
    [ -d "$e" ] && check "$e/bin/python"
  done
}

# Walk every plausible place conda-style environments live, using $PATTERN.
scan_roots() {
  [ -n "$PY" ] && return 0
  scan_root "$HOME/.conda/envs"
  scan_root "$HOME/micromamba/envs"
  if [ -n "${MAMBA_ROOT_PREFIX:-}" ]; then scan_root "$MAMBA_ROOT_PREFIX/envs"; fi
  if [ -n "${CONDA_ENVS_PATH:-}" ]; then scan_root "$CONDA_ENVS_PATH"; fi
  if [ -n "${FW_ENV_DIRS:-}" ]; then
    local r
    for r in $FW_ENV_DIRS; do scan_root "$r"; done
  fi
  local base dist
  for base in "$HOME" /opt /usr/local /srv; do
    for dist in miniforge3 mambaforge miniconda3 anaconda3 miniconda anaconda conda micromamba; do
      scan_root "$base/$dist/envs"
    done
  done
}

# One line of "<manager> env list --json": strip quotes/commas/brackets,
# then each remaining word is a candidate env path.
json_line() {
  local p
  for p in $(printf '%s' "$1" | tr -d '",{}[]'); do
    [ -n "$PY" ] && return 0
    [ -x "$p/bin/python" ] && check "$p/bin/python" force
  done
}

# Conda/micromamba know about envs_dirs entries we cannot guess; also try PATH.
ask_managers() {
  debug "asking conda/micromamba for their env lists"
  local line
  if command -v conda >/dev/null 2>&1; then
    while read -r line; do json_line "$line"; done < <(conda env list --json 2>/dev/null)
  fi
  if command -v micromamba >/dev/null 2>&1; then
    while read -r line; do json_line "$line"; done < <(micromamba env list --json 2>/dev/null)
  fi
  if command -v python >/dev/null 2>&1; then check "$(command -v python)" force; fi
  if command -v python3 >/dev/null 2>&1; then check "$(command -v python3)" force; fi
}

# --- 1. explicit override ---------------------------------------------------
if [ -n "${FW_PYTHON:-}" ]; then check "$FW_PYTHON" force; fi

# --- 2. whatever is activated in this shell ---------------------------------
if [ -n "${CONDA_PREFIX:-}" ]; then check "$CONDA_PREFIX/bin/python" force; fi
if [ -n "${VIRTUAL_ENV:-}" ]; then check "$VIRTUAL_ENV/bin/python" force; fi

# --- 3. virtual environments next to the repository -------------------------
for v in .venv* venv env ../*; do
  case "$v" in
    .venv*|venv|env|../.venv*|../*feabas*) [ -d "$v" ] && check "$PWD/$v/bin/python" force ;;
  esac
done

# --- 4. conda-style envs, feabas-looking names first ------------------------
if [ -z "$LISTONLY" ]; then
  PATTERN="*feabas*"
  scan_roots
fi

# --- 5. ... then anything else that has PySide6 -----------------------------
PATTERN="*"
scan_roots

# --- 6. last resort: ask the package managers themselves --------------------
if [ -z "$PY" ]; then ask_managers; fi

if [ -n "$LISTONLY" ]; then
  echo
  echo "Pick one with:  FW_PYTHON=<path> ./start_gui.sh"
  exit 0
fi

if [ -z "$PY" ]; then
  echo
  echo "Could not find a Python environment with PySide6 installed."
  echo "Run tools/install.sh once to create the 'feabas-workbench' environment,"
  echo "or create a virtual environment here:"
  echo "    python3 -m venv .venv && .venv/bin/python -m pip install -e ."
  echo "or point this script at an existing one:"
  echo "    FW_PYTHON=/path/to/env/bin/python ./start_gui.sh"
  echo
  echo '"./start_gui.sh --envs" lists what was found, "FW_DEBUG=1" shows the search.'
  echo
  exit 1
fi

echo "Starting FEABAS Workbench with $PY"
exec "$PY" -m feabas_workbench "$@"
