@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------------------
rem Start the FEABAS Workbench GUI (Windows).
rem
rem Double-click, or from a terminal:
rem     start_gui.bat                      open the last project
rem     start_gui.bat --project D:\path    open a specific project folder
rem     start_gui.bat --page 3             jump straight to a window (0..6)
rem     start_gui.bat --envs               list every environment that would work
rem
rem Search order:
rem     0  start_gui.local.bat next to this script: the interpreter tools\install.bat installed
rem        into (delete that file to go back to pure discovery)
rem     1  %FW_PYTHON%
rem     2  the environment already active in this shell (conda or venv)
rem     3  a venv next to this script (.venv, .venv-*, venv, env, ..\.venv*)
rem     4  conda/mamba/micromamba envs whose name contains "feabas"
rem     5  any other env under those roots that has PySide6
rem     6  "conda env list --json" and PATH (covers custom envs_dirs)
rem
rem Knobs:
rem     set FW_PYTHON=C:\path\to\env\python.exe      force one interpreter
rem     set FW_ENV_DIRS="D:\envs" "\\server\envs"    extra folders full of envs
rem     set FW_DEBUG=1                               show every candidate tried
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

set "PY="
set "LISTONLY="
set "SEEN="
if /i "%~1"=="--envs" set "LISTONLY=1"
if defined LISTONLY echo Environments that can run the workbench:

rem --- 0. the installation tools\install.bat recorded -------------------------
rem An FW_PYTHON set in the shell still wins; a recorded interpreter that no longer
rem exists is skipped by :check and the search below takes over.
if not defined FW_PYTHON if exist "%~dp0start_gui.local.bat" call "%~dp0start_gui.local.bat"

rem --- 1. explicit override ---------------------------------------------------
if defined FW_PYTHON call :check "%FW_PYTHON%" force

rem --- 2. whatever is activated in this shell ---------------------------------
if defined CONDA_PREFIX call :check "%CONDA_PREFIX%\python.exe" force
if defined VIRTUAL_ENV call :check "%VIRTUAL_ENV%\Scripts\python.exe" force

rem --- 3. virtual environments next to the repository -------------------------
for /d %%v in (".venv*" "venv" "env" "..\.venv*" "..\*feabas*") do (
  call :check "%%~fv\Scripts\python.exe" force
  call :check "%%~fv\python.exe" force
)

rem --- 4. conda-style envs, feabas-looking names first ------------------------
if not defined LISTONLY (
  set "PATTERN=*feabas*"
  call :scan_roots
)

rem --- 5. ... then anything else that has PySide6 -----------------------------
set "PATTERN=*"
call :scan_roots

rem --- 6. last resort: ask the package managers themselves --------------------
if not defined PY call :ask_managers

if defined LISTONLY (
  echo.
  echo Pick one with:  set FW_PYTHON^=^<path^>
  exit /b 0
)

if not defined PY (
  echo.
  echo Could not find a Python environment with PySide6 installed.
  echo Run tools\install.bat once to create the 'feabas-workbench' environment,
  echo or create a virtual environment here:
  echo     python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -e .
  echo or point this script at an existing one:
  echo     set FW_PYTHON=C:\path\to\env\python.exe
  echo.
  echo "start_gui.bat --envs" lists what was found, "set FW_DEBUG=1" shows the search.
  echo.
  pause
  exit /b 1
)

echo Starting FEABAS Workbench with "%PY%"
"%PY%" -m feabas_workbench %*
set "RC=!errorlevel!"
if not "!RC!"=="0" (
  echo.
  echo The workbench exited with an error ^(code !RC!^) - see the messages above.
  pause
)
exit /b !RC!

rem ===========================================================================
rem Walk every plausible place conda-style environments live, using !PATTERN!.
:scan_roots
if defined PY exit /b
call :scan_root "%USERPROFILE%\.conda\envs"
call :scan_root "%USERPROFILE%\micromamba\envs"
call :scan_root "%LOCALAPPDATA%\micromamba\envs"
if defined MAMBA_ROOT_PREFIX call :scan_root "%MAMBA_ROOT_PREFIX%\envs"
if defined CONDA_ENVS_PATH call :scan_root "%CONDA_ENVS_PATH%"
if defined FW_ENV_DIRS (
  for %%r in (%FW_ENV_DIRS%) do call :scan_root "%%~r"
)
for %%b in ("%LOCALAPPDATA%" "%USERPROFILE%" "%ProgramData%" "%ProgramFiles%" "C:" "D:" "E:") do (
  for %%d in (miniforge3 mambaforge miniconda3 anaconda3 miniconda anaconda conda micromamba) do (
    call :scan_root "%%~b\%%d\envs"
  )
)
exit /b

rem Every subfolder of one env root that matches !PATTERN!.
:scan_root
if defined PY exit /b
if not exist "%~1\" exit /b
if defined FW_DEBUG echo [debug] root %~1
for /d %%e in ("%~1\!PATTERN!") do call :check "%%~fe\python.exe"
exit /b

rem Test one interpreter. Pass "force" to skip the cheap site-packages pre-filter.
:check
if defined PY exit /b
set "cand=%~1"
if not exist "%cand%" exit /b
if /i not "%~2"=="force" (
  set "sp="
  if exist "%~dp1Lib\site-packages\PySide6\" set "sp=1"
  if exist "%~dp1..\Lib\site-packages\PySide6\" set "sp=1"
  if not defined sp (
    if defined FW_DEBUG echo [debug] skip %cand%  ^(no PySide6 in site-packages^)
    exit /b
  )
)
if defined FW_DEBUG echo [debug] test %cand%
"%cand%" -c "import PySide6" >nul 2>&1
if errorlevel 1 exit /b
if defined LISTONLY (
  rem the roots overlap (a micromamba env is reachable two ways), so list each interpreter once
  echo !SEEN! | findstr /I /L /C:"!cand!" >nul
  if errorlevel 1 (
    echo     !cand!
    set "SEEN=!SEEN!;!cand!"
  )
  exit /b
)
set "PY=%cand%"
exit /b

rem Conda/micromamba know about envs_dirs entries we cannot guess; also try PATH.
:ask_managers
if defined FW_DEBUG echo [debug] asking conda/micromamba for their env lists
for /f "usebackq delims=" %%l in (`conda env list --json 2^>nul`) do call :json_line "%%l"
for /f "usebackq delims=" %%l in (`micromamba env list --json 2^>nul`) do call :json_line "%%l"
for /f "usebackq delims=" %%l in (`where python 2^>nul`) do call :check "%%l"
for /f "usebackq delims=" %%l in (`where python3 2^>nul`) do call :check "%%l"
exit /b

rem One line of "<manager> env list --json": strip quotes/commas, unescape \\.
:json_line
if defined PY exit /b
set ln=%*
set ln=!ln:"=!
set ln=!ln:,=!
set ln=!ln:\\=\!
for /f "tokens=* delims= " %%p in ("!ln!") do (
  if exist "%%p\python.exe" call :check "%%p\python.exe"
)
exit /b
