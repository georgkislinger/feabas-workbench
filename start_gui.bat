@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------------------
rem Start the FEABAS Workbench GUI (Windows).
rem
rem Double-click, or from a terminal:
rem     start_gui.bat                       open the last project
rem     start_gui.bat --project D:\path      open a specific project folder
rem     start_gui.bat --page 3               jump straight to a window (0..6)
rem
rem Looks for a .venv in this folder or a conda environment that has PySide6
rem installed. To force one:
rem     set FW_PYTHON=C:\path\to\env\python.exe  &  start_gui.bat
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

set PY=
if defined FW_PYTHON (
  if exist "%FW_PYTHON%" set PY=%FW_PYTHON%
)

rem A plain venv in the repository root, if there is one.
if not defined PY (
  if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import PySide6" >nul 2>&1
    if !errorlevel! equ 0 set PY=%~dp0.venv\Scripts\python.exe
  )
)

rem Conda environments. They live either under <conda root>\envs or - when conda's
rem envs_dirs puts them there - under %USERPROFILE%\.conda\envs, so scan both.
if not defined PY (
  set ENVROOTS="%USERPROFILE%\.conda\envs"
  for %%b in ("%LOCALAPPDATA%" "%USERPROFILE%" "C:\ProgramData" "C:") do (
    for %%d in (miniforge3 mambaforge miniconda3 anaconda3 Miniconda3 Anaconda3) do (
      if exist "%%~b\%%d\envs" set ENVROOTS=!ENVROOTS! "%%~b\%%d\envs"
    )
  )
  for %%r in (!ENVROOTS!) do (
    for %%e in (feabas-gui feabas-workbench feabas_env feabas) do (
      if not defined PY (
        if exist "%%~r\%%e\python.exe" (
          "%%~r\%%e\python.exe" -c "import PySide6" >nul 2>&1
          if !errorlevel! equ 0 set PY=%%~r\%%e\python.exe
        )
      )
    )
  )
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
  pause
  exit /b 1
)

echo Starting FEABAS Workbench with "%PY%"
"%PY%" -m feabas_workbench %*
if errorlevel 1 (
  echo.
  echo The workbench exited with an error - see the messages above.
  pause
)
exit /b %errorlevel%
