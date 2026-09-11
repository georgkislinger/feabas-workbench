@echo off
setlocal
rem ---------------------------------------------------------------------------
rem FEABAS Workbench installer (Windows)
rem Creates the GUI environment with conda (Miniforge/Miniconda) and starts the
rem workbench. FEABAS and deep-learning environments are installed from the
rem Setup page inside the app (or by hand, see README).
rem ---------------------------------------------------------------------------
set ENVNAME=feabas-workbench
set HERE=%~dp0..
set CONDA=
rem conda.exe on PATH wins; otherwise look in the usual install roots (same list as core\envs.py)
for %%n in (conda.exe mamba.exe micromamba.exe) do (
  if not defined CONDA if not "%%~$PATH:n"=="" set CONDA=%%~$PATH:n
)
for %%b in ("%LOCALAPPDATA%" "%USERPROFILE%" "C:\ProgramData" "C:") do (
  for %%d in (miniforge3 mambaforge miniconda3 anaconda3 Miniconda3 Anaconda3) do (
    if exist "%%~b\%%d\Scripts\conda.exe" if not defined CONDA set CONDA=%%~b\%%d\Scripts\conda.exe
  )
)
if not defined CONDA (
  echo No conda found. Install Miniforge from https://conda-forge.org/download/ and re-run.
  pause
  exit /b 1
)
echo Using %CONDA%
"%CONDA%" env list | findstr /B /C:"%ENVNAME% " >nul
if errorlevel 1 (
  echo Creating environment %ENVNAME% ...
  "%CONDA%" create -y -n %ENVNAME% --override-channels -c conda-forge python=3.11 pip
  if errorlevel 1 goto :fail
)
for /f "delims=" %%i in ('"%CONDA%" run -n %ENVNAME% python -c "import sys; print(sys.executable)"') do set PY=%%i
echo Installing the workbench into %PY%
"%PY%" -m pip install --index-url https://pypi.org/simple -e "%HERE%"
if errorlevel 1 goto :fail
rem desktop shortcut
set SC=%USERPROFILE%\Desktop\FEABAS Workbench.lnk
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SC%'); $s.TargetPath='%PY%'; $s.Arguments='-m feabas_workbench'; $s.WorkingDirectory='%HERE%'; $s.Save()"
echo.
echo Done. Starting the workbench (also on the Desktop shortcut).
"%PY%" -m feabas_workbench
exit /b 0
:fail
echo Installation failed. See messages above.
pause
exit /b 1
