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
rem conda.exe on PATH wins; otherwise look in the usual install roots (same list as core\envs.py).
rem %%~$PATH:n is unquoted on purpose (the if argument needs the bare path), so the whole
rem comparison must be quoted instead: an empty result would otherwise read `if not ==""`.
for %%n in (conda.exe mamba.exe micromamba.exe) do (
  if not defined CONDA if not "%%~$PATH:n" == "" set "CONDA=%%~$PATH:n"
)
for %%b in ("%LOCALAPPDATA%" "%USERPROFILE%" "C:\ProgramData" "C:") do (
  for %%d in (miniforge3 mambaforge miniconda3 anaconda3 Miniconda3 Anaconda3) do (
    if exist "%%~b\%%d\Scripts\conda.exe" if not defined CONDA set "CONDA=%%~b\%%d\Scripts\conda.exe"
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
rem Ask the environment where its python.exe is (it may live under <conda>\envs or ~\.conda\envs).
rem Not via `for /f ('...')`: that runs the command through `cmd /c`, which strips the outer quotes
rem of a quoted command line and breaks it. Redirecting to a file avoids that entirely.
set PY=
set PYFILE=%TEMP%\feabas_workbench_python.txt
"%CONDA%" run -n %ENVNAME% python -c "import sys; print(sys.executable)" > "%PYFILE%"
if errorlevel 1 goto :fail
set /p PY=<"%PYFILE%"
del "%PYFILE%" >nul 2>&1
if not defined PY (
  echo Could not determine the python.exe of environment %ENVNAME%.
  goto :fail
)
if not exist "%PY%" (
  echo The reported interpreter does not exist: %PY%
  goto :fail
)
echo Installing the workbench into %PY%
"%PY%" -m pip install --index-url https://pypi.org/simple -e "%HERE%"
if errorlevel 1 goto :fail
rem desktop shortcut. Writing a small .ps1 and running it with -File avoids all quoting problems:
rem -Command would make PowerShell re-parse the string and an apostrophe in a path (e.g. a profile
rem like C:\Users\Ben's) would break it. The paths are passed as environment variables instead.
set SC=%USERPROFILE%\Desktop\FEABAS Workbench.lnk
set "PS1=%TEMP%\feabas_workbench_shortcut.ps1"
> "%PS1%" echo $s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:FW_SC)
>>"%PS1%" echo $s.TargetPath = $env:FW_PY
>>"%PS1%" echo $s.Arguments = '-m feabas_workbench'
>>"%PS1%" echo $s.WorkingDirectory = $env:FW_HERE
>>"%PS1%" echo $s.Save()
set "FW_SC=%SC%" & set "FW_PY=%PY%" & set "FW_HERE=%HERE%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
if errorlevel 1 echo Could not write the Desktop shortcut - start the app with start_gui.bat instead.
del "%PS1%" >nul 2>&1
echo.
echo Done. Starting the workbench (also on the Desktop shortcut).
"%PY%" -m feabas_workbench
exit /b 0
:fail
echo Installation failed. See messages above.
pause
exit /b 1
