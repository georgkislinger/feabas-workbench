@echo off
setlocal enabledelayedexpansion
rem ---------------------------------------------------------------------------
rem FEABAS Workbench installer (Windows)
rem
rem Creates the GUI environment 'feabas-workbench' with micromamba / mamba / conda
rem (downloading a standalone micromamba if the PC has none - no Python needed),
rem installs the workbench into it, records that interpreter in start_gui.local.bat
rem (so start_gui.bat is hard-wired to this installation and only falls back to its
rem own search if the environment disappears), puts a shortcut on the Desktop and
rem starts the app. FEABAS and deep-learning environments are installed from the
rem Setup page inside the app (or by hand, see README).
rem
rem Knobs:
rem     set FW_CONDA=C:\path\to\micromamba.exe     use this package manager
rem     set FW_DEBUG=1                              show every place searched
rem ---------------------------------------------------------------------------
set "ENVNAME=feabas-workbench"
set "HERE=%~dp0.."
for %%i in ("%HERE%") do set "HERE=%%~fi"
set "CONDA="

rem --- 1. explicit override -----------------------------------------------------
if defined FW_CONDA call :try "%FW_CONDA%"

rem --- 2. package managers on PATH; micromamba/mamba first (faster, no Anaconda channel terms)
for %%n in (micromamba.exe mamba.exe conda.exe mamba.bat conda.bat) do call :from_path %%n

rem --- 3. what 'conda init' / 'micromamba shell init' left in the environment ---
if defined CONDA_EXE call :try "%CONDA_EXE%"
if defined MAMBA_EXE call :try "%MAMBA_EXE%"
if defined MAMBA_ROOT_PREFIX call :try "%MAMBA_ROOT_PREFIX%\micromamba.exe"
if defined MAMBA_ROOT_PREFIX call :try "%MAMBA_ROOT_PREFIX%\Library\bin\micromamba.exe"

rem --- 4. the usual install folders, micromamba first, then conda-style roots on every drive
call :try "%LOCALAPPDATA%\micromamba\micromamba.exe"
call :try "%USERPROFILE%\micromamba\micromamba.exe"
call :try "%USERPROFILE%\micromamba\Library\bin\micromamba.exe"
call :try "%LOCALAPPDATA%\Programs\micromamba\micromamba.exe"
call :try "%USERPROFILE%\.local\bin\micromamba.exe"
call :try "%APPDATA%\FeabasWorkbench\micromamba\micromamba.exe"
for %%b in ("%LOCALAPPDATA%" "%USERPROFILE%" "%ProgramData%" "%ProgramFiles%" "C:" "D:" "E:") do (
  for %%d in (miniforge3 mambaforge miniconda3 anaconda3 miniconda anaconda Miniconda3 Anaconda3 micromamba) do (
    call :try "%%~b\%%d\Scripts\conda.exe"
    call :try "%%~b\%%d\condabin\conda.bat"
    call :try "%%~b\%%d\Library\bin\micromamba.exe"
    call :try "%%~b\%%d\micromamba.exe"
  )
)

rem --- 5. nothing at all (a fresh PC): fetch a standalone micromamba, like the Setup page does ----
if not defined CONDA call :get_micromamba

if not defined CONDA (
  echo.
  echo No conda, mamba or micromamba found: looked on PATH, in CONDA_EXE / MAMBA_EXE, and in the usual
  echo install folders on C:, D: and E: ^(set FW_DEBUG=1 to see every path tried^), and the micromamba
  echo download failed. Check the internet connection, install Miniforge from
  echo https://conda-forge.org/download/, or point this script at a package manager:
  echo     set FW_CONDA=C:\path\to\micromamba.exe
  echo.
  pause
  exit /b 1
)
echo Using %CONDA%

rem --- does the environment exist? Ask it for its interpreter (that also tells us where it lives:
rem     micromamba and conda keep their environments in different folders). --------------------
call :env_python
if not defined PY (
  echo Creating environment %ENVNAME% ...
  call "%CONDA%" create -y -n %ENVNAME% --override-channels -c conda-forge python=3.11 pip
  if errorlevel 1 goto :fail
  call :env_python
)
if not defined PY (
  echo Could not determine the python.exe of environment %ENVNAME%.
  goto :fail
)
echo Installing the workbench from %HERE%
echo                       into %PY%
"%PY%" -m pip install --index-url https://pypi.org/simple -e "%HERE%"
if errorlevel 1 goto :fail

rem --- hard-wire the launcher to this installation ----------------------------------------------
rem start_gui.bat reads start_gui.local.bat before it searches for environments, so this
rem interpreter is used from now on; if it ever disappears the launcher's own search takes over.
set "LOCAL=%HERE%\start_gui.local.bat"
> "%LOCAL%" echo @rem Written by tools\install.bat on %DATE% %TIME:~0,5%: the environment the workbench was installed into.
>>"%LOCAL%" echo @rem start_gui.bat reads this first. Delete the file to go back to automatic discovery.
>>"%LOCAL%" echo set "FW_PYTHON=%PY%"
echo Recorded the interpreter in %LOCAL%

rem --- desktop shortcut. Writing a small .ps1 and running it with -File avoids all quoting problems:
rem -Command would make PowerShell re-parse the string and an apostrophe in a path (e.g. a profile
rem like C:\Users\Ben's) would break it. The paths are passed as environment variables instead.
set "SC=%USERPROFILE%\Desktop\FEABAS Workbench.lnk"
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
echo Done. Starting the workbench (later: start_gui.bat or the Desktop shortcut).
"%PY%" -m feabas_workbench
exit /b 0

:fail
echo Installation failed. See messages above.
pause
exit /b 1

rem ===========================================================================
rem Take the first candidate that exists.
:try
if defined CONDA exit /b
if defined FW_DEBUG echo [debug] try %~1
if exist "%~1" set "CONDA=%~1"
exit /b

rem Download a standalone micromamba (single exe, no installer) into the workbench's settings folder,
rem where the Setup page looks for it too. Needs curl.exe and tar.exe, both part of Windows 10 1803+.
:get_micromamba
set "MMDIR=%APPDATA%\FeabasWorkbench\micromamba"
if exist "%MMDIR%\Library\bin\micromamba.exe" (
  set "CONDA=%MMDIR%\Library\bin\micromamba.exe"
  exit /b
)
where curl.exe >nul 2>&1 || exit /b
where tar.exe >nul 2>&1 || exit /b
echo No conda/mamba/micromamba found - downloading micromamba to %MMDIR% ...
if not exist "%MMDIR%" mkdir "%MMDIR%"
curl.exe -L --fail --silent --show-error "https://micro.mamba.pm/api/micromamba/win-64/latest" -o "%MMDIR%\micromamba.tar.bz2"
if errorlevel 1 exit /b
tar.exe -xf "%MMDIR%\micromamba.tar.bz2" -C "%MMDIR%"
if errorlevel 1 exit /b
del "%MMDIR%\micromamba.tar.bz2" >nul 2>&1
if exist "%MMDIR%\Library\bin\micromamba.exe" set "CONDA=%MMDIR%\Library\bin\micromamba.exe"
exit /b

rem Look a file name up on PATH (%~$PATH:1 does the search).
:from_path
if defined CONDA exit /b
if defined FW_DEBUG echo [debug] PATH lookup %1
if not "%~$PATH:1"=="" set "CONDA=%~$PATH:1"
exit /b

rem The interpreter of environment ENVNAME, via the package manager itself, or nothing.
rem Not via `for /f ('...')`: that runs the command through `cmd /c`, which strips the outer quotes
rem of a quoted command line and breaks it. Redirecting to a file avoids that entirely.
:env_python
set "PY="
set "PYFILE=%TEMP%\feabas_workbench_python.txt"
call "%CONDA%" run -n %ENVNAME% python -c "import sys; print(sys.executable)" > "%PYFILE%" 2>nul
if errorlevel 1 (
  del "%PYFILE%" >nul 2>&1
  exit /b
)
set /p PY=<"%PYFILE%"
del "%PYFILE%" >nul 2>&1
if defined PY if not exist "%PY%" set "PY="
exit /b
