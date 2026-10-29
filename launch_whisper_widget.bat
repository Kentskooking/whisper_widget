@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [error] Missing project Python: "%PYTHON_EXE%"
    echo [hint] Follow the setup steps in README.md.
    goto :error
)

echo [check] Project runtime and model files...
"%PYTHON_EXE%" -u -m app.preflight
if errorlevel 1 goto :error
"%PYTHON_EXE%" -m pip check
if errorlevel 1 goto :error

if /i "%~1"=="--check" (
    echo [check] Startup checks passed.
    endlocal & exit /b 0
)

echo [run] Starting Bark Widget supervisor...
echo [run] Supervisor restarts are reported here and in runtime\logs\supervisor_log.txt.
"%PYTHON_EXE%" -u -m app.launcher
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :runtime_error
endlocal & exit /b 0

:error
echo [error] Startup validation failed. See the original error above.
echo [hint] Setup is explicit; the launcher does not create or repair the environment.
set "EXIT_CODE=1"
goto :show_error

:runtime_error
echo [error] Application exited with code %EXIT_CODE%.

:show_error
if /i not "%~1"=="--check" pause
endlocal & exit /b %EXIT_CODE%
