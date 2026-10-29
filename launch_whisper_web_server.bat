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
"%PYTHON_EXE%" -u -m app.preflight --web
if errorlevel 1 goto :error
"%PYTHON_EXE%" -m pip check
if errorlevel 1 goto :error

if defined WHISPER_WEB_SSL_CERTFILE if not defined WHISPER_WEB_SSL_KEYFILE (
    echo [error] WHISPER_WEB_SSL_CERTFILE requires WHISPER_WEB_SSL_KEYFILE.
    goto :error
)
if defined WHISPER_WEB_SSL_KEYFILE if not defined WHISPER_WEB_SSL_CERTFILE (
    echo [error] WHISPER_WEB_SSL_KEYFILE requires WHISPER_WEB_SSL_CERTFILE.
    goto :error
)

if /i "%~1"=="--check" (
    echo [check] Startup checks passed.
    endlocal & exit /b 0
)

if "%WHISPER_WEB_HOST%"=="" set "WHISPER_WEB_HOST=127.0.0.1"
if "%WHISPER_WEB_PORT%"=="" set "WHISPER_WEB_PORT=8765"
set "WHISPER_WEB_SCHEME=http"
if defined WHISPER_WEB_SSL_CERTFILE set "WHISPER_WEB_SCHEME=https"

echo [run] Starting Bark Web Server at %WHISPER_WEB_SCHEME%://%WHISPER_WEB_HOST%:%WHISPER_WEB_PORT% ...
if "%WHISPER_WEB_SCHEME%"=="https" (
    echo [run] HTTPS enabled with %WHISPER_WEB_SSL_CERTFILE%
    "%PYTHON_EXE%" -u -m app.web.standalone --host "%WHISPER_WEB_HOST%" --port "%WHISPER_WEB_PORT%" --ssl-certfile "%WHISPER_WEB_SSL_CERTFILE%" --ssl-keyfile "%WHISPER_WEB_SSL_KEYFILE%"
) else (
    "%PYTHON_EXE%" -u -m app.web.standalone --host "%WHISPER_WEB_HOST%" --port "%WHISPER_WEB_PORT%"
)
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
