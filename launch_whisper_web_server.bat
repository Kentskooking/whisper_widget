@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [setup] Creating virtual environment in %VENV_DIR%...
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        py -3 -m venv "%VENV_DIR%" || goto :error
    ) else (
        python -m venv "%VENV_DIR%" || goto :error
    )
)

if not exist "%PYTHON_EXE%" goto :error

echo [setup] Checking required packages...
"%PYTHON_EXE%" -c "import importlib.metadata as m, fastapi, uvicorn, multipart, whisper, torch, torchaudio, numpy; m.version('openai-whisper')" >nul 2>nul
if not %ERRORLEVEL% EQU 0 (
    echo [setup] Installing or repairing dependencies...
    "%PYTHON_EXE%" -m pip install --upgrade pip "setuptools<81" || goto :error
    "%PYTHON_EXE%" -m pip install --no-build-isolation -r requirements.txt || goto :error
)

if "%WHISPER_WEB_HOST%"=="" set "WHISPER_WEB_HOST=127.0.0.1"
if "%WHISPER_WEB_PORT%"=="" set "WHISPER_WEB_PORT=8765"
set "WHISPER_WEB_SCHEME=http"
if not "%WHISPER_WEB_SSL_CERTFILE%"=="" if not "%WHISPER_WEB_SSL_KEYFILE%"=="" set "WHISPER_WEB_SCHEME=https"

echo [run] Starting Whisper Web Server at %WHISPER_WEB_SCHEME%://%WHISPER_WEB_HOST%:%WHISPER_WEB_PORT% ...
if "%WHISPER_WEB_SCHEME%"=="https" (
    echo [run] HTTPS enabled with %WHISPER_WEB_SSL_CERTFILE%
    "%PYTHON_EXE%" web_server.py --host "%WHISPER_WEB_HOST%" --port "%WHISPER_WEB_PORT%" --ssl-certfile "%WHISPER_WEB_SSL_CERTFILE%" --ssl-keyfile "%WHISPER_WEB_SSL_KEYFILE%"
) else (
    "%PYTHON_EXE%" web_server.py --host "%WHISPER_WEB_HOST%" --port "%WHISPER_WEB_PORT%"
)
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

:error
echo [error] Failed to create or use %VENV_DIR%.
echo [hint] In %VENV_DIR%, run:
echo        python -m pip install --upgrade pip "setuptools^<81"
echo        python -m pip install --no-build-isolation -r requirements.txt
endlocal & exit /b 1
