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
"%PYTHON_EXE%" -c "import importlib.metadata as m, customtkinter, whisper, pyaudio, keyboard, pyperclip, torch, torchaudio, soundfile, numpy; m.version('openai-whisper')" >nul 2>nul
if not %ERRORLEVEL% EQU 0 (
    echo [setup] Installing or repairing dependencies...
    "%PYTHON_EXE%" -m pip install --upgrade pip "setuptools<81" || goto :error
    "%PYTHON_EXE%" -m pip install --no-build-isolation -r requirements.txt || goto :error
)

echo [run] Starting Whisper Widget...
"%PYTHON_EXE%" whisper_widget.py
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

:error
echo [error] Failed to create or use %VENV_DIR%.
echo [hint] In %VENV_DIR%, run:
echo        python -m pip install --upgrade pip "setuptools^<81"
echo        python -m pip install --no-build-isolation -r requirements.txt
endlocal & exit /b 1
