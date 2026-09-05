@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto :error

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [error] Missing project Python: "%PYTHON_EXE%"
    echo [hint] Follow the explicit uv setup steps in README.md.
    goto :error
)

echo [check] Project Python...
"%PYTHON_EXE%" -u -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable); sys.exit('This project requires Python 3.11') if sys.version_info[:2] != (3, 11) else None; sys.exit('This project requires 64-bit Python') if sys.maxsize <= 2**32 else None"
if errorlevel 1 goto :error

echo [check] Required packages and NVIDIA CUDA...
"%PYTHON_EXE%" -u -c "import sys, importlib.metadata as m, customtkinter, whisper, pyaudio, keyboard, torch, torchaudio, soundfile, numpy, fastapi, uvicorn, multipart; print('Whisper:', m.version('openai-whisper')); print('PyTorch:', torch.__version__); sys.exit('NVIDIA CUDA is unavailable; startup stopped') if not torch.cuda.is_available() else None; print('GPU:', torch.cuda.get_device_name(0)); torch.zeros(1, device='cuda:0').item(); sys.exit('The SoundFile audio backend is unavailable') if 'soundfile' not in torchaudio.list_audio_backends() else None"
if errorlevel 1 goto :error

echo [check] Package consistency...
"%PYTHON_EXE%" -m pip check
if errorlevel 1 goto :error

echo [check] FFmpeg...
"%PYTHON_EXE%" -u -c "import sys, shutil, subprocess; executable = shutil.which('ffmpeg'); sys.exit('FFmpeg is missing from PATH; open a new terminal after installation') if not executable else None; print('FFmpeg:', executable); subprocess.run([executable, '-version'], check=True)"
if errorlevel 1 goto :error

if /i "%~1"=="--check" (
    echo [check] Startup checks passed.
    endlocal & exit /b 0
)

echo [run] Starting Whisper Widget supervisor...
echo [run] Supervisor restarts are reported here and in sidecache\runtime\supervisor_log.txt.
"%PYTHON_EXE%" -u whisper_supervisor.py
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
