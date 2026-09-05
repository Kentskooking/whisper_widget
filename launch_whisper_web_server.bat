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

echo [check] Application modules...
"%PYTHON_EXE%" -u -c "import sys, importlib.util; import app.desktop, app.supervisor, app.transcription_service, app.web.embedded, app.workers.clipboard; modules = ('app.workers.vad', 'app.workers.transcribe', 'app.web.standalone'); missing = [name for name in modules if importlib.util.find_spec(name) is None]; sys.exit('Missing application modules: ' + ', '.join(missing)) if missing else None"
if errorlevel 1 goto :error

echo [check] Generated data layout...
"%PYTHON_EXE%" -u -c "from app.paths import check_data_layout; check_data_layout()"
if errorlevel 1 goto :error

echo [check] Package consistency...
"%PYTHON_EXE%" -m pip check
if errorlevel 1 goto :error

echo [check] FFmpeg...
"%PYTHON_EXE%" -u -c "import sys, shutil, subprocess; executable = shutil.which('ffmpeg'); sys.exit('FFmpeg is missing from PATH; open a new terminal after installation') if not executable else None; print('FFmpeg:', executable); subprocess.run([executable, '-version'], check=True)"
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

echo [run] Starting Whisper Web Server at %WHISPER_WEB_SCHEME%://%WHISPER_WEB_HOST%:%WHISPER_WEB_PORT% ...
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
