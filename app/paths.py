"""Repository paths and process entry points shared by the application."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_MODULE = "app.desktop"
VAD_WORKER_MODULE = "app.workers.vad"
WHISPER_WORKER_MODULE = "app.workers.transcribe"
CLIPBOARD_WORKER_MODULE = "app.workers.clipboard"
