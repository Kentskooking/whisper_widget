"""Shared locations for source modules and local application data."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_MODULE = "app.desktop"
VAD_WORKER_MODULE = "app.workers.vad"
WHISPER_WORKER_MODULE = "app.workers.transcribe"
CLIPBOARD_WORKER_MODULE = "app.workers.clipboard"

LOG_DIR = Path("data/transcriptions")
DEBUG_AUDIO_DIR = Path("data/debug_audio")
RAW_AUDIO_BACKUP_DIR = Path("data/raw_audio_backups")
FAILED_AUDIO_DIR = Path("data/failed_recordings")
RUNTIME_LOG_DIR = Path("runtime/logs")
RUNTIME_STATE_DIR = Path("runtime/state")
RECORDING_WORK_DIR = Path("runtime/work/recording")
CHUNKED_TEMP_DIR = Path("runtime/work/chunks")
WEB_WORK_DIR = Path("runtime/work/web")
EVENT_LOG_FILE = RUNTIME_LOG_DIR / "event_log.txt"
WEB_EVENT_LOG_FILE = RUNTIME_LOG_DIR / "web_event_log.txt"
WIDGET_HEARTBEAT_FILE = "widget_heartbeat.json"
WIDGET_RECOVERY_FILE = "widget_recovery.json"
FATAL_PYTHON_LOG_FILE = "python_fatal.log"
MIGRATION_MARKER = Path("runtime/data_migration_in_progress.json")

DATA_PLACEHOLDERS = {
    LOG_DIR: "transcriptions_will_appear_here.txt",
    DEBUG_AUDIO_DIR: "debug_audio_will_appear_here.txt",
    RAW_AUDIO_BACKUP_DIR: "raw_audio_backups_will_appear_here.txt",
    FAILED_AUDIO_DIR: "failed_recordings_will_appear_here.txt",
}

# Old relative paths map to their new locations; used only by explicit migration.
LEGACY_DIRECTORIES = {
    "transcriptions": LOG_DIR,
    "debug_audio": DEBUG_AUDIO_DIR,
    "raw_audio_backups": RAW_AUDIO_BACKUP_DIR,
    "chunked_transcription_work": CHUNKED_TEMP_DIR,
    "web_transcription_work": WEB_WORK_DIR,
}


def legacy_data_moves(root=REPO_ROOT):
    root = Path(root)
    candidates = [(root / old, root / new) for old, new in LEGACY_DIRECTORIES.items()]
    for pattern, destination in (
        ("event_log*.txt", RUNTIME_LOG_DIR),
        ("web_event_log*.txt", RUNTIME_LOG_DIR),
        ("temp_*.wav", RECORDING_WORK_DIR),
        ("failed_recording_*.wav", FAILED_AUDIO_DIR),
        ("failed_speech_only_*.wav", FAILED_AUDIO_DIR),
    ):
        candidates.extend((path, root / destination / path.name) for path in sorted(root.glob(pattern)))
    for path in sorted((root / "sidecache/runtime").glob("*")):
        destination = RUNTIME_LOG_DIR if path.name in ("supervisor_log.txt", FATAL_PYTHON_LOG_FILE) else RUNTIME_STATE_DIR
        candidates.append((path, root / destination / path.name))
    return [(source, target) for source, target in candidates if source.exists() or source.is_symlink()]


def check_data_layout(root=REPO_ROOT):
    """Stop before creating output if an explicit migration is needed."""
    root = Path(root)
    if (root / MIGRATION_MARKER).exists():
        raise RuntimeError(
            f"An unfinished data migration needs inspection: {root / MIGRATION_MARKER}. "
            "See docs/storage.md; startup stopped to preserve recovery data."
        )
    moves = legacy_data_moves(root)
    if moves:
        names = ", ".join(str(source.relative_to(root)) for source, _ in moves)
        raise RuntimeError(
            f"Legacy generated data found: {names}. Stop Whisper, then run "
            ".venv\\Scripts\\python.exe -m tools.migrate_data to preview migration; "
            "add --apply to move it. See docs/storage.md."
        )
