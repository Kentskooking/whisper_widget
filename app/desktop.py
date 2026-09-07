import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import ctypes
import customtkinter as ctk
import tkinter as tk
import pyaudio
import wave
import threading
import time
import sys
from app.paths import (
    REPO_ROOT, VAD_WORKER_MODULE, WHISPER_WORKER_MODULE, CLIPBOARD_WORKER_MODULE,
    LOG_DIR, EVENT_LOG_FILE, DEBUG_AUDIO_DIR, RAW_AUDIO_BACKUP_DIR, FAILED_AUDIO_DIR,
    CHUNKED_TEMP_DIR, WEB_WORK_DIR, RECORDING_WORK_DIR, RUNTIME_STATE_DIR,
    RUNTIME_LOG_DIR, WIDGET_HEARTBEAT_FILE, WIDGET_RECOVERY_FILE,
    FATAL_PYTHON_LOG_FILE, check_data_layout,
)
import shutil
import subprocess
import json
import re
import math
import struct
import io
import tempfile
import warnings
import faulthandler
from queue import Queue, Empty, Full
from ctypes import wintypes
from datetime import datetime

try:
    import winsound
except Exception:
    winsound = None

try:
    import keyboard
except Exception:
    keyboard = None

# Suppress Torch FutureWarnings (cleaner UI)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "auto"
WHISPER_WORKER_READY_TIMEOUT_SECONDS = 120
WHISPER_TRANSCRIBE_TIMEOUT_SECONDS = 180
HOTKEY = "f8"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
EVENT_LOG_MAX_BYTES = 5 * 1024 * 1024
EVENT_LOG_BACKUP_COUNT = 2
EVENT_LOG_HEADER = "timestamp | event | details\n"
SAVE_DEBUG_AUDIO = True
WHISPER_LANGUAGE = None
WHISPER_NO_SPEECH_THRESHOLD = None
VAD_PAD_MS = 400
VAD_MIN_SPEECH_MS = 150
VAD_MERGE_GAP_MS = 600
VAD_REQUEST_TIMEOUT_SECONDS = 180
VAD_WORKER_READY_TIMEOUT_SECONDS = 30
CHUNKED_TRANSCRIPTION_ENABLED = True
CHUNKED_CHUNK_SECONDS = 30.0
CHUNKED_OVERLAP_SECONDS = 1.5
CHUNKED_QUEUE_MAXSIZE = 0
CHUNKED_FINALIZE_TIMEOUT_SECONDS = 60
CHUNKED_KEEP_TEMP_ON_SUCCESS = False
CHUNKED_KEEP_TEMP_ON_FAILURE = True
CHUNKED_SAVE_DEBUG_ARTIFACTS = True
CHUNKED_WHISPER_MAX_ATTEMPTS = 2
WEB_SERVER_ENABLED = os.environ.get("WHISPER_WIDGET_WEB_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
WEB_SERVER_HOST = os.environ.get("WHISPER_WEB_HOST", "127.0.0.1")
WEB_SERVER_PORT = int(os.environ.get("WHISPER_WEB_PORT", "8765"))
WEB_SERVER_SSL_CERTFILE = os.environ.get("WHISPER_WEB_SSL_CERTFILE") or None
WEB_SERVER_SSL_KEYFILE = os.environ.get("WHISPER_WEB_SSL_KEYFILE") or None
CLIPBOARD_COPY_TIMEOUT_SECONDS = 1.0
SOUND_CUE_TONES = {
    "notification": [(660, 80)],
    "ready": [(880, 80)],
    "unmute": [(880, 60)],
    "record_start": [(660, 80)],
    "record_stop": [(440, 80)],
    "success": [(660, 80), (880, 80)],
}

# Win32 constants for stable global hotkey + topmost behavior
IS_WINDOWS = os.name == "nt"
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_TOPMOST_FLAGS = SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW
HWND_TOPMOST = -1
FILE_ATTRIBUTE_HIDDEN = 0x2
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.RegisterHotKey.argtypes = [wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL

    user32.UnregisterHotKey.argtypes = [wintypes.HWND, wintypes.INT]
    user32.UnregisterHotKey.restype = wintypes.BOOL

    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int

    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL

    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.UINT
    ]
    user32.SetWindowPos.restype = wintypes.BOOL

    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD

    kernel32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.SetFileAttributesW.restype = wintypes.BOOL

    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
else:
    user32 = None
    kernel32 = None


_fatal_log_file = None


def enable_fatal_crash_logging():
    global _fatal_log_file

    try:
        runtime_dir = os.path.join(str(REPO_ROOT), RUNTIME_LOG_DIR)
        os.makedirs(runtime_dir, exist_ok=True)
        crash_log_path = os.path.join(runtime_dir, FATAL_PYTHON_LOG_FILE)
        _fatal_log_file = open(crash_log_path, "a", encoding="utf-8", buffering=1)
        _fatal_log_file.write(
            f"\n{datetime.now().isoformat(timespec='seconds')} | fatal_logging_enabled | pid={os.getpid()}\n"
        )
        faulthandler.enable(file=_fatal_log_file, all_threads=True)
    except Exception:
        _fatal_log_file = None


def hotkey_token_to_vk(token: str):
    token = token.lower()
    named_keys = {
        "esc": 0x1B,
        "escape": 0x1B,
        "tab": 0x09,
        "enter": 0x0D,
        "return": 0x0D,
        "space": 0x20,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "insert": 0x2D,
        "delete": 0x2E,
        "home": 0x24,
        "end": 0x23,
        "pageup": 0x21,
        "pgup": 0x21,
        "pagedown": 0x22,
        "pgdn": 0x22,
    }
    if token in named_keys:
        return named_keys[token]

    if len(token) == 1 and token.isalpha():
        return ord(token.upper())
    if len(token) == 1 and token.isdigit():
        return ord(token)

    if token.startswith("f") and token[1:].isdigit():
        num = int(token[1:])
        if 1 <= num <= 24:
            return 0x70 + (num - 1)

    return None


def parse_hotkey_for_win32(hotkey: str):
    tokens = [tok.strip().lower() for tok in hotkey.split("+") if tok.strip()]
    if not tokens:
        return None

    mods = MOD_NOREPEAT
    vk = None
    for token in tokens:
        if token in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif token == "shift":
            mods |= MOD_SHIFT
        elif token in ("alt", "menu"):
            mods |= MOD_ALT
        elif token in ("win", "windows", "super"):
            mods |= MOD_WIN
        else:
            if vk is not None:
                return None
            vk = hotkey_token_to_vk(token)
            if vk is None:
                return None

    if vk is None:
        return None
    return mods, vk


def write_json_atomic(path: str, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    last_error = None

    for attempt in range(6):
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(temp_path, path)
            return
        except OSError as e:
            last_error = e
            winerror = getattr(e, "winerror", None)
            if winerror not in (5, 32):
                break
            time.sleep(min(0.5, 0.05 * (2 ** attempt)))
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    if last_error is not None:
        raise last_error


class EventLogger:
    """Thread-safe event logger for runtime diagnostics."""
    def __init__(self, path: str, max_bytes: int = EVENT_LOG_MAX_BYTES, backup_count: int = EVENT_LOG_BACKUP_COUNT):
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = max(0, backup_count)
        self._lock = threading.Lock()
        self._enabled = True
        self._initialize()

    def _initialize(self):
        try:
            with self._lock:
                self._ensure_log_file_locked()
        except Exception as e:
            self._enabled = False
            print(f"Event log init failed: {e}")
            return

    def _archive_path(self, index: int) -> str:
        root, ext = os.path.splitext(self.path)
        return f"{root}.{index}{ext}"

    def _apply_hidden_attribute(self, target_path: str):
        if IS_WINDOWS:
            try:
                attrs = kernel32.GetFileAttributesW(target_path)
                if attrs != INVALID_FILE_ATTRIBUTES and not (attrs & FILE_ATTRIBUTE_HIDDEN):
                    kernel32.SetFileAttributesW(target_path, attrs | FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                print(f"Event log hide flag failed: {e}")

    def _write_header_locked(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(EVENT_LOG_HEADER)
        self._apply_hidden_attribute(self.path)

    def _rotate_locked(self):
        if self.backup_count > 0:
            oldest_archive = self._archive_path(self.backup_count)
            if os.path.exists(oldest_archive):
                os.remove(oldest_archive)

            for index in range(self.backup_count - 1, 0, -1):
                archive_path = self._archive_path(index)
                next_archive_path = self._archive_path(index + 1)
                if os.path.exists(archive_path):
                    os.replace(archive_path, next_archive_path)
                    self._apply_hidden_attribute(next_archive_path)

            if os.path.exists(self.path):
                first_archive_path = self._archive_path(1)
                os.replace(self.path, first_archive_path)
                self._apply_hidden_attribute(first_archive_path)
        elif os.path.exists(self.path):
            os.remove(self.path)

        self._write_header_locked()

    def _ensure_log_file_locked(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
            self._rotate_locked()
            return

        if not os.path.exists(self.path):
            self._write_header_locked()
            return

        if os.path.getsize(self.path) == 0:
            self._write_header_locked()
            return

        self._apply_hidden_attribute(self.path)

    def log(self, event: str, **fields):
        if not self._enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        safe_fields = []
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value).replace("\n", "\\n").replace("\r", "\\r").replace("|", "/")
            safe_fields.append(f"{key}={text}")

        details = " ".join(safe_fields)
        line = f"{timestamp} | {event}"
        if details:
            line += f" | {details}"
        line += "\n"

        try:
            with self._lock:
                self._ensure_log_file_locked()
                current_size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
                if current_size + len(line.encode("utf-8")) > self.max_bytes:
                    self._rotate_locked()
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            self._enabled = False
            print(f"Event log write failed: {e}")
class PersistentVADWorkerClient:
    """Keeps Silero VAD in a separate long-lived process to avoid repeated cold starts."""

    def __init__(self, worker_module: str, workdir: str, log_fn):
        self.worker_module = worker_module
        self.workdir = workdir
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._response_queue = Queue()
        self._ready_event = threading.Event()
        self._ready = False
        self._last_start_error = None
        self._request_id = 0

    def _log(self, event, **fields):
        try:
            self.log_fn(event, **fields)
        except Exception:
            pass

    def _is_current_process(self, process):
        return process is not None and process is self._process

    def _reset_state_locked(self):
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._response_queue = Queue()
        self._ready_event = threading.Event()
        self._ready = False
        self._last_start_error = None

    def _start_locked(self):
        command = [
            sys.executable,
            "-u",
            "-m",
            self.worker_module,
            "--server",
        ]
        self._reset_state_locked()
        process = subprocess.Popen(
            command,
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process
        self._stdout_thread = threading.Thread(target=self._stdout_loop, args=(process,), daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, args=(process,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._log("vad_worker_started", pid=process.pid)

    def _stdout_loop(self, process):
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except Exception as e:
                    self._log("vad_worker_stdout_invalid", pid=process.pid, error=e, line=line[-400:])
                    continue

                message_type = message.get("type")
                if message_type == "ready":
                    with self._lock:
                        is_current = self._is_current_process(process)
                        if is_current:
                            self._ready = bool(message.get("ok"))
                            self._last_start_error = None if self._ready else (message.get("error") or "unknown worker startup error")
                            self._ready_event.set()
                    if not is_current:
                        continue
                    if self._ready:
                        self._log("vad_worker_ready", pid=process.pid)
                    else:
                        self._log("vad_worker_ready_failed", pid=process.pid, error=self._last_start_error)
                elif message_type == "response":
                    self._response_queue.put(message)
                elif message_type == "shutdown_ack":
                    self._log("vad_worker_shutdown_ack", pid=process.pid)
                else:
                    self._log("vad_worker_message_unknown", pid=process.pid, message_type=message_type)
        finally:
            returncode = process.poll()
            with self._lock:
                is_current = self._is_current_process(process)
                if is_current:
                    self._ready = False
                    if not self._ready_event.is_set():
                        self._last_start_error = f"worker exited before ready (code {returncode})"
                        self._ready_event.set()
            if is_current:
                self._log("vad_worker_exited", pid=process.pid, returncode=returncode)

    def _stderr_loop(self, process):
        try:
            for raw_line in process.stderr:
                line = raw_line.strip()
                if line:
                    self._log("vad_worker_stderr", pid=process.pid, line=line[-400:])
        except Exception as e:
            self._log("vad_worker_stderr_failed", pid=getattr(process, "pid", None), error=e)

    def ensure_ready(self, timeout_seconds: int = VAD_WORKER_READY_TIMEOUT_SECONDS):
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._start_locked()
            ready_event = self._ready_event

        ready = ready_event.wait(timeout_seconds)
        with self._lock:
            process = self._process
            if not ready:
                self._terminate_locked(process, reason="startup_timeout")
                raise RuntimeError(f"VAD worker did not become ready within {timeout_seconds}s")
            if process is None or process.poll() is not None:
                error = self._last_start_error or "VAD worker exited during startup"
                raise RuntimeError(error)
            if not self._ready:
                raise RuntimeError(self._last_start_error or "VAD worker failed to start")

    def is_ready(self):
        with self._lock:
            return bool(self._ready and self._process is not None and self._process.poll() is None)

    def run(
        self,
        in_wav_path: str,
        out_wav_path: str,
        sample_rate: int = 16000,
        pad_ms: int = 250,
        min_speech_ms: int = 150,
        merge_gap_ms: int = 400,
        speech_prob_threshold: float = 0.5,
        timeout_seconds: int = VAD_REQUEST_TIMEOUT_SECONDS,
        return_details: bool = False,
    ) -> float | dict:
        self.ensure_ready()

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or not self._ready:
                raise RuntimeError("VAD worker is not available")

            self._request_id += 1
            request_id = self._request_id
            response_queue = self._response_queue
            request = {
                "type": "run",
                "request_id": request_id,
                "input": os.path.abspath(in_wav_path),
                "output": os.path.abspath(out_wav_path),
                "sample_rate": sample_rate,
                "pad_ms": pad_ms,
                "min_speech_ms": min_speech_ms,
                "merge_gap_ms": merge_gap_ms,
                "threshold": speech_prob_threshold,
            }
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
            except Exception as e:
                self._terminate_locked(process, reason="request_send_failed")
                raise RuntimeError("VAD worker pipe failed while sending request") from e

        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    if self._is_current_process(process):
                        self._terminate_locked(process, reason="request_timeout")
                raise RuntimeError(f"VAD worker timed out after {timeout_seconds}s")

            try:
                message = response_queue.get(timeout=remaining)
            except Empty:
                with self._lock:
                    process_alive = self._is_current_process(process) and process.poll() is None
                if not process_alive:
                    raise RuntimeError("VAD worker exited while processing audio")
                continue

            if message.get("request_id") != request_id:
                self._log(
                    "vad_worker_response_unexpected",
                    expected_request_id=request_id,
                    actual_request_id=message.get("request_id"),
                )
                continue

            if not message.get("ok"):
                error_text = message.get("error") or "VAD worker reported an unknown error"
                raise RuntimeError(error_text)

            speech_secs = float(message["speech_secs"])
            if return_details:
                return {
                    "speech_secs": speech_secs,
                    "segments": list(message.get("segments") or []),
                }
            return speech_secs

    def _terminate_locked(self, process, reason: str):
        if process is None:
            return
        if self._is_current_process(process):
            self._reset_state_locked()

        try:
            if process.poll() is None and process.stdin:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass

        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except Exception:
                pass

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2.0)
            except Exception:
                pass

        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass

        self._log("vad_worker_terminated", pid=getattr(process, "pid", None), reason=reason, returncode=process.poll())

    def close(self):
        with self._lock:
            process = self._process
            self._terminate_locked(process, reason="close")


class PersistentWhisperWorkerClient:
    """Keeps Whisper/PyTorch in a separate process so native crashes do not kill the UI."""

    def __init__(self, worker_module: str, workdir: str, model_size: str, device: str, log_fn):
        self.worker_module = worker_module
        self.workdir = workdir
        self.model_size = model_size
        self.device = device
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._response_queue = Queue()
        self._ready_event = threading.Event()
        self._ready = False
        self._last_start_error = None
        self._last_ready_payload = {}
        self._request_id = 0

    def _log(self, event, **fields):
        try:
            self.log_fn(event, **fields)
        except Exception:
            pass

    def _is_current_process(self, process):
        return process is not None and process is self._process

    def _reset_state_locked(self):
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._response_queue = Queue()
        self._ready_event = threading.Event()
        self._ready = False
        self._last_start_error = None
        self._last_ready_payload = {}

    def _start_locked(self):
        command = [
            sys.executable,
            "-u",
            "-m",
            self.worker_module,
            "--server",
            "--model",
            self.model_size,
            "--device",
            self.device,
        ]
        self._reset_state_locked()
        process = subprocess.Popen(
            command,
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process
        self._stdout_thread = threading.Thread(target=self._stdout_loop, args=(process,), daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, args=(process,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._log("whisper_worker_started", pid=process.pid, model=self.model_size, device=self.device)

    def _stdout_loop(self, process):
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except Exception as e:
                    self._log("whisper_worker_stdout_invalid", pid=process.pid, error=e, line=line[-400:])
                    continue

                message_type = message.get("type")
                if message_type == "ready":
                    ready_payload = dict(message)
                    ready_payload["launcher_pid"] = process.pid
                    ready_payload["worker_pid"] = message.get("pid")
                    with self._lock:
                        is_current = self._is_current_process(process)
                        if is_current:
                            self._ready = bool(message.get("ok"))
                            self._last_ready_payload = ready_payload
                            self._last_start_error = None if self._ready else (message.get("error") or "unknown worker startup error")
                            self._ready_event.set()
                    if not is_current:
                        continue
                    if self._ready:
                        self._log(
                            "whisper_worker_ready",
                            pid=process.pid,
                            launcher_pid=process.pid,
                            worker_pid=message.get("pid"),
                            model=message.get("model"),
                            device=message.get("device"),
                            load_device=message.get("load_device"),
                            model_dtype=message.get("model_dtype"),
                            layer_norm_dtype=message.get("layer_norm_dtype"),
                            cuda_allocated_mb=message.get("cuda_allocated_mb"),
                            cuda_reserved_mb=message.get("cuda_reserved_mb"),
                            load_seconds=f"{float(message.get('load_seconds') or 0.0):.3f}",
                        )
                    else:
                        self._log("whisper_worker_ready_failed", pid=process.pid, error=self._last_start_error)
                elif message_type == "response":
                    self._response_queue.put(message)
                elif message_type == "shutdown_ack":
                    self._log("whisper_worker_shutdown_ack", pid=process.pid)
                else:
                    self._log("whisper_worker_message_unknown", pid=process.pid, message_type=message_type)
        finally:
            returncode = process.poll()
            with self._lock:
                is_current = self._is_current_process(process)
                if is_current:
                    self._ready = False
                    if not self._ready_event.is_set():
                        self._last_start_error = f"worker exited before ready (code {returncode})"
                        self._ready_event.set()
            if is_current:
                self._log("whisper_worker_exited", pid=process.pid, returncode=returncode)

    def _stderr_loop(self, process):
        try:
            for raw_line in process.stderr:
                line = raw_line.strip()
                if line:
                    self._log("whisper_worker_stderr", pid=process.pid, line=line[-400:])
        except Exception as e:
            self._log("whisper_worker_stderr_failed", pid=getattr(process, "pid", None), error=e)

    def ensure_ready(self, timeout_seconds: int = WHISPER_WORKER_READY_TIMEOUT_SECONDS):
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._start_locked()
            ready_event = self._ready_event

        ready = ready_event.wait(timeout_seconds)
        with self._lock:
            process = self._process
            if not ready:
                self._terminate_locked(process, reason="startup_timeout")
                raise RuntimeError(f"Whisper worker did not become ready within {timeout_seconds}s")
            if process is None or process.poll() is not None:
                error = self._last_start_error or "Whisper worker exited during startup"
                raise RuntimeError(error)
            if not self._ready:
                raise RuntimeError(self._last_start_error or "Whisper worker failed to start")
            return dict(self._last_ready_payload)

    def is_ready(self):
        with self._lock:
            return bool(self._ready and self._process is not None and self._process.poll() is None)

    def ready_payload(self):
        with self._lock:
            return dict(self._last_ready_payload)

    def restart(self, reason: str = "restart_requested"):
        with self._lock:
            process = self._process
            self._terminate_locked(process, reason=reason)

    def transcribe(
        self,
        audio_path: str,
        options: dict,
        timeout_seconds: int = WHISPER_TRANSCRIBE_TIMEOUT_SECONDS,
    ) -> dict:
        with self._request_lock:
            ready_timeout = min(
                WHISPER_WORKER_READY_TIMEOUT_SECONDS,
                max(5, timeout_seconds),
            )
            self.ensure_ready(timeout_seconds=ready_timeout)
            with self._lock:
                process = self._process
                if process is None or process.poll() is not None or not self._ready:
                    raise RuntimeError("Whisper worker is not available")

                self._request_id += 1
                request_id = self._request_id
                response_queue = self._response_queue
                request = {
                    "type": "transcribe",
                    "request_id": request_id,
                    "audio_path": os.path.abspath(audio_path),
                    "options": options or {},
                }
                try:
                    process.stdin.write(json.dumps(request) + "\n")
                    process.stdin.flush()
                except Exception as e:
                    self._terminate_locked(process, reason="request_send_failed")
                    raise RuntimeError("Whisper worker pipe failed while sending request") from e

            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._lock:
                        if self._is_current_process(process):
                            self._terminate_locked(process, reason="request_timeout")
                    raise RuntimeError(f"Whisper worker timed out after {timeout_seconds}s")

                try:
                    message = response_queue.get(timeout=min(remaining, 0.25))
                except Empty:
                    with self._lock:
                        process_alive = self._is_current_process(process) and process.poll() is None
                    if not process_alive:
                        raise RuntimeError(f"Whisper worker exited while processing audio (code {process.poll()})")
                    continue

                if message.get("request_id") != request_id:
                    self._log(
                        "whisper_worker_response_unexpected",
                        expected_request_id=request_id,
                        actual_request_id=message.get("request_id"),
                    )
                    continue

                if not message.get("ok"):
                    error_text = message.get("error") or "Whisper worker reported an unknown error"
                    raise RuntimeError(error_text)

                return message

    def _terminate_locked(self, process, reason: str):
        if process is None:
            return
        if self._is_current_process(process):
            self._reset_state_locked()

        try:
            if process.poll() is None and process.stdin:
                process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass

        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except Exception:
                pass

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2.0)
            except Exception:
                pass

        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass

        self._log("whisper_worker_terminated", pid=getattr(process, "pid", None), reason=reason, returncode=process.poll())

    def close(self):
        with self._lock:
            process = self._process
            self._terminate_locked(process, reason="close")


class ChunkCaptureSpooler:
    def __init__(
        self,
        output_dir: str,
        sample_rate: int,
        channels: int,
        sample_width: int,
        chunk_seconds: float,
        overlap_seconds: float,
        log_fn,
    ):
        self.output_dir = output_dir
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self.chunk_samples = max(1, int(round(float(chunk_seconds) * self.sample_rate)))
        requested_overlap = max(0, int(round(float(overlap_seconds) * self.sample_rate)))
        self.overlap_samples = min(requested_overlap, max(0, self.chunk_samples - 1))
        self.bytes_per_sample_frame = self.channels * self.sample_width
        self.log_fn = log_fn
        os.makedirs(self.output_dir, exist_ok=True)
        self.buffer = bytearray()
        self.buffer_start_sample = 0
        self.total_samples_received = 0
        self.last_chunk_end_sample = 0
        self.next_chunk_index = 0
        self.chunks = []

    def add_frame(self, data: bytes):
        if not data:
            return []

        if len(data) % self.bytes_per_sample_frame != 0:
            raise ValueError("audio frame byte length is not aligned to sample frame size")

        self.buffer.extend(data)
        self.total_samples_received += len(data) // self.bytes_per_sample_frame
        return self._flush_complete_chunks()

    def finalize(self):
        if self.total_samples_received > self.last_chunk_end_sample and self.buffer:
            self._write_chunk(
                start_sample=self.buffer_start_sample,
                end_sample=self.total_samples_received,
                is_final=True,
            )
        return list(self.chunks)

    def _flush_complete_chunks(self):
        written = []
        while self.total_samples_received - self.buffer_start_sample >= self.chunk_samples:
            start_sample = self.buffer_start_sample
            end_sample = start_sample + self.chunk_samples
            chunk = self._write_chunk(
                start_sample=start_sample,
                end_sample=end_sample,
                is_final=False,
            )
            written.append(chunk)

            keep_start_sample = end_sample - self.overlap_samples
            drop_samples = keep_start_sample - self.buffer_start_sample
            drop_bytes = drop_samples * self.bytes_per_sample_frame
            del self.buffer[:drop_bytes]
            self.buffer_start_sample = keep_start_sample

        return written

    def _write_chunk(self, start_sample: int, end_sample: int, is_final: bool):
        sample_count = max(0, end_sample - start_sample)
        byte_count = sample_count * self.bytes_per_sample_frame
        chunk_audio = bytes(self.buffer[:byte_count])
        chunk_index = self.next_chunk_index
        self.next_chunk_index += 1

        filename = f"chunk_{chunk_index:04d}_{start_sample:010d}_{end_sample:010d}.wav"
        path = os.path.join(self.output_dir, filename)
        with wave.open(path, "wb") as writer:
            writer.setnchannels(self.channels)
            writer.setsampwidth(self.sample_width)
            writer.setframerate(self.sample_rate)
            writer.writeframes(chunk_audio)

        chunk = {
            "index": chunk_index,
            "path": path,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "seconds": sample_count / self.sample_rate,
            "is_final": is_final,
        }
        self.chunks.append(chunk)
        self.last_chunk_end_sample = max(self.last_chunk_end_sample, end_sample)

        self.log_fn(
            "created",
            chunk_index=chunk_index,
            file=path,
            start_sample=start_sample,
            end_sample=end_sample,
            seconds=f"{chunk['seconds']:.3f}",
            final=is_final,
        )
        return chunk


class WhisperWidget(ctk.CTk):
    def __init__(self):
        check_data_layout()
        super().__init__()
        self.base_dir = str(REPO_ROOT)
        self.runtime_state_dir = os.path.join(self.base_dir, RUNTIME_STATE_DIR)
        self.runtime_heartbeat_path = os.path.join(self.runtime_state_dir, WIDGET_HEARTBEAT_FILE)
        self.runtime_recovery_path = os.path.join(self.runtime_state_dir, WIDGET_RECOVERY_FILE)
        self.runtime_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.runtime_state_lock = threading.Lock()

        # Ensure Log Directory Exists
        self.log_dir = os.path.join(self.base_dir, LOG_DIR)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.runtime_state_dir, exist_ok=True)
        self.event_log_path = os.path.join(self.base_dir, EVENT_LOG_FILE)
        os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
        self.recording_work_dir = os.path.join(self.base_dir, RECORDING_WORK_DIR)
        self.failed_audio_dir = os.path.join(self.base_dir, FAILED_AUDIO_DIR)
        os.makedirs(self.recording_work_dir, exist_ok=True)
        os.makedirs(self.failed_audio_dir, exist_ok=True)
        self.debug_audio_dir = os.path.join(self.base_dir, DEBUG_AUDIO_DIR)
        self.raw_audio_backup_dir = os.path.join(self.base_dir, RAW_AUDIO_BACKUP_DIR)
        self.chunked_temp_dir = os.path.join(self.base_dir, CHUNKED_TEMP_DIR)
        self.web_work_dir = os.path.join(self.base_dir, WEB_WORK_DIR)
        if SAVE_DEBUG_AUDIO:
            os.makedirs(self.debug_audio_dir, exist_ok=True)
        os.makedirs(self.raw_audio_backup_dir, exist_ok=True)
        os.makedirs(self.web_work_dir, exist_ok=True)
        self.event_logger = EventLogger(self.event_log_path)
        self.log_event(
            "app_start",
            pid=os.getpid(),
            python=sys.version.split()[0],
            hotkey=HOTKEY,
            model=MODEL_SIZE
        )
        self.log_event("event_log_initialized", file=self.event_log_path)
        self.log_event(
            "chunked_transcription_config",
            enabled=CHUNKED_TRANSCRIPTION_ENABLED,
            chunk_seconds=CHUNKED_CHUNK_SECONDS,
            overlap_seconds=CHUNKED_OVERLAP_SECONDS,
            queue_maxsize=CHUNKED_QUEUE_MAXSIZE,
            finalize_timeout_seconds=CHUNKED_FINALIZE_TIMEOUT_SECONDS,
            whisper_max_attempts=CHUNKED_WHISPER_MAX_ATTEMPTS,
            save_debug_artifacts=CHUNKED_SAVE_DEBUG_ARTIFACTS,
            temp_dir=self.chunked_temp_dir,
        )
        self.log_event(
            "whisper_worker_config",
            model=MODEL_SIZE,
            device=WHISPER_DEVICE,
            ready_timeout_seconds=WHISPER_WORKER_READY_TIMEOUT_SECONDS,
            transcribe_timeout_seconds=WHISPER_TRANSCRIBE_TIMEOUT_SECONDS,
        )
        # Window Setup
        self.title("Whisper")
        self.geometry("120x150+50+50") # Circle plus status/copy footer
        self.resizable(False, False)
        self.attributes("-topmost", True) # Always on top
        self.overrideredirect(True) # Frameless window

        # Dragging Logic
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.do_drag)

        # Dark Mode
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State Variables
        self.is_recording = False
        self.is_processing = False
        self.is_muted = False  # New Mute State
        self.last_ui_state = None
        self.last_transcription_path = os.path.join(self.runtime_state_dir, "last_transcript.txt")
        self.last_transcription_lock = threading.Lock()
        self.manual_copy_lock = threading.Lock()
        self.last_transcription = self.load_last_transcription()
        self.audio_frames = []
        self.ui_queue = Queue()
        self.shutdown_event = threading.Event()
        self.hotkey_mode = None
        self.hotkey_listener_thread = None
        self.hotkey_thread_id = None
        self.hotkey_id = 1
        self.hotkey_ready_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.stream_lock = threading.Lock()
        self.record_thread = None
        self.sound_lock = threading.Lock()
        self.temp_filename = os.path.join(self.recording_work_dir, "temp_recording.wav")
        self.temp_speech_filename = os.path.join(self.recording_work_dir, "temp_speech_only.wav")
        self.current_raw_backup_path = None
        self.chunk_spooler = None
        self.chunk_capture_chunks = []
        self.chunk_transcribe_queue = None
        self.chunk_transcribe_thread = None
        self.chunk_transcribe_results = {}
        self.chunk_transcribe_lock = threading.Lock()
        self.chunk_pipeline_failed = False
        self.web_pipeline_lock = threading.Lock()
        self.web_server = None
        self.startup_recovery_thread = None
        self.recovery_state = {"active": False}
        self.vad_worker_module = VAD_WORKER_MODULE
        self.whisper_worker_module = WHISPER_WORKER_MODULE
        self.vad_client = PersistentVADWorkerClient(
            worker_module=self.vad_worker_module,
            workdir=str(REPO_ROOT),
            log_fn=self.log_event,
        )
        self.whisper_client = PersistentWhisperWorkerClient(
            worker_module=self.whisper_worker_module,
            workdir=str(REPO_ROOT),
            model_size=MODEL_SIZE,
            device=WHISPER_DEVICE,
            log_fn=self.log_event,
        )
        self.log_event("audio_init_ok")
        
        # UI Elements
        self.record_btn = ctk.CTkButton(
            self,
            text="READY",
            width=100,
            height=100,
            corner_radius=50, # Circular
            fg_color="#2e7d32", # Green
            hover_color="#1b5e20",
            command=self.toggle_recording,
            font=("Arial", 14, "bold")
        )
        self.record_btn.grid(row=0, column=0, padx=10, pady=10)

        # Mute Indicator (Overlay)
        self.mute_label = ctk.CTkLabel(self, text="", font=("Arial", 20), bg_color="transparent", text_color="red")
        self.mute_label.place(relx=0.8, rely=0.2, anchor="center")

        # Load Model in Background
        self.whisper_ready = False
        self.status_label = ctk.CTkLabel(self, text="Loading Model...", font=("Arial", 10))
        self.status_label.grid(row=1, column=0, padx=(30, 4), pady=(0, 5))
        self.create_copy_button()
        self.update_ui_state("loading")
        self.write_runtime_heartbeat(reason="app_initialized")
        
        threading.Thread(target=self.load_model, daemon=True).start()

        # Bind Keys
        self.bind("<space>", lambda event: self.toggle_recording())
        self.bind("m", lambda event: self.toggle_mute())  # Bind M for Mute
        
        # Keep UI updates and topmost behavior stable over long sessions
        self.after(50, self.process_ui_queue)
        self.after(2000, self.maintain_topmost)
        self.bind("<FocusOut>", lambda _event: self.reassert_topmost())
        self.bind("<Map>", lambda _event: self.reassert_topmost())

        # Start Global Hotkey Listener
        self.register_hotkey()
        self.reassert_topmost()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_web_server()
        self.log_event("app_ready")
        self.write_runtime_heartbeat(reason="app_ready")

    def create_copy_button(self):
        self.copy_button = ctk.CTkFrame(self, width=26, height=24, fg_color="transparent")
        self.copy_button.grid(row=1, column=0, sticky="sw", padx=(4, 0), pady=(0, 5))
        self.copy_button.pack_propagate(False)
        background = self._apply_appearance_mode(self.cget("fg_color"))
        self.copy_icon = tk.Canvas(self.copy_button, bg=background, highlightthickness=0,
                                   cursor="hand2")
        self.copy_icon.pack(fill="both", expand=True)

        def draw(event):
            scale_x, scale_y = event.width / 26, event.height / 24
            self.copy_icon.delete("paper")
            for left, top, right, bottom in ((5, 3, 15, 16), (9, 7, 20, 20)):
                self.copy_icon.create_rectangle(
                    left * scale_x, top * scale_y, right * scale_x, bottom * scale_y,
                    fill=background, outline="#aeb5bd", width=1.3 * min(scale_x, scale_y),
                    tags="paper",
                )

        self.copy_icon.bind("<Configure>", draw)
        self.copy_icon.bind("<Enter>", lambda _event: self.copy_icon.itemconfigure(
            "paper", outline="#ffffff"))
        self.copy_icon.bind("<Leave>", lambda _event: self.copy_icon.itemconfigure(
            "paper", outline="#aeb5bd"))
        # Stop these events before the toplevel's drag bindings see them.
        self.copy_icon.bind("<ButtonPress-1>", self.copy_last_transcription)
        self.copy_icon.bind("<B1-Motion>", lambda _event: "break")
        self.copy_icon.bind("<ButtonRelease-1>", lambda _event: "break")

    def start_drag(self, event):
        self.x = event.x
        self.y = event.y
        self.log_event("window_drag_start", x=self.winfo_x(), y=self.winfo_y())

    def do_drag(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"{x}+{y}")

    def log_event(self, event, **fields):
        try:
            fields["thread"] = threading.current_thread().name
            self.event_logger.log(event, **fields)
        except Exception:
            pass

    def log_chunk_event(self, event, chunk_index=None, **fields):
        if chunk_index is not None:
            fields["chunk_index"] = chunk_index
        self.log_event(f"chunk_{event}", **fields)

    def start_web_server(self):
        if not WEB_SERVER_ENABLED:
            self.log_event("web_widget_server_disabled")
            return

        try:
            from app.web.embedded import WhisperWidgetWebServer

            self.web_server = WhisperWidgetWebServer(
                widget=self,
                host=WEB_SERVER_HOST,
                port=WEB_SERVER_PORT,
                work_dir=self.web_work_dir,
                ssl_certfile=WEB_SERVER_SSL_CERTFILE,
                ssl_keyfile=WEB_SERVER_SSL_KEYFILE,
                log_fn=self.log_event,
            )
            self.web_server.start()
            self.log_event("web_widget_server_started", host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)
        except Exception as e:
            self.web_server = None
            self.log_event("web_widget_server_start_failed", host=WEB_SERVER_HOST, port=WEB_SERVER_PORT, error=e)

    def stop_web_server(self):
        if self.web_server is None:
            return

        try:
            self.web_server.stop()
        except Exception as e:
            self.log_event("web_widget_server_stop_failed", error=e)
        finally:
            self.web_server = None

    def web_status_payload(self):
        ready_payload = self.whisper_client.ready_payload()
        return {
            "ok": True,
            "model_ready": self.is_whisper_ready(),
            "vad_ready": self.vad_client.is_ready(),
            "busy": bool(self.is_recording or self.is_processing or self.web_pipeline_lock.locked()),
            "recording": bool(self.is_recording),
            "processing": bool(self.is_processing),
            "model": ready_payload.get("model") or MODEL_SIZE,
            "device": ready_payload.get("device") or WHISPER_DEVICE,
            "worker_pid": ready_payload.get("worker_pid"),
            "launcher_pid": ready_payload.get("launcher_pid"),
        }

    def web_chunk_config_payload(self):
        sample_width = self.p.get_sample_size(pyaudio.paInt16)
        chunk_samples = max(1, int(round(CHUNKED_CHUNK_SECONDS * SAMPLE_RATE)))
        requested_overlap = max(0, int(round(CHUNKED_OVERLAP_SECONDS * SAMPLE_RATE)))
        overlap_samples = min(requested_overlap, max(0, chunk_samples - 1))
        return {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": sample_width,
            "chunk_seconds": CHUNKED_CHUNK_SECONDS,
            "overlap_seconds": CHUNKED_OVERLAP_SECONDS,
            "chunk_samples": chunk_samples,
            "overlap_samples": overlap_samples,
        }

    def build_runtime_heartbeat_payload(self, status=None, reason=None):
        listener_alive = bool(self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive())
        payload = {
            "pid": os.getpid(),
            "session_id": self.runtime_session_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "status": status or self.last_ui_state or "starting",
            "recording": bool(self.is_recording),
            "processing": bool(self.is_processing),
            "recovery_active": bool((self.recovery_state or {}).get("active")),
            "hotkey_mode": self.hotkey_mode,
            "hotkey_listener_alive": listener_alive,
            "raw_backup_path": self.current_raw_backup_path,
            "chunk_dir": self.chunked_temp_dir,
            "chunk_dir_exists": os.path.isdir(self.chunked_temp_dir),
        }
        if reason:
            payload["reason"] = reason
        return payload

    def write_runtime_heartbeat(self, status=None, reason=None):
        try:
            payload = self.build_runtime_heartbeat_payload(status=status, reason=reason)
            with self.runtime_state_lock:
                write_json_atomic(self.runtime_heartbeat_path, payload)
        except Exception as e:
            self.log_event("runtime_heartbeat_write_failed", error=e)

    def remove_runtime_heartbeat(self):
        try:
            if os.path.exists(self.runtime_heartbeat_path):
                os.remove(self.runtime_heartbeat_path)
        except Exception as e:
            self.log_event("runtime_heartbeat_remove_failed", error=e)

    def update_recovery_state(self, **fields):
        now = datetime.now().isoformat(timespec="seconds")
        payload = dict(self.recovery_state or {})
        payload.update(fields)
        payload.setdefault("active", True)
        payload.setdefault("started_at", now)
        payload["updated_at"] = now
        payload["pid"] = os.getpid()
        payload["session_id"] = self.runtime_session_id
        if payload.get("mode") == "chunked":
            payload["chunk_dir"] = self.chunked_temp_dir
        if self.current_raw_backup_path and not payload.get("raw_backup_path"):
            payload["raw_backup_path"] = self.current_raw_backup_path

        self.recovery_state = payload
        try:
            with self.runtime_state_lock:
                write_json_atomic(self.runtime_recovery_path, payload)
        except Exception as e:
            self.log_event(
                "recovery_state_write_failed",
                mode=payload.get("mode"),
                stage=payload.get("stage"),
                error=e,
            )
            return False
        return True

    def clear_recovery_state(self, reason="cleared"):
        had_state = os.path.exists(self.runtime_recovery_path)
        self.recovery_state = {"active": False}
        try:
            with self.runtime_state_lock:
                if os.path.exists(self.runtime_recovery_path):
                    os.remove(self.runtime_recovery_path)
        except Exception as e:
            self.log_event("recovery_state_clear_failed", reason=reason, error=e)
            return

        if had_state:
            self.log_event("recovery_state_cleared", reason=reason)

    def parse_chunk_recovery_entry(self, path):
        match = re.match(r"^chunk_(\d{4})_(\d+)_(\d+)\.wav$", os.path.basename(path))
        if not match:
            return None

        chunk_index = int(match.group(1))
        start_sample = int(match.group(2))
        end_sample = int(match.group(3))
        sample_count = max(0, end_sample - start_sample)
        return {
            "index": chunk_index,
            "path": path,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "seconds": sample_count / float(SAMPLE_RATE),
            "is_final": False,
        }

    def load_startup_recovery_candidate(self):
        if not os.path.exists(self.runtime_recovery_path):
            return None

        try:
            with open(self.runtime_recovery_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            self.log_event("startup_recovery_state_invalid", error=e)
            self.clear_recovery_state(reason="invalid_startup_state")
            return None

        if not state.get("active"):
            self.clear_recovery_state(reason="inactive_startup_state")
            return None

        chunk_dir = state.get("chunk_dir") or self.chunked_temp_dir
        chunk_candidates = []
        if chunk_dir and os.path.isdir(chunk_dir):
            for entry in os.listdir(chunk_dir):
                parsed = self.parse_chunk_recovery_entry(os.path.join(chunk_dir, entry))
                if parsed is not None:
                    chunk_candidates.append(parsed)

        if chunk_candidates:
            chunk_candidates.sort(key=lambda item: item["index"])
            if chunk_candidates:
                chunk_candidates[-1]["is_final"] = True
            return {
                "kind": "chunked",
                "state": state,
                "chunk_dir": chunk_dir,
                "chunks": chunk_candidates,
            }

        raw_backup_path = state.get("raw_backup_path")
        if raw_backup_path and os.path.exists(raw_backup_path):
            return {
                "kind": "raw_backup",
                "state": state,
                "raw_path": raw_backup_path,
            }

        if os.path.exists(self.temp_filename):
            return {
                "kind": "temp_recording",
                "state": state,
                "raw_path": self.temp_filename,
            }

        self.log_event(
            "startup_recovery_stale_state",
            mode=state.get("mode"),
            chunk_dir=chunk_dir,
            raw_backup_path=raw_backup_path,
        )
        self.clear_recovery_state(reason="stale_startup_state")
        return None

    def reset_temp_audio_files(self, include_raw=False, reason="reset"):
        temp_paths = [
            self.temp_speech_filename,
        ]
        if include_raw:
            temp_paths.insert(0, self.temp_filename)

        for path in temp_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                os.remove(path)
                self.log_event("temp_file_removed", file=path, reason=reason)
            except Exception as e:
                self.log_event("temp_file_remove_failed", file=path, reason=reason, error=e)

    def is_safe_child_path(self, path, parent_dir):
        try:
            path_real = os.path.realpath(path)
            parent_real = os.path.realpath(parent_dir)
            return path_real != parent_real and os.path.commonpath([path_real, parent_real]) == parent_real
        except Exception:
            return False

    def prepare_chunked_temp_dir(self):
        os.makedirs(self.chunked_temp_dir, exist_ok=True)
        self.log_chunk_event("temp_dir_ready", dir=self.chunked_temp_dir)
        return self.chunked_temp_dir

    def cleanup_chunked_temp_dir(self, keep=False, reason="success"):
        if keep:
            self.log_chunk_event("temp_dir_preserved", dir=self.chunked_temp_dir, reason=reason)
            return

        if not os.path.exists(self.chunked_temp_dir):
            return

        if not self.is_safe_child_path(self.chunked_temp_dir, self.base_dir):
            self.log_chunk_event("temp_dir_cleanup_refused", dir=self.chunked_temp_dir, reason="unsafe_path")
            return

        try:
            shutil.rmtree(self.chunked_temp_dir)
            self.log_chunk_event("temp_dir_removed", dir=self.chunked_temp_dir, reason=reason)
        except Exception as e:
            self.log_chunk_event("temp_dir_remove_failed", dir=self.chunked_temp_dir, reason=reason, error=e)

    def start_chunk_capture(self):
        self.chunk_spooler = None
        self.chunk_capture_chunks = []
        self.chunk_pipeline_failed = False
        if not CHUNKED_TRANSCRIPTION_ENABLED:
            return

        try:
            self.cleanup_chunked_temp_dir(keep=False, reason="recording_start")
            temp_dir = self.prepare_chunked_temp_dir()
            sample_width = self.p.get_sample_size(pyaudio.paInt16)
            self.chunk_spooler = ChunkCaptureSpooler(
                output_dir=temp_dir,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width=sample_width,
                chunk_seconds=CHUNKED_CHUNK_SECONDS,
                overlap_seconds=CHUNKED_OVERLAP_SECONDS,
                log_fn=self.log_chunk_event,
            )
            self.start_chunk_transcription_queue()
            self.log_chunk_event(
                "capture_started",
                chunk_seconds=CHUNKED_CHUNK_SECONDS,
                overlap_seconds=CHUNKED_OVERLAP_SECONDS,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width=sample_width,
            )
            self.update_recovery_state(
                active=True,
                mode="chunked",
                stage="recording",
                chunk_count=0,
            )
        except Exception as e:
            self.chunk_spooler = None
            self.log_chunk_event("capture_start_failed", error=e)

    def record_chunk_frame(self, data):
        if self.chunk_spooler is None:
            return

        try:
            chunks = self.chunk_spooler.add_frame(data)
            for chunk in chunks:
                self.chunk_capture_chunks.append(chunk)
                if not self.enqueue_chunk_for_transcription(chunk):
                    self.chunk_pipeline_failed = True
            if chunks:
                self.update_recovery_state(
                    active=True,
                    mode="chunked",
                    stage="recording",
                    chunk_count=len(self.chunk_capture_chunks),
                )
        except Exception as e:
            self.log_chunk_event("capture_frame_failed", error=e)
            self.chunk_spooler = None
            self.chunk_pipeline_failed = True

    def finalize_chunk_capture(self, reason="recording_stop"):
        if self.chunk_spooler is None:
            return []

        try:
            existing_chunks = len(self.chunk_capture_chunks)
            all_chunks = self.chunk_spooler.finalize()
            new_chunks = all_chunks[existing_chunks:]
            for chunk in new_chunks:
                self.chunk_capture_chunks.append(chunk)
                if not self.enqueue_chunk_for_transcription(chunk):
                    self.chunk_pipeline_failed = True
            self.log_chunk_event(
                "capture_finalized",
                chunks=len(self.chunk_capture_chunks),
                reason=reason,
            )
            self.update_recovery_state(
                active=True,
                mode="chunked",
                stage="processing",
                chunk_count=len(self.chunk_capture_chunks),
                capture_finalize_reason=reason,
            )
            return list(self.chunk_capture_chunks)
        except Exception as e:
            self.log_chunk_event("capture_finalize_failed", reason=reason, error=e)
            self.chunk_pipeline_failed = True
            return []
        finally:
            self.chunk_spooler = None

    def chunk_artifact_path(self, chunk, suffix):
        root, _ = os.path.splitext(chunk["path"])
        return f"{root}_{suffix}.wav"

    def base_transcribe_options(self):
        return {
            "condition_on_previous_text": False,
            "language": WHISPER_LANGUAGE,
            "logprob_threshold": -1.0,
            "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        }

    def process_chunk_audio(self, chunk):
        chunk_index = chunk.get("index")
        source_path = chunk["path"]
        started = time.perf_counter()
        result = {
            "ok": False,
            "chunk_index": chunk_index,
            "source_path": source_path,
            "speech_secs": 0.0,
            "processed_path": None,
            "no_speech": False,
            "error": None,
            "timings": {},
        }

        self.log_chunk_event("processing_start", chunk_index=chunk_index, file=source_path)
        try:
            speech_path = self.chunk_artifact_path(chunk, "speech_only")

            vad_started = time.perf_counter()
            speech_secs = self.vad_client.run(
                in_wav_path=source_path,
                out_wav_path=speech_path,
                sample_rate=SAMPLE_RATE,
                pad_ms=VAD_PAD_MS,
                min_speech_ms=VAD_MIN_SPEECH_MS,
                merge_gap_ms=VAD_MERGE_GAP_MS,
                speech_prob_threshold=0.5,
            )
            vad_seconds = time.perf_counter() - vad_started
            result["speech_secs"] = speech_secs
            result["timings"]["vad_seconds"] = vad_seconds
            self.log_chunk_event(
                "vad_complete",
                chunk_index=chunk_index,
                speech_secs=f"{speech_secs:.3f}",
                processing_seconds=f"{vad_seconds:.3f}",
            )

            if speech_secs <= 0.0:
                result["ok"] = True
                result["no_speech"] = True
                result["timings"]["total_seconds"] = time.perf_counter() - started
                self.log_chunk_event(
                    "processing_complete",
                    chunk_index=chunk_index,
                    no_speech=True,
                    total_seconds=f"{result['timings']['total_seconds']:.3f}",
                )
                return result

            processed_source = speech_path
            result["ok"] = True
            result["processed_path"] = processed_source
            result["timings"]["total_seconds"] = time.perf_counter() - started
            self.log_chunk_event(
                "processing_complete",
                chunk_index=chunk_index,
                file=processed_source,
                no_speech=False,
                total_seconds=f"{result['timings']['total_seconds']:.3f}",
            )
            return result
        except Exception as e:
            result["error"] = str(e)
            result["timings"]["total_seconds"] = time.perf_counter() - started
            self.log_chunk_event(
                "failed",
                chunk_index=chunk_index,
                error=e,
                total_seconds=f"{result['timings']['total_seconds']:.3f}",
            )
            return result

    def start_chunk_transcription_queue(self):
        queue_maxsize = max(0, int(CHUNKED_QUEUE_MAXSIZE))
        queue = Queue(maxsize=queue_maxsize)
        results = {}
        self.chunk_transcribe_queue = queue
        self.chunk_transcribe_results = results
        self.chunk_transcribe_thread = threading.Thread(
            target=self.chunk_transcription_worker_loop,
            args=(queue, results),
            daemon=True,
            name="chunk_transcription_worker",
        )
        self.chunk_transcribe_thread.start()
        self.log_chunk_event("transcription_queue_started", queue_maxsize=queue_maxsize)

    def enqueue_chunk_for_transcription(self, chunk):
        if self.chunk_transcribe_queue is None:
            self.log_chunk_event("queue_missing", chunk_index=chunk.get("index"))
            return False

        try:
            self.chunk_transcribe_queue.put(chunk, block=False)
            self.log_chunk_event("queued", chunk_index=chunk.get("index"), file=chunk.get("path"))
            return True
        except Full:
            self.log_chunk_event("queue_full", chunk_index=chunk.get("index"), file=chunk.get("path"))
            return False

    def finish_chunk_transcription_queue(self, timeout_seconds=CHUNKED_FINALIZE_TIMEOUT_SECONDS):
        queue = self.chunk_transcribe_queue
        thread = self.chunk_transcribe_thread
        if queue is None or thread is None:
            return []

        queue.put(None)
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            self.log_chunk_event("transcription_queue_join_timeout", timeout_seconds=timeout_seconds)
            raise TimeoutError(f"chunk transcription queue did not finish within {timeout_seconds}s")
        else:
            self.log_chunk_event("transcription_queue_finished", results=len(self.chunk_transcribe_results))

        self.chunk_transcribe_queue = None
        self.chunk_transcribe_thread = None
        with self.chunk_transcribe_lock:
            return [self.chunk_transcribe_results[index] for index in sorted(self.chunk_transcribe_results)]

    def chunk_transcription_worker_loop(self, queue, results):
        while True:
            chunk = queue.get()
            try:
                if chunk is None:
                    return

                result = self.process_and_transcribe_chunk(chunk)
                with self.chunk_transcribe_lock:
                    results[chunk.get("index")] = result
            finally:
                queue.task_done()

    def process_and_transcribe_chunk(self, chunk):
        chunk_index = chunk.get("index")
        result = {
            "ok": False,
            "chunk_index": chunk_index,
            "text": "",
            "source_path": chunk.get("path"),
            "transcribed_path": None,
            "processing": None,
            "error": None,
            "timings": {},
        }

        processing_result = self.process_chunk_audio(chunk)
        result["processing"] = processing_result
        if processing_result.get("no_speech"):
            result["ok"] = True
            self.log_chunk_event("whisper_skipped_no_speech", chunk_index=chunk_index)
            return result

        transcribe_path = processing_result.get("processed_path")
        stage = "processed_chunk"
        if not processing_result.get("ok") or not transcribe_path or not os.path.exists(transcribe_path):
            transcribe_path = chunk.get("path")
            stage = "raw_chunk_fallback"
            self.log_chunk_event(
                "processing_failed_fallback_raw",
                chunk_index=chunk_index,
                error=processing_result.get("error"),
            )

        if not self.is_whisper_ready():
            result["error"] = "model_not_loaded"
            self.log_chunk_event("whisper_failed", chunk_index=chunk_index, stage=stage, error=result["error"])
            return result

        result["transcribed_path"] = transcribe_path
        max_attempts = max(1, int(CHUNKED_WHISPER_MAX_ATTEMPTS))
        request_timeout = min(
            WHISPER_TRANSCRIBE_TIMEOUT_SECONDS,
            max(10, CHUNKED_FINALIZE_TIMEOUT_SECONDS - 25),
        )
        result["timings"]["whisper_attempt_timeout_seconds"] = request_timeout
        for whisper_attempt in range(1, max_attempts + 1):
            try:
                self.log_chunk_event(
                    "whisper_start",
                    chunk_index=chunk_index,
                    stage=stage,
                    source=transcribe_path,
                    attempt=whisper_attempt,
                    max_attempts=max_attempts,
                )
                started = time.perf_counter()
                whisper_result = self.transcribe_with_whisper(
                    transcribe_path,
                    self.base_transcribe_options(),
                    timeout_seconds=request_timeout,
                )
                elapsed = time.perf_counter() - started
                text = (whisper_result.get("text") or "").strip()
                result["text"] = text
                result["timings"]["whisper_seconds"] = elapsed
                result["timings"]["whisper_attempts"] = whisper_attempt
                result["ok"] = True

                if text:
                    self.log_chunk_event(
                        "whisper_success",
                        chunk_index=chunk_index,
                        stage=stage,
                        chars=len(text),
                        processing_seconds=f"{elapsed:.3f}",
                        attempt=whisper_attempt,
                    )
                else:
                    self.log_chunk_event(
                        "whisper_no_text",
                        chunk_index=chunk_index,
                        stage=stage,
                        processing_seconds=f"{elapsed:.3f}",
                        attempt=whisper_attempt,
                    )
                return result
            except Exception as e:
                result["error"] = str(e)
                result["timings"]["whisper_attempts"] = whisper_attempt
                self.log_chunk_event(
                    "whisper_failed",
                    chunk_index=chunk_index,
                    stage=stage,
                    error=e,
                    attempt=whisper_attempt,
                    max_attempts=max_attempts,
                )
                if whisper_attempt >= max_attempts:
                    return result

                self.log_chunk_event(
                    "whisper_retry_restart",
                    chunk_index=chunk_index,
                    stage=stage,
                    next_attempt=whisper_attempt + 1,
                )
                self.restart_whisper_worker(reason="chunk_whisper_retry")

        return result

    def transcript_words_for_match(self, text):
        words = []
        for raw_word in text.split():
            normalized = raw_word.strip().lower().strip(".,!?;:\"'()[]{}")
            if normalized:
                words.append(normalized)
        return words

    def trailing_word_punctuation(self, word):
        word = (word or "").strip()
        while word and word[-1] in "\"')]}":
            word = word[:-1]

        punctuation = ""
        while word and word[-1] in ".,!?;:":
            punctuation = word[-1] + punctuation
            word = word[:-1]
        return punctuation

    def transfer_discarded_overlap_punctuation(self, existing_text, next_original_words, overlap_words):
        if overlap_words <= 0 or not next_original_words:
            return existing_text

        existing_words = existing_text.split()
        if not existing_words:
            return existing_text

        retained_word = existing_words[-1]
        discarded_word = next_original_words[overlap_words - 1]
        if self.trailing_word_punctuation(retained_word):
            return existing_text

        punctuation = self.trailing_word_punctuation(discarded_word)
        if not punctuation:
            return existing_text

        return f"{existing_text}{punctuation}"

    def merge_transcript_overlap(self, existing_text, next_text, max_overlap_words=16):
        existing_text = (existing_text or "").strip()
        next_text = (next_text or "").strip()
        if not existing_text:
            return next_text
        if not next_text:
            return existing_text

        existing_words = self.transcript_words_for_match(existing_text)
        next_words = self.transcript_words_for_match(next_text)
        max_overlap = min(max_overlap_words, len(existing_words), len(next_words))
        overlap_words = 0
        for size in range(max_overlap, 0, -1):
            if existing_words[-size:] == next_words[:size]:
                overlap_words = size
                break

        if overlap_words == 0:
            return f"{existing_text} {next_text}"

        next_original_words = next_text.split()
        remaining_next = " ".join(next_original_words[overlap_words:]).strip()
        existing_text = self.transfer_discarded_overlap_punctuation(
            existing_text,
            next_original_words,
            overlap_words,
        )
        if not remaining_next:
            return existing_text
        if existing_text[-1:] in ".!?" and remaining_next[:1].islower():
            existing_text = existing_text.rstrip(".!?")
        return f"{existing_text} {remaining_next}"

    def stitch_chunk_transcripts(self, chunk_results):
        stitched = ""
        for result in sorted(chunk_results, key=lambda item: item.get("chunk_index", -1)):
            if not result.get("ok"):
                continue
            text = (result.get("text") or "").strip()
            if not text:
                continue
            stitched = self.merge_transcript_overlap(stitched, text)
        return stitched.strip()

    def copy_chunk_debug_artifacts(self, artifact_dir, chunk_results):
        chunks_dir = os.path.join(artifact_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        copied_by_chunk = {}

        for result in sorted(chunk_results, key=lambda item: item.get("chunk_index", -1)):
            chunk_index = result.get("chunk_index")
            if chunk_index is None:
                continue

            source_path = result.get("source_path")
            candidates = []
            if source_path:
                root, _ = os.path.splitext(source_path)
                candidates.extend([
                    ("raw", source_path),
                    ("speech_only", f"{root}_speech_only.wav"),
                ])

            processing = result.get("processing") or {}
            for label, path in (
                ("processed", processing.get("processed_path")),
                ("transcribed", result.get("transcribed_path")),
            ):
                if path:
                    candidates.append((label, path))

            copied = {}
            seen_paths = set()
            for label, path in candidates:
                if not path:
                    continue
                normalized_path = os.path.abspath(path)
                if normalized_path in seen_paths or not os.path.exists(normalized_path):
                    continue
                seen_paths.add(normalized_path)

                destination = os.path.join(chunks_dir, f"chunk_{int(chunk_index):04d}_{label}.wav")
                try:
                    shutil.copy2(normalized_path, destination)
                    copied[label] = destination
                except Exception as e:
                    copied[f"{label}_copy_error"] = str(e)
                    self.log_chunk_event(
                        "debug_artifact_copy_failed",
                        chunk_index=chunk_index,
                        label=label,
                        source=normalized_path,
                        error=e,
                    )

            copied_by_chunk[chunk_index] = copied

        return copied_by_chunk

    def write_chunk_transcript_debug(self, chunk_results, stitched_text, reason):
        if not SAVE_DEBUG_AUDIO or not CHUNKED_SAVE_DEBUG_ARTIFACTS:
            return None

        try:
            debug_dir = os.path.join(self.debug_audio_dir, "chunked_transcripts")
            os.makedirs(debug_dir, exist_ok=True)
            artifact_root = os.path.join(self.debug_audio_dir, "chunked_runs")
            os.makedirs(artifact_root, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            run_name = f"chunked_{timestamp}_{reason}"
            artifact_dir = os.path.join(artifact_root, run_name)
            os.makedirs(artifact_dir, exist_ok=True)
            copied_artifacts = self.copy_chunk_debug_artifacts(artifact_dir, chunk_results)
            output_path = os.path.join(debug_dir, f"{run_name}.json")
            payload = {
                "reason": reason,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "artifact_dir": artifact_dir,
                "stitched_chars": len(stitched_text or ""),
                "stitched_text": stitched_text or "",
                "chunks": [],
            }

            for result in sorted(chunk_results, key=lambda item: item.get("chunk_index", -1)):
                processing = result.get("processing") or {}
                payload["chunks"].append({
                    "chunk_index": result.get("chunk_index"),
                    "ok": result.get("ok"),
                    "text": result.get("text") or "",
                    "chars": len(result.get("text") or ""),
                    "source_path": result.get("source_path"),
                    "transcribed_path": result.get("transcribed_path"),
                    "error": result.get("error"),
                    "timings": result.get("timings") or {},
                    "debug_artifacts": copied_artifacts.get(result.get("chunk_index"), {}),
                    "processing": {
                        "ok": processing.get("ok"),
                        "no_speech": processing.get("no_speech"),
                        "speech_secs": processing.get("speech_secs"),
                        "processed_path": processing.get("processed_path"),
                        "error": processing.get("error"),
                        "timings": processing.get("timings") or {},
                    },
                })

            transcript_path = os.path.join(artifact_dir, "stitched_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(stitched_text or "")

            manifest_path = os.path.join(artifact_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.log_chunk_event(
                "transcript_debug_saved",
                file=output_path,
                artifact_dir=artifact_dir,
                chunks=len(payload["chunks"]),
                reason=reason,
            )
            return output_path
        except Exception as e:
            self.log_chunk_event("transcript_debug_save_failed", reason=reason, error=e)
            return None

    def log_transcription(self, text):
        """Appends transcription to a daily log file."""
        self.remember_transcription(text)
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            filename = os.path.join(self.log_dir, f"{date_str}.txt")
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {text}\n")
            self.log_event("transcription_log_saved", chars=len(text), file=filename)
        except Exception as e:
            print(f"Failed to log transcription: {e}")
            self.log_event("transcription_log_failed", error=e)

    def print_transcription_to_terminal(self, text):
        if not text:
            return
        print("\n[transcription]")
        print(text)
        print("[/transcription]", flush=True)

    def load_last_transcription(self):
        try:
            with open(self.last_transcription_path, encoding="utf-8") as source:
                return source.read()
        except FileNotFoundError:
            return ""
        except Exception as error:
            self.log_event("last_transcription_load_failed", error=error)
            return ""

    def remember_transcription(self, text):
        if not text:
            return
        with self.last_transcription_lock:
            # Keep the in-memory copy even if saving or automatic copying fails.
            self.last_transcription = text
            temporary_path = self.last_transcription_path + ".tmp"
            try:
                with open(temporary_path, "w", encoding="utf-8") as output:
                    output.write(text)
                os.replace(temporary_path, self.last_transcription_path)
            except Exception as error:
                self.log_event("last_transcription_save_failed", error=error)

    def copy_last_transcription(self, _event=None):
        if self.shutdown_event.is_set():
            return "break"
        with self.last_transcription_lock:
            text = self.last_transcription
        if not text or not self.manual_copy_lock.acquire(blocking=False):
            return "break"

        def copy():
            try:
                copied = self.copy_transcription_to_clipboard(text)
                self.log_event("manual_clipboard_copy_finished", copied=copied, chars=len(text))
            finally:
                self.manual_copy_lock.release()

        try:
            threading.Thread(target=copy, name="manual_clipboard_copy", daemon=True).start()
        except Exception:
            self.manual_copy_lock.release()
            raise
        return "break"

    def copy_transcription_to_clipboard(self, text):
        """Attempt one clipboard write outside the widget process."""
        if not text:
            return False

        self.log_event(
            "clipboard_copy_start",
            chars=len(text),
            timeout_seconds=CLIPBOARD_COPY_TIMEOUT_SECONDS,
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", CLIPBOARD_WORKER_MODULE],
                cwd=str(REPO_ROOT),
                input=text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=CLIPBOARD_COPY_TIMEOUT_SECONDS,
                check=False,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired:
            self.log_event(
                "clipboard_copy_abandoned",
                reason="timeout",
                chars=len(text),
                timeout_seconds=CLIPBOARD_COPY_TIMEOUT_SECONDS,
            )
            return False
        except Exception as e:
            self.log_event(
                "clipboard_copy_abandoned",
                reason="worker_start_failed",
                chars=len(text),
                error=e,
            )
            return False

        if result.returncode != 0:
            error = (result.stderr or result.stdout or "clipboard worker failed").strip()
            self.log_event(
                "clipboard_copy_abandoned",
                reason="clipboard_busy" if result.returncode == 2 else "worker_failed",
                chars=len(text),
                returncode=result.returncode,
                error=error[-400:],
            )
            return False

        self.log_event("clipboard_copy_success", chars=len(text))
        return True

    def load_model(self):
        """Starts the persistent Whisper worker on a background thread."""
        try:
            print("Starting Whisper worker...")
            self.log_event(
                "model_load_start",
                device=WHISPER_DEVICE,
                model=MODEL_SIZE,
                worker=self.whisper_worker_module,
            )
            ready_payload = self.whisper_client.ensure_ready(
                timeout_seconds=WHISPER_WORKER_READY_TIMEOUT_SECONDS,
            )
            self.whisper_ready = True
            try:
                self.log_event("vad_worker_prewarm_start")
                self.vad_client.ensure_ready()
                self.log_event("vad_worker_prewarm_success")
            except Exception as vad_error:
                print(f"VAD worker prewarm failed: {vad_error}")
                self.log_event("vad_worker_prewarm_failed", error=vad_error)
            recovery_candidate = self.load_startup_recovery_candidate()
            if recovery_candidate is None:
                self.ui_call(self.update_ui_state, "ready")
            print(
                "Whisper worker ready "
                f"({ready_payload.get('device')}, {ready_payload.get('model_dtype')})."
            )
            self.log_event(
                "model_load_success",
                device=ready_payload.get("device"),
                load_device=ready_payload.get("load_device"),
                model_dtype=ready_payload.get("model_dtype"),
                layer_norm_dtype=ready_payload.get("layer_norm_dtype"),
                cuda_allocated_mb=ready_payload.get("cuda_allocated_mb"),
                cuda_reserved_mb=ready_payload.get("cuda_reserved_mb"),
                model=ready_payload.get("model"),
                launcher_pid=ready_payload.get("launcher_pid"),
                worker_pid=ready_payload.get("pid"),
            )
            if recovery_candidate is None:
                self.play_cue("ready")
            else:
                self.log_event(
                    "startup_recovery_candidate_found",
                    kind=recovery_candidate.get("kind"),
                    chunks=len(recovery_candidate.get("chunks") or []),
                    raw_path=recovery_candidate.get("raw_path"),
                )
                self.ui_call(self.start_startup_recovery, recovery_candidate)
        except Exception as e:
            self.whisper_ready = False
            self.ui_call(self.status_label.configure, text="Error Loading")
            print(f"Error loading model: {e}")
            self.log_event("model_load_failed", error=e)

    def start_startup_recovery(self, candidate):
        if self.startup_recovery_thread is not None or self.is_processing:
            return

        self.is_recording = False
        self.is_processing = True
        self.update_ui_state("recovering")
        self.write_runtime_heartbeat(reason="startup_recovery_start")
        self.startup_recovery_thread = threading.Thread(
            target=self.run_startup_recovery,
            args=(candidate,),
            daemon=True,
            name="startup_recovery",
        )
        self.startup_recovery_thread.start()

    def run_startup_recovery(self, candidate):
        try:
            kind = candidate.get("kind")
            if kind == "chunked":
                self.recover_chunked_session(candidate)
            elif kind in ("raw_backup", "temp_recording"):
                self.recover_single_session(candidate)
            else:
                raise RuntimeError(f"unsupported_recovery_kind={kind}")
        except Exception as e:
            self.log_event("startup_recovery_failed", kind=candidate.get("kind"), error=e)
            self.clear_recovery_state(reason="startup_recovery_failed")
            self.ui_call(self.finish_processing)
        finally:
            self.startup_recovery_thread = None

    def recover_chunked_session(self, candidate):
        chunk_results = []
        chunk_list = list(candidate.get("chunks") or [])
        self.log_event(
            "startup_recovery_chunked_begin",
            chunks=len(chunk_list),
            chunk_dir=candidate.get("chunk_dir"),
        )
        try:
            for chunk in chunk_list:
                result = self.process_and_transcribe_chunk(chunk)
                chunk_results.append(result)

            failed_results = [result for result in chunk_results if not result.get("ok")]
            if failed_results:
                raise RuntimeError(f"chunk_recovery_failures={len(failed_results)}")

            text = self.stitch_chunk_transcripts(chunk_results)
            self.write_chunk_transcript_debug(chunk_results, text, reason="recovery_success")
            if text:
                print("\n[transcription]")
                print(text)
                print("[/transcription]", flush=True)
                self.log_transcription(text)
                clipboard_copied = self.copy_transcription_to_clipboard(text)
                if clipboard_copied:
                    print("Copied transcription to clipboard.")
                self.log_event(
                    "chunked_transcription_success",
                    chunks=len(chunk_results),
                    chars=len(text),
                    recovered=True,
                    clipboard_copied=clipboard_copied,
                )
                if clipboard_copied:
                    self.ui_call(self.record_btn.configure, text="COPIED", fg_color="#1565c0")
                else:
                    self.ui_call(self.record_btn.configure, text="SAVED", fg_color="#ef6c00")
                self.play_cue("success")
                time.sleep(1)
            else:
                self.log_event("chunked_transcription_empty", chunks=len(chunk_results), recovered=True)

            self.log_event(
                "chunked_transcription_pipeline_finish",
                success=True,
                chunks=len(chunk_results),
                chars=len(text),
                recovered=True,
            )
            self.cleanup_chunked_temp_dir(
                keep=CHUNKED_KEEP_TEMP_ON_SUCCESS,
                reason="recovery_success",
            )
            self.clear_recovery_state(reason="startup_recovery_chunked_success")
            self.ui_call(self.finish_processing)
        except Exception:
            self.write_chunk_transcript_debug(chunk_results, "", reason="recovery_failure")
            self.cleanup_chunked_temp_dir(
                keep=CHUNKED_KEEP_TEMP_ON_FAILURE,
                reason="recovery_failure",
            )
            raise

    def recover_single_session(self, candidate):
        raw_path = candidate.get("raw_path")
        if not raw_path or not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing recovery audio: {raw_path}")

        self.log_event("startup_recovery_single_begin", file=raw_path, kind=candidate.get("kind"))
        self.claim_raw_audio_backup(raw_path)
        self.transcribe_saved_audio(raw_path, source_label=f"startup_recovery_{candidate.get('kind')}")

    def is_whisper_ready(self):
        return self.whisper_client.is_ready()

    def transcribe_with_whisper(self, audio_path, options, timeout_seconds=WHISPER_TRANSCRIBE_TIMEOUT_SECONDS):
        try:
            return self.whisper_client.transcribe(
                audio_path,
                options or {},
                timeout_seconds=timeout_seconds,
            )
        finally:
            self.whisper_ready = self.whisper_client.is_ready()

    def restart_whisper_worker(self, reason="restart_requested"):
        self.whisper_ready = False
        self.whisper_client.restart(reason=reason)

    def ui_call(self, fn, *args, **kwargs):
        """Runs UI work on the Tk main thread."""
        if threading.current_thread() is threading.main_thread():
            fn(*args, **kwargs)
            return
        self.ui_queue.put((fn, args, kwargs))

    def process_ui_queue(self):
        """Drains queued UI actions from background threads."""
        while True:
            try:
                fn, args, kwargs = self.ui_queue.get_nowait()
            except Empty:
                break
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(f"UI queue callback failed: {e}")
                self.log_event("ui_queue_callback_failed", error=e)

        if not self.shutdown_event.is_set():
            self.after(50, self.process_ui_queue)

    def reassert_topmost(self):
        try:
            self.attributes("-topmost", True)
            self.lift()
            if IS_WINDOWS:
                hwnd = int(self.winfo_id())
                if hwnd:
                    ok = user32.SetWindowPos(
                        wintypes.HWND(hwnd),
                        wintypes.HWND(HWND_TOPMOST),
                        0,
                        0,
                        0,
                        0,
                        SWP_TOPMOST_FLAGS
                    )
                    if not ok:
                        print(f"SetWindowPos failed: {ctypes.get_last_error()}")
                        self.log_event("set_window_pos_failed", win_error=ctypes.get_last_error())
        except Exception as e:
            print(f"Topmost reassert failed: {e}")
            self.log_event("topmost_reassert_failed", error=e)

    def maintain_topmost(self):
        """Periodically reassert topmost so long-running sessions stay pinned."""
        if self.shutdown_event.is_set():
            return
        self.reassert_topmost()
        self.ensure_hotkey_is_alive()
        listener_alive = bool(self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive())
        self.log_event(
            "runtime_heartbeat",
            hotkey_mode=self.hotkey_mode,
            hotkey_listener_alive=listener_alive,
            recording=self.is_recording,
            processing=self.is_processing
        )
        self.write_runtime_heartbeat(reason="periodic_heartbeat")
        self.after(10000, self.maintain_topmost)

    def on_hotkey_pressed(self):
        self.log_event("hotkey_pressed", hotkey=HOTKEY, mode=self.hotkey_mode)
        self.ui_call(self.toggle_recording)

    def ensure_hotkey_is_alive(self):
        if self.hotkey_mode == "win32":
            if self.hotkey_listener_thread and not self.hotkey_listener_thread.is_alive():
                print("Win32 hotkey loop stopped; re-registering...")
                self.log_event("hotkey_listener_dead", mode="win32")
                self.register_hotkey()

    def register_hotkey(self):
        self.log_event("hotkey_register_attempt", hotkey=HOTKEY)
        self.unregister_hotkey()

        if IS_WINDOWS and self.register_win32_hotkey():
            self.hotkey_mode = "win32"
            print(f"Global hotkey registered via Win32: {HOTKEY.upper()}")
            self.log_event("hotkey_register_success", mode="win32", hotkey=HOTKEY)
            return

        if keyboard is not None:
            try:
                keyboard.add_hotkey(HOTKEY, self.on_hotkey_pressed)
                self.hotkey_mode = "keyboard"
                print(f"Global hotkey registered via keyboard: {HOTKEY.upper()}")
                self.log_event("hotkey_register_success", mode="keyboard", hotkey=HOTKEY)
                return
            except Exception as e:
                print(f"Keyboard hotkey registration failed: {e}")
                self.log_event("hotkey_register_failed", mode="keyboard", error=e)

        self.hotkey_mode = None
        self.ui_call(self.status_label.configure, text="Hotkey Error")
        print("Global hotkey unavailable.")
        self.log_event("hotkey_register_failed", mode="none")

    def register_win32_hotkey(self):
        parsed = parse_hotkey_for_win32(HOTKEY)
        if parsed is None:
            print(f"Unsupported Win32 hotkey format: {HOTKEY}")
            self.log_event("hotkey_parse_failed", hotkey=HOTKEY)
            return False

        mods, vk = parsed
        self.hotkey_ready_event.clear()
        self.hotkey_listener_thread = threading.Thread(
            target=self.win32_hotkey_loop,
            args=(mods, vk),
            daemon=True
        )
        self.hotkey_listener_thread.start()

        # Wait briefly for registration status from listener thread
        ready = self.hotkey_ready_event.wait(timeout=2.0)
        if not ready:
            self.log_event("hotkey_register_timeout", mode="win32")
        return self.hotkey_mode == "win32"

    def win32_hotkey_loop(self, mods, vk):
        self.hotkey_thread_id = kernel32.GetCurrentThreadId()
        self.log_event("win32_hotkey_loop_start", thread_id=self.hotkey_thread_id, vk=vk, mods=mods)

        registered = user32.RegisterHotKey(None, self.hotkey_id, mods, vk)
        if not registered:
            print(f"Win32 RegisterHotKey failed: {ctypes.get_last_error()}")
            self.hotkey_mode = None
            self.hotkey_ready_event.set()
            self.log_event("win32_register_hotkey_failed", win_error=ctypes.get_last_error())
            return

        self.hotkey_mode = "win32"
        self.hotkey_ready_event.set()

        msg = wintypes.MSG()
        while not self.shutdown_event.is_set():
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == -1:
                print(f"Win32 GetMessageW failed: {ctypes.get_last_error()}")
                self.log_event("win32_getmessage_failed", win_error=ctypes.get_last_error())
                break
            if result == 0:
                self.log_event("win32_hotkey_loop_quit_message")
                break
            if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
                self.on_hotkey_pressed()

        try:
            user32.UnregisterHotKey(None, self.hotkey_id)
        except Exception as e:
            print(f"Win32 UnregisterHotKey failed: {e}")
            self.log_event("win32_unregister_hotkey_failed", error=e)

        self.hotkey_thread_id = None
        if not self.shutdown_event.is_set():
            self.hotkey_mode = None
        self.log_event("win32_hotkey_loop_end")

    def unregister_hotkey(self):
        self.log_event("hotkey_unregister_attempt", mode=self.hotkey_mode)
        if self.hotkey_mode == "keyboard" and keyboard is not None:
            try:
                keyboard.unhook_all_hotkeys()
            except Exception as e:
                print(f"Keyboard hotkey cleanup failed: {e}")
                self.log_event("hotkey_unregister_failed", mode="keyboard", error=e)
            finally:
                self.hotkey_mode = None
                self.log_event("hotkey_unregistered", mode="keyboard")
                return

        if self.hotkey_mode == "win32" and self.hotkey_thread_id:
            try:
                user32.PostThreadMessageW(self.hotkey_thread_id, WM_QUIT, 0, 0)
            except Exception as e:
                print(f"Win32 hotkey stop failed: {e}")
                self.log_event("hotkey_unregister_failed", mode="win32", error=e)

        if self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive():
            self.hotkey_listener_thread.join(timeout=1.0)

        self.hotkey_thread_id = None
        self.hotkey_mode = None
        self.log_event("hotkey_unregistered", mode="win32_or_none")

    def play_cue(self, cue_name: str = "notification"):
        """Queues a non-blocking notification cue."""
        if self.is_muted:
            return

        threading.Thread(
            target=self._play_cue_sync,
            args=(cue_name,),
            daemon=True,
        ).start()

    def play_sound(self, freq, duration):
        """Queues a non-blocking notification tone."""
        if IS_WINDOWS and winsound is not None:
            self.play_cue("notification")
            return
        self.play_sound_sequence([(freq, duration)])

    def play_sound_sequence(self, tones):
        if self.is_muted:
            return

        threading.Thread(
            target=self._play_sound_sequence_sync,
            args=(list(tones),),
            daemon=True
        ).start()

    def _play_cue_sync(self, cue_name: str):
        try:
            with self.sound_lock:
                tones = SOUND_CUE_TONES.get(cue_name, SOUND_CUE_TONES["notification"])
                if self._play_windows_generated_cue(tones):
                    return

                self._play_generated_tones(tones)
        except Exception as e:
            print(f"Sound Error: {e}")
            self.log_event("sound_error", error=e)

    def _play_windows_generated_cue(self, tones) -> bool:
        if not (IS_WINDOWS and winsound is not None):
            return False

        try:
            wav_bytes = self._build_cue_wav_bytes(tones)
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_SYNC)
            return True
        except Exception as e:
            self.log_event("windows_generated_cue_failed", error=e)
            return False

    def _play_sound_sequence_sync(self, tones):
        try:
            with self.sound_lock:
                self._play_generated_tones(tones)
        except Exception as e:
            print(f"Sound Error: {e}")
            self.log_event("sound_error", error=e)

    def _play_generated_tones(self, tones):
        rate = 44100

        for freq, duration in tones:
            byte_stream = self._build_tone_pcm_bytes(freq, duration, rate)

            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                output=True,
            )
            try:
                stream.write(byte_stream)
            finally:
                stream.stop_stream()
                stream.close()

    def _build_cue_wav_bytes(self, tones):
        rate = 44100
        cue_buffer = io.BytesIO()
        with wave.open(cue_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            for index, (freq, duration) in enumerate(tones):
                if index > 0:
                    wav_file.writeframes(b"\x00\x00" * int(rate * 0.03))
                wav_file.writeframes(self._build_tone_pcm_bytes(freq, duration, rate))
        return cue_buffer.getvalue()

    @staticmethod
    def _build_tone_pcm_bytes(freq, duration_ms, rate):
        duration_ms = int(duration_ms)
        num_samples = int(rate * (duration_ms / 1000.0))
        ramp_samples = max(1, min(num_samples // 2, int(rate * 0.008)))
        audio_data = []

        for x in range(num_samples):
            envelope = min(
                1.0,
                x / ramp_samples,
                (num_samples - x - 1) / ramp_samples,
            )
            sample = 0.16 * envelope * math.sin(2 * math.pi * float(freq) * (x / rate))
            audio_data.append(struct.pack('<h', int(sample * 32767.0)))

        return b''.join(audio_data)

    def build_transcribe_attempts(self, speech_secs, processed_source):
        return self.build_transcribe_attempts_for_paths(
            raw_path=self.temp_filename,
            speech_path=self.temp_speech_filename,
            speech_secs=speech_secs,
            processed_source=processed_source,
        )

    def build_transcribe_attempts_for_paths(
        self,
        raw_path,
        speech_path,
        speech_secs,
        processed_source=None,
    ):
        base_options = self.base_transcribe_options()
        attempt_candidates = []
        processed_source = processed_source or speech_path

        if speech_secs > 0.0 and processed_source:
            attempt_candidates.append(("speech_only_primary", processed_source, {}))
            if processed_source != speech_path and speech_path and os.path.exists(speech_path):
                attempt_candidates.append(("speech_only_raw_fallback", speech_path, {}))

        attempt_candidates.append(("raw_full_fallback", raw_path, {}))
        attempt_candidates.append((
            "raw_full_permissive",
            raw_path,
            {
                "compression_ratio_threshold": None,
                "logprob_threshold": None,
                "no_speech_threshold": None,
            }
        ))

        attempts = []
        seen = set()
        for label, path, overrides in attempt_candidates:
            if not path or not os.path.exists(path):
                continue

            signature = (
                os.path.abspath(path),
                tuple(sorted((key, repr(value)) for key, value in overrides.items())),
            )
            if signature in seen:
                continue

            seen.add(signature)
            options = dict(base_options)
            options.update(overrides)
            attempts.append({
                "label": label,
                "path": path,
                "options": options,
            })

        return attempts

    def save_debug_audio_files(self):
        if not SAVE_DEBUG_AUDIO:
            return

        capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        capture_dir = os.path.join(self.debug_audio_dir, capture_id)
        saved_files = 0

        try:
            os.makedirs(capture_dir, exist_ok=False)
            for source_path, output_name in [
                (self.temp_filename, "raw.wav"),
                (self.temp_speech_filename, "speech_only.wav"),
            ]:
                if not os.path.exists(source_path):
                    continue
                shutil.copy2(source_path, os.path.join(capture_dir, output_name))
                saved_files += 1

            if saved_files == 0:
                os.rmdir(capture_dir)
                self.log_event("debug_audio_capture_empty")
                return

            self.log_event("debug_audio_capture_saved", dir=capture_dir, files=saved_files)
        except Exception as e:
            self.log_event("debug_audio_capture_failed", error=e)

    def save_web_debug_audio_files(self, raw_path, speech_path, request_id):
        if not SAVE_DEBUG_AUDIO:
            return None

        capture_dir = os.path.join(self.debug_audio_dir, "web", request_id)
        saved_files = 0
        try:
            os.makedirs(capture_dir, exist_ok=False)
            for source_path, output_name in [
                (raw_path, "raw.wav"),
                (speech_path, "speech_only.wav"),
            ]:
                if not source_path or not os.path.exists(source_path):
                    continue
                shutil.copy2(source_path, os.path.join(capture_dir, output_name))
                saved_files += 1

            if saved_files == 0:
                os.rmdir(capture_dir)
                self.log_event("web_debug_audio_capture_empty", request_id=request_id)
                return None

            self.log_event(
                "web_debug_audio_capture_saved",
                request_id=request_id,
                dir=capture_dir,
                files=saved_files,
            )
            return capture_dir
        except Exception as e:
            self.log_event("web_debug_audio_capture_failed", request_id=request_id, error=e)
            return None

    def analyze_web_vad(self, wav_path, source_label="web_vad_upload", request_id=None):
        request_id = request_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if self.is_recording or self.is_processing:
            self.log_event(
                "web_vad_rejected_busy",
                recording=self.is_recording,
                processing=self.is_processing,
            )
            return {
                "ok": False,
                "busy": True,
                "speech_detected": None,
                "speech_secs": 0.0,
                "segments": [],
                "error": "Widget is busy recording or processing.",
                "timings": {},
                "request_id": request_id,
            }

        if not self.web_pipeline_lock.acquire(blocking=False):
            self.log_event("web_vad_rejected_lock_busy")
            return {
                "ok": False,
                "busy": True,
                "speech_detected": None,
                "speech_secs": 0.0,
                "segments": [],
                "error": "Another web audio request is already running.",
                "timings": {},
                "request_id": request_id,
            }

        wav_path = os.path.abspath(wav_path)
        request_dir = None
        started = time.perf_counter()
        timings = {}
        success = False
        marked_processing = False

        try:
            request_dir = tempfile.mkdtemp(prefix=f"vad_{request_id}_", dir=self.web_work_dir)
            raw_path = os.path.join(request_dir, "raw.wav")
            speech_path = os.path.join(request_dir, "speech_only.wav")
            self.is_processing = True
            marked_processing = True
            self.ui_call(self.update_ui_state, "processing")
            self.write_runtime_heartbeat(reason="web_vad_start")
            self.log_event(
                "web_vad_start",
                request_id=request_id,
                source=wav_path,
                source_label=source_label,
            )

            self.vad_client.ensure_ready(timeout_seconds=VAD_WORKER_READY_TIMEOUT_SECONDS)
            shutil.copy2(wav_path, raw_path)

            vad_started = time.perf_counter()
            vad_result = self.vad_client.run(
                in_wav_path=raw_path,
                out_wav_path=speech_path,
                sample_rate=SAMPLE_RATE,
                pad_ms=VAD_PAD_MS,
                min_speech_ms=VAD_MIN_SPEECH_MS,
                merge_gap_ms=VAD_MERGE_GAP_MS,
                speech_prob_threshold=0.5,
                timeout_seconds=VAD_REQUEST_TIMEOUT_SECONDS,
                return_details=True,
            )
            timings["vad_seconds"] = time.perf_counter() - vad_started
            timings["total_seconds"] = time.perf_counter() - started
            speech_secs = float(vad_result["speech_secs"])
            segments = list(vad_result.get("segments") or [])
            speech_detected = speech_secs > 0.0
            success = True
            self.log_event(
                "web_vad_finish",
                request_id=request_id,
                speech_detected=speech_detected,
                speech_secs=f"{speech_secs:.3f}",
                total_seconds=f"{timings['total_seconds']:.3f}",
            )
            return {
                "ok": True,
                "speech_detected": speech_detected,
                "speech_secs": speech_secs,
                "segments": segments,
                "method": "existing_widget_vad",
                "timings": timings,
                "request_id": request_id,
            }
        except Exception as e:
            timings["total_seconds"] = time.perf_counter() - started
            self.log_event(
                "web_vad_failed",
                request_id=request_id,
                error=e,
                total_seconds=f"{timings['total_seconds']:.3f}",
            )
            return {
                "ok": False,
                "speech_detected": None,
                "speech_secs": 0.0,
                "segments": [],
                "timings": timings,
                "request_id": request_id,
                "error": str(e),
            }
        finally:
            if request_dir and success:
                try:
                    shutil.rmtree(request_dir)
                    self.log_event("web_vad_temp_removed", request_id=request_id, dir=request_dir)
                except Exception as cleanup_error:
                    self.log_event(
                        "web_vad_temp_remove_failed",
                        request_id=request_id,
                        dir=request_dir,
                        error=cleanup_error,
                    )
            elif request_dir:
                self.log_event("web_vad_temp_preserved", request_id=request_id, dir=request_dir)
            try:
                if marked_processing:
                    self.ui_call(self.finish_processing)
            finally:
                self.web_pipeline_lock.release()

    def transcribe_web_wav(self, wav_path, source_label="web_upload"):
        if self.is_recording or self.is_processing:
            self.log_event(
                "web_transcription_rejected_busy",
                recording=self.is_recording,
                processing=self.is_processing,
            )
            return {
                "ok": False,
                "busy": True,
                "text": "",
                "error": "Widget is busy recording or processing.",
                "timings": {},
            }

        if not self.web_pipeline_lock.acquire(blocking=False):
            self.log_event("web_transcription_rejected_lock_busy")
            return {
                "ok": False,
                "busy": True,
                "text": "",
                "error": "Another web transcription is already running.",
                "timings": {},
            }

        wav_path = os.path.abspath(wav_path)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        request_dir = tempfile.mkdtemp(prefix=f"{request_id}_", dir=self.web_work_dir)
        raw_path = os.path.join(request_dir, "raw.wav")
        speech_path = os.path.join(request_dir, "speech_only.wav")
        started = time.perf_counter()
        timings = {}
        attempts_payload = []
        text = ""
        success = False
        speech_secs = 0.0
        completed_attempts = 0

        try:
            self.is_processing = True
            self.ui_call(self.update_ui_state, "processing")
            self.write_runtime_heartbeat(reason="web_transcription_start")
            self.log_event(
                "web_transcription_pipeline_start",
                request_id=request_id,
                source=wav_path,
                source_label=source_label,
            )

            self.whisper_client.ensure_ready(timeout_seconds=WHISPER_WORKER_READY_TIMEOUT_SECONDS)
            self.whisper_ready = self.whisper_client.is_ready()
            shutil.copy2(wav_path, raw_path)
            self.log_event("web_transcription_source_staged", request_id=request_id, file=raw_path)

            vad_started = time.perf_counter()
            try:
                speech_secs = self.vad_client.run(
                    in_wav_path=raw_path,
                    out_wav_path=speech_path,
                    sample_rate=SAMPLE_RATE,
                    pad_ms=VAD_PAD_MS,
                    min_speech_ms=VAD_MIN_SPEECH_MS,
                    merge_gap_ms=VAD_MERGE_GAP_MS,
                    speech_prob_threshold=0.5,
                )
                timings["vad_seconds"] = time.perf_counter() - vad_started
                self.log_event(
                    "web_vad_complete",
                    request_id=request_id,
                    speech_secs=f"{speech_secs:.3f}",
                    processing_seconds=f"{timings['vad_seconds']:.3f}",
                )
            except Exception as e:
                timings["vad_seconds"] = time.perf_counter() - vad_started
                speech_secs = -1.0
                self.log_event("web_vad_failed_fallback_raw", request_id=request_id, error=e)

            if speech_secs == 0.0:
                self.log_event("web_vad_no_speech_detected", request_id=request_id)
            elif speech_secs > 0.0:
                self.log_event("web_vad_speech_source_ready", request_id=request_id, file=speech_path)

            debug_dir = self.save_web_debug_audio_files(raw_path, speech_path, request_id)
            attempts = self.build_transcribe_attempts_for_paths(
                raw_path=raw_path,
                speech_path=speech_path,
                speech_secs=speech_secs,
                processed_source=speech_path if speech_secs > 0.0 else None,
            )
            whisper_total = 0.0

            for attempt_number, attempt in enumerate(attempts, start=1):
                attempt_started = time.perf_counter()
                attempt_payload = {
                    "attempt": attempt_number,
                    "stage": attempt["label"],
                    "ok": False,
                    "text_chars": 0,
                    "seconds": 0.0,
                }
                try:
                    self.log_event(
                        "web_transcribe_attempt_start",
                        request_id=request_id,
                        attempt=attempt_number,
                        stage=attempt["label"],
                        source=attempt["path"],
                    )
                    result = self.transcribe_with_whisper(
                        attempt["path"],
                        attempt["options"],
                    )
                    completed_attempts += 1
                    elapsed = time.perf_counter() - attempt_started
                    whisper_total += elapsed
                    attempt_text = (result.get("text") or "").strip()
                    attempt_payload.update(
                        {
                            "ok": True,
                            "text_chars": len(attempt_text),
                            "seconds": elapsed,
                        }
                    )
                    attempts_payload.append(attempt_payload)

                    if attempt_text:
                        text = attempt_text
                        success = True
                        self.log_event(
                            "web_transcribe_attempt_success",
                            request_id=request_id,
                            attempt=attempt_number,
                            stage=attempt["label"],
                            chars=len(text),
                            processing_seconds=f"{elapsed:.3f}",
                        )
                        break

                    self.log_event(
                        "web_transcribe_attempt_no_text",
                        request_id=request_id,
                        attempt=attempt_number,
                        stage=attempt["label"],
                        processing_seconds=f"{elapsed:.3f}",
                    )
                except Exception as e:
                    elapsed = time.perf_counter() - attempt_started
                    whisper_total += elapsed
                    attempt_payload.update(
                        {
                            "ok": False,
                            "error": str(e),
                            "seconds": elapsed,
                        }
                    )
                    attempts_payload.append(attempt_payload)
                    self.log_event(
                        "web_transcribe_attempt_failed",
                        request_id=request_id,
                        attempt=attempt_number,
                        stage=attempt["label"],
                        error=e,
                    )
                    time.sleep(0.5)

            timings["whisper_seconds"] = whisper_total
            if not success and completed_attempts > 0:
                success = True
                self.log_event(
                    "web_transcribe_all_attempts_empty",
                    request_id=request_id,
                    attempts=completed_attempts,
                )

            if text:
                self.print_transcription_to_terminal(text)
                self.log_transcription(text)

            timings["total_seconds"] = time.perf_counter() - started
            self.log_event(
                "web_transcription_pipeline_finish",
                request_id=request_id,
                success=success,
                chars=len(text),
                total_seconds=f"{timings['total_seconds']:.3f}",
            )
            return {
                "ok": success,
                "text": text,
                "speech_secs": speech_secs,
                "timings": timings,
                "attempts": attempts_payload,
                "debug_dir": debug_dir,
                "request_id": request_id,
                **({} if success else {"error": "No transcription attempt completed successfully"}),
            }
        except Exception as e:
            timings["total_seconds"] = time.perf_counter() - started
            self.log_event(
                "web_transcription_pipeline_failed",
                request_id=request_id,
                error=e,
                total_seconds=f"{timings['total_seconds']:.3f}",
            )
            return {
                "ok": False,
                "text": "",
                "speech_secs": speech_secs,
                "timings": timings,
                "attempts": attempts_payload,
                "request_id": request_id,
                "error": str(e),
            }
        finally:
            keep_temp = not success
            if keep_temp:
                self.log_event(
                    "web_transcription_temp_preserved",
                    request_id=request_id,
                    dir=request_dir,
                    success=success,
                )
            else:
                try:
                    shutil.rmtree(request_dir)
                    self.log_event(
                        "web_transcription_temp_removed",
                        request_id=request_id,
                        dir=request_dir,
                    )
                except Exception as cleanup_error:
                    self.log_event(
                        "web_transcription_temp_remove_failed",
                        request_id=request_id,
                        dir=request_dir,
                        error=cleanup_error,
                    )
            self.ui_call(self.finish_processing)
            self.web_pipeline_lock.release()

    def create_raw_audio_backup(self, source_path):
        capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        backup_path = os.path.join(self.raw_audio_backup_dir, f"recording_{capture_id}.wav")
        self.current_raw_backup_path = None

        try:
            shutil.copy2(source_path, backup_path)
            self.current_raw_backup_path = backup_path
            self.log_event("raw_audio_backup_saved", file=backup_path, source=source_path)
            self.update_recovery_state(
                active=True,
                mode="single",
                stage="raw_backup_saved",
                raw_backup_path=backup_path,
            )
            return backup_path
        except Exception as e:
            self.log_event("raw_audio_backup_failed", source=source_path, error=e)
            return None

    def save_raw_audio_backup(self):
        return self.create_raw_audio_backup(self.temp_filename)

    def claim_raw_audio_backup(self, source_path):
        source_path = os.path.abspath(source_path)
        raw_backup_root = os.path.abspath(self.raw_audio_backup_dir)
        try:
            if os.path.commonpath([source_path, raw_backup_root]) == raw_backup_root:
                self.current_raw_backup_path = source_path
                self.update_recovery_state(
                    active=True,
                    mode="single",
                    stage="raw_backup_claimed",
                    raw_backup_path=source_path,
                )
                return source_path
        except Exception:
            pass

        return self.create_raw_audio_backup(source_path)

    def remove_raw_audio_backup(self):
        backup_path = self.current_raw_backup_path
        self.current_raw_backup_path = None
        if not backup_path or not os.path.exists(backup_path):
            return

        try:
            os.remove(backup_path)
            self.log_event("raw_audio_backup_removed", file=backup_path)
        except Exception as e:
            self.log_event("raw_audio_backup_remove_failed", file=backup_path, error=e)

    def preserve_raw_audio_backup(self) -> bool:
        backup_path = self.current_raw_backup_path
        self.current_raw_backup_path = None
        if not backup_path or not os.path.exists(backup_path):
            return False

        print(f"Transcription Failed. Audio saved to: {backup_path}")
        self.log_event("transcription_failed_audio_preserved", file=backup_path)

        if os.path.exists(self.temp_filename):
            try:
                os.remove(self.temp_filename)
                self.log_event("temp_file_removed", file=self.temp_filename)
            except Exception:
                self.log_event("temp_file_remove_failed", file=self.temp_filename)

        return True

    def toggle_mute(self):
        """Toggles mute state and updates UI."""
        self.is_muted = not self.is_muted
        self.log_event("mute_toggled", muted=self.is_muted)
        if self.is_muted:
            self.mute_label.configure(text="🔇")
        else:
            self.mute_label.configure(text="")
            self.play_cue("unmute")

    def update_ui_state(self, state):
        """Updates button color and text based on state."""
        if state != self.last_ui_state:
            self.log_event("ui_state_changed", state=state)
            self.last_ui_state = state
        if state == "loading":
            self.record_btn.configure(text="LOADING", fg_color="#555555", state="disabled")
            self.status_label.configure(text="Loading...")
        elif state == "ready":
            self.record_btn.configure(text="READY", fg_color="#2e7d32", hover_color="#1b5e20", state="normal")
            self.status_label.configure(text="Press F8")
        elif state == "recording":
            self.record_btn.configure(text="STOP", fg_color="#c62828", hover_color="#b71c1c")
            self.status_label.configure(text="Recording...")
        elif state == "processing":
            self.record_btn.configure(text="...", fg_color="#f9a825", state="disabled")
            self.status_label.configure(text="Transcribing...")
        elif state == "recovering":
            self.record_btn.configure(text="RETRY", fg_color="#ef6c00", state="disabled")
            self.status_label.configure(text="Recovering...")
        self.write_runtime_heartbeat(reason="ui_state_changed")

    def toggle_recording(self):
        """Main toggle logic."""
        if threading.current_thread() is not threading.main_thread():
            self.ui_call(self.toggle_recording)
            return

        if not self.is_whisper_ready():
            self.log_event("toggle_ignored_model_not_ready")
            return # Model not loaded yet

        if not self.is_recording:
            if not self.is_processing:
                self.log_event("toggle_start_recording")
                self.start_recording()
        else:
            self.log_event("toggle_stop_recording")
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.is_processing = False
        self.audio_frames = []
        self.reset_temp_audio_files(include_raw=True, reason="recording_start")
        self.update_recovery_state(
            active=True,
            mode="chunked" if CHUNKED_TRANSCRIPTION_ENABLED else "single",
            stage="recording",
            raw_backup_path=None,
        )
        self.start_chunk_capture()
        self.update_ui_state("recording")
        self.log_event("recording_started")

        # Open Stream
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            self.log_event("microphone_stream_opened", sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
            self.record_thread = threading.Thread(
                target=self.record_loop,
                daemon=True,
                name="record_loop",
            )
            self.record_thread.start()
            self.play_cue("record_start")
        except Exception as e:
            print(f"Microphone error: {e}")
            self.log_event("microphone_error", error=e)
            self.is_recording = False
            self.clear_recovery_state(reason="microphone_error")
            self.update_ui_state("ready")

    def record_loop(self):
        while self.is_recording:
            try:
                with self.stream_lock:
                    stream = self.stream
                    if stream is None:
                        break
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.audio_frames.append(data)
                self.record_chunk_frame(data)
            except Exception as e:
                print(f"Record loop error: {e}")
                self.log_event("record_loop_error", error=e)
                if self.is_recording:
                    self.ui_call(self.stop_recording)
                break

    def stop_recording(self):
        self.is_recording = False
        self.log_event("recording_stopped", frames=len(self.audio_frames))

        record_thread = self.record_thread
        if record_thread is not None and record_thread is not threading.current_thread():
            record_thread.join(timeout=2.0)
            if record_thread.is_alive():
                self.log_event("record_thread_join_timeout")
            else:
                self.record_thread = None

        acquired_stream_lock = self.stream_lock.acquire(timeout=2.0)
        if not acquired_stream_lock:
            self.log_event("microphone_stream_close_skipped", reason="stream_lock_timeout")
        else:
            try:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                    self.log_event("microphone_stream_closed")
            finally:
                self.stream_lock.release()
        self.finalize_chunk_capture()

        self.play_cue("record_stop")

        self.update_ui_state("processing")
        self.is_processing = True
        self.write_runtime_heartbeat(reason="stop_recording")
        self.update_recovery_state(
            active=True,
            mode="chunked" if CHUNKED_TRANSCRIPTION_ENABLED else "single",
            stage="processing",
            chunk_count=len(self.chunk_capture_chunks),
        )
        self.log_event("transcription_pipeline_start")

        # Save and Transcribe in Background
        if (
            CHUNKED_TRANSCRIPTION_ENABLED
            and self.chunk_transcribe_queue is not None
            and self.chunk_capture_chunks
        ):
            threading.Thread(target=self.finalize_chunked_transcription, daemon=True).start()
        else:
            threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def finalize_chunked_transcription(self):
        self.log_event("chunked_transcription_pipeline_start", chunks=len(self.chunk_capture_chunks))
        chunk_results = []

        try:
            if self.chunk_pipeline_failed:
                raise RuntimeError("chunk_pipeline_marked_failed")

            chunk_results = self.finish_chunk_transcription_queue()
            failed_results = [result for result in chunk_results if not result.get("ok")]
            if failed_results:
                raise RuntimeError(f"chunk_transcription_failures={len(failed_results)}")

            text = self.stitch_chunk_transcripts(chunk_results)
            self.write_chunk_transcript_debug(chunk_results, text, reason="success")
            if text:
                print("\n[transcription]")
                print(text)
                print("[/transcription]", flush=True)
                self.log_transcription(text)
                clipboard_copied = self.copy_transcription_to_clipboard(text)
                if clipboard_copied:
                    print("Copied transcription to clipboard.")
                self.log_event(
                    "chunked_transcription_success",
                    chunks=len(chunk_results),
                    chars=len(text),
                    clipboard_copied=clipboard_copied,
                )
                if clipboard_copied:
                    self.ui_call(self.record_btn.configure, text="COPIED", fg_color="#1565c0")
                else:
                    self.ui_call(self.record_btn.configure, text="SAVED", fg_color="#ef6c00")
                self.play_cue("success")
                time.sleep(1)
            else:
                self.log_event("chunked_transcription_empty", chunks=len(chunk_results))

            self.log_event(
                "chunked_transcription_pipeline_finish",
                success=True,
                chunks=len(chunk_results),
                chars=len(text),
            )
            self.cleanup_chunked_temp_dir(
                keep=CHUNKED_KEEP_TEMP_ON_SUCCESS,
                reason="success",
            )
            self.clear_recovery_state(reason="chunked_transcription_success")
            self.ui_call(self.finish_processing)
        except Exception as e:
            self.log_event("chunked_transcription_failed_fallback_batch", error=e)
            with self.chunk_transcribe_lock:
                partial_results = [
                    self.chunk_transcribe_results[index]
                    for index in sorted(self.chunk_transcribe_results)
                ]
            self.write_chunk_transcript_debug(partial_results or chunk_results, "", reason="failure")

            worker_active = (
                self.chunk_transcribe_thread is not None
                and self.chunk_transcribe_thread.is_alive()
            )
            if worker_active:
                try:
                    self.finish_chunk_transcription_queue(timeout_seconds=30)
                    worker_active = False
                except Exception as queue_error:
                    worker_active = (
                        self.chunk_transcribe_thread is not None
                        and self.chunk_transcribe_thread.is_alive()
                    )
                    self.log_event("chunked_transcription_queue_shutdown_failed", error=queue_error)

            self.cleanup_chunked_temp_dir(
                keep=CHUNKED_KEEP_TEMP_ON_FAILURE,
                reason="fallback_batch",
            )
            if worker_active:
                self.log_event("chunked_transcription_fallback_skipped_worker_active")
                self.clear_recovery_state(reason="chunked_fallback_skipped_worker_active")
                self.ui_call(self.finish_processing)
                return

            self.update_recovery_state(
                active=True,
                mode="single",
                stage="fallback_batch_transcription",
            )
            self.transcribe_audio()

    def transcribe_audio(self):
        self.log_event("transcribe_thread_started")
        try:
            self.reset_temp_audio_files(include_raw=False, reason="transcribe_audio_start")
            wf = wave.open(self.temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            self.log_event("audio_saved", file=self.temp_filename, frames=len(self.audio_frames))
            self.save_raw_audio_backup()
        except Exception as e:
            print(f"Error saving WAV: {e}")
            self.log_event("audio_save_failed", error=e)
            self.clear_recovery_state(reason="audio_save_failed")
            self.ui_call(self.finish_processing)
            return

        self.transcribe_saved_audio(self.temp_filename, source_label="live_recording")

    def transcribe_saved_audio(self, raw_source_path, source_label="saved_audio"):
        raw_source_path = os.path.abspath(raw_source_path)
        active_input_path = os.path.abspath(self.temp_filename)

        try:
            self.reset_temp_audio_files(
                include_raw=(raw_source_path != active_input_path),
                reason="transcribe_saved_audio_start",
            )
            if raw_source_path != active_input_path:
                shutil.copy2(raw_source_path, self.temp_filename)
                self.log_event(
                    "transcribe_source_staged",
                    source=raw_source_path,
                    staged_file=self.temp_filename,
                    source_label=source_label,
                )
        except Exception as e:
            self.log_event(
                "transcribe_source_stage_failed",
                source=raw_source_path,
                source_label=source_label,
                error=e,
            )
            self.clear_recovery_state(reason="transcribe_source_stage_failed")
            self.ui_call(self.finish_processing)
            return

        self.update_recovery_state(
            active=True,
            mode="single",
            stage="processing_saved_audio",
            source_label=source_label,
            raw_backup_path=self.current_raw_backup_path,
        )

        speech_secs = 0.0
        processed_source = None
        self.log_event("vad_start", file=self.temp_filename)
        try:
            speech_secs = self.vad_client.run(
                in_wav_path=self.temp_filename,
                out_wav_path=self.temp_speech_filename,
                sample_rate=SAMPLE_RATE,
                pad_ms=VAD_PAD_MS,
                min_speech_ms=VAD_MIN_SPEECH_MS,
                merge_gap_ms=VAD_MERGE_GAP_MS,
                speech_prob_threshold=0.5,
            )
            self.log_event("vad_complete", speech_secs=f"{speech_secs:.3f}")
        except Exception as e:
            print(f"VAD failed, falling back to raw audio: {e}")
            self.log_event("vad_failed_fallback_raw", error=e)
            speech_secs = -1.0

        success = False

        if speech_secs == 0.0:
            print("No speech detected by VAD, trying fallback transcription.")
            self.log_event("vad_no_speech_detected")
        elif speech_secs > 0.0:
            processed_source = self.temp_speech_filename
            self.log_event("vad_speech_source_ready", file=processed_source)

        self.save_debug_audio_files()

        completed_attempts = 0
        attempts = self.build_transcribe_attempts(speech_secs, processed_source)

        for attempt_number, attempt in enumerate(attempts, start=1):
            try:
                self.log_event(
                    "transcribe_attempt_start",
                    attempt=attempt_number,
                    stage=attempt["label"],
                    source=attempt["path"]
                )
                result = self.transcribe_with_whisper(
                    attempt["path"],
                    attempt["options"],
                )
                completed_attempts += 1
                text = (result.get("text") or "").strip()

                if text:
                    print("\n[transcription]")
                    print(text)
                    print("[/transcription]", flush=True)
                    self.log_transcription(text)
                    clipboard_copied = self.copy_transcription_to_clipboard(text)
                    if clipboard_copied:
                        print("Copied transcription to clipboard.")
                    self.log_event(
                        "transcribe_attempt_success",
                        attempt=attempt_number,
                        stage=attempt["label"],
                        chars=len(text),
                        clipboard_copied=clipboard_copied,
                    )
                    if clipboard_copied:
                        self.ui_call(self.record_btn.configure, text="COPIED", fg_color="#1565c0")
                    else:
                        self.ui_call(self.record_btn.configure, text="SAVED", fg_color="#ef6c00")
                    self.play_cue("success")
                    time.sleep(1)
                    success = True
                    break

                print("No text detected.")
                self.log_event(
                    "transcribe_attempt_no_text",
                    attempt=attempt_number,
                    stage=attempt["label"]
                )
            except Exception as e:
                print(f"Attempt {attempt_number} failed: {e}")
                self.log_event(
                    "transcribe_attempt_failed",
                    attempt=attempt_number,
                    stage=attempt["label"],
                    error=e
                )
                time.sleep(0.5)

        if not success and completed_attempts > 0:
            success = True
            self.log_event("transcribe_all_attempts_empty", attempts=completed_attempts)

        self.log_event("transcription_pipeline_finish", success=success)
        self.ui_call(self.finish_processing)

        if success:
            self.remove_raw_audio_backup()
            for f in [
                self.temp_filename,
                self.temp_speech_filename,
            ]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        self.log_event("temp_file_removed", file=f)
                    except:
                        self.log_event("temp_file_remove_failed", file=f)
                        pass
            self.clear_recovery_state(reason="transcription_pipeline_complete")
        else:
            raw_backup_preserved = self.preserve_raw_audio_backup()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = os.path.join(self.failed_audio_dir, f"failed_recording_{timestamp}.wav")
            if not raw_backup_preserved:
                try:
                    os.rename(self.temp_filename, failed_filename)
                    print(f"Transcription Failed. Audio saved to: {failed_filename}")
                    self.log_event("transcription_failed_audio_saved", file=failed_filename)
                except Exception as e:
                    print(f"Could not save backup file: {e}")
                    self.log_event("transcription_failed_backup_save_error", error=e)

            if os.path.exists(self.temp_speech_filename):
                try:
                    failed_speech_filename = os.path.join(self.failed_audio_dir, f"failed_speech_only_{timestamp}.wav")
                    os.rename(self.temp_speech_filename, failed_speech_filename)
                    self.log_event("transcription_failed_speech_saved", file=failed_speech_filename)
                except:
                    self.log_event("transcription_failed_speech_save_failed")
                    pass
            self.clear_recovery_state(reason="transcription_pipeline_complete")

    def finish_processing(self):
        self.is_processing = False
        self.update_ui_state("ready")
        self.log_event("processing_finished")
        self.write_runtime_heartbeat(reason="processing_finished")

    def on_close(self):
        self.log_event("app_close_start")
        if not self.is_recording and not self.is_processing:
            self.clear_recovery_state(reason="clean_shutdown")
        else:
            self.log_event(
                "app_close_preserving_recovery_state",
                recording=self.is_recording,
                processing=self.is_processing,
            )
        self.write_runtime_heartbeat(status="stopping", reason="app_close_start")
        self.shutdown_event.set()
        self.unregister_hotkey()
        self.stop_web_server()
        self.whisper_client.close()
        self.vad_client.close()
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self.p.terminate()
            self.log_event("audio_shutdown_ok")
        except Exception as e:
            print(f"Audio cleanup failed: {e}")
            self.log_event("audio_shutdown_failed", error=e)
        self.log_event("app_close_end")
        self.remove_runtime_heartbeat()
        self.destroy()

if __name__ == "__main__":
    check_data_layout()
    enable_fatal_crash_logging()
    app = WhisperWidget()
    app.mainloop()
