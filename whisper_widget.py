import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import ctypes
import customtkinter as ctk
import whisper
import pyaudio
import wave
import audioop
import threading
import time
import sys
import shutil
import subprocess
import json
import pyperclip
import math
import struct
import numpy as np
import torch
import warnings
from queue import Queue, Empty
from ctypes import wintypes
from datetime import datetime
import soundfile as sf
try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    import keyboard
except Exception:
    keyboard = None

# Suppress Torch FutureWarnings (cleaner UI)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MODEL_SIZE = "large-v3"
HOTKEY = "f8"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
LOG_DIR = "transcriptions"
EVENT_LOG_FILE = "event_log.txt"
EVENT_LOG_MAX_BYTES = 5 * 1024 * 1024
EVENT_LOG_BACKUP_COUNT = 2
EVENT_LOG_HEADER = "timestamp | event | details\n"
DEBUG_AUDIO_DIR = "debug_audio"
SAVE_DEBUG_AUDIO = True
RAW_AUDIO_BACKUP_DIR = "raw_audio_backups"
WHISPER_LANGUAGE = "en"
WHISPER_NO_SPEECH_THRESHOLD = None
VAD_PAD_MS = 400
VAD_MIN_SPEECH_MS = 150
VAD_MERGE_GAP_MS = 600
VAD_REQUEST_TIMEOUT_SECONDS = 180
VAD_WORKER_READY_TIMEOUT_SECONDS = 30
NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS = 180
NOISE_REDUCTION_ENABLED = True
NOISE_REDUCTION_BACKEND = "webrtc_apm_wsl"
NOISE_REDUCTION_WEBRTC_PRESET = "light"
NOISE_REDUCTION_WEBRTC_DISTRO = "Ubuntu-22.04"
NOISE_REDUCTION_PROP_DECREASE = 0.85
NOISE_REDUCTION_CHUNK_SECONDS = 10.0
NOISE_REDUCTION_PADDING_SECONDS = 2.0
NOISE_REDUCTION_N_FFT = 512
NORMALIZE_AUDIO_ENABLED = True
NORMALIZE_TARGET_PEAK_DBFS = -4.0
NORMALIZE_MAX_GAIN_DB = 16.0
CHUNKED_TRANSCRIPTION_ENABLED = False
CHUNKED_CHUNK_SECONDS = 30.0
CHUNKED_OVERLAP_SECONDS = 1.5
CHUNKED_QUEUE_MAXSIZE = 4
CHUNKED_TEMP_DIR = "chunked_transcription_work"
CHUNKED_KEEP_TEMP_ON_SUCCESS = False
CHUNKED_KEEP_TEMP_ON_FAILURE = True

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

    def __init__(self, worker_script_path: str, workdir: str, log_fn):
        self.worker_script_path = worker_script_path
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
            self.worker_script_path,
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
    ) -> float:
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

            return float(message["speech_secs"])

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


def reduce_noise_wav(
    in_wav_path: str,
    out_wav_path: str,
    prop_decrease: float = 0.85,
    chunk_seconds: float = 10.0,
    padding_seconds: float = 2.0,
    n_fft: int = 512,
):
    """Writes a denoised mono WAV for Whisper preprocessing."""
    if nr is None:
        raise RuntimeError("noisereduce is not installed")

    audio, sample_rate = sf.read(in_wav_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    chunk_size = max(n_fft, int(sample_rate * chunk_seconds))
    padding = max(n_fft, int(sample_rate * padding_seconds))

    reduced_audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=False,
        prop_decrease=prop_decrease,
        chunk_size=chunk_size,
        padding=padding,
        n_fft=n_fft,
    )
    sf.write(out_wav_path, reduced_audio, sample_rate, subtype="PCM_16")


def reduce_noise_wav_subprocess(
    worker_script_path: str,
    in_wav_path: str,
    out_wav_path: str,
    prop_decrease: float = 0.85,
    chunk_seconds: float = 10.0,
    padding_seconds: float = 2.0,
    n_fft: int = 512,
    timeout_seconds: float = NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS,
):
    """Runs noise reduction in a child process so native crashes cannot kill the UI."""
    command = [
        sys.executable,
        worker_script_path,
        "--input",
        in_wav_path,
        "--output",
        out_wav_path,
        "--prop-decrease",
        str(prop_decrease),
        "--chunk-seconds",
        str(chunk_seconds),
        "--padding-seconds",
        str(padding_seconds),
        "--n-fft",
        str(n_fft),
    ]
    return run_json_subprocess(
        command=command,
        cwd=os.path.dirname(worker_script_path) or None,
        timeout_seconds=timeout_seconds,
    )


def run_json_subprocess(
    command,
    cwd: str | None = None,
    timeout_seconds: float = NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS,
):
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        details = [f"returncode={completed.returncode}"]
        if stderr:
            details.append(f"stderr={stderr[:400]}")
        if stdout:
            details.append(f"stdout={stdout[:400]}")
        raise RuntimeError(" ".join(details))

    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_worker_output={stdout[:400]}") from exc


def reduce_noise_wav_webrtc_apm_subprocess(
    wrapper_script_path: str,
    in_wav_path: str,
    out_wav_path: str,
    preset: str = NOISE_REDUCTION_WEBRTC_PRESET,
    distro: str = NOISE_REDUCTION_WEBRTC_DISTRO,
    timeout_seconds: float = NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS,
):
    """Runs WebRTC APM denoise via the local wrapper so the UI process stays isolated."""
    command = [
        sys.executable,
        wrapper_script_path,
        "--input",
        in_wav_path,
        "--output",
        out_wav_path,
        "--preset",
        preset,
    ]
    if distro:
        command.extend(["--distro", distro])

    return run_json_subprocess(
        command=command,
        cwd=os.path.dirname(wrapper_script_path) or None,
        timeout_seconds=timeout_seconds,
    )


def normalize_wav(
    in_wav_path: str,
    out_wav_path: str,
    target_peak_dbfs: float = NORMALIZE_TARGET_PEAK_DBFS,
    max_gain_db: float = NORMALIZE_MAX_GAIN_DB,
    target_sample_rate: int | None = SAMPLE_RATE,
):
    """Writes a conservatively peak-normalized mono PCM WAV.

    Keep this path independent of torch/torchaudio/CUDA. A Python exception here
    can be handled by the caller; a native library access violation cannot.
    """
    with wave.open(in_wav_path, "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        compression = reader.getcomptype()
        frames = reader.readframes(reader.getnframes())

    if compression != "NONE":
        raise ValueError(f"unsupported WAV compression: {compression}")
    if sample_width != 2:
        raise ValueError(f"normalize_wav only supports 16-bit PCM, got {sample_width * 8}-bit")
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"normalize_wav only supports mono/stereo WAV, got {channels} channels")

    input_sample_rate = int(sample_rate)
    if target_sample_rate and int(sample_rate) != int(target_sample_rate):
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            int(sample_rate),
            int(target_sample_rate),
            None,
        )
        sample_rate = int(target_sample_rate)

    full_scale = float((1 << (8 * sample_width - 1)) - 1)
    peak = float(audioop.max(frames, sample_width)) if frames else 0.0
    if peak <= 0.0:
        normalized_frames = frames
        gain = 1.0
        output_peak = 0.0
    else:
        target_peak = full_scale * (10 ** (target_peak_dbfs / 20.0))
        max_gain = 10 ** (max_gain_db / 20.0)
        gain = min(max_gain, max(1.0, target_peak / peak))
        normalized_frames = audioop.mul(frames, sample_width, gain)
        output_peak = float(audioop.max(normalized_frames, sample_width)) if normalized_frames else 0.0

    with wave.open(out_wav_path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(sample_width)
        writer.setframerate(int(sample_rate))
        writer.writeframes(normalized_frames)

    input_peak_dbfs = 20.0 * math.log10(max(peak / full_scale, 1e-12))
    output_peak_dbfs = 20.0 * math.log10(max(output_peak / full_scale, 1e-12))
    return {
        "gain_db": 20.0 * math.log10(max(gain, 1e-12)),
        "input_peak_dbfs": input_peak_dbfs,
        "output_peak_dbfs": output_peak_dbfs,
        "sample_rate": int(sample_rate),
        "input_sample_rate": input_sample_rate,
    }


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
        super().__init__()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.noise_reduce_worker_script = os.path.join(self.base_dir, "noise_reduce_worker.py")
        self.webrtc_apm_wrapper_script = os.path.join(self.base_dir, "webrtc_apm_wav_wrapper.py")
        
        # Ensure Log Directory Exists
        os.makedirs(LOG_DIR, exist_ok=True)
        self.event_log_path = os.path.join(self.base_dir, EVENT_LOG_FILE)
        self.debug_audio_dir = os.path.join(self.base_dir, DEBUG_AUDIO_DIR)
        self.raw_audio_backup_dir = os.path.join(self.base_dir, RAW_AUDIO_BACKUP_DIR)
        self.chunked_temp_dir = os.path.join(self.base_dir, CHUNKED_TEMP_DIR)
        if SAVE_DEBUG_AUDIO:
            os.makedirs(self.debug_audio_dir, exist_ok=True)
        os.makedirs(self.raw_audio_backup_dir, exist_ok=True)
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
            temp_dir=self.chunked_temp_dir,
        )

        # Window Setup
        self.title("Whisper")
        self.geometry("120x120+50+50") # Small square
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
        self.sound_lock = threading.Lock()
        self.temp_filename = "temp_recording.wav"
        self.temp_denoised_filename = "temp_denoised.wav"
        self.temp_normalized_filename = "temp_normalized.wav"
        self.temp_speech_filename = "temp_speech_only.wav"
        self.current_raw_backup_path = None
        self.chunk_spooler = None
        self.chunk_capture_chunks = []
        self.vad_worker_path = os.path.join(self.base_dir, "vad_worker.py")
        self.vad_client = PersistentVADWorkerClient(
            worker_script_path=self.vad_worker_path,
            workdir=self.base_dir,
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
        self.model = None
        self.status_label = ctk.CTkLabel(self, text="Loading Model...", font=("Arial", 10))
        self.status_label.grid(row=1, column=0, pady=(0, 5))
        
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
        self.log_event("app_ready")

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
            self.log_chunk_event(
                "capture_started",
                chunk_seconds=CHUNKED_CHUNK_SECONDS,
                overlap_seconds=CHUNKED_OVERLAP_SECONDS,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                sample_width=sample_width,
            )
        except Exception as e:
            self.chunk_spooler = None
            self.log_chunk_event("capture_start_failed", error=e)

    def record_chunk_frame(self, data):
        if self.chunk_spooler is None:
            return

        try:
            self.chunk_spooler.add_frame(data)
        except Exception as e:
            self.log_chunk_event("capture_frame_failed", error=e)
            self.chunk_spooler = None

    def finalize_chunk_capture(self, reason="recording_stop"):
        if self.chunk_spooler is None:
            return

        try:
            self.chunk_capture_chunks = self.chunk_spooler.finalize()
            self.log_chunk_event(
                "capture_finalized",
                chunks=len(self.chunk_capture_chunks),
                reason=reason,
            )
        except Exception as e:
            self.log_chunk_event("capture_finalize_failed", reason=reason, error=e)
        finally:
            self.chunk_spooler = None

    def log_transcription(self, text):
        """Appends transcription to a daily log file."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            filename = os.path.join(LOG_DIR, f"{date_str}.txt")
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {text}\n")
            self.log_event("transcription_log_saved", chars=len(text), file=filename)
        except Exception as e:
            print(f"Failed to log transcription: {e}")
            self.log_event("transcription_log_failed", error=e)

    def load_model(self):
        """Loads the Whisper model on a background thread."""
# Note: cuda:1 actually maps to A6000 due to CUDA_VISIBLE_DEVICES remapping
        try:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            self.log_event("model_load_start", device=device, model=MODEL_SIZE)
            self.model = whisper.load_model(MODEL_SIZE, device=device)
            try:
                self.log_event("vad_worker_prewarm_start")
                self.vad_client.ensure_ready()
                self.log_event("vad_worker_prewarm_success")
            except Exception as vad_error:
                print(f"VAD worker prewarm failed: {vad_error}")
                self.log_event("vad_worker_prewarm_failed", error=vad_error)
            self.ui_call(self.update_ui_state, "ready")
            print("Model loaded.")
            self.log_event("model_load_success", device=device)
            self.play_sound(1000, 100)
        except Exception as e:
            self.ui_call(self.status_label.configure, text="Error Loading")
            print(f"Error loading model: {e}")
            self.log_event("model_load_failed", error=e)

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

    def play_sound(self, freq, duration):
        """Queues a non-blocking notification tone."""
        self.play_sound_sequence([(freq, duration)])

    def play_sound_sequence(self, tones):
        if self.is_muted:
            return

        threading.Thread(
            target=self._play_sound_sequence_sync,
            args=(list(tones),),
            daemon=True
        ).start()

    def _play_sound_sequence_sync(self, tones):
        try:
            with self.sound_lock:
                for freq, duration in tones:
                    duration_ms = duration
                    rate = 44100

                    num_samples = int(rate * (duration_ms / 1000.0))
                    audio_data = []
                    for x in range(num_samples):
                        sample = 0.38 * math.sin(2 * math.pi * freq * (x / rate))
                        packed_sample = struct.pack('<h', int(sample * 32767.0))
                        audio_data.append(packed_sample)

                    byte_stream = b''.join(audio_data)

                    stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        output=True
                    )
                    stream.write(byte_stream)
                    stream.stop_stream()
                    stream.close()
        except Exception as e:
            print(f"Sound Error: {e}")
            self.log_event("sound_error", error=e)

    def build_transcribe_attempts(self, speech_secs, processed_source):
        base_options = {
            "condition_on_previous_text": False,
            "language": WHISPER_LANGUAGE,
            "logprob_threshold": -1.0,
            "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        }
        attempt_candidates = []

        if speech_secs > 0.0 and processed_source:
            attempt_candidates.append(("speech_only_primary", processed_source, {}))
            if processed_source != self.temp_denoised_filename and os.path.exists(self.temp_denoised_filename):
                attempt_candidates.append(("speech_only_denoised_fallback", self.temp_denoised_filename, {}))
            if processed_source != self.temp_speech_filename and os.path.exists(self.temp_speech_filename):
                attempt_candidates.append(("speech_only_raw_fallback", self.temp_speech_filename, {}))

        attempt_candidates.append(("raw_full_fallback", self.temp_filename, {}))
        attempt_candidates.append((
            "raw_full_permissive",
            self.temp_filename,
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
                (self.temp_denoised_filename, "denoised.wav"),
                (self.temp_normalized_filename, "normalized.wav"),
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

    def save_raw_audio_backup(self):
        capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        backup_path = os.path.join(self.raw_audio_backup_dir, f"recording_{capture_id}.wav")
        self.current_raw_backup_path = None

        try:
            shutil.copy2(self.temp_filename, backup_path)
            self.current_raw_backup_path = backup_path
            self.log_event("raw_audio_backup_saved", file=backup_path)
        except Exception as e:
            self.log_event("raw_audio_backup_failed", error=e)

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
            self.play_sound(1000, 50)

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

    def toggle_recording(self):
        """Main toggle logic."""
        if threading.current_thread() is not threading.main_thread():
            self.ui_call(self.toggle_recording)
            return

        if self.model is None:
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
        self.audio_frames = []
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
            threading.Thread(target=self.record_loop, daemon=True).start()
            self.play_sound(800, 100) # High Beep
        except Exception as e:
            print(f"Microphone error: {e}")
            self.log_event("microphone_error", error=e)
            self.is_recording = False
            self.update_ui_state("ready")

    def record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
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

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.finalize_chunk_capture()

        self.play_sound(400, 100) # Low Beep

        self.update_ui_state("processing")
        self.is_processing = True
        self.log_event("transcription_pipeline_start")

        # Save and Transcribe in Background
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def transcribe_audio(self):
        self.log_event("transcribe_thread_started")
        # Save to WAV
        try:
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
            self.ui_call(self.finish_processing)
            return

        # -------- VAD FIRST (CPU) --------
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

            if NOISE_REDUCTION_ENABLED:
                noise_backend = NOISE_REDUCTION_BACKEND
                self.log_event("noise_reduction_start", source=processed_source, backend=noise_backend)
                try:
                    if noise_backend == "webrtc_apm_wsl":
                        if not os.path.exists(self.webrtc_apm_wrapper_script):
                            raise FileNotFoundError(f"Missing WebRTC wrapper: {self.webrtc_apm_wrapper_script}")

                        worker_result = reduce_noise_wav_webrtc_apm_subprocess(
                            self.webrtc_apm_wrapper_script,
                            processed_source,
                            self.temp_denoised_filename,
                            preset=NOISE_REDUCTION_WEBRTC_PRESET,
                            distro=NOISE_REDUCTION_WEBRTC_DISTRO,
                            timeout_seconds=NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS,
                        )
                        processed_source = self.temp_denoised_filename
                        self.log_event(
                            "noise_reduction_complete",
                            file=processed_source,
                            backend=noise_backend,
                            preset=NOISE_REDUCTION_WEBRTC_PRESET,
                            output_sample_rate_hz=worker_result.get("output_sample_rate_hz"),
                            output_audio_seconds=worker_result.get("seconds"),
                            processing_seconds=worker_result.get("elapsed_seconds"),
                        )
                    elif noise_backend == "noisereduce_subprocess":
                        if nr is None:
                            raise RuntimeError("noisereduce is not installed")

                        worker_result = reduce_noise_wav_subprocess(
                            self.noise_reduce_worker_script,
                            processed_source,
                            self.temp_denoised_filename,
                            prop_decrease=NOISE_REDUCTION_PROP_DECREASE,
                            chunk_seconds=NOISE_REDUCTION_CHUNK_SECONDS,
                            padding_seconds=NOISE_REDUCTION_PADDING_SECONDS,
                            n_fft=NOISE_REDUCTION_N_FFT,
                            timeout_seconds=NOISE_REDUCTION_REQUEST_TIMEOUT_SECONDS,
                        )
                        processed_source = self.temp_denoised_filename
                        self.log_event(
                            "noise_reduction_complete",
                            file=processed_source,
                            backend=noise_backend,
                            chunk_seconds=NOISE_REDUCTION_CHUNK_SECONDS,
                            padding_seconds=NOISE_REDUCTION_PADDING_SECONDS,
                            n_fft=NOISE_REDUCTION_N_FFT,
                            output_audio_seconds=worker_result.get("seconds"),
                        )
                    else:
                        raise RuntimeError(f"unsupported_noise_reduction_backend={noise_backend}")
                except Exception as e:
                    print(f"Noise reduction failed, falling back to speech-only audio: {e}")
                    self.log_event(
                        "noise_reduction_failed_fallback_speech_only",
                        backend=noise_backend,
                        error=e,
                    )

            if NORMALIZE_AUDIO_ENABLED and processed_source and os.path.exists(processed_source):
                self.log_event("normalize_start", source=processed_source)
                try:
                    stats = normalize_wav(
                        processed_source,
                        self.temp_normalized_filename,
                        target_peak_dbfs=NORMALIZE_TARGET_PEAK_DBFS,
                        max_gain_db=NORMALIZE_MAX_GAIN_DB,
                    )
                    processed_source = self.temp_normalized_filename
                    self.log_event(
                        "normalize_complete",
                        file=processed_source,
                        gain_db=f"{stats['gain_db']:.2f}",
                        input_peak_dbfs=f"{stats['input_peak_dbfs']:.2f}",
                        output_peak_dbfs=f"{stats['output_peak_dbfs']:.2f}",
                        input_sample_rate_hz=stats["input_sample_rate"],
                        output_sample_rate_hz=stats["sample_rate"],
                    )
                except Exception as e:
                    print(f"Normalization failed, falling back to processed speech-only audio: {e}")
                    self.log_event("normalize_failed_fallback_processed", error=e)
        elif NORMALIZE_AUDIO_ENABLED:
            self.log_event("normalize_skipped", source="no_speech_or_vad_failure")

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
                result = self.model.transcribe(
                    attempt["path"],
                    **attempt["options"],
                )
                completed_attempts += 1
                text = result["text"].strip()

                if text:
                    print("\n[transcription]")
                    print(text)
                    print("[/transcription]", flush=True)
                    pyperclip.copy(text)
                    print("Copied transcription to clipboard.")
                    self.log_event(
                        "transcribe_attempt_success",
                        attempt=attempt_number,
                        stage=attempt["label"],
                        chars=len(text)
                    )

                    self.log_transcription(text)

                    self.ui_call(self.record_btn.configure, text="COPIED", fg_color="#1565c0")
                    self.play_sound_sequence([(1000, 100), (1500, 100)])
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

        # Cleanup Logic
        if success:
            self.remove_raw_audio_backup()
            for f in [
                self.temp_filename,
                self.temp_denoised_filename,
                self.temp_normalized_filename,
                self.temp_speech_filename,
            ]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        self.log_event("temp_file_removed", file=f)
                    except:
                        self.log_event("temp_file_remove_failed", file=f)
                        pass
        else:
            # Failure: Keep the pre-VAD backup if it was captured.
            raw_backup_preserved = self.preserve_raw_audio_backup()

            # Failure: Save the file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = f"failed_recording_{timestamp}.wav"
            if not raw_backup_preserved:
                try:
                    os.rename(self.temp_filename, failed_filename)
                    print(f"Transcription Failed. Audio saved to: {failed_filename}")
                    self.log_event("transcription_failed_audio_saved", file=failed_filename)
                except Exception as e:
                    print(f"Could not save backup file: {e}")
                    self.log_event("transcription_failed_backup_save_error", error=e)

            if os.path.exists(self.temp_denoised_filename):
                try:
                    failed_denoised_filename = f"failed_denoised_{timestamp}.wav"
                    os.rename(self.temp_denoised_filename, failed_denoised_filename)
                    self.log_event("transcription_failed_denoised_saved", file=failed_denoised_filename)
                except:
                    self.log_event("transcription_failed_denoised_save_failed")
                    pass

            if os.path.exists(self.temp_normalized_filename):
                try:
                    failed_normalized_filename = f"failed_normalized_{timestamp}.wav"
                    os.rename(self.temp_normalized_filename, failed_normalized_filename)
                    self.log_event("transcription_failed_normalized_saved", file=failed_normalized_filename)
                except:
                    self.log_event("transcription_failed_normalized_save_failed")
                    pass

            if os.path.exists(self.temp_speech_filename):
                try:
                    failed_speech_filename = f"failed_speech_only_{timestamp}.wav"
                    os.rename(self.temp_speech_filename, failed_speech_filename)
                    self.log_event("transcription_failed_speech_saved", file=failed_speech_filename)
                except:
                    self.log_event("transcription_failed_speech_save_failed")
                    pass

    def finish_processing(self):
        self.is_processing = False
        self.update_ui_state("ready")
        self.log_event("processing_finished")

    def on_close(self):
        self.log_event("app_close_start")
        self.shutdown_event.set()
        self.unregister_hotkey()
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
        self.destroy()

if __name__ == "__main__":
    app = WhisperWidget()
    app.mainloop()
