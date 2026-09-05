from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
import json
import os
import shutil
import subprocess
import sys
from app.paths import (
    REPO_ROOT, VAD_WORKER_MODULE, WHISPER_WORKER_MODULE, LOG_DIR,
    WEB_EVENT_LOG_FILE as EVENT_LOG_FILE, DEBUG_AUDIO_DIR, WEB_WORK_DIR, check_data_layout,
)
import tempfile
import threading
import time


MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "auto"
WHISPER_WORKER_READY_TIMEOUT_SECONDS = 120
WHISPER_TRANSCRIBE_TIMEOUT_SECONDS = 180
SAMPLE_RATE = 16000
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


class EventLogger:
    """Thread-safe event logger for headless runtime diagnostics."""

    def __init__(
        self,
        path: str,
        max_bytes: int = EVENT_LOG_MAX_BYTES,
        backup_count: int = EVENT_LOG_BACKUP_COUNT,
    ):
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

    def _archive_path(self, index: int) -> str:
        root, ext = os.path.splitext(self.path)
        return f"{root}.{index}{ext}"

    def _write_header_locked(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(EVENT_LOG_HEADER)

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

            if os.path.exists(self.path):
                os.replace(self.path, self._archive_path(1))
        elif os.path.exists(self.path):
            os.remove(self.path)

        self._write_header_locked()

    def _ensure_log_file_locked(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
            self._rotate_locked()
            return

        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            self._write_header_locked()

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
    """Keeps Silero VAD in a separate long-lived process."""

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
        self._stdout_thread = threading.Thread(
            target=self._stdout_loop,
            args=(process,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            args=(process,),
            daemon=True,
        )
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
                            if self._ready:
                                self._last_start_error = None
                            else:
                                self._last_start_error = (
                                    message.get("error") or "unknown worker startup error"
                                )
                            self._ready_event.set()
                    if not is_current:
                        continue
                    if self._ready:
                        self._log("vad_worker_ready", pid=process.pid)
                    else:
                        self._log(
                            "vad_worker_ready_failed",
                            pid=process.pid,
                            error=self._last_start_error,
                        )
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
        sample_rate: int = SAMPLE_RATE,
        pad_ms: int = VAD_PAD_MS,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        merge_gap_ms: int = VAD_MERGE_GAP_MS,
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

        self._log(
            "vad_worker_terminated",
            pid=getattr(process, "pid", None),
            reason=reason,
            returncode=process.poll(),
        )

    def close(self):
        with self._lock:
            process = self._process
            self._terminate_locked(process, reason="close")


class PersistentWhisperWorkerClient:
    """Keeps Whisper/PyTorch in a separate process so native crashes do not kill the server."""

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
        self._stdout_thread = threading.Thread(
            target=self._stdout_loop,
            args=(process,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            args=(process,),
            daemon=True,
        )
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
                            if self._ready:
                                self._last_start_error = None
                            else:
                                self._last_start_error = (
                                    message.get("error") or "unknown worker startup error"
                                )
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
                        self._log(
                            "whisper_worker_ready_failed",
                            pid=process.pid,
                            error=self._last_start_error,
                        )
                elif message_type == "response":
                    self._response_queue.put(message)
                elif message_type == "shutdown_ack":
                    self._log("whisper_worker_shutdown_ack", pid=process.pid)
                else:
                    self._log(
                        "whisper_worker_message_unknown",
                        pid=process.pid,
                        message_type=message_type,
                    )
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
            ready_timeout = min(WHISPER_WORKER_READY_TIMEOUT_SECONDS, max(5, timeout_seconds))
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
                        raise RuntimeError(
                            f"Whisper worker exited while processing audio (code {process.poll()})"
                        )
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

        self._log(
            "whisper_worker_terminated",
            pid=getattr(process, "pid", None),
            reason=reason,
            returncode=process.poll(),
        )

    def close(self):
        with self._lock:
            process = self._process
            self._terminate_locked(process, reason="close")


@dataclass
class TranscriptionServiceConfig:
    model_size: str = MODEL_SIZE
    whisper_device: str = WHISPER_DEVICE
    sample_rate: int = SAMPLE_RATE
    whisper_language: str | None = WHISPER_LANGUAGE
    whisper_no_speech_threshold: float | None = WHISPER_NO_SPEECH_THRESHOLD
    save_debug_audio: bool = SAVE_DEBUG_AUDIO
    keep_temp_on_success: bool = False
    keep_temp_on_failure: bool = True


class TranscriptionService:
    def __init__(self, base_dir: str | None = None, config: TranscriptionServiceConfig | None = None):
        self.base_dir = os.path.abspath(base_dir) if base_dir is not None else str(REPO_ROOT)
        check_data_layout(self.base_dir)
        self.config = config or TranscriptionServiceConfig()
        self.log_dir = os.path.join(self.base_dir, LOG_DIR)
        self.debug_audio_dir = os.path.join(self.base_dir, DEBUG_AUDIO_DIR)
        self.work_dir = os.path.join(self.base_dir, WEB_WORK_DIR)
        self.event_log_path = os.path.join(self.base_dir, EVENT_LOG_FILE)
        self.vad_worker_module = VAD_WORKER_MODULE
        self.whisper_worker_module = WHISPER_WORKER_MODULE
        self._pipeline_lock = threading.Lock()

        os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        if self.config.save_debug_audio:
            os.makedirs(self.debug_audio_dir, exist_ok=True)

        self.event_logger = EventLogger(self.event_log_path)
        self.vad_client = PersistentVADWorkerClient(
            worker_module=self.vad_worker_module,
            workdir=str(REPO_ROOT),
            log_fn=self.log_event,
        )
        self.whisper_client = PersistentWhisperWorkerClient(
            worker_module=self.whisper_worker_module,
            workdir=str(REPO_ROOT),
            model_size=self.config.model_size,
            device=self.config.whisper_device,
            log_fn=self.log_event,
        )
        self.log_event(
            "web_service_initialized",
            model=self.config.model_size,
            device=self.config.whisper_device,
            work_dir=self.work_dir,
        )

    def log_event(self, event, **fields):
        try:
            fields["thread"] = threading.current_thread().name
            self.event_logger.log(event, **fields)
        except Exception:
            pass

    def status(self) -> dict:
        ready_payload = self.whisper_client.ready_payload()
        return {
            "ok": True,
            "model_ready": self.whisper_client.is_ready(),
            "vad_ready": self.vad_client.is_ready(),
            "model": ready_payload.get("model") or self.config.model_size,
            "device": ready_payload.get("device") or self.config.whisper_device,
            "worker_pid": ready_payload.get("worker_pid"),
            "launcher_pid": ready_payload.get("launcher_pid"),
        }

    def ensure_ready(self) -> dict:
        self.log_event("web_service_ensure_ready_start")
        ready_payload = self.whisper_client.ensure_ready(
            timeout_seconds=WHISPER_WORKER_READY_TIMEOUT_SECONDS,
        )
        self.vad_client.ensure_ready(timeout_seconds=VAD_WORKER_READY_TIMEOUT_SECONDS)
        self.log_event(
            "web_service_ensure_ready_success",
            model=ready_payload.get("model"),
            device=ready_payload.get("device"),
        )
        return self.status()

    def base_transcribe_options(self) -> dict:
        return {
            "condition_on_previous_text": False,
            "language": self.config.whisper_language,
            "logprob_threshold": -1.0,
            "no_speech_threshold": self.config.whisper_no_speech_threshold,
        }

    def build_transcribe_attempts(
        self,
        raw_path: str,
        speech_path: str,
        speech_secs: float,
    ) -> list[dict]:
        base_options = self.base_transcribe_options()
        attempt_candidates = []

        if speech_secs > 0.0 and os.path.exists(speech_path):
            attempt_candidates.append(("speech_only_primary", speech_path, {}))

        attempt_candidates.append(("raw_full_fallback", raw_path, {}))
        attempt_candidates.append(
            (
                "raw_full_permissive",
                raw_path,
                {
                    "compression_ratio_threshold": None,
                    "logprob_threshold": None,
                    "no_speech_threshold": None,
                },
            )
        )

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
            attempts.append(
                {
                    "label": label,
                    "path": path,
                    "options": options,
                }
            )

        return attempts

    def log_transcription(self, text: str):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            filename = os.path.join(self.log_dir, f"{date_str}.txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {text}\n")
            self.log_event("web_transcription_log_saved", chars=len(text), file=filename)
        except Exception as e:
            self.log_event("web_transcription_log_failed", error=e)

    def save_debug_audio_files(self, raw_path: str, speech_path: str, request_id: str):
        if not self.config.save_debug_audio:
            return None

        capture_dir = os.path.join(self.debug_audio_dir, "web", request_id)
        saved_files = 0
        try:
            os.makedirs(capture_dir, exist_ok=False)
            for source_path, output_name in [
                (raw_path, "raw.wav"),
                (speech_path, "speech_only.wav"),
            ]:
                if not os.path.exists(source_path):
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

    def transcribe_wav(self, wav_path: str, source_label: str = "web_upload") -> dict:
        wav_path = os.path.abspath(wav_path)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        request_dir = tempfile.mkdtemp(prefix=f"{request_id}_", dir=self.work_dir)
        raw_path = os.path.join(request_dir, "raw.wav")
        speech_path = os.path.join(request_dir, "speech_only.wav")
        started = time.perf_counter()
        timings = {}
        attempts_payload = []
        text = ""
        success = False
        error_text = None
        speech_secs = 0.0

        with self._pipeline_lock:
            self.log_event(
                "web_transcription_pipeline_start",
                request_id=request_id,
                source=wav_path,
                source_label=source_label,
            )
            try:
                shutil.copy2(wav_path, raw_path)
                self.log_event("web_transcription_source_staged", request_id=request_id, file=raw_path)

                vad_started = time.perf_counter()
                try:
                    speech_secs = self.vad_client.run(
                        in_wav_path=raw_path,
                        out_wav_path=speech_path,
                        sample_rate=self.config.sample_rate,
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

                debug_dir = self.save_debug_audio_files(raw_path, speech_path, request_id)
                attempts = self.build_transcribe_attempts(raw_path, speech_path, speech_secs)
                completed_attempts = 0
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
                        result = self.whisper_client.transcribe(
                            attempt["path"],
                            attempt["options"],
                            timeout_seconds=WHISPER_TRANSCRIBE_TIMEOUT_SECONDS,
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
                    self.log_transcription(text)

                if not success:
                    error_text = "No transcription attempt completed successfully"

                timings["total_seconds"] = time.perf_counter() - started
                result_payload = {
                    "ok": success,
                    "text": text,
                    "speech_secs": speech_secs,
                    "timings": timings,
                    "attempts": attempts_payload,
                    "debug_dir": debug_dir,
                    "request_id": request_id,
                }
                if error_text:
                    result_payload["error"] = error_text

                self.log_event(
                    "web_transcription_pipeline_finish",
                    request_id=request_id,
                    success=success,
                    chars=len(text),
                    total_seconds=f"{timings['total_seconds']:.3f}",
                )
                return result_payload
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
                    "temp_dir": request_dir,
                }
            finally:
                should_keep = (
                    self.config.keep_temp_on_success
                    if success
                    else self.config.keep_temp_on_failure
                )
                if should_keep:
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

    def close(self):
        self.log_event("web_service_close_start")
        self.whisper_client.close()
        self.vad_client.close()
        self.log_event("web_service_close_end")
