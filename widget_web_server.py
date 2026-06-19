from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import threading
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONVERT_TIMEOUT_SECONDS = 180


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper Widget Web</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #181b20;
      --panel-2: #20242b;
      --text: #f2f4f8;
      --muted: #aab2c0;
      --border: #343a45;
      --green: #2f8f5b;
      --green-hover: #27784d;
      --red: #b94444;
      --red-hover: #9f3838;
      --blue: #4176d8;
      --blue-hover: #345fb0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(900px, 100%);
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 4px 0;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      text-align: right;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
    }
    button {
      min-width: 112px;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    #recordButton { background: var(--green); }
    #recordButton:hover:not(:disabled) { background: var(--green-hover); }
    #stopButton { background: var(--red); }
    #stopButton:hover:not(:disabled) { background: var(--red-hover); }
    #copyButton { background: var(--blue); }
    #copyButton:hover:not(:disabled) { background: var(--blue-hover); }
    label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      min-height: 42px;
      padding: 0 4px;
      user-select: none;
    }
    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--blue);
    }
    textarea {
      width: 100%;
      min-height: min(54vh, 520px);
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 14px;
      font: 16px/1.5 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }
    .meta {
      min-height: 22px;
      color: var(--muted);
    }
    .error { color: #ffb4b4; }
    .recording { color: #ffd596; }
    @media (max-width: 620px) {
      main { padding: 14px; }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status { text-align: left; }
      button { flex: 1 1 130px; }
      label { flex-basis: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Whisper Widget Web</h1>
      <div id="status" class="status">Checking widget</div>
    </header>
    <section class="toolbar" aria-label="Recorder controls">
      <button id="recordButton" type="button">Record</button>
      <button id="stopButton" type="button" disabled>Stop</button>
      <button id="copyButton" type="button" disabled>Copy</button>
      <label><input id="autoCopy" type="checkbox"> Auto-copy</label>
    </section>
    <textarea id="transcript" spellcheck="true" placeholder="Transcript"></textarea>
    <div id="meta" class="meta"></div>
  </main>
  <script>
    const recordButton = document.getElementById("recordButton");
    const stopButton = document.getElementById("stopButton");
    const copyButton = document.getElementById("copyButton");
    const autoCopy = document.getElementById("autoCopy");
    const statusEl = document.getElementById("status");
    const transcriptEl = document.getElementById("transcript");
    const metaEl = document.getElementById("meta");

    let mediaRecorder = null;
    let mediaStream = null;
    let audioChunks = [];
    let recordingMimeType = "";
    let recordingStartedAt = 0;

    function setStatus(text, className = "") {
      statusEl.textContent = text;
      statusEl.className = className ? `status ${className}` : "status";
    }

    function preferredMimeType() {
      const choices = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus"
      ];
      for (const choice of choices) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(choice)) {
          return choice;
        }
      }
      return "";
    }

    function extensionForMime(mimeType) {
      if (mimeType.includes("mp4")) return "mp4";
      if (mimeType.includes("ogg")) return "ogg";
      return "webm";
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health", { cache: "no-store" });
        const data = await response.json();
        if (data.busy) {
          setStatus("Widget busy");
        } else if (data.model_ready) {
          setStatus(`Ready: ${data.model} on ${data.device}`);
        } else {
          setStatus("Widget loading model");
        }
      } catch (error) {
        setStatus("Widget unavailable", "error");
      }
    }

    async function startRecording() {
      if (!window.isSecureContext) {
        setStatus("Use HTTPS or localhost for microphone", "error");
        return;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Microphone API unavailable", "error");
        return;
      }

      try {
        audioChunks = [];
        transcriptEl.value = "";
        metaEl.textContent = "";
        copyButton.disabled = true;
        recordButton.disabled = true;
        stopButton.disabled = true;
        setStatus("Opening microphone");

        recordingMimeType = preferredMimeType();
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        const options = recordingMimeType ? { mimeType: recordingMimeType } : {};
        mediaRecorder = new MediaRecorder(mediaStream, options);
        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            audioChunks.push(event.data);
          }
        };
        mediaRecorder.onstart = () => {
          recordingStartedAt = performance.now();
          stopButton.disabled = false;
          setStatus("Recording", "recording");
        };
        mediaRecorder.onstop = uploadRecording;
        mediaRecorder.start();
      } catch (error) {
        setStatus(error.message || "Recording failed", "error");
        recordButton.disabled = false;
        stopButton.disabled = true;
        stopMediaStream();
      }
    }

    function stopRecording() {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        setStatus("Finalizing recording");
        stopButton.disabled = true;
        try {
          mediaRecorder.requestData();
        } catch (error) {
          // Some browsers do not allow requestData in every state.
        }
        mediaRecorder.stop();
        return;
      }

      recordButton.disabled = false;
      stopButton.disabled = true;
      stopMediaStream();
    }

    function stopMediaStream() {
      if (!mediaStream) return;
      for (const track of mediaStream.getTracks()) {
        track.stop();
      }
      mediaStream = null;
    }

    async function uploadRecording() {
      stopMediaStream();
      recordButton.disabled = false;

      if (!audioChunks.length) {
        setStatus("No audio captured", "error");
        return;
      }

      const blob = new Blob(audioChunks, { type: recordingMimeType || "audio/webm" });
      const extension = extensionForMime(blob.type);
      const formData = new FormData();
      formData.append("audio", blob, `recording.${extension}`);

      setStatus("Transcribing");
      try {
        const started = performance.now();
        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || `HTTP ${response.status}`);
        }

        transcriptEl.value = data.text || "";
        copyButton.disabled = !transcriptEl.value;
        const elapsed = ((performance.now() - started) / 1000).toFixed(2);
        const total = data.timings && data.timings.total_seconds
          ? Number(data.timings.total_seconds).toFixed(2)
          : elapsed;
        const recordedSeconds = recordingStartedAt
          ? ((performance.now() - recordingStartedAt) / 1000).toFixed(1)
          : "";
        metaEl.textContent = recordedSeconds
          ? `Recorded ${recordedSeconds}s / server ${total}s`
          : `Server ${total}s`;
        setStatus(data.text ? "Ready" : "No text returned");

        if (autoCopy.checked && data.text) {
          try {
            await navigator.clipboard.writeText(data.text);
            setStatus("Copied");
          } catch (error) {
            setStatus("Ready");
          }
        }
      } catch (error) {
        setStatus(error.message || "Transcription failed", "error");
      }
    }

    async function copyTranscript() {
      const text = transcriptEl.value;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        setStatus("Copied");
      } catch (error) {
        transcriptEl.focus();
        transcriptEl.select();
        setStatus("Select text to copy", "error");
      }
    }

    recordButton.addEventListener("click", startRecording);
    stopButton.addEventListener("click", stopRecording);
    copyButton.addEventListener("click", copyTranscript);
    refreshHealth();
    window.setInterval(refreshHealth, 10000);
  </script>
</body>
</html>
"""


def safe_upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix or len(suffix) > 12:
        return ".webm"
    allowed = set(".abcdefghijklmnopqrstuvwxyz0123456789")
    if any(char not in allowed for char in suffix):
        return ".webm"
    return suffix


def save_upload(upload: UploadFile, output_dir: str) -> tuple[str, int]:
    suffix = safe_upload_suffix(upload.filename)
    output_path = os.path.join(output_dir, f"upload{suffix}")
    total_bytes = 0
    with open(output_path, "wb") as output:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError("Upload is too large")
            output.write(chunk)

    if total_bytes == 0:
        raise ValueError("Upload is empty")

    return output_path, total_bytes


def convert_to_wav(input_path: str, output_path: str) -> float:
    started = time.perf_counter()
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg was not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {CONVERT_TIMEOUT_SECONDS}s") from e

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-1200:]
        raise RuntimeError(f"ffmpeg conversion failed: {stderr or completed.returncode}")

    return time.perf_counter() - started


class WhisperWidgetWebServer:
    def __init__(
        self,
        widget,
        host: str,
        port: int,
        work_dir: str,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        log_fn,
    ):
        self.widget = widget
        self.host = host
        self.port = int(port)
        self.work_dir = work_dir
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.log_fn = log_fn
        self.server = None
        self.thread = None
        os.makedirs(self.work_dir, exist_ok=True)

    def log_event(self, event, **fields):
        try:
            self.log_fn(event, **fields)
        except Exception:
            pass

    def create_app(self):
        app = FastAPI(title="Whisper Widget Web")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse(INDEX_HTML)

        @app.get("/health")
        def health():
            return self.widget.web_status_payload()

        @app.post("/api/transcribe")
        def transcribe(audio: UploadFile = File(...)):
            upload_root = os.path.join(self.work_dir, "uploads")
            os.makedirs(upload_root, exist_ok=True)
            request_dir = tempfile.mkdtemp(prefix="upload_", dir=upload_root)
            upload_path = None
            wav_path = os.path.join(request_dir, "converted.wav")
            success = False

            try:
                upload_path, upload_bytes = save_upload(audio, request_dir)
                self.log_event(
                    "web_widget_upload_saved",
                    file=upload_path,
                    bytes=upload_bytes,
                    content_type=audio.content_type,
                )
                convert_seconds = convert_to_wav(upload_path, wav_path)
                self.log_event(
                    "web_widget_upload_converted",
                    source=upload_path,
                    output=wav_path,
                    convert_seconds=f"{convert_seconds:.3f}",
                )
                result = self.widget.transcribe_web_wav(wav_path, source_label="web_upload")
                timings = dict(result.get("timings") or {})
                timings["convert_seconds"] = convert_seconds
                result["timings"] = timings
                success = bool(result.get("ok"))
                status_code = 200 if success else 409 if result.get("busy") else 500
                return JSONResponse(result, status_code=status_code)
            except Exception as e:
                self.log_event("web_widget_upload_transcription_failed", file=upload_path, error=e)
                return JSONResponse(
                    {
                        "ok": False,
                        "text": "",
                        "error": str(e),
                        "timings": {},
                    },
                    status_code=500,
                )
            finally:
                if success:
                    try:
                        shutil.rmtree(request_dir)
                        self.log_event("web_widget_upload_temp_removed", dir=request_dir)
                    except Exception as e:
                        self.log_event("web_widget_upload_temp_remove_failed", dir=request_dir, error=e)
                else:
                    self.log_event("web_widget_upload_temp_preserved", dir=request_dir)

        return app

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        import uvicorn

        app = self.create_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            ssl_certfile=self.ssl_certfile,
            ssl_keyfile=self.ssl_keyfile,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run,
            daemon=True,
            name="widget_web_server",
        )
        self.thread.start()
        self.log_event("web_widget_server_start_requested", host=self.host, port=self.port)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.log_event("web_widget_server_stop_requested", host=self.host, port=self.port)
