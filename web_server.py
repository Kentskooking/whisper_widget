from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from transcription_service import TranscriptionService


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONVERT_TIMEOUT_SECONDS = 180
PRELOAD_ON_START = os.environ.get("WHISPER_WEB_PRELOAD", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper Web</title>
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
      --amber: #c28a2c;
    }
    * {
      box-sizing: border-box;
    }
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
    #recordButton {
      background: var(--green);
    }
    #recordButton:hover:not(:disabled) {
      background: var(--green-hover);
    }
    #stopButton {
      background: var(--red);
    }
    #stopButton:hover:not(:disabled) {
      background: var(--red-hover);
    }
    #copyButton {
      background: var(--blue);
    }
    #copyButton:hover:not(:disabled) {
      background: var(--blue-hover);
    }
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
    .error {
      color: #ffb4b4;
    }
    .recording {
      color: #ffd596;
    }
    @media (max-width: 620px) {
      main {
        padding: 14px;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status {
        text-align: left;
      }
      button {
        flex: 1 1 130px;
      }
      label {
        flex-basis: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Whisper Web</h1>
      <div id="status" class="status">Checking server</div>
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
        if (data.model_ready) {
          setStatus(`Ready: ${data.model} on ${data.device}`);
        } else {
          setStatus("Model cold");
        }
      } catch (error) {
        setStatus("Server unavailable", "error");
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
        metaEl.textContent = `Server ${total}s`;
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


service = TranscriptionService()


def prewarm_service():
    try:
        service.ensure_ready()
    except Exception as e:
        service.log_event("web_service_prewarm_failed", error=e)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    del app_instance
    if PRELOAD_ON_START:
        thread = threading.Thread(target=prewarm_service, daemon=True, name="web_service_prewarm")
        thread.start()
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="Whisper Web", lifespan=lifespan)


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
                raise HTTPException(status_code=413, detail="Upload is too large")
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Upload is empty")

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


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health():
    return service.status()


@app.post("/api/transcribe")
def transcribe(audio: UploadFile = File(...)):
    upload_root = os.path.join(service.work_dir, "uploads")
    os.makedirs(upload_root, exist_ok=True)
    request_dir = tempfile.mkdtemp(prefix="upload_", dir=upload_root)
    upload_path = None
    wav_path = os.path.join(request_dir, "converted.wav")
    success = False

    try:
        upload_path, upload_bytes = save_upload(audio, request_dir)
        service.log_event(
            "web_upload_saved",
            file=upload_path,
            bytes=upload_bytes,
            content_type=audio.content_type,
        )
        convert_seconds = convert_to_wav(upload_path, wav_path)
        service.log_event(
            "web_upload_converted",
            source=upload_path,
            output=wav_path,
            convert_seconds=f"{convert_seconds:.3f}",
        )
        result = service.transcribe_wav(wav_path, source_label="web_upload")
        timings = dict(result.get("timings") or {})
        timings["convert_seconds"] = convert_seconds
        result["timings"] = timings
        success = bool(result.get("ok"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        service.log_event("web_upload_transcription_failed", file=upload_path, error=e)
        return {
            "ok": False,
            "text": "",
            "error": str(e),
            "timings": {},
        }
    finally:
        if success:
            try:
                shutil.rmtree(request_dir)
                service.log_event("web_upload_temp_removed", dir=request_dir)
            except Exception as e:
                service.log_event("web_upload_temp_remove_failed", dir=request_dir, error=e)
        else:
            service.log_event("web_upload_temp_preserved", dir=request_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Whisper web server.")
    parser.add_argument("--host", default=os.environ.get("WHISPER_WEB_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WHISPER_WEB_PORT", DEFAULT_PORT)),
    )
    parser.add_argument("--ssl-certfile", default=os.environ.get("WHISPER_WEB_SSL_CERTFILE"))
    parser.add_argument("--ssl-keyfile", default=os.environ.get("WHISPER_WEB_SSL_KEYFILE"))
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Start the Whisper and VAD workers during server startup.",
    )
    return parser.parse_args()


def main() -> int:
    global PRELOAD_ON_START
    args = parse_args()
    PRELOAD_ON_START = PRELOAD_ON_START or args.preload
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
