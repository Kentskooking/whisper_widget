# Bark Widget

A compact Windows speech-to-text widget using
[ConwayResearch/Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro).
This branch retains the Whisper widget's interface and HTTP API while replacing
recognition with Bark Pro, an **English ASR model** based on Parakeet TDT.
It is unrelated to Suno's Bark speech generator.

## Features

- Always-on-top, draggable widget with the existing recording/status controls.
- **F8** globally or **Space** while focused to start/stop recording.
- **M** to mute sound cues; automatic clipboard delivery and a copy-again button.
- Latest completed transcript survives restarts and is shared with web clients.
- Buffered transcription while recording, with original pauses and quiet speech preserved.
- Recording recovery, failed-audio retention, debug audio, daily transcripts,
  rotating diagnostic logs, and the existing process supervisor.
- Embedded browser recorder, file uploads, chunked sessions, and separate VAD analysis.
- Configurable PyTorch device; no GPU model, UUID, or workstation path is required.

## Requirements

- Windows with 64-bit **Python 3.11 or later** and Tk support.
  Choose a Python release supported by the wheels you install.
- A compatible PyTorch installation. A CUDA-capable build/device is selected
  automatically when available; otherwise the backend uses CPU float32.
  CPU transcription can be substantially slower.
- [FFmpeg](https://ffmpeg.org/download.html) on PATH for uploaded media conversion.
- A working microphone for desktop recording.
- Disk space for the model (about 1.3 GB of weights), dependencies, and recordings.

This is a Windows desktop application. Hardware support follows the selected
PyTorch build; there is no claim that every GPU/backend works. The device options
implemented here are `auto`, `cpu`, `cuda`, and `cuda:N`.

## Install

Clone this branch and open PowerShell in the repository root. Create a local
environment using the Python installation you chose; replace the example path:

```powershell
& "C:\Path\To\Python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Choose the appropriate PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/).
Run its installation command through `.\.venv\Scripts\python.exe -m pip`
instead of a global `pip`. Then install the application dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.download_models
```

The download command obtains Bark Pro from Hugging Face into the normal user
cache. It does not start inference, recording, or a server. Silero VAD's weights
are included with its Python package. Runtime workers load local files only.
For an existing/offline model, set `BARK_MODEL_DIR` instead of downloading again.

Dependencies use compatibility ranges, with no CUDA wheel index or Python patch
version forced by this repository. Transformers is limited to the **5.19 release
family** because the buffered decoder uses its Parakeet cache interfaces.
Broadening that range requires checking those interfaces and the decoder tests.
There is no workstation-specific lock file.

## Run

```powershell
.\launch_whisper_widget.bat
```

The existing launcher name is retained. It validates the environment, then starts
the supervisor and the same compact widget, now labeled Bark. Wait for READY,
record with F8, and paste after COPIED. If clipboard delivery fails, the completed
text remains saved and available through the copy button.

Check startup requirements without opening a window or loading model weights:

```powershell
.\launch_whisper_widget.bat --check
```

Launchers use this checkout's `.venv\Scripts\python.exe`. They never install
packages, download models, or repair environments automatically. Close another
widget using F8 before starting; a hotkey conflict is reported rather than sharing
a recording shortcut between instances. The launcher also prevents a second Bark
widget instance.

## Device and model configuration

Set variables in the shell that launches the widget, or in your own wrapper.
The application does not require or read a `.env` file.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BARK_DEVICE` | `auto` | First visible CUDA device, otherwise CPU; or explicitly `cpu` / `cuda:N`. |
| `BARK_DTYPE` | `auto` | Native CUDA BF16 when supported, otherwise float32. Explicit CUDA float16/BF16 or float32 is also available. |
| `BARK_MODEL_DIR` | Unset | Local directory containing the Bark model, processor, tokenizer, and generation files. |
| `BARK_MODEL_REVISION` | `4b5c51e04aa8d16dda9ecdff71bc29f7360149e4` | Snapshot of the same Bark checkpoint used by download/cache lookup. |
| `HF_HOME` | Hugging Face default | Optional model cache location. |
| `CUDA_VISIBLE_DEVICES` | Inherited | Optional device visibility control; device indices follow this setting. |
| `BARK_CPU_THREADS` | `2` | PyTorch CPU thread count in the recognition worker. |
| `BARK_READY_TIMEOUT` | `120` | Worker startup timeout in seconds. |
| `BARK_TRANSCRIBE_TIMEOUT` | `180` | Per-request inference timeout in seconds. |
| `BARK_FINALIZE_TIMEOUT` | `180` | Desktop queue completion timeout in seconds. |

For example:

```powershell
$env:BARK_DEVICE = "cuda:1"
$env:BARK_MODEL_DIR = "D:\Models\Underdog-Bark-3.0-Pro"
.\launch_whisper_widget.bat
```

Use `BARK_DEVICE=cpu` to select CPU explicitly. Increase transcription and
finalization timeouts together if slower hardware needs more time. Selecting a
missing device fails with a diagnostic. The application does not choose a GPU by
name, reserve hardware, or change other applications' GPU allocations.

## Audio behavior

The microphone records mono 16 kHz PCM16. Silero runs on CPU to find pauses; it
does **not** remove silence before transcription. Capture seeks a confirmed pause
near 30 seconds, with a bounded fallback at 40 seconds. Forced cuts retain the
original 1.5-second overlap for recovery.

Bark decodes at most 10 seconds of new audio per internal window, with up to
10 seconds of left context and 5 seconds of right context. Decoder state carries
through the recording, and overlapping context is never emitted twice.
Stopping flushes available audio immediately without waiting for future silence
or context. Finished text comes from the accumulated tokens, preserving spoken
repetitions. Long-file requests and recovery use an independent bounded decoder.

Bark Pro is English-only. Capitalization, punctuation, and wording can differ
from Whisper or whole-recording inference. No punctuation repair, repetition
filter, denoise, or gain processing is added. Configure microphone cleanup
upstream through your Windows input device if desired.

## Browser recorder and API

The desktop widget starts the embedded server at:

```text
http://127.0.0.1:8765
```

It shares the desktop workers and completed transcript history. Existing routes
and `WHISPER_*` web configuration names remain compatible:

- `WHISPER_WIDGET_WEB_ENABLED=0` disables the embedded server.
- `WHISPER_WEB_HOST` and `WHISPER_WEB_PORT` change the bind address and port.
- `WHISPER_WEB_SSL_CERTFILE` and `WHISPER_WEB_SSL_KEYFILE` must be set together.

See [the API contract](docs/API.md) for uploads, chunk sessions, latest transcript,
VAD analysis, and busy/error handling. A remote browser needs HTTPS for microphone
access; use your own private HTTPS proxy or Tailscale Serve configuration.
The API has no built-in authentication: keep the default loopback binding or
provide your own access boundary before enabling remote access.

`launch_whisper_web_server.bat` retains the separate diagnostic browser/file
upload server. It loads its own worker and offers `/`, `/health`, and
`/api/transcribe`; the full chunked/history/VAD API belongs to the desktop's
embedded server. Do not run both on the same port.

## Storage and recovery

Audio and transcript output remains under `data/`; logs, pending recordings,
and recovery state remain under `runtime/`. Preserve those directories when
changing versions. See [storage.md](docs/storage.md) for paths and migration.

Remote session status is persisted under `runtime/work/web/session_state/`.
Interrupted sessions report failure after restart and retain received audio.
Decoder caches are never guessed or restored after worker loss: recovery uses
the original recording. Cancelled sessions keep the pipeline occupied until
in-flight work stops, then release decoder state.

## Development

Install the optional developer requirements and run the repository verifier:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\tools\check_all.ps1
```

It checks in-memory syntax compilation, Ruff, regression tests, and Git patch
hygiene. Tests include sample continuity, buffered state/retries, recovery,
clipboard history, and process entry points. Model-backed/microphone acceptance
is separate from those regression checks.

See [implementation notes](docs/bark-port.md) for the source port and validation
scope. Model weights and generated recordings are excluded from Git. The model
has its own license and attribution requirements; retain its downloaded
`NOTICE` and consult the linked model card when redistributing model assets.
