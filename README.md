# Whisper Widget

A standalone, always-on-top desktop widget for instant speech-to-text transcription using OpenAI's Whisper model.

## Features
- **Always-on-top Widget:** Minimalistic UI that floats over other windows.
- **Global Hotkey:** Press **F8** anywhere to start/stop recording.
- **Drag & Drop:** Click and drag the widget to position it anywhere.
- **Auto Copy:** Transcription is automatically copied to your clipboard.
- **Copy Again:** The bottom-left copy icon copies the latest completed transcript,
  including after restarting the widget. The footer still displays **Press F8**.
- **Smart Transcription:**
  - **VAD First:** Uses Silero VAD on the raw recording to isolate speech before Whisper.
  - **Simple Audio Path:** Sends VAD speech-only audio directly to Whisper without in-app denoise or normalization.
  - **Progressive Fallbacks:** Falls back to raw full-audio transcription attempts if needed.
- **Transcription Logs:** Saves daily transcription logs to the `data/transcriptions/` folder.
- **Debug Audio Capture:** Saves `raw.wav` and `speech_only.wav` in timestamped folders under `data/debug_audio/` for comparison and model testing.
- **Runtime Event Logs:** Writes runtime diagnostics to `runtime/logs/event_log.txt` without storing transcription text.
  - The active log rotates at 5 MB and retains `event_log.1.txt` and `event_log.2.txt`.
  - On Windows the active log and rotated archives are marked hidden. View them with `Get-Content -Force .\runtime\logs\event_log.txt`.
- **Web Recorder MVP:** Runs a browser UI that records from the browser device microphone, uploads the completed recording, and returns the transcript from this machine.

## Setup
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and make
FFmpeg available on your system PATH. This project uses 64-bit Python 3.11.
The `.python-version` file records that version for uv; other projects can use
different Python versions.

Initialize the environment explicitly from the repository root:
```bat
uv venv --python 3.11 --seed .venv
.venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<81"
.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
```
`requirements.txt` is pinned to this project's CUDA 12.1 PyTorch wheels.
Create a fresh `.venv` after moving the project to another machine.

### Windows launch flow
```bat
launch_whisper_widget.bat
```
Both launchers use only this repository's `.venv\Scripts\python.exe`.
They check Python 3.11, runtime imports, package consistency, NVIDIA CUDA,
the SoundFile audio backend, and FFmpeg before starting. A failed check prints
its original diagnostic and exits with a nonzero status. On a normal launch,
the console waits for a key after failure so the diagnostic remains visible.
Setup and dependency repair are explicit commands, never automatic startup actions.

Run the checks without starting the app or pausing on failure:
```bat
launch_whisper_widget.bat --check
launch_whisper_web_server.bat --check
```

The widget launcher starts `app.supervisor`, which still restarts crashed
or unresponsive widget processes. Each restart is printed in the console and
recorded in `runtime/logs/supervisor_log.txt`; repeated crashes eventually
stop the supervisor with a nonzero exit status.
The standalone web launcher rejects an incomplete HTTPS certificate/key pair.

### Embedded web recorder
The desktop widget now starts the web recorder in the same process by default.
Use the normal widget launcher:
```bat
launch_whisper_widget.bat
```

By default the embedded web server binds to `127.0.0.1:8765`:
```text
http://127.0.0.1:8765
```

The browser records from the device where the page is open, then sends the completed audio file to the already-running widget. Web requests use the widget's existing VAD and Whisper worker clients, so the model is not loaded a second time.

Agents and applications can call the widget directly without using the browser recorder. See [Whisper Widget HTTP API](docs/API.md) for the tailnet URL, endpoint contract, error handling, and client examples.

The agent API also provides `POST /api/vad` for the widget's VAD decision and timestamped speech segments without invoking Whisper transcription.

Override the bind address or port with:
```bat
set WHISPER_WEB_HOST=127.0.0.1
set WHISPER_WEB_PORT=8765
launch_whisper_widget.bat
```

Disable the embedded web server with:
```bat
set WHISPER_WIDGET_WEB_ENABLED=0
launch_whisper_widget.bat
```

`launch_whisper_web_server.bat` still exists as a standalone diagnostic server, but it owns a separate Whisper worker and should not be used for the normal widget-backed web flow.

For another Tailscale device, expose the local server through Tailscale Serve HTTPS so browser microphone and clipboard APIs run in a secure context:
```bat
tailscale serve --bg 8765
tailscale serve status
```

Then open the HTTPS Tailscale Serve URL shown by `tailscale serve status`.

If Tailscale Serve is not available, run the web server directly with a Tailscale certificate:
```bat
mkdir sidecache\certs
tailscale cert --cert-file sidecache\certs\<machine-fqdn>.crt --key-file sidecache\certs\<machine-fqdn>.key <machine-fqdn>
set WHISPER_WEB_HOST=0.0.0.0
set WHISPER_WEB_PORT=8765
set WHISPER_WEB_SSL_CERTFILE=sidecache\certs\<machine-fqdn>.crt
set WHISPER_WEB_SSL_KEYFILE=sidecache\certs\<machine-fqdn>.key
launch_whisper_widget.bat
```

Then open `https://<machine-fqdn>:8765` from the other device.

For direct tailnet-IP connectivity testing without Tailscale Serve, bind the server to all interfaces:
```bat
set WHISPER_WEB_HOST=0.0.0.0
set WHISPER_WEB_PORT=8765
launch_whisper_widget.bat
```

Then open `http://<this-machine-tailscale-ip>:8765` from the other device. This proves network reachability, but browser microphone access may still be blocked because raw tailnet IP HTTP is not a secure context. If it times out while localhost works, check that Windows Firewall allows inbound TCP traffic on the selected port.

### Dev checks
Install the dev-only verifier tooling into the same `.venv`:
```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the deterministic local checks:
```powershell
.\tools\check_all.ps1
```

That runner performs:
- in-memory Python syntax compilation for repo `.py` files that are not git-ignored
- `ruff check` on the same repo `.py` files with a minimal low-noise ruleset (`E9` and `F`)
- regression checks in `tests/` for process entry points and path handling
- `git diff --check` for whitespace and patch hygiene

### Microphone Cleanup
The app does not run its own denoise or normalization stages. Configure any
microphone cleanup upstream at the Windows input-device level, such as selecting
a virtual microphone that already applies the desired processing.

## Controls
- **Click & Drag:** Move the widget.
- **F8:** Toggle recording (Global Hotkey).
- **Spacebar:** Toggle recording (When widget is focused).
- **M:** Toggle Mute (Disable sound feedback).
- **Copy icon (bottom left):** Copy the last completed transcript again without
  starting/stopping recording. Works for desktop and embedded web transcripts.
  The latest text is saved locally in `runtime/state/last_transcript.txt` before
  automatic copying, so a clipboard failure does not lose it.

## Configuration
You can edit the `Configuration` section at the top of `app/desktop.py` to change:
- `MODEL_SIZE` (default: `large-v3`) - The default is Whisper's multilingual large model.
- `WHISPER_LANGUAGE` (default: `None`) - Auto-detect the spoken language; set a language code such as `en` to force one language.
- `HOTKEY` (default: `f8`) - Change global shortcut.
- `CUDA_VISIBLE_DEVICES` - Adjust GPU targeting if you have multiple GPUs.
- `SAVE_DEBUG_AUDIO` - Keep or disable timestamped debug audio capture under `data/debug_audio/`.
- `EVENT_LOG_MAX_BYTES`, `EVENT_LOG_BACKUP_COUNT` - Control event log rotation size and retained archives.

Valid `MODEL_SIZE` values from the installed `openai-whisper` package:

```text
tiny.en
tiny
base.en
base
small.en
small
medium.en
medium
large-v1
large-v2
large-v3
large
large-v3-turbo
turbo
```

## Source layout

- `app/desktop.py`: desktop widget.
- `app/supervisor.py`: widget process monitoring.
- `app/transcription_service.py`: standalone transcription service.
- `app/workers/`: clipboard, VAD, and Whisper subprocesses.
- `app/web/`: embedded recorder and standalone diagnostic server.
- `app/paths.py`: repository location, generated-data paths, and process module names.
- `tools/`: repository verification and explicit data migration commands.
- `docs/`: tracked project documentation, including the HTTP API.
- `tests/`: source layout and process entry-point checks.

Run application processes as modules from the repository root, using the
project environment; for example, `.venv\Scripts\python.exe -m app.supervisor`.
The Windows launchers set the working directory explicitly.

## Generated files

Saved transcripts and audio live under `data/`; logs, recovery state, and
processing files live under `runtime/`. Generated contents are ignored by Git;
named text placeholders keep the `data/` subfolders in fresh checkouts.
Recovery state and work files can contain unfinished recordings, so preserve them
together. See [storage locations and migration](docs/storage.md) for the full
layout, retention behavior, and the explicit migration command for older checkouts.
