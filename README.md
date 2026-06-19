# Whisper Widget

A standalone, always-on-top desktop widget for instant speech-to-text transcription using OpenAI's Whisper model.

## Features
- **Always-on-top Widget:** Minimalistic UI that floats over other windows.
- **Global Hotkey:** Press **F8** anywhere to start/stop recording.
- **Drag & Drop:** Click and drag the widget to position it anywhere.
- **Auto Copy:** Transcription is automatically copied to your clipboard.
- **Smart Transcription:**
  - **VAD First:** Uses Silero VAD on the raw recording to isolate speech before Whisper.
  - **Simple Audio Path:** Sends VAD speech-only audio directly to Whisper without in-app denoise or normalization.
  - **Progressive Fallbacks:** Falls back to raw full-audio transcription attempts if needed.
- **Transcription Logs:** Saves daily transcription logs to the `transcriptions` folder.
- **Debug Audio Capture:** Saves `raw.wav` and `speech_only.wav` in timestamped folders under `debug_audio/` for comparison and model testing.
- **Runtime Event Logs:** Writes runtime diagnostics to `event_log.txt` in the project root without storing transcription text.
  - The active log rotates at 5 MB and retains `event_log.1.txt` and `event_log.2.txt`.
  - On Windows the active log and rotated archives are marked hidden. View them with `Get-Content -Force .\\event_log.txt`.
- **Web Recorder MVP:** Runs a browser UI that records from the browser device microphone, uploads the completed recording, and returns the transcript from this machine.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You need FFmpeg installed and added to your system PATH)*

2. Run the widget:
   ```bash
   python whisper_widget.py
   ```

### Recommended Windows launch flow (venv isolated)
Use the included launcher to keep this app isolated from system Python package drift:
```bat
launch_whisper_widget.bat
```
On first run it creates `.venv`, installs pinned dependencies from `requirements.txt`, then launches the app.
`requirements.txt` is pinned to the CUDA 12.1 PyTorch wheels used by this project.
Note: the virtual environment folder name is `.venv` (with a leading dot), not `venv`.

If you want to install manually in the same way as the launcher:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip "setuptools<81"
python -m pip install --no-build-isolation -r requirements.txt
```

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
python -m pip install -r requirements-dev.txt
```

Run the deterministic local checks:
```powershell
.\tools\check_all.ps1
```

That runner performs:
- in-memory Python syntax compilation for repo `.py` files that are not git-ignored
- `ruff check` on the same repo `.py` files with a minimal low-noise ruleset (`E9` and `F`)
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

## Configuration
You can edit the `Configuration` section at the top of `whisper_widget.py` to change:
- `MODEL_SIZE` (default: `turbo`) - Common alternatives to test are `large-v3` and `medium`.
- `HOTKEY` (default: `f8`) - Change global shortcut.
- `CUDA_VISIBLE_DEVICES` - Adjust GPU targeting if you have multiple GPUs.
- `SAVE_DEBUG_AUDIO` - Keep or disable timestamped debug audio capture under `debug_audio/`.
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
