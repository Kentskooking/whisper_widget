# Whisper Widget

A standalone, always-on-top desktop widget for instant speech-to-text transcription using OpenAI's Whisper model.

## Features
- **Always-on-top Widget:** Minimalistic UI that floats over other windows.
- **Global Hotkey:** Press **F8** anywhere to start/stop recording.
- **Drag & Drop:** Click and drag the widget to position it anywhere.
- **Auto Copy:** Transcription is automatically copied to your clipboard.
- **Smart Transcription:**
  - **VAD (Voice Activity Detection):** Uses Silero VAD to strip silence before processing, improving accuracy and speed.
  - **Retry Logic:** Automatically retries transcription if it fails.
- **Logs:** Saves daily transcription logs to the `transcriptions` folder.
  - Runtime event diagnostics are also written to `event_log.txt` in the project root (no transcription text is stored there).
  - On Windows the event log file is marked hidden. View it with `Get-Content -Force .\\event_log.txt`.

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

## Controls
- **Click & Drag:** Move the widget.
- **F8:** Toggle recording (Global Hotkey).
- **Spacebar:** Toggle recording (When widget is focused).
- **M:** Toggle Mute (Disable sound feedback).

## Configuration
You can edit the `Configuration` section at the top of `whisper_widget.py` to change:
- `MODEL_SIZE` (default: `large-v3`) - Change to `base`, `small`, or `medium` for faster performance on lower-end hardware.
- `HOTKEY` (default: `f8`) - Change global shortcut.
- `CUDA_VISIBLE_DEVICES` - Adjust GPU targeting if you have multiple GPUs.
