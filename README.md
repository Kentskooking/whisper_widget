# Whisper Widget

A standalone, always-on-top desktop widget for instant speech-to-text transcription using OpenAI's Whisper model.

## Features
- **Always-on-top Widget:** Minimalistic UI that floats over other windows.
- **Global Hotkey:** Press **F8** anywhere to start/stop recording.
- **Drag & Drop:** Click and drag the widget to position it anywhere.
- **Auto Copy:** Transcription is automatically copied to your clipboard.
- **Smart Transcription:**
  - **VAD First:** Uses Silero VAD on the raw recording to isolate speech before heavier processing.
  - **Speech Cleanup:** Applies chunked `noisereduce` denoising and conservative peak normalization to the speech-only audio before Whisper.
  - **Progressive Fallbacks:** Falls back across processed speech-only audio and raw full-audio transcription attempts if needed.
- **Transcription Logs:** Saves daily transcription logs to the `transcriptions` folder.
- **Debug Audio Capture:** Saves `raw.wav`, `speech_only.wav`, `denoised.wav`, and `normalized.wav` in timestamped folders under `debug_audio/` for comparison and model testing.
- **Runtime Event Logs:** Writes runtime diagnostics to `event_log.txt` in the project root without storing transcription text.
  - The active log rotates at 5 MB and retains `event_log.1.txt` and `event_log.2.txt`.
  - On Windows the active log and rotated archives are marked hidden. View them with `Get-Content -Force .\\event_log.txt`.

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
- `MODEL_SIZE` (default: `large-v3`) - Common alternatives to test are `turbo` and `medium`.
- `HOTKEY` (default: `f8`) - Change global shortcut.
- `CUDA_VISIBLE_DEVICES` - Adjust GPU targeting if you have multiple GPUs.
- `NOISE_REDUCTION_ENABLED`, `NOISE_REDUCTION_PROP_DECREASE`, `NOISE_REDUCTION_CHUNK_SECONDS`, `NOISE_REDUCTION_PADDING_SECONDS`, `NOISE_REDUCTION_N_FFT` - Tune denoising behavior and long-recording stability.
- `NORMALIZE_AUDIO_ENABLED`, `NORMALIZE_TARGET_PEAK_DBFS`, `NORMALIZE_MAX_GAIN_DB` - Tune conservative peak normalization after denoise.
- `SAVE_DEBUG_AUDIO` - Keep or disable timestamped debug audio capture under `debug_audio/`.
- `EVENT_LOG_MAX_BYTES`, `EVENT_LOG_BACKUP_COUNT` - Control event log rotation size and retained archives.
