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
