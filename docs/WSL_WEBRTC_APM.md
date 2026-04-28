# WSL WebRTC APM Helper

The widget's default noise-reduction backend is `webrtc_apm_wsl`. It runs a
small native C++ WebRTC Audio Processing Module helper inside WSL, then returns
the denoised WAV to the Windows widget process.

This helper is not a long-running server. The Python wrapper starts `wsl.exe`,
runs a bash script, builds the native helper if needed, processes one WAV, and
exits.

## Files

- `whisper_widget.py` - uses `NOISE_REDUCTION_BACKEND = "webrtc_apm_wsl"` by
  default and calls the wrapper during recording processing.
- `webrtc_apm_wav_wrapper.py` - Windows/Python bridge. Converts Windows paths to
  `/mnt/<drive>/...` paths and invokes WSL.
- `tools/build_webrtc_apm_wsl.sh` - compiles the native helper into
  `sidecache/webrtc_apm_build/webrtc_apm_wav`.
- `tools/run_webrtc_apm_wsl.sh` - build-if-needed runner used by the wrapper.
- `tools/webrtc_apm_wav.cpp` - native C++ WAV processor using WebRTC APM noise
  suppression and libsndfile.

## WSL Prerequisites

The default distro name is `Ubuntu-22.04`. Install WSL and Ubuntu from Windows
first if they are not already available:

```powershell
wsl --list --verbose
```

Inside the WSL distro, install the native build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y libwebrtc-audio-processing-dev libsndfile1-dev g++ pkg-config cmake
```

The build script also checks these dependencies and prints the same install
command if any are missing.

## First Build

From the repo root on Windows:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/EDEN/Python\ Gui/Whisper_web_server/whisper_widget && ./tools/build_webrtc_apm_wsl.sh"
```

The script prints the compiled helper path. By default the binary lands under:

```text
sidecache/webrtc_apm_build/webrtc_apm_wav
```

`sidecache/` is intentionally ignored by git because it is a local build cache.

If the repo lives somewhere else, adjust the `/mnt/...` path. You can also run
the script directly from inside WSL after changing into the repo directory.

## Running The Helper Directly

The wrapper is the same entry point used by the widget and bakeoff harness:

```powershell
python .\webrtc_apm_wav_wrapper.py --input .\debug_audio\some_run\speech_only.wav --output .\debug_audio\some_run\webrtc.wav --preset light
```

Use `--preset heavy` for stronger WebRTC noise suppression:

```powershell
python .\webrtc_apm_wav_wrapper.py --input input.wav --output output.wav --preset heavy
```

If your WSL distro name is different:

```powershell
python .\webrtc_apm_wav_wrapper.py --input input.wav --output output.wav --preset light --distro Ubuntu-24.04
```

Successful runs print JSON to stdout with processor details such as preset,
input sample rate, output sample rate, channels, frame count, and duration. The
native helper writes mono 48 kHz WAV output; the widget's later normalization
stage resamples back to the configured widget sample rate.

## Widget Configuration

The relevant constants live near the top of `whisper_widget.py`:

```python
NOISE_REDUCTION_ENABLED = True
NOISE_REDUCTION_BACKEND = "webrtc_apm_wsl"
NOISE_REDUCTION_WEBRTC_PRESET = "light"
NOISE_REDUCTION_WEBRTC_DISTRO = "Ubuntu-22.04"
```

Use `"light"` for the normal recording path. `"heavy"` is available for
comparison, but it can remove more speech detail.

If the WSL helper is unavailable and you need to run without it, change
`NOISE_REDUCTION_BACKEND` to `"noisereduce_subprocess"` or set
`NOISE_REDUCTION_ENABLED = False`.

## Bakeoff Usage

`audio_bakeoff.py` includes WebRTC variants by default:

```powershell
python .\audio_bakeoff.py --recent-backups 1 --variants webrtc_apm_light_wsl,webrtc_apm_heavy_wsl
```

Outputs are written to `bakeoff_outputs/`, which is ignored by git. Use these
runs for listening comparisons before changing the widget defaults.

## Troubleshooting

`wsl.exe` cannot find the distro:

```powershell
wsl --list --verbose
```

Set `NOISE_REDUCTION_WEBRTC_DISTRO` in `whisper_widget.py`, or pass
`--distro <name>` to `webrtc_apm_wav_wrapper.py`.

Missing `pkg-config`, `webrtc-audio-processing`, or `sndfile`:

```bash
sudo apt-get update
sudo apt-get install -y libwebrtc-audio-processing-dev libsndfile1-dev g++ pkg-config cmake
```

Permission denied running shell scripts inside WSL:

```bash
chmod +x tools/*.sh
```

Stale or broken native binary:

```bash
rm -rf sidecache/webrtc_apm_build
./tools/build_webrtc_apm_wsl.sh
```

Paths with spaces are supported by the wrapper, but when running commands
manually in WSL, quote or escape repo paths such as `Python\ Gui`.

## Git Hygiene

Keep source files tracked:

- `webrtc_apm_wav_wrapper.py`
- `tools/build_webrtc_apm_wsl.sh`
- `tools/run_webrtc_apm_wsl.sh`
- `tools/webrtc_apm_wav.cpp`
- `docs/WSL_WEBRTC_APM.md`

Keep local artifacts ignored:

- `sidecache/`
- `debug_audio/`
- `raw_audio_backups/`
- `bakeoff_outputs/`
- `*.wav` and other local audio/media files
- `event_log*.txt`

Before pushing, check that only source and docs changes are present:

```powershell
git status --short
git diff --check
```
