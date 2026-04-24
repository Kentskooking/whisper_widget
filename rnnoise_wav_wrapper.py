import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 48000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a WAV to 48 kHz mono PCM, run RNNoise in WSL, and write a WAV result."
    )
    parser.add_argument("--input", required=True, help="Input WAV path.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--distro", default="Ubuntu-22.04", help="WSL distro name.")
    return parser.parse_args()


def load_mono_wav(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return np.ascontiguousarray(resample_poly(audio, up, down).astype(np.float32))


def write_raw_pcm16le(path: Path, audio: np.ndarray):
    clipped = np.clip(audio, -1.0, 0.9999695)
    pcm16 = (clipped * 32768.0).astype("<i2")
    pcm16.tofile(path)


def read_raw_pcm16le(path: Path):
    pcm16 = np.fromfile(path, dtype="<i2")
    return np.ascontiguousarray((pcm16.astype(np.float32) / 32768.0), dtype=np.float32)


def to_wsl_path(path: Path):
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("/mnt/"):
        return text
    windows_path = PureWindowsPath(resolved)
    drive = windows_path.drive.rstrip(":").lower()
    tail = "/".join(windows_path.parts[1:])
    return f"/mnt/{drive}/{tail}"


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    repo_root = Path(__file__).resolve().parent
    runner_script = repo_root / "tools" / "run_rnnoise_demo_wsl.sh"

    audio, sample_rate = load_mono_wav(input_path)
    prepared_audio = resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rnnoise_",
        dir=str(output_path.parent),
        ignore_cleanup_errors=True,
    ) as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_raw = temp_dir_path / "input.raw"
        output_raw = temp_dir_path / "output.raw"
        write_raw_pcm16le(input_raw, prepared_audio)

        if os.name == "nt":
            command = [
                "wsl.exe",
                "-d",
                args.distro,
                "--",
                "/bin/bash",
                to_wsl_path(runner_script),
                to_wsl_path(input_raw),
                to_wsl_path(output_raw),
            ]
        else:
            command = [
                "/bin/bash",
                str(runner_script),
                str(input_raw.resolve()),
                str(output_raw.resolve()),
            ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"RNNoise WSL runner failed with exit code {exc.returncode}: "
                f"stdout={exc.stdout.strip()} stderr={exc.stderr.strip()}"
            ) from exc
        denoised_audio = read_raw_pcm16le(output_raw)
        sf.write(output_path, denoised_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")

    print(
        json.dumps(
            {
                "frames": int(denoised_audio.shape[0]),
                "sample_rate": TARGET_SAMPLE_RATE,
                "seconds": round(float(denoised_audio.shape[0]) / float(TARGET_SAMPLE_RATE), 3),
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    )


if __name__ == "__main__":
    main()
