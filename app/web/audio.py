"""Audio upload validation and format conversion shared by the web sessions."""
from pathlib import Path
import os
import subprocess
import time
import wave

from fastapi import UploadFile

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONVERT_TIMEOUT_SECONDS = 180


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


def validate_chunk_wav(path: str, sample_rate: int, channels: int, sample_width: int, frames: int):
    with wave.open(path, "rb") as reader:
        actual_channels = reader.getnchannels()
        actual_sample_width = reader.getsampwidth()
        actual_sample_rate = reader.getframerate()
        actual_frames = reader.getnframes()

    if actual_channels != channels:
        raise ValueError(f"Chunk channel mismatch: expected {channels}, got {actual_channels}")
    if actual_sample_width != sample_width:
        raise ValueError(
            f"Chunk sample width mismatch: expected {sample_width}, got {actual_sample_width}"
        )
    if actual_sample_rate != sample_rate:
        raise ValueError(f"Chunk sample rate mismatch: expected {sample_rate}, got {actual_sample_rate}")
    if actual_frames != frames:
        raise ValueError(f"Chunk frame mismatch: expected {frames}, got {actual_frames}")


def save_chunk_upload(upload: UploadFile, output_path: str) -> int:
    total_bytes = 0
    with open(output_path, "wb") as output:
        while True:
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError("Chunk upload is too large")
            output.write(chunk)

    if total_bytes == 0:
        raise ValueError("Chunk upload is empty")

    return total_bytes
