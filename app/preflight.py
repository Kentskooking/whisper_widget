"""Startup checks without model loading, downloads, or microphone access."""
import argparse
import importlib
import importlib.metadata
import os
import shutil
import struct
import sys


def check(desktop=True):
    if sys.version_info < (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("Use 64-bit Python 3.11 or later")
    if desktop and os.name != "nt":
        raise RuntimeError("This desktop widget targets Windows")
    from packaging.version import Version

    version = Version(importlib.metadata.version("transformers"))
    if not Version("5.19") <= version < Version("5.20"):
        raise RuntimeError("Buffered Bark requires transformers>=5.19,<5.20; see requirements.txt")
    for name in ("torch", "numpy", "soundfile", "soxr", "librosa", "silero_vad",
                 "fastapi", "uvicorn", "multipart"):
        importlib.import_module(name)
    # These imports are part of the decoder's compatibility contract.
    from transformers import AutoProcessor, ParakeetForTDT  # noqa: F401
    from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache  # noqa: F401
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderModelOutput  # noqa: F401
    import torch
    from app.bark_backend import resolve_device, resolve_model_path
    from app.paths import check_data_layout

    device = resolve_device(torch, os.environ.get("BARK_DEVICE", "auto"))
    model = resolve_model_path()
    if desktop:
        for name in ("pyaudio", "customtkinter", "keyboard"):
            importlib.import_module(name)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required on PATH for uploaded media conversion")
    if bool(os.environ.get("WHISPER_WEB_SSL_CERTFILE")) != bool(os.environ.get("WHISPER_WEB_SSL_KEYFILE")):
        raise RuntimeError("Set both WHISPER_WEB_SSL_CERTFILE and WHISPER_WEB_SSL_KEYFILE")
    check_data_layout()
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | device {device}")
    print(f"Bark model: {model}")
    print("Startup checks passed; model inference and microphone access have not been exercised.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="Check the standalone server")
    args = parser.parse_args()
    check(desktop=not args.web)


if __name__ == "__main__":
    main()
