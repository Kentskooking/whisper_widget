"""Bark Pro ASR with configurable PyTorch device and bounded buffered decoding."""
from pathlib import Path
import os
import time
import uuid

MODEL_ID = "ConwayResearch/Underdog-Bark-3.0-Pro"
MODEL_REVISION = "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4"
MODEL_FILES = ["*.json", "*.safetensors", "NOTICE", "README.md"]


def configure_environment():
    # Respect the user's CUDA_VISIBLE_DEVICES, HF_HOME and offline settings.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_model_path():
    """Resolve an explicit local directory or the previously downloaded snapshot."""
    configured = os.environ.get("BARK_MODEL_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"BARK_MODEL_DIR must contain the Bark Pro model files: {path}")
        return path
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(
            MODEL_ID, revision=os.environ.get("BARK_MODEL_REVISION", MODEL_REVISION),
            allow_patterns=MODEL_FILES, local_files_only=True,
        ))
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError("Cached snapshot is incomplete")
        return path
    except Exception as error:
        raise RuntimeError(
            "Bark Pro is not cached. Run .venv\\Scripts\\python.exe -m app.download_models "
            "or set BARK_MODEL_DIR to a downloaded model directory."
        ) from error


def resolve_device(torch, requested="auto"):
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cpu":
        return "cpu"
    if device.type != "cuda":
        raise ValueError("BARK_DEVICE must be auto, cpu, cuda, or cuda:N")
    index = device.index if device.index is not None else 0
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f"Requested PyTorch device is unavailable: cuda:{index}")
    return f"cuda:{index}"


def resolve_dtype(torch, device, requested="auto"):
    requested = requested.strip().lower()
    bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported(including_emulation=False)
    if requested == "auto":
        return torch.bfloat16 if bf16 else torch.float32
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16" and bf16:
        return torch.bfloat16
    if requested == "float16" and device.startswith("cuda"):
        return torch.float16
    raise ValueError(f"Unsupported BARK_DTYPE={requested!r} for {device}; use auto or float32")


class BarkEngine:
    def __init__(self, model_id=MODEL_ID, device="auto"):
        if model_id not in (MODEL_ID, "bark-pro"):
            raise ValueError(f"Unsupported model {model_id!r}; this backend uses {MODEL_ID}")
        configure_environment()
        model_path = resolve_model_path()
        import torch
        from transformers import AutoProcessor, ParakeetForTDT

        self.torch = torch
        self.device = resolve_device(torch, device)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        self.dtype = resolve_dtype(torch, self.device, os.environ.get("BARK_DTYPE", "auto"))
        threads = int(os.environ.get("BARK_CPU_THREADS", "2"))
        if threads < 1:
            raise ValueError("BARK_CPU_THREADS must be positive")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
        torch.manual_seed(0)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = ParakeetForTDT.from_pretrained(
            model_path, dtype=self.dtype, local_files_only=True, trust_remote_code=False,
        ).to(self.device).eval()
        self.reset_stream()
        self.synchronize()

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def reset_stream(self):
        from app.bark_buffered import BufferedBark
        self.buffered = BufferedBark(self)

    def transcribe(self, path, **options):
        import numpy as np
        import soundfile as sf
        import soxr

        if set(options) - {"buffered", "language"}:
            raise ValueError("Whisper decoding options are not supported by Bark")
        if options.get("language") not in (None, "en", "english"):
            raise ValueError("Bark Pro is an English speech recognition model")
        started = time.perf_counter()
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if not len(audio):
            raise ValueError("Empty audio cannot be transcribed")
        duration = len(audio) / rate
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples")
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if options.get("buffered") is not None:
            result = self.buffered.transcribe(audio, rate, options["buffered"])
        elif peak < 0.0001:
            result = {"text": ""}
        else:
            if rate != 16000:
                audio = soxr.resample(audio, rate, 16000)
            # Bound encoder windows for long files/recovery, independently of
            # any active recording's prediction cache. Preserve original audio.
            from app.bark_buffered import BufferedBark
            result = BufferedBark(self).transcribe(audio, 16000, {
                "stream_id": "file-" + uuid.uuid4().hex, "index": 0,
                "window_start": 0, "window_end": len(audio),
                "core_start": 0, "core_end": len(audio), "final": True,
            })
            # This request is a complete file, not a continuation of a stream.
            result = {"text": result["buffered"]["cumulative_text"]}
        return dict(result, model=MODEL_ID, duration=duration, seconds=time.perf_counter() - started,
                    peak=peak, rms=rms)
