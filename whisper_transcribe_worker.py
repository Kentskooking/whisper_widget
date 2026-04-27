import argparse
import contextlib
import json
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

warnings.filterwarnings("ignore", category=FutureWarning)


def emit_message(payload: dict):
    sys.__stdout__.write(json.dumps(payload) + "\n")
    sys.__stdout__.flush()


@contextlib.contextmanager
def redirect_library_stdout():
    with contextlib.redirect_stdout(sys.stderr):
        yield


def resolve_device(requested_device: str, torch_module):
    requested_device = (requested_device or "auto").strip().lower()
    if requested_device != "auto":
        return requested_device

    if not torch_module.cuda.is_available():
        return "cpu"

    # Preserve the app's historical preference for cuda:1 when two GPUs are visible.
    return "cuda:1" if torch_module.cuda.device_count() > 1 else "cuda:0"


def load_worker_model(model_size: str, requested_device: str):
    import torch
    import whisper

    device = resolve_device(requested_device, torch)
    started = time.perf_counter()
    with redirect_library_stdout():
        model = whisper.load_model(model_size, device=device)
    return model, device, time.perf_counter() - started


def run_server(args) -> int:
    try:
        model, device, load_seconds = load_worker_model(args.model, args.device)
        emit_message({
            "type": "ready",
            "ok": True,
            "pid": os.getpid(),
            "model": args.model,
            "device": device,
            "load_seconds": load_seconds,
        })
    except Exception as e:
        emit_message({
            "type": "ready",
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as e:
            emit_message({
                "type": "response",
                "ok": False,
                "error": f"invalid request json: {e}",
            })
            continue

        request_type = request.get("type")
        if request_type == "shutdown":
            emit_message({"type": "shutdown_ack"})
            return 0

        request_id = request.get("request_id")
        if request_type != "transcribe":
            emit_message({
                "type": "response",
                "request_id": request_id,
                "ok": False,
                "error": f"unsupported request type: {request_type}",
            })
            continue

        try:
            audio_path = request["audio_path"]
            options = request.get("options") or {}
            started = time.perf_counter()
            with redirect_library_stdout():
                result = model.transcribe(audio_path, **options)
            elapsed = time.perf_counter() - started
            emit_message({
                "type": "response",
                "request_id": request_id,
                "ok": True,
                "text": (result.get("text") or "").strip(),
                "elapsed_seconds": elapsed,
            })
        except Exception as e:
            emit_message({
                "type": "response",
                "request_id": request_id,
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.server:
        parser.error("--server is required")

    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
