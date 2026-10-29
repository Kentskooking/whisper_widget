import argparse
import contextlib
import json
import os
import sys
import time
import traceback
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


def emit_message(payload: dict):
    sys.__stdout__.write(json.dumps(payload) + "\n")
    sys.__stdout__.flush()


@contextlib.contextmanager
def redirect_library_stdout():
    with contextlib.redirect_stdout(sys.stderr):
        yield


def load_worker_model(model_size: str, requested_device: str):
    from app.bark_backend import BarkEngine
    started = time.perf_counter()
    with redirect_library_stdout():
        model = BarkEngine(model_size, requested_device)
    torch = model.torch
    cuda_memory = {}
    if model.device.startswith("cuda"):
        cuda_memory = {
            "cuda_allocated_mb": round(torch.cuda.memory_allocated(model.device) / 1024 / 1024, 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved(model.device) / 1024 / 1024, 1),
            "gpu_name": torch.cuda.get_device_name(model.device),
        }
    return model, model.device, model.device, str(model.dtype), None, cuda_memory, time.perf_counter() - started


def run_server(args) -> int:
    try:
        (
            model,
            device,
            load_device,
            model_dtype,
            layer_norm_dtype,
            cuda_memory,
            load_seconds,
        ) = load_worker_model(args.model, args.device)
        emit_message({
            "type": "ready",
            "ok": True,
            "pid": os.getpid(),
            "model": args.model,
            "device": device,
            "load_device": load_device,
            "model_dtype": model_dtype,
            "layer_norm_dtype": layer_norm_dtype,
            "load_seconds": load_seconds,
            **cuda_memory,
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
        if request_type == "reset_stream":
            model.reset_stream()
            emit_message({"type": "response", "request_id": request_id, "ok": True})
            continue
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
                **({"buffered": result["buffered"]} if "buffered" in result else {}),
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
    from app.bark_backend import MODEL_ID
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default=os.environ.get("BARK_DEVICE", "auto"))
    args = parser.parse_args()

    if not args.server:
        parser.error("--server is required")

    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
