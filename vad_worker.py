import argparse
import contextlib
import json
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchaudio  # noqa: F401


_silero_model = None
_silero_utils = None


def emit_message(payload: dict):
    sys.__stdout__.write(json.dumps(payload) + "\n")
    sys.__stdout__.flush()


@contextlib.contextmanager
def redirect_library_stdout():
    with contextlib.redirect_stdout(sys.stderr):
        yield


def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None or _silero_utils is None:
        with redirect_library_stdout():
            _silero_model, _silero_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        _silero_model.to("cpu")
    return _silero_model, _silero_utils


def vad_extract_speech_only(
    in_wav_path: str,
    out_wav_path: str,
    sample_rate: int = 16000,
    pad_ms: int = 250,
    min_speech_ms: int = 150,
    merge_gap_ms: int = 400,
    speech_prob_threshold: float = 0.5,
) -> float:
    model, utils = load_silero_vad()
    (get_speech_timestamps, save_audio, read_audio, _, collect_chunks) = utils

    wav = read_audio(in_wav_path, sampling_rate=sample_rate)

    speech_ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=speech_prob_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=merge_gap_ms,
    )

    if not speech_ts:
        return 0.0

    pad = int(sample_rate * (pad_ms / 1000.0))
    n = wav.numel()

    padded = []
    for seg in speech_ts:
        start = max(0, seg["start"] - pad)
        end = min(n, seg["end"] + pad)
        padded.append({"start": start, "end": end})

    padded.sort(key=lambda x: x["start"])
    merged = [padded[0]]
    for seg in padded[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"]:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg)

    speech_audio = collect_chunks(merged, wav)
    save_audio(out_wav_path, speech_audio, sampling_rate=sample_rate)

    total_samples = sum(seg["end"] - seg["start"] for seg in merged)
    return total_samples / sample_rate


def write_result(result_file: str, payload: dict):
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def run_once(args) -> int:
    try:
        speech_secs = vad_extract_speech_only(
            in_wav_path=args.input,
            out_wav_path=args.output,
            sample_rate=args.sample_rate,
            pad_ms=args.pad_ms,
            min_speech_ms=args.min_speech_ms,
            merge_gap_ms=args.merge_gap_ms,
            speech_prob_threshold=args.threshold,
        )
        write_result(args.result_file, {"ok": True, "speech_secs": speech_secs})
        return 0
    except Exception as e:
        write_result(
            args.result_file,
            {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def run_server() -> int:
    try:
        load_silero_vad()
        emit_message({"type": "ready", "ok": True, "pid": os.getpid()})
    except Exception as e:
        emit_message(
            {
                "type": "ready",
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as e:
            emit_message(
                {
                    "type": "response",
                    "ok": False,
                    "error": f"invalid request json: {e}",
                }
            )
            continue

        request_type = request.get("type")
        if request_type == "shutdown":
            emit_message({"type": "shutdown_ack"})
            return 0

        request_id = request.get("request_id")
        if request_type != "run":
            emit_message(
                {
                    "type": "response",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"unsupported request type: {request_type}",
                }
            )
            continue

        try:
            with redirect_library_stdout():
                speech_secs = vad_extract_speech_only(
                    in_wav_path=request["input"],
                    out_wav_path=request["output"],
                    sample_rate=int(request["sample_rate"]),
                    pad_ms=int(request["pad_ms"]),
                    min_speech_ms=int(request["min_speech_ms"]),
                    merge_gap_ms=int(request["merge_gap_ms"]),
                    speech_prob_threshold=float(request["threshold"]),
                )
            emit_message(
                {
                    "type": "response",
                    "request_id": request_id,
                    "ok": True,
                    "speech_secs": speech_secs,
                }
            )
        except Exception as e:
            emit_message(
                {
                    "type": "response",
                    "request_id": request_id,
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--pad-ms", type=int)
    parser.add_argument("--min-speech-ms", type=int)
    parser.add_argument("--merge-gap-ms", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--result-file")
    args = parser.parse_args()

    if args.server:
        return run_server()

    required = [
        ("--input", args.input),
        ("--output", args.output),
        ("--sample-rate", args.sample_rate),
        ("--pad-ms", args.pad_ms),
        ("--min-speech-ms", args.min_speech_ms),
        ("--merge-gap-ms", args.merge_gap_ms),
        ("--threshold", args.threshold),
        ("--result-file", args.result_file),
    ]
    missing = [name for name, value in required if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
