import argparse
import json
import os
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchaudio  # noqa: F401


_silero_model = None
_silero_utils = None


def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None or _silero_utils is None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--pad-ms", type=int, required=True)
    parser.add_argument("--min-speech-ms", type=int, required=True)
    parser.add_argument("--merge-gap-ms", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
