import argparse
import audioop
import json
import math
import os
import sys
import traceback
import wave


def emit(payload: dict):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def normalize_wav(
    in_wav_path: str,
    out_wav_path: str,
    target_peak_dbfs: float,
    max_gain_db: float,
    target_sample_rate: int | None,
):
    with wave.open(in_wav_path, "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        compression = reader.getcomptype()
        frames = reader.readframes(reader.getnframes())

    if compression != "NONE":
        raise ValueError(f"unsupported WAV compression: {compression}")
    if sample_width != 2:
        raise ValueError(f"normalize_wav only supports 16-bit PCM, got {sample_width * 8}-bit")
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"normalize_wav only supports mono/stereo WAV, got {channels} channels")

    input_sample_rate = int(sample_rate)
    if target_sample_rate and int(sample_rate) != int(target_sample_rate):
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            int(sample_rate),
            int(target_sample_rate),
            None,
        )
        sample_rate = int(target_sample_rate)

    full_scale = float((1 << (8 * sample_width - 1)) - 1)
    peak = float(audioop.max(frames, sample_width)) if frames else 0.0
    if peak <= 0.0:
        normalized_frames = frames
        gain = 1.0
        output_peak = 0.0
    else:
        target_peak = full_scale * (10 ** (target_peak_dbfs / 20.0))
        max_gain = 10 ** (max_gain_db / 20.0)
        gain = min(max_gain, max(1.0, target_peak / peak))
        normalized_frames = audioop.mul(frames, sample_width, gain)
        output_peak = float(audioop.max(normalized_frames, sample_width)) if normalized_frames else 0.0

    out_wav_path = os.path.abspath(out_wav_path)
    output_dir = os.path.dirname(out_wav_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with wave.open(out_wav_path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(sample_width)
        writer.setframerate(int(sample_rate))
        writer.writeframes(normalized_frames)

    input_peak_dbfs = 20.0 * math.log10(max(peak / full_scale, 1e-12))
    output_peak_dbfs = 20.0 * math.log10(max(output_peak / full_scale, 1e-12))
    return {
        "gain_db": 20.0 * math.log10(max(gain, 1e-12)),
        "input_peak_dbfs": input_peak_dbfs,
        "output_peak_dbfs": output_peak_dbfs,
        "sample_rate": int(sample_rate),
        "input_sample_rate": input_sample_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-peak-dbfs", type=float, required=True)
    parser.add_argument("--max-gain-db", type=float, required=True)
    parser.add_argument("--target-sample-rate", type=int)
    args = parser.parse_args()

    try:
        result = normalize_wav(
            in_wav_path=args.input,
            out_wav_path=args.output,
            target_peak_dbfs=args.target_peak_dbfs,
            max_gain_db=args.max_gain_db,
            target_sample_rate=args.target_sample_rate,
        )
        emit(result)
        return 0
    except Exception as e:
        emit({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
