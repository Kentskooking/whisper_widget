import argparse
import json
import sys
import warnings

import soundfile as sf

try:
    import noisereduce as nr
except Exception:
    nr = None


warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run noise reduction on a WAV file in an isolated child process."
    )
    parser.add_argument("--input", required=True, help="Input WAV path.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--prop-decrease", type=float, default=0.85)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--padding-seconds", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=512)
    return parser.parse_args()


def reduce_noise_wav(
    in_wav_path: str,
    out_wav_path: str,
    prop_decrease: float,
    chunk_seconds: float,
    padding_seconds: float,
    n_fft: int,
):
    if nr is None:
        raise RuntimeError("noisereduce is not installed")

    audio, sample_rate = sf.read(in_wav_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    chunk_size = max(n_fft, int(sample_rate * chunk_seconds))
    padding = max(n_fft, int(sample_rate * padding_seconds))

    reduced_audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=False,
        prop_decrease=prop_decrease,
        chunk_size=chunk_size,
        padding=padding,
        n_fft=n_fft,
    )
    sf.write(out_wav_path, reduced_audio, sample_rate, subtype="PCM_16")
    return {
        "frames": int(reduced_audio.shape[0]),
        "sample_rate": int(sample_rate),
        "seconds": round(float(reduced_audio.shape[0]) / float(sample_rate), 3),
    }


def main():
    args = parse_args()
    result = reduce_noise_wav(
        in_wav_path=args.input,
        out_wav_path=args.output,
        prop_decrease=args.prop_decrease,
        chunk_seconds=args.chunk_seconds,
        padding_seconds=args.padding_seconds,
        n_fft=args.n_fft,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise
