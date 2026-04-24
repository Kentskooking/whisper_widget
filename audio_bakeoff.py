import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DEFAULT_TARGET_PEAK_DBFS = -4.0
DEFAULT_MAX_GAIN_DB = 14.0
DEFAULT_VARIANTS = [
    {
        "name": "raw_normalized",
        "type": "passthrough",
    },
    {
        "name": "nr_light",
        "type": "noise_reduce_worker",
        "prop_decrease": 0.55,
        "chunk_seconds": 8.0,
        "padding_seconds": 1.5,
        "n_fft": 512,
    },
    {
        "name": "nr_widget_default",
        "type": "noise_reduce_worker",
        "prop_decrease": 0.85,
        "chunk_seconds": 10.0,
        "padding_seconds": 2.0,
        "n_fft": 512,
    },
    {
        "name": "nr_heavy",
        "type": "noise_reduce_worker",
        "prop_decrease": 0.95,
        "chunk_seconds": 12.0,
        "padding_seconds": 2.5,
        "n_fft": 1024,
    },
    {
        "name": "rnnoise_wsl",
        "type": "rnnoise_wsl",
        "input_sample_rate": 48000,
    },
    {
        "name": "webrtc_apm_light_wsl",
        "type": "webrtc_apm_wsl",
        "preset": "light",
    },
    {
        "name": "webrtc_apm_heavy_wsl",
        "type": "webrtc_apm_wsl",
        "preset": "heavy",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a labeled bakeoff set of denoised WAVs from saved recordings. "
            "Each output is normalized using the same peak-normalization strategy as the widget."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input WAV paths. If omitted, the newest raw backup is used.",
    )
    parser.add_argument(
        "--recent-backups",
        type=int,
        default=0,
        help="Use the N most recent files from raw_audio_backups/ instead of explicit inputs.",
    )
    parser.add_argument(
        "--raw-backup-dir",
        default="raw_audio_backups",
        help="Directory used by --recent-backups and the default fallback.",
    )
    parser.add_argument(
        "--out-dir",
        default="bakeoff_outputs",
        help="Root output directory for generated bakeoff runs.",
    )
    parser.add_argument(
        "--variant-config",
        help="Optional JSON file describing variants. Defaults include passthrough and noisereduce worker variants.",
    )
    parser.add_argument(
        "--variants",
        help="Comma-separated variant names to run from the loaded variant set.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip the post-denoise normalization step.",
    )
    parser.add_argument(
        "--target-peak-dbfs",
        type=float,
        default=DEFAULT_TARGET_PEAK_DBFS,
        help="Peak target used by the normalization stage.",
    )
    parser.add_argument(
        "--max-gain-db",
        type=float,
        default=DEFAULT_MAX_GAIN_DB,
        help="Maximum gain applied by the normalization stage.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "variant"


def read_variant_config(path: str | None):
    if not path:
        return list(DEFAULT_VARIANTS)

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{config_path} does not contain a non-empty 'variants' list")
    return variants


def select_variants(variants, variant_csv: str | None):
    if not variant_csv:
        return [variant for variant in variants if variant.get("enabled", True)]

    wanted = {item.strip() for item in variant_csv.split(",") if item.strip()}
    selected = [variant for variant in variants if variant.get("name") in wanted]
    missing = sorted(wanted.difference({variant.get("name") for variant in selected}))
    if missing:
        raise ValueError(f"Unknown variant names: {', '.join(missing)}")
    return selected


def resolve_input_paths(args):
    explicit = [Path(path).resolve() for path in args.inputs]
    if explicit:
        return explicit

    backup_dir = Path(args.raw_backup_dir)
    if not backup_dir.exists():
        raise FileNotFoundError(f"Raw backup directory not found: {backup_dir}")

    wavs = sorted(
        [path for path in backup_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {backup_dir}")

    if args.recent_backups > 0:
        return wavs[: args.recent_backups]

    return [wavs[0]]


def load_mono_wav(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def write_mono_wav(path: Path, audio: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return np.ascontiguousarray(resample_poly(audio, up, down).astype(np.float32))


def normalize_wav(
    in_wav_path: Path,
    out_wav_path: Path,
    target_peak_dbfs: float,
    max_gain_db: float,
):
    audio, sample_rate = sf.read(in_wav_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        sf.write(out_wav_path, audio, sample_rate, subtype="PCM_16")
        return {
            "gain_db": 0.0,
            "input_peak_dbfs": float("-inf"),
            "output_peak_dbfs": float("-inf"),
        }

    target_peak = 10 ** (target_peak_dbfs / 20.0)
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = min(max_gain, max(1.0, target_peak / peak))
    normalized_audio = np.clip(audio * gain, -0.999, 0.999)
    output_peak = float(np.max(np.abs(normalized_audio))) if normalized_audio.size else 0.0
    sf.write(out_wav_path, normalized_audio, sample_rate, subtype="PCM_16")

    return {
        "gain_db": 20.0 * math.log10(max(gain, 1e-12)),
        "input_peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
        "output_peak_dbfs": 20.0 * math.log10(max(output_peak, 1e-12)),
    }


def prepare_variant_input(source_wav: Path, work_dir: Path, variant: dict):
    target_sample_rate = variant.get("input_sample_rate")
    if not target_sample_rate:
        return source_wav

    audio, source_rate = load_mono_wav(source_wav)
    prepared_audio = resample_audio(audio, source_rate, int(target_sample_rate))
    prepared_path = work_dir / f"{slugify(variant['name'])}__prepared.wav"
    write_mono_wav(prepared_path, prepared_audio, int(target_sample_rate))
    return prepared_path


def run_passthrough(prepared_input: Path, working_output: Path):
    shutil.copyfile(prepared_input, working_output)
    return {
        "processor": "passthrough",
    }


def run_noise_reduce_worker(variant: dict, worker_script: Path, prepared_input: Path, working_output: Path):
    command = [
        sys.executable,
        str(worker_script),
        "--input",
        str(prepared_input),
        "--output",
        str(working_output),
        "--prop-decrease",
        str(float(variant.get("prop_decrease", 0.85))),
        "--chunk-seconds",
        str(float(variant.get("chunk_seconds", 10.0))),
        "--padding-seconds",
        str(float(variant.get("padding_seconds", 2.0))),
        "--n-fft",
        str(int(variant.get("n_fft", 512))),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    details = json.loads(result.stdout.strip() or "{}")
    details["processor"] = "noise_reduce_worker"
    return details


def run_command_variant(variant: dict, prepared_input: Path, working_output: Path, work_dir: Path):
    template = variant.get("command")
    if not isinstance(template, list) or not template:
        raise ValueError(f"Variant {variant['name']} is missing a non-empty command list")

    context = {
        "input_wav": str(prepared_input),
        "output_wav": str(working_output),
        "work_dir": str(work_dir),
    }
    command = [item.format(**context) for item in template]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return {
        "processor": "command",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_rnnoise_wsl_wrapper(variant: dict, prepared_input: Path, working_output: Path):
    wrapper_script = Path(__file__).resolve().with_name("rnnoise_wav_wrapper.py")
    command = [
        sys.executable,
        str(wrapper_script),
        "--input",
        str(prepared_input),
        "--output",
        str(working_output),
    ]
    distro = variant.get("wsl_distro")
    if distro:
        command.extend(["--distro", str(distro)])

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    details = json.loads(result.stdout.strip() or "{}")
    details["processor"] = "rnnoise_wsl"
    return details


def run_webrtc_apm_wsl_wrapper(variant: dict, prepared_input: Path, working_output: Path):
    wrapper_script = Path(__file__).resolve().with_name("webrtc_apm_wav_wrapper.py")
    command = [
        sys.executable,
        str(wrapper_script),
        "--input",
        str(prepared_input),
        "--output",
        str(working_output),
        "--preset",
        str(variant.get("preset", "light")),
    ]
    distro = variant.get("wsl_distro")
    if distro:
        command.extend(["--distro", str(distro)])

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    details = json.loads(result.stdout.strip() or "{}")
    details["processor"] = "webrtc_apm_wsl"
    return details


def process_variant(
    source_wav: Path,
    variant: dict,
    case_dir: Path,
    work_dir: Path,
    worker_script: Path,
    args,
):
    variant_name = slugify(variant["name"])
    prepared_input = prepare_variant_input(source_wav, work_dir, variant)
    working_output = work_dir / f"{variant_name}__working.wav"
    final_output = case_dir / f"{source_wav.stem}__{variant_name}.wav"

    started_at = time.perf_counter()
    result = {
        "name": variant["name"],
        "type": variant["type"],
        "status": "ok",
        "source": str(source_wav),
        "prepared_input": str(prepared_input),
        "output": str(final_output),
    }

    if variant["type"] == "passthrough":
        processor_details = run_passthrough(prepared_input, working_output)
    elif variant["type"] == "noise_reduce_worker":
        processor_details = run_noise_reduce_worker(variant, worker_script, prepared_input, working_output)
    elif variant["type"] == "rnnoise_wsl":
        processor_details = run_rnnoise_wsl_wrapper(variant, prepared_input, working_output)
    elif variant["type"] == "webrtc_apm_wsl":
        processor_details = run_webrtc_apm_wsl_wrapper(variant, prepared_input, working_output)
    elif variant["type"] == "command":
        processor_details = run_command_variant(variant, prepared_input, working_output, work_dir)
    else:
        raise ValueError(f"Unsupported variant type: {variant['type']}")

    result["processor_details"] = processor_details
    result["processor_seconds"] = round(time.perf_counter() - started_at, 3)

    if args.skip_normalize:
        shutil.copyfile(working_output, final_output)
        result["normalized"] = False
    else:
        normalize_stats = normalize_wav(
            working_output,
            final_output,
            target_peak_dbfs=args.target_peak_dbfs,
            max_gain_db=args.max_gain_db,
        )
        result["normalized"] = True
        result["normalize_stats"] = {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in normalize_stats.items()
        }

    result["final_seconds"] = round(time.perf_counter() - started_at, 3)
    return result


def write_playlist(case_dir: Path, outputs):
    playlist_path = case_dir / "listen_order.m3u"
    lines = ["#EXTM3U"]
    for output in outputs:
        lines.append(Path(output).name)
    playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_paths = resolve_input_paths(args)
    variants = select_variants(read_variant_config(args.variant_config), args.variants)
    worker_script = Path(__file__).resolve().with_name("noise_reduce_worker.py")
    run_root = Path(args.out_dir) / datetime.now().strftime("bakeoff_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_root": str(run_root.resolve()),
        "inputs": [],
        "variants": variants,
        "target_peak_dbfs": args.target_peak_dbfs,
        "max_gain_db": args.max_gain_db,
        "normalized": not args.skip_normalize,
    }

    for source_wav in input_paths:
        case_dir = run_root / source_wav.stem
        work_dir = case_dir / "_work"
        case_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        case_summary = {
            "source": str(source_wav),
            "case_dir": str(case_dir.resolve()),
            "results": [],
        }
        outputs = []
        for variant in variants:
            if not variant.get("enabled", True):
                case_summary["results"].append(
                    {
                        "name": variant["name"],
                        "type": variant["type"],
                        "status": "skipped",
                        "reason": "disabled",
                    }
                )
                continue

            try:
                result = process_variant(source_wav, variant, case_dir, work_dir, worker_script, args)
                case_summary["results"].append(result)
                outputs.append(result["output"])
                print(f"[ok] {source_wav.name} -> {variant['name']} -> {result['output']}")
            except Exception as exc:
                case_summary["results"].append(
                    {
                        "name": variant["name"],
                        "type": variant["type"],
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"[failed] {source_wav.name} -> {variant['name']} -> {exc}", file=sys.stderr)

        write_playlist(case_dir, outputs)
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(json.dumps(case_summary, indent=2), encoding="utf-8")
        summary["inputs"].append(case_summary)

    run_manifest = run_root / "run_manifest.json"
    run_manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run manifest: {run_manifest.resolve()}")


if __name__ == "__main__":
    main()
