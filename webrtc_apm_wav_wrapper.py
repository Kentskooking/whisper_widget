import argparse
import json
import os
import subprocess
import time
from pathlib import Path, PureWindowsPath


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the native WSL WebRTC APM WAV helper from the bakeoff harness."
    )
    parser.add_argument("--input", required=True, help="Input WAV path.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--preset", choices=("light", "heavy"), default="light")
    parser.add_argument("--distro", default="Ubuntu-22.04", help="WSL distro name.")
    return parser.parse_args()


def to_wsl_path(path: Path):
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("/mnt/"):
        return text
    windows_path = PureWindowsPath(resolved)
    drive = windows_path.drive.rstrip(":").lower()
    tail = "/".join(windows_path.parts[1:])
    return f"/mnt/{drive}/{tail}"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    repo_root = Path(__file__).resolve().parent
    runner_script = repo_root / "tools" / "run_webrtc_apm_wsl.sh"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if os.name == "nt":
        command = [
            "wsl.exe",
            "-d",
            args.distro,
            "--",
            "/bin/bash",
            to_wsl_path(runner_script),
            "--input",
            to_wsl_path(input_path),
            "--output",
            to_wsl_path(output_path),
            "--preset",
            args.preset,
        ]
    else:
        command = [
            "/bin/bash",
            str(runner_script),
            "--input",
            str(input_path.resolve()),
            "--output",
            str(output_path.resolve()),
            "--preset",
            args.preset,
        ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"WebRTC APM helper failed with exit code {exc.returncode}: "
            f"stdout={exc.stdout.strip()} stderr={exc.stderr.strip()}"
        ) from exc

    helper_stdout = result.stdout.strip()
    try:
        details = json.loads(helper_stdout or "{}")
    except json.JSONDecodeError:
        details = {"stdout": helper_stdout}

    details["preset"] = args.preset
    details["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if result.stderr.strip():
        details["stderr"] = result.stderr.strip()

    print(json.dumps(details))


if __name__ == "__main__":
    main()
