from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_python_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
        "tools/*.py",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        paths.append(REPO_ROOT / stripped)
    return sorted(paths)


def compile_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def main() -> int:
    try:
        files = repo_python_files()
    except subprocess.CalledProcessError as exc:
        print("Failed to list repo Python files.", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        return 1

    if not files:
        print("No repo Python files found.")
        return 0

    failures = 0
    for path in files:
        try:
            compile_file(path)
            print(f"OK   {path.relative_to(REPO_ROOT)}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {path.relative_to(REPO_ROOT)}: {exc}", file=sys.stderr)

    if failures:
        print(f"Syntax check failed for {failures} file(s).", file=sys.stderr)
        return 1

    print(f"Syntax check passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
