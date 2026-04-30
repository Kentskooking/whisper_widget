# Agent Notes

This file applies to the whole repository.

## Environment
- Use the repo-local virtual environment at `.venv`.
- Keep runtime dependencies in `requirements.txt`.
- Keep developer-only verification tools in `requirements-dev.txt`.

## Verification
- For any code change, run `.\tools\check_all.ps1` from the repo root before close-out.
- If the verifier is not available yet in the local `.venv`, install it with:
  - `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`
- Treat a failing verifier as a blocker unless the user explicitly tells you not to fix it.

## What The Verifier Does
- Runs `tools/check_syntax.py` to compile repo Python files in memory without `.pyc` side effects.
- Runs `ruff check` against repo Python files that are not git-ignored.
- Runs `git diff --check` for whitespace and patch hygiene.

## Reporting
- If you could not run the verifier, say so explicitly and explain why.
- If you ran it, summarize the result in the final handoff.
