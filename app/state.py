"""Atomic persistent session state, without importing the desktop UI."""
import json
import os
import threading
import time


def write_json_atomic(path: str, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    last_error = None

    for attempt in range(6):
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(temp_path, path)
            return
        except OSError as e:
            last_error = e
            winerror = getattr(e, "winerror", None)
            if winerror not in (5, 32):
                break
            time.sleep(min(0.5, 0.05 * (2 ** attempt)))
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    if last_error is not None:
        raise last_error
