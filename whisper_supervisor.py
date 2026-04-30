import json
import os
import subprocess
import sys
import time
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "sidecache", "runtime")
HEARTBEAT_PATH = os.path.join(RUNTIME_DIR, "widget_heartbeat.json")
SUPERVISOR_LOG_PATH = os.path.join(RUNTIME_DIR, "supervisor_log.txt")
CHILD_SCRIPT = os.path.join(BASE_DIR, "whisper_widget.py")
POLL_INTERVAL_SECONDS = 1.0
STARTUP_GRACE_SECONDS = 180.0
HEARTBEAT_STALE_SECONDS = 45.0
CRASH_WINDOW_SECONDS = 600.0
MAX_CRASHES_IN_WINDOW = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0


def log(message: str, **fields):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details = " ".join(
        f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}"
        for key, value in fields.items()
        if value is not None
    )
    line = f"{timestamp} | {message}"
    if details:
        line += f" | {details}"
    print(line, flush=True)
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(SUPERVISOR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def heartbeat_snapshot(launch_wall_time: float, session_id: str | None = None):
    if not os.path.exists(HEARTBEAT_PATH):
        return None

    payload = load_json(HEARTBEAT_PATH)
    if not isinstance(payload, dict):
        return None

    try:
        modified_at = os.path.getmtime(HEARTBEAT_PATH)
    except OSError:
        return None

    # Ignore stale heartbeat files left behind from a previous session.
    if modified_at + 1.0 < launch_wall_time:
        return None

    if session_id is not None and payload.get("session_id") != session_id:
        return None

    payload["_age_seconds"] = max(0.0, time.time() - modified_at)
    return payload


def kill_process_tree(pid: int, force: bool):
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def stop_child(child: subprocess.Popen, reason: str, force: bool):
    if child.poll() is not None:
        return child.returncode

    log("child_stop_requested", pid=child.pid, reason=reason, force=force)
    if not force:
        try:
            child.terminate()
            child.wait(timeout=10)
            return child.returncode
        except Exception:
            pass

    try:
        kill_process_tree(child.pid, force=True)
    except Exception as exc:
        log("child_force_kill_failed", pid=child.pid, reason=reason, error=exc)

    try:
        child.wait(timeout=15)
    except Exception:
        pass
    return child.poll()


def launch_child():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WHISPER_WIDGET_SUPERVISED"] = "1"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    child = subprocess.Popen(
        [sys.executable, CHILD_SCRIPT],
        cwd=BASE_DIR,
        env=env,
        creationflags=creationflags,
    )
    log("child_started", pid=child.pid, python=sys.executable, script=CHILD_SCRIPT)
    return child


def restart_delay(crashes_in_window: int):
    exponent = max(0, crashes_in_window - 1)
    return min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** exponent))


def main():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    crash_times = []

    while True:
        remove_file(HEARTBEAT_PATH)
        child = launch_child()
        launch_time = time.monotonic()
        launch_wall_time = time.time()
        bound_session_id = None
        restart_reason = None
        restart_details = {}

        try:
            while True:
                heartbeat = heartbeat_snapshot(launch_wall_time, session_id=bound_session_id)
                if heartbeat is not None and bound_session_id is None:
                    bound_session_id = heartbeat.get("session_id")
                    log(
                        "child_heartbeat_bound",
                        child_pid=child.pid,
                        heartbeat_pid=heartbeat.get("pid"),
                        session_id=bound_session_id,
                    )

                returncode = child.poll()
                if returncode is not None:
                    if returncode == 0:
                        log("child_exited_cleanly", pid=child.pid, returncode=returncode)
                        return 0
                    restart_reason = "child_exit"
                    restart_details = {"returncode": returncode}
                    break

                elapsed = time.monotonic() - launch_time
                if heartbeat is None:
                    if elapsed > STARTUP_GRACE_SECONDS:
                        restart_reason = "heartbeat_missing"
                        restart_details = {"elapsed_seconds": f"{elapsed:.1f}"}
                        break
                else:
                    age_seconds = heartbeat.get("_age_seconds")
                    if age_seconds is not None and age_seconds > HEARTBEAT_STALE_SECONDS:
                        restart_reason = "heartbeat_stale"
                        restart_details = {
                            "age_seconds": f"{age_seconds:.1f}",
                            "status": heartbeat.get("status"),
                            "recording": heartbeat.get("recording"),
                            "processing": heartbeat.get("processing"),
                            "recovery_active": heartbeat.get("recovery_active"),
                        }
                        break

                time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("supervisor_interrupt", pid=child.pid)
            stop_child(child, reason="supervisor_interrupt", force=False)
            return 130

        if restart_reason in ("heartbeat_missing", "heartbeat_stale"):
            stop_child(child, reason=restart_reason, force=True)

        now = time.monotonic()
        crash_times = [timestamp for timestamp in crash_times if now - timestamp <= CRASH_WINDOW_SECONDS]
        crash_times.append(now)
        crashes_in_window = len(crash_times)
        returncode = child.poll()

        log(
            "child_restart_scheduled",
            pid=child.pid,
            reason=restart_reason,
            returncode=returncode,
            crashes_in_window=crashes_in_window,
            **restart_details,
        )

        if crashes_in_window > MAX_CRASHES_IN_WINDOW:
            log(
                "supervisor_giving_up",
                reason="crash_loop",
                crashes_in_window=crashes_in_window,
            )
            return returncode if returncode not in (None, 0) else 1

        delay = restart_delay(crashes_in_window)
        log("child_restart_wait", seconds=f"{delay:.1f}")
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            log("supervisor_interrupt_during_backoff")
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
