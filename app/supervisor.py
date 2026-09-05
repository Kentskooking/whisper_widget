import json
import os
import subprocess
import sys
from app.paths import (
    REPO_ROOT, DESKTOP_MODULE, RUNTIME_STATE_DIR, RUNTIME_LOG_DIR,
    WIDGET_HEARTBEAT_FILE, check_data_layout,
)
import time
import ctypes
import traceback
from datetime import datetime


BASE_DIR = str(REPO_ROOT)
RUNTIME_DIR = os.path.join(BASE_DIR, RUNTIME_STATE_DIR)
HEARTBEAT_PATH = os.path.join(RUNTIME_DIR, WIDGET_HEARTBEAT_FILE)
SUPERVISOR_LOG_PATH = os.path.join(BASE_DIR, RUNTIME_LOG_DIR, "supervisor_log.txt")
POLL_INTERVAL_SECONDS = 1.0
STARTUP_GRACE_SECONDS = 180.0
HEARTBEAT_STALE_SECONDS = 45.0
CRASH_WINDOW_SECONDS = 600.0
MAX_CRASHES_IN_WINDOW = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

if os.name == "nt":
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    STILL_ACTIVE = 259
    WAIT_TIMEOUT = 0x00000102
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    _kernel32.GetExitCodeProcess.restype = ctypes.c_int
    _kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int


class RuntimeProcess:
    def __init__(self, pid: int):
        self.pid = int(pid)
        self.handle = None

        if os.name != "nt":
            return

        desired_access = PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        handle = _kernel32.OpenProcess(desired_access, False, self.pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess failed for pid={self.pid}")
        self.handle = handle

    def poll(self):
        if os.name != "nt":
            try:
                os.kill(self.pid, 0)
                return None
            except OSError:
                return 1

        if not self.handle:
            return None

        wait_result = _kernel32.WaitForSingleObject(self.handle, 0)
        if wait_result == WAIT_TIMEOUT:
            return None

        exit_code = ctypes.c_uint32()
        if not _kernel32.GetExitCodeProcess(self.handle, ctypes.byref(exit_code)):
            raise OSError(ctypes.get_last_error(), f"GetExitCodeProcess failed for pid={self.pid}")
        if exit_code.value == STILL_ACTIVE:
            return None
        return int(exit_code.value)

    def close(self):
        if self.handle:
            _kernel32.CloseHandle(self.handle)
            self.handle = None


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
    try:
        os.makedirs(os.path.dirname(SUPERVISOR_LOG_PATH), exist_ok=True)
        with open(SUPERVISOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        print(f"{timestamp} | supervisor_log_write_failed | error={exc}", flush=True)


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


def stop_runtime_process(runtime_process: RuntimeProcess | None, reason: str, force: bool):
    if runtime_process is None:
        return None

    try:
        returncode = runtime_process.poll()
    except Exception as exc:
        log("runtime_poll_failed", pid=runtime_process.pid, reason=reason, error=exc)
        returncode = None
    if returncode is not None:
        return returncode

    log("runtime_stop_requested", pid=runtime_process.pid, reason=reason, force=force)
    try:
        kill_process_tree(runtime_process.pid, force=force)
    except Exception as exc:
        log("runtime_force_kill_failed", pid=runtime_process.pid, reason=reason, error=exc)
        return runtime_process.poll()

    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            returncode = runtime_process.poll()
        except Exception as exc:
            log("runtime_poll_failed", pid=runtime_process.pid, reason=reason, error=exc)
            return None
        if returncode is not None:
            return returncode
        time.sleep(0.25)
    try:
        return runtime_process.poll()
    except Exception as exc:
        log("runtime_poll_failed", pid=runtime_process.pid, reason=reason, error=exc)
        return None


def safe_poll(process, description: str):
    if process is None:
        return None

    try:
        return process.poll()
    except Exception as exc:
        log("process_poll_failed", process=description, error=exc)
        return None


def launch_child():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WHISPER_WIDGET_SUPERVISED"] = "1"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    child = subprocess.Popen(
        [sys.executable, "-u", "-m", DESKTOP_MODULE],
        cwd=BASE_DIR,
        env=env,
        creationflags=creationflags,
    )
    log("child_started", pid=child.pid, python=sys.executable, module=DESKTOP_MODULE)
    return child


def restart_delay(crashes_in_window: int):
    exponent = max(0, crashes_in_window - 1)
    return min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** exponent))


def main():
    check_data_layout()
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    crash_times = []
    log("supervisor_started", pid=os.getpid(), python=sys.executable, module=DESKTOP_MODULE)

    while True:
        remove_file(HEARTBEAT_PATH)
        child = launch_child()
        runtime_process = None
        runtime_attach_failed_pid = None
        launch_time = time.monotonic()
        launch_wall_time = time.time()
        bound_session_id = None
        restart_reason = None
        restart_details = {}
        launcher_exit_logged = False

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
                if heartbeat is not None and runtime_process is None:
                    heartbeat_pid = heartbeat.get("pid")
                    if heartbeat_pid not in (None, child.pid):
                        try:
                            runtime_process = RuntimeProcess(heartbeat_pid)
                            runtime_attach_failed_pid = None
                            log(
                                "child_runtime_attached",
                                launcher_pid=child.pid,
                                runtime_pid=runtime_process.pid,
                                session_id=bound_session_id,
                            )
                        except Exception as exc:
                            if runtime_attach_failed_pid != heartbeat_pid:
                                log(
                                    "child_runtime_attach_failed",
                                    launcher_pid=child.pid,
                                    runtime_pid=heartbeat_pid,
                                    session_id=bound_session_id,
                                    error=exc,
                                )
                                runtime_attach_failed_pid = heartbeat_pid

                launcher_returncode = child.poll()
                if runtime_process is not None and launcher_returncode is not None and not launcher_exit_logged:
                    log(
                        "child_launcher_exited",
                        launcher_pid=child.pid,
                        runtime_pid=runtime_process.pid,
                        returncode=launcher_returncode,
                    )
                    launcher_exit_logged = True

                monitored_process = runtime_process or child
                monitored_pid = runtime_process.pid if runtime_process is not None else child.pid
                returncode = monitored_process.poll()
                if returncode is not None:
                    if returncode == 0:
                        detached_runtime_pid = heartbeat.get("pid") if heartbeat is not None else None
                        detached_runtime_pending = detached_runtime_pid not in (None, child.pid) and runtime_process is None
                        if runtime_process is None and (bound_session_id is None or detached_runtime_pending):
                            elapsed = time.monotonic() - launch_time
                            if elapsed <= STARTUP_GRACE_SECONDS:
                                if not launcher_exit_logged:
                                    log(
                                        "child_launcher_exited_awaiting_heartbeat",
                                        launcher_pid=child.pid,
                                        heartbeat_pid=detached_runtime_pid,
                                        returncode=returncode,
                                        elapsed_seconds=f"{elapsed:.1f}",
                                    )
                                    launcher_exit_logged = True
                                time.sleep(POLL_INTERVAL_SECONDS)
                                continue
                        log("child_exited_cleanly", pid=monitored_pid, returncode=returncode)
                        if runtime_process is not None:
                            runtime_process.close()
                        return 0
                    restart_reason = "child_exit"
                    restart_details = {"monitored_pid": monitored_pid}
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
        except Exception as exc:
            restart_reason = "supervisor_monitor_error"
            restart_details = {
                "error": repr(exc),
                "traceback": traceback.format_exc()[-1200:],
            }
            log("supervisor_monitor_error", **restart_details)
        except KeyboardInterrupt:
            log("supervisor_interrupt", pid=child.pid)
            stop_runtime_process(runtime_process, reason="supervisor_interrupt", force=False)
            stop_child(child, reason="supervisor_interrupt", force=False)
            if runtime_process is not None:
                runtime_process.close()
            return 130

        if restart_reason in ("heartbeat_missing", "heartbeat_stale", "supervisor_monitor_error"):
            stop_runtime_process(runtime_process, reason=restart_reason, force=True)
            stop_child(child, reason=restart_reason, force=True)

        now = time.monotonic()
        crash_times = [timestamp for timestamp in crash_times if now - timestamp <= CRASH_WINDOW_SECONDS]
        crash_times.append(now)
        crashes_in_window = len(crash_times)
        returncode = (
            safe_poll(runtime_process, "runtime")
            if runtime_process is not None
            else safe_poll(child, "child")
        )

        log(
            "child_restart_scheduled",
            pid=runtime_process.pid if runtime_process is not None else child.pid,
            reason=restart_reason,
            returncode=returncode,
            crashes_in_window=crashes_in_window,
            **restart_details,
        )

        if runtime_process is not None:
            runtime_process.close()

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
