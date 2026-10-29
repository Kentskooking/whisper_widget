"""Single-instance guard and process cleanup for the existing widget."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import subprocess


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class SingleInstance:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateMutexW(None, False, "Local\\BarkWidget")
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            self.close()
            raise RuntimeError("Bark Widget is already running. Use its existing window.")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None



def stop_owned_process(process):
    """Terminate only a worker created by this widget, including Windows venv children."""
    if process.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   capture_output=True, timeout=15, creationflags=NO_WINDOW, check=True)
    process.wait(timeout=10)
