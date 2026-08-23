import ctypes
import os
import sys
from ctypes import wintypes


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
EXIT_COPY_FAILED = 1
EXIT_CLIPBOARD_BUSY = 2


class ClipboardBusyError(RuntimeError):
    pass


def copy_text_once(text: str) -> None:
    """Attempt one Windows clipboard write without retrying."""
    if os.name != "nt":
        raise RuntimeError("Windows clipboard worker requires Windows")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    hwnd = user32.CreateWindowExW(
        0,
        "STATIC",
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
    )
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())

    clipboard_open = False
    allocation = None
    try:
        if not user32.OpenClipboard(hwnd):
            raise ClipboardBusyError(f"OpenClipboard failed: {ctypes.get_last_error()}")
        clipboard_open = True

        if not user32.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())

        buffer = ctypes.create_unicode_buffer(text)
        allocation = kernel32.GlobalAlloc(GMEM_MOVEABLE, ctypes.sizeof(buffer))
        if not allocation:
            raise ctypes.WinError(ctypes.get_last_error())

        pointer = kernel32.GlobalLock(allocation)
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            ctypes.memmove(pointer, ctypes.addressof(buffer), ctypes.sizeof(buffer))
        finally:
            kernel32.GlobalUnlock(allocation)

        if not user32.SetClipboardData(CF_UNICODETEXT, allocation):
            raise ctypes.WinError(ctypes.get_last_error())

        # Windows owns the allocation after SetClipboardData succeeds.
        allocation = None
    finally:
        if allocation:
            kernel32.GlobalFree(allocation)
        if clipboard_open:
            user32.CloseClipboard()
        user32.DestroyWindow(hwnd)


def main() -> int:
    text = sys.stdin.read()
    try:
        copy_text_once(text)
    except ClipboardBusyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CLIPBOARD_BUSY
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_COPY_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
