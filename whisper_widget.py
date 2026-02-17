import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import ctypes
import customtkinter as ctk
import whisper
import pyaudio
import wave
import threading
import time
import sys
import pyperclip
import math
import struct
import torch
import warnings
from queue import Queue, Empty
from ctypes import wintypes
from datetime import datetime
import torchaudio
import soundfile as sf

try:
    import keyboard
except Exception:
    keyboard = None

# Suppress Torch FutureWarnings (cleaner UI)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MODEL_SIZE = "large-v3"
HOTKEY = "f8"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
LOG_DIR = "transcriptions"
EVENT_LOG_FILE = "event_log.txt"

# Win32 constants for stable global hotkey + topmost behavior
IS_WINDOWS = os.name == "nt"
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_TOPMOST_FLAGS = SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW
HWND_TOPMOST = -1
FILE_ATTRIBUTE_HIDDEN = 0x2
INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.RegisterHotKey.argtypes = [wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL

    user32.UnregisterHotKey.argtypes = [wintypes.HWND, wintypes.INT]
    user32.UnregisterHotKey.restype = wintypes.BOOL

    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int

    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL

    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.UINT
    ]
    user32.SetWindowPos.restype = wintypes.BOOL

    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD

    kernel32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.SetFileAttributesW.restype = wintypes.BOOL

    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
else:
    user32 = None
    kernel32 = None


def hotkey_token_to_vk(token: str):
    token = token.lower()
    named_keys = {
        "esc": 0x1B,
        "escape": 0x1B,
        "tab": 0x09,
        "enter": 0x0D,
        "return": 0x0D,
        "space": 0x20,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "insert": 0x2D,
        "delete": 0x2E,
        "home": 0x24,
        "end": 0x23,
        "pageup": 0x21,
        "pgup": 0x21,
        "pagedown": 0x22,
        "pgdn": 0x22,
    }
    if token in named_keys:
        return named_keys[token]

    if len(token) == 1 and token.isalpha():
        return ord(token.upper())
    if len(token) == 1 and token.isdigit():
        return ord(token)

    if token.startswith("f") and token[1:].isdigit():
        num = int(token[1:])
        if 1 <= num <= 24:
            return 0x70 + (num - 1)

    return None


def parse_hotkey_for_win32(hotkey: str):
    tokens = [tok.strip().lower() for tok in hotkey.split("+") if tok.strip()]
    if not tokens:
        return None

    mods = MOD_NOREPEAT
    vk = None
    for token in tokens:
        if token in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif token == "shift":
            mods |= MOD_SHIFT
        elif token in ("alt", "menu"):
            mods |= MOD_ALT
        elif token in ("win", "windows", "super"):
            mods |= MOD_WIN
        else:
            if vk is not None:
                return None
            vk = hotkey_token_to_vk(token)
            if vk is None:
                return None

    if vk is None:
        return None
    return mods, vk


class EventLogger:
    """Thread-safe event logger for runtime diagnostics."""
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._enabled = True
        self._initialize()

    def _initialize(self):
        try:
            with open(self.path, "a", encoding="utf-8"):
                pass
            if os.path.getsize(self.path) == 0:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write("timestamp | event | details\n")
        except Exception as e:
            self._enabled = False
            print(f"Event log init failed: {e}")
            return

        if IS_WINDOWS:
            try:
                attrs = kernel32.GetFileAttributesW(self.path)
                if attrs != INVALID_FILE_ATTRIBUTES and not (attrs & FILE_ATTRIBUTE_HIDDEN):
                    kernel32.SetFileAttributesW(self.path, attrs | FILE_ATTRIBUTE_HIDDEN)
            except Exception as e:
                print(f"Event log hide flag failed: {e}")

    def log(self, event: str, **fields):
        if not self._enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        safe_fields = []
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value).replace("\n", "\\n").replace("\r", "\\r").replace("|", "/")
            safe_fields.append(f"{key}={text}")

        details = " ".join(safe_fields)
        line = f"{timestamp} | {event}"
        if details:
            line += f" | {details}"
        line += "\n"

        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            self._enabled = False
            print(f"Event log write failed: {e}")

# ---------------- VAD (Silero, CPU) ----------------

_silero_model = None
_silero_utils = None

def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None or _silero_utils is None:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        _silero_model.to("cpu")
    return _silero_model, _silero_utils


def vad_extract_speech_only(
    in_wav_path: str,
    out_wav_path: str,
    sample_rate: int = 16000,
    pad_ms: int = 250,
    min_speech_ms: int = 150,
    merge_gap_ms: int = 400,
    speech_prob_threshold: float = 0.5,
) -> float:
    """
    Creates a speech-only WAV using Silero VAD.
    Returns total detected speech duration (seconds).
    Returns 0.0 if no speech detected.
    """

    model, utils = load_silero_vad()
    (get_speech_timestamps, save_audio, read_audio, _, collect_chunks) = utils

    wav = read_audio(in_wav_path, sampling_rate=sample_rate)

    speech_ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=speech_prob_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=merge_gap_ms,
    )

    if not speech_ts:
        return 0.0

    pad = int(sample_rate * (pad_ms / 1000.0))
    n = wav.numel()

    padded = []
    for seg in speech_ts:
        start = max(0, seg["start"] - pad)
        end = min(n, seg["end"] + pad)
        padded.append({"start": start, "end": end})

    # Merge overlapping segments
    padded.sort(key=lambda x: x["start"])
    merged = [padded[0]]
    for seg in padded[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"]:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg)

    speech_audio = collect_chunks(merged, wav)
    save_audio(out_wav_path, speech_audio, sampling_rate=sample_rate)

    total_samples = sum(seg["end"] - seg["start"] for seg in merged)
    return total_samples / sample_rate

class WhisperWidget(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Ensure Log Directory Exists
        os.makedirs(LOG_DIR, exist_ok=True)
        self.event_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EVENT_LOG_FILE)
        self.event_logger = EventLogger(self.event_log_path)
        self.log_event(
            "app_start",
            pid=os.getpid(),
            python=sys.version.split()[0],
            hotkey=HOTKEY,
            model=MODEL_SIZE
        )
        self.log_event("event_log_initialized", file=self.event_log_path)

        # Window Setup
        self.title("Whisper")
        self.geometry("120x120+50+50") # Small square
        self.resizable(False, False)
        self.attributes("-topmost", True) # Always on top
        self.overrideredirect(True) # Frameless window

        # Dragging Logic
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.do_drag)

        # Dark Mode
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State Variables
        self.is_recording = False
        self.is_processing = False
        self.is_muted = False  # New Mute State
        self.last_ui_state = None
        self.audio_frames = []
        self.ui_queue = Queue()
        self.shutdown_event = threading.Event()
        self.hotkey_mode = None
        self.hotkey_listener_thread = None
        self.hotkey_thread_id = None
        self.hotkey_id = 1
        self.hotkey_ready_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.temp_filename = "temp_recording.wav"
        self.temp_speech_filename = "temp_speech_only.wav"
        self.log_event("audio_init_ok")
        
        # UI Elements
        self.record_btn = ctk.CTkButton(
            self,
            text="READY",
            width=100,
            height=100,
            corner_radius=50, # Circular
            fg_color="#2e7d32", # Green
            hover_color="#1b5e20",
            command=self.toggle_recording,
            font=("Arial", 14, "bold")
        )
        self.record_btn.grid(row=0, column=0, padx=10, pady=10)

        # Mute Indicator (Overlay)
        self.mute_label = ctk.CTkLabel(self, text="", font=("Arial", 20), bg_color="transparent", text_color="red")
        self.mute_label.place(relx=0.8, rely=0.2, anchor="center")

        # Load Model in Background
        self.model = None
        self.status_label = ctk.CTkLabel(self, text="Loading Model...", font=("Arial", 10))
        self.status_label.grid(row=1, column=0, pady=(0, 5))
        
        threading.Thread(target=self.load_model, daemon=True).start()

        # Bind Keys
        self.bind("<space>", lambda event: self.toggle_recording())
        self.bind("m", lambda event: self.toggle_mute())  # Bind M for Mute
        
        # Keep UI updates and topmost behavior stable over long sessions
        self.after(50, self.process_ui_queue)
        self.after(2000, self.maintain_topmost)
        self.bind("<FocusOut>", lambda _event: self.reassert_topmost())
        self.bind("<Map>", lambda _event: self.reassert_topmost())

        # Start Global Hotkey Listener
        self.register_hotkey()
        self.reassert_topmost()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.log_event("app_ready")

    def start_drag(self, event):
        self.x = event.x
        self.y = event.y
        self.log_event("window_drag_start", x=self.winfo_x(), y=self.winfo_y())

    def do_drag(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"{x}+{y}")

    def log_event(self, event, **fields):
        try:
            fields["thread"] = threading.current_thread().name
            self.event_logger.log(event, **fields)
        except Exception:
            pass

    def log_transcription(self, text):
        """Appends transcription to a daily log file."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            filename = os.path.join(LOG_DIR, f"{date_str}.txt")
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {text}\n")
            self.log_event("transcription_log_saved", chars=len(text), file=filename)
        except Exception as e:
            print(f"Failed to log transcription: {e}")
            self.log_event("transcription_log_failed", error=e)

    def load_model(self):
        """Loads the Whisper model on a background thread."""
# Note: cuda:1 actually maps to A6000 due to CUDA_VISIBLE_DEVICES remapping
        try:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            self.log_event("model_load_start", device=device, model=MODEL_SIZE)
            self.model = whisper.load_model(MODEL_SIZE, device=device)
            self.ui_call(self.update_ui_state, "ready")
            print("Model loaded.")
            self.log_event("model_load_success", device=device)
            self.play_sound(1000, 100)
        except Exception as e:
            self.ui_call(self.status_label.configure, text="Error Loading")
            print(f"Error loading model: {e}")
            self.log_event("model_load_failed", error=e)

    def ui_call(self, fn, *args, **kwargs):
        """Runs UI work on the Tk main thread."""
        if threading.current_thread() is threading.main_thread():
            fn(*args, **kwargs)
            return
        self.ui_queue.put((fn, args, kwargs))

    def process_ui_queue(self):
        """Drains queued UI actions from background threads."""
        while True:
            try:
                fn, args, kwargs = self.ui_queue.get_nowait()
            except Empty:
                break
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(f"UI queue callback failed: {e}")
                self.log_event("ui_queue_callback_failed", error=e)

        if not self.shutdown_event.is_set():
            self.after(50, self.process_ui_queue)

    def reassert_topmost(self):
        try:
            self.attributes("-topmost", True)
            self.lift()
            if IS_WINDOWS:
                hwnd = int(self.winfo_id())
                if hwnd:
                    ok = user32.SetWindowPos(
                        wintypes.HWND(hwnd),
                        wintypes.HWND(HWND_TOPMOST),
                        0,
                        0,
                        0,
                        0,
                        SWP_TOPMOST_FLAGS
                    )
                    if not ok:
                        print(f"SetWindowPos failed: {ctypes.get_last_error()}")
                        self.log_event("set_window_pos_failed", win_error=ctypes.get_last_error())
        except Exception as e:
            print(f"Topmost reassert failed: {e}")
            self.log_event("topmost_reassert_failed", error=e)

    def maintain_topmost(self):
        """Periodically reassert topmost so long-running sessions stay pinned."""
        if self.shutdown_event.is_set():
            return
        self.reassert_topmost()
        self.ensure_hotkey_is_alive()
        listener_alive = bool(self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive())
        self.log_event(
            "runtime_heartbeat",
            hotkey_mode=self.hotkey_mode,
            hotkey_listener_alive=listener_alive,
            recording=self.is_recording,
            processing=self.is_processing
        )
        self.after(10000, self.maintain_topmost)

    def on_hotkey_pressed(self):
        self.log_event("hotkey_pressed", hotkey=HOTKEY, mode=self.hotkey_mode)
        self.ui_call(self.toggle_recording)

    def ensure_hotkey_is_alive(self):
        if self.hotkey_mode == "win32":
            if self.hotkey_listener_thread and not self.hotkey_listener_thread.is_alive():
                print("Win32 hotkey loop stopped; re-registering...")
                self.log_event("hotkey_listener_dead", mode="win32")
                self.register_hotkey()

    def register_hotkey(self):
        self.log_event("hotkey_register_attempt", hotkey=HOTKEY)
        self.unregister_hotkey()

        if IS_WINDOWS and self.register_win32_hotkey():
            self.hotkey_mode = "win32"
            print(f"Global hotkey registered via Win32: {HOTKEY.upper()}")
            self.log_event("hotkey_register_success", mode="win32", hotkey=HOTKEY)
            return

        if keyboard is not None:
            try:
                keyboard.add_hotkey(HOTKEY, self.on_hotkey_pressed)
                self.hotkey_mode = "keyboard"
                print(f"Global hotkey registered via keyboard: {HOTKEY.upper()}")
                self.log_event("hotkey_register_success", mode="keyboard", hotkey=HOTKEY)
                return
            except Exception as e:
                print(f"Keyboard hotkey registration failed: {e}")
                self.log_event("hotkey_register_failed", mode="keyboard", error=e)

        self.hotkey_mode = None
        self.ui_call(self.status_label.configure, text="Hotkey Error")
        print("Global hotkey unavailable.")
        self.log_event("hotkey_register_failed", mode="none")

    def register_win32_hotkey(self):
        parsed = parse_hotkey_for_win32(HOTKEY)
        if parsed is None:
            print(f"Unsupported Win32 hotkey format: {HOTKEY}")
            self.log_event("hotkey_parse_failed", hotkey=HOTKEY)
            return False

        mods, vk = parsed
        self.hotkey_ready_event.clear()
        self.hotkey_listener_thread = threading.Thread(
            target=self.win32_hotkey_loop,
            args=(mods, vk),
            daemon=True
        )
        self.hotkey_listener_thread.start()

        # Wait briefly for registration status from listener thread
        ready = self.hotkey_ready_event.wait(timeout=2.0)
        if not ready:
            self.log_event("hotkey_register_timeout", mode="win32")
        return self.hotkey_mode == "win32"

    def win32_hotkey_loop(self, mods, vk):
        self.hotkey_thread_id = kernel32.GetCurrentThreadId()
        self.log_event("win32_hotkey_loop_start", thread_id=self.hotkey_thread_id, vk=vk, mods=mods)

        registered = user32.RegisterHotKey(None, self.hotkey_id, mods, vk)
        if not registered:
            print(f"Win32 RegisterHotKey failed: {ctypes.get_last_error()}")
            self.hotkey_mode = None
            self.hotkey_ready_event.set()
            self.log_event("win32_register_hotkey_failed", win_error=ctypes.get_last_error())
            return

        self.hotkey_mode = "win32"
        self.hotkey_ready_event.set()

        msg = wintypes.MSG()
        while not self.shutdown_event.is_set():
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == -1:
                print(f"Win32 GetMessageW failed: {ctypes.get_last_error()}")
                self.log_event("win32_getmessage_failed", win_error=ctypes.get_last_error())
                break
            if result == 0:
                self.log_event("win32_hotkey_loop_quit_message")
                break
            if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
                self.on_hotkey_pressed()

        try:
            user32.UnregisterHotKey(None, self.hotkey_id)
        except Exception as e:
            print(f"Win32 UnregisterHotKey failed: {e}")
            self.log_event("win32_unregister_hotkey_failed", error=e)

        self.hotkey_thread_id = None
        if not self.shutdown_event.is_set():
            self.hotkey_mode = None
        self.log_event("win32_hotkey_loop_end")

    def unregister_hotkey(self):
        self.log_event("hotkey_unregister_attempt", mode=self.hotkey_mode)
        if self.hotkey_mode == "keyboard" and keyboard is not None:
            try:
                keyboard.unhook_all_hotkeys()
            except Exception as e:
                print(f"Keyboard hotkey cleanup failed: {e}")
                self.log_event("hotkey_unregister_failed", mode="keyboard", error=e)
            finally:
                self.hotkey_mode = None
                self.log_event("hotkey_unregistered", mode="keyboard")
                return

        if self.hotkey_mode == "win32" and self.hotkey_thread_id:
            try:
                user32.PostThreadMessageW(self.hotkey_thread_id, WM_QUIT, 0, 0)
            except Exception as e:
                print(f"Win32 hotkey stop failed: {e}")
                self.log_event("hotkey_unregister_failed", mode="win32", error=e)

        if self.hotkey_listener_thread and self.hotkey_listener_thread.is_alive():
            self.hotkey_listener_thread.join(timeout=1.0)

        self.hotkey_thread_id = None
        self.hotkey_mode = None
        self.log_event("hotkey_unregistered", mode="win32_or_none")

    def play_sound(self, freq, duration):
        """Plays sound if not muted using PyAudio (generates sine wave)."""
        if not self.is_muted:
            try:
                duration_ms = duration  # Duration is passed in ms
                rate = 44100
                
                # Generate Sine Wave
                num_samples = int(rate * (duration_ms / 1000.0))
                audio_data = []
                for x in range(num_samples):
                    sample = 0.38 * math.sin(2 * math.pi * freq * (x / rate)) # 0.38 amplitude
                    # Pack as 16-bit PCM (little endian)
                    packed_sample = struct.pack('<h', int(sample * 32767.0))
                    audio_data.append(packed_sample)
                
                byte_stream = b''.join(audio_data)

                # Play Stream
                stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    output=True
                )
                stream.write(byte_stream)
                stream.stop_stream()
                stream.close()

            except Exception as e:
                print(f"Sound Error: {e}")
                self.log_event("sound_error", error=e)

    def toggle_mute(self):
        """Toggles mute state and updates UI."""
        self.is_muted = not self.is_muted
        self.log_event("mute_toggled", muted=self.is_muted)
        if self.is_muted:
            self.mute_label.configure(text="🔇")
        else:
            self.mute_label.configure(text="")
            self.play_sound(1000, 50)

    def update_ui_state(self, state):
        """Updates button color and text based on state."""
        if state != self.last_ui_state:
            self.log_event("ui_state_changed", state=state)
            self.last_ui_state = state
        if state == "loading":
            self.record_btn.configure(text="LOADING", fg_color="#555555", state="disabled")
            self.status_label.configure(text="Loading...")
        elif state == "ready":
            self.record_btn.configure(text="READY", fg_color="#2e7d32", hover_color="#1b5e20", state="normal")
            self.status_label.configure(text="Press F8")
        elif state == "recording":
            self.record_btn.configure(text="STOP", fg_color="#c62828", hover_color="#b71c1c")
            self.status_label.configure(text="Recording...")
        elif state == "processing":
            self.record_btn.configure(text="...", fg_color="#f9a825", state="disabled")
            self.status_label.configure(text="Transcribing...")

    def toggle_recording(self):
        """Main toggle logic."""
        if threading.current_thread() is not threading.main_thread():
            self.ui_call(self.toggle_recording)
            return

        if self.model is None:
            self.log_event("toggle_ignored_model_not_ready")
            return # Model not loaded yet

        if not self.is_recording:
            if not self.is_processing:
                self.log_event("toggle_start_recording")
                self.start_recording()
        else:
            self.log_event("toggle_stop_recording")
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_frames = []
        self.update_ui_state("recording")
        self.log_event("recording_started")
        
        self.play_sound(800, 100) # High Beep

        # Open Stream
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            self.log_event("microphone_stream_opened", sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
            threading.Thread(target=self.record_loop, daemon=True).start()
        except Exception as e:
            print(f"Microphone error: {e}")
            self.log_event("microphone_error", error=e)
            self.is_recording = False
            self.update_ui_state("ready")

    def record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                print(f"Record loop error: {e}")
                self.log_event("record_loop_error", error=e)
                if self.is_recording:
                    self.ui_call(self.stop_recording)
                break

    def stop_recording(self):
        self.is_recording = False
        self.log_event("recording_stopped", frames=len(self.audio_frames))
        
        self.play_sound(400, 100) # Low Beep
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.update_ui_state("processing")
        self.is_processing = True
        self.log_event("transcription_pipeline_start")

        # Save and Transcribe in Background
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def transcribe_audio(self):
        self.log_event("transcribe_thread_started")
        # Save to WAV
        try:
            wf = wave.open(self.temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            self.log_event("audio_saved", file=self.temp_filename, frames=len(self.audio_frames))
        except Exception as e:
            print(f"Error saving WAV: {e}")
            self.log_event("audio_save_failed", error=e)
            self.ui_call(self.finish_processing)
            return

        # -------- VAD FIRST (CPU) --------
        speech_secs = 0.0
        self.log_event("vad_start", file=self.temp_filename)
        try:
            speech_secs = vad_extract_speech_only(
                in_wav_path=self.temp_filename,
                out_wav_path=self.temp_speech_filename,
                sample_rate=SAMPLE_RATE,
                pad_ms=250,
                min_speech_ms=150,
                merge_gap_ms=400,
                speech_prob_threshold=0.5,
            )
            self.log_event("vad_complete", speech_secs=f"{speech_secs:.3f}")
        except Exception as e:
            print(f"VAD failed, falling back to raw audio: {e}")
            self.log_event("vad_failed_fallback_raw", error=e)
            speech_secs = -1.0

        success = False
        
        if speech_secs == 0.0:
            print("No speech detected (VAD).")
            self.log_event("vad_no_speech_detected")
            success = True
        else:
            wav_for_whisper = (
                self.temp_speech_filename if speech_secs > 0 else self.temp_filename
            )

            for attempt in range(1, 4): # 3 Attempts
                try:
                    self.log_event("transcribe_attempt_start", attempt=attempt, source=wav_for_whisper)
                    # print(f"Transcription Attempt {attempt}...")
                    result = self.model.transcribe(
                        wav_for_whisper,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0
                    )
                    text = result["text"].strip()
                    
                    if text:
                        print("\n[transcription]")
                        print(text)
                        print("[/transcription]", flush=True)
                        pyperclip.copy(text)
                        print("Copied transcription to clipboard.")
                        self.log_event("transcribe_attempt_success", attempt=attempt, chars=len(text))
                        
                        # Log to file
                        self.log_transcription(text)

                        # Success Notification
                        self.ui_call(self.record_btn.configure, text="COPIED", fg_color="#1565c0")
                        self.play_sound(1000, 100)
                        self.play_sound(1500, 100)
                        time.sleep(1) 
                        success = True
                        break # Success!
                    else:
                        print("No text detected.")
                        self.log_event("transcribe_attempt_no_text", attempt=attempt)
                        success = True # Treated as success (empty audio), just didn't copy
                        break 

                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    self.log_event("transcribe_attempt_failed", attempt=attempt, error=e)
                    time.sleep(0.5) # Brief pause before retry

        self.log_event("transcription_pipeline_finish", success=success)
        self.ui_call(self.finish_processing)

        # Cleanup Logic
        if success:
            for f in [self.temp_filename, self.temp_speech_filename]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        self.log_event("temp_file_removed", file=f)
                    except:
                        self.log_event("temp_file_remove_failed", file=f)
                        pass
        else:
            # Failure: Save the file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = f"failed_recording_{timestamp}.wav"
            try:
                os.rename(self.temp_filename, failed_filename)
                print(f"Transcription Failed. Audio saved to: {failed_filename}")
                self.log_event("transcription_failed_audio_saved", file=failed_filename)
                
                if os.path.exists(self.temp_speech_filename):
                    try:
                        failed_speech_filename = f"failed_speech_only_{timestamp}.wav"
                        os.rename(self.temp_speech_filename, failed_speech_filename)
                        self.log_event("transcription_failed_speech_saved", file=failed_speech_filename)
                    except:
                        self.log_event("transcription_failed_speech_save_failed")
                        pass
            except Exception as e:
                print(f"Could not save backup file: {e}")
                self.log_event("transcription_failed_backup_save_error", error=e)

    def finish_processing(self):
        self.is_processing = False
        self.update_ui_state("ready")
        self.log_event("processing_finished")

    def on_close(self):
        self.log_event("app_close_start")
        self.shutdown_event.set()
        self.unregister_hotkey()
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self.p.terminate()
            self.log_event("audio_shutdown_ok")
        except Exception as e:
            print(f"Audio cleanup failed: {e}")
            self.log_event("audio_shutdown_failed", error=e)
        self.log_event("app_close_end")
        self.destroy()

if __name__ == "__main__":
    app = WhisperWidget()
    app.mainloop()
