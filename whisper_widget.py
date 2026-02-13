import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import customtkinter as ctk
import whisper
import pyaudio
import wave
import threading
import time
import keyboard
import pyperclip
import math
import struct
import torch
import warnings
from queue import Queue, Empty
from datetime import datetime
import torchaudio
import soundfile as sf

# Suppress Torch FutureWarnings (cleaner UI)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MODEL_SIZE = "large-v3"
HOTKEY = "f8"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
LOG_DIR = "transcriptions"

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
        self.audio_frames = []
        self.ui_queue = Queue()
        self.shutdown_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.temp_filename = "temp_recording.wav"
        self.temp_speech_filename = "temp_speech_only.wav"
        
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
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_drag(self, event):
        self.x = event.x
        self.y = event.y

    def do_drag(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"{x}+{y}")

    def log_transcription(self, text):
        """Appends transcription to a daily log file."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            filename = os.path.join(LOG_DIR, f"{date_str}.txt")
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] {text}\n")
        except Exception as e:
            print(f"Failed to log transcription: {e}")

    def load_model(self):
        """Loads the Whisper model on a background thread."""
# Note: cuda:1 actually maps to A6000 due to CUDA_VISIBLE_DEVICES remapping
        try:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            self.model = whisper.load_model(MODEL_SIZE, device=device)
            self.ui_call(self.update_ui_state, "ready")
            print("Model loaded.")
            self.play_sound(1000, 100)
        except Exception as e:
            self.ui_call(self.status_label.configure, text="Error Loading")
            print(f"Error loading model: {e}")

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

        if not self.shutdown_event.is_set():
            self.after(50, self.process_ui_queue)

    def reassert_topmost(self):
        try:
            self.attributes("-topmost", True)
            self.lift()
        except Exception as e:
            print(f"Topmost reassert failed: {e}")

    def maintain_topmost(self):
        """Periodically reassert topmost so long-running sessions stay pinned."""
        if self.shutdown_event.is_set():
            return
        self.reassert_topmost()
        self.after(10000, self.maintain_topmost)

    def on_hotkey_pressed(self):
        self.ui_call(self.toggle_recording)

    def register_hotkey(self):
        try:
            keyboard.add_hotkey(HOTKEY, self.on_hotkey_pressed)
            print(f"Global hotkey registered: {HOTKEY.upper()}")
        except Exception as e:
            self.ui_call(self.status_label.configure, text="Hotkey Error")
            print(f"Hotkey registration failed: {e}")

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

    def toggle_mute(self):
        """Toggles mute state and updates UI."""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_label.configure(text="🔇")
        else:
            self.mute_label.configure(text="")
            self.play_sound(1000, 50)

    def update_ui_state(self, state):
        """Updates button color and text based on state."""
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
            return # Model not loaded yet

        if not self.is_recording:
            if not self.is_processing:
                self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_frames = []
        self.update_ui_state("recording")
        
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
            threading.Thread(target=self.record_loop, daemon=True).start()
        except Exception as e:
            print(f"Microphone error: {e}")
            self.is_recording = False
            self.update_ui_state("ready")

    def record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                print(f"Record loop error: {e}")
                if self.is_recording:
                    self.ui_call(self.stop_recording)
                break

    def stop_recording(self):
        self.is_recording = False
        
        self.play_sound(400, 100) # Low Beep
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.update_ui_state("processing")
        self.is_processing = True

        # Save and Transcribe in Background
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def transcribe_audio(self):
        # Save to WAV
        try:
            wf = wave.open(self.temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
        except Exception as e:
            print(f"Error saving WAV: {e}")
            self.ui_call(self.finish_processing)
            return

        # -------- VAD FIRST (CPU) --------
        speech_secs = 0.0
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
        except Exception as e:
            print(f"VAD failed, falling back to raw audio: {e}")
            speech_secs = -1.0

        success = False
        
        if speech_secs == 0.0:
            print("No speech detected (VAD).")
            success = True
        else:
            wav_for_whisper = (
                self.temp_speech_filename if speech_secs > 0 else self.temp_filename
            )

            for attempt in range(1, 4): # 3 Attempts
                try:
                    # print(f"Transcription Attempt {attempt}...")
                    result = self.model.transcribe(
                        wav_for_whisper,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0
                    )
                    text = result["text"].strip()
                    
                    if text:
                        pyperclip.copy(text)
                        print(f"Copied: {text}")
                        
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
                        success = True # Treated as success (empty audio), just didn't copy
                        break 

                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    time.sleep(0.5) # Brief pause before retry

        self.ui_call(self.finish_processing)

        # Cleanup Logic
        if success:
            for f in [self.temp_filename, self.temp_speech_filename]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        else:
            # Failure: Save the file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_filename = f"failed_recording_{timestamp}.wav"
            try:
                os.rename(self.temp_filename, failed_filename)
                print(f"Transcription Failed. Audio saved to: {failed_filename}")
                
                if os.path.exists(self.temp_speech_filename):
                    try:
                        os.rename(self.temp_speech_filename, f"failed_speech_only_{timestamp}.wav")
                    except:
                        pass
            except Exception as e:
                print(f"Could not save backup file: {e}")

    def finish_processing(self):
        self.is_processing = False
        self.update_ui_state("ready")

    def on_close(self):
        self.shutdown_event.set()
        try:
            keyboard.unhook_all_hotkeys()
        except Exception as e:
            print(f"Hotkey cleanup failed: {e}")
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self.p.terminate()
        except Exception as e:
            print(f"Audio cleanup failed: {e}")
        self.destroy()

if __name__ == "__main__":
    app = WhisperWidget()
    app.mainloop()
