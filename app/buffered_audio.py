"""Context windows around the existing, sample-exact capture chunks."""
from pathlib import Path
import uuid
import wave

from app.bark_buffered import FRAME_SAMPLES, LEFT_SAMPLES, RIGHT_SAMPLES


class BufferedAudioSpooler:
    """Keep the capture policy; delay inference until right context is available.

    Raw chunks (including the legacy forced overlap) remain recovery artifacts.
    Decoded cores partition the recording exactly once. Context is read-only
    lookaround and is never stitched into the transcript a second time.
    """

    def __init__(self, capture):
        if (capture.sample_rate, capture.channels, capture.sample_width) != (16000, 1, 2):
            raise ValueError("Buffered Bark requires 16 kHz mono PCM16")
        self.capture = capture
        self.left = LEFT_SAMPLES
        self.right = RIGHT_SAMPLES
        self.stream_id = uuid.uuid4().hex
        self.history = bytearray()
        self.history_start = 0
        self.pending = []
        self.chunks = []
        self.core_end = 0
        self.closed = False

    @property
    def pause_search_enabled(self):
        return self.capture.pause_search_enabled

    @pause_search_enabled.setter
    def pause_search_enabled(self, value):
        self.capture.pause_search_enabled = value

    def add_frame(self, data):
        if self.closed:
            raise RuntimeError("Buffered capture is closed")
        if len(data) % 2:
            raise ValueError("PCM16 samples must be aligned")
        self.history.extend(data)
        self.pending.extend(self.capture.add_frame(data))
        return self._ready(final=False)

    def finalize(self):
        if not self.closed:
            count = len(self.capture.chunks)
            self.capture.finalize()
            self.pending.extend(self.capture.chunks[count:])
            self._ready(final=True)
            self.closed = True
            self.history.clear()
        return list(self.chunks)

    def _ready(self, final):
        ready = []
        total = self.capture.total_samples_received
        while self.pending:
            chunk = self.pending[0]
            end = chunk["end_sample"]
            if not final and total < end + self.right:
                break
            # A common 80 ms encoder grid avoids frame gaps/duplication when a
            # pause midpoint is not aligned to the encoder subsampling stride.
            start = max(0, self.core_end - self.left) // FRAME_SAMPLES * FRAME_SAMPLES
            window_end = min(total, end + self.right)
            if start < self.history_start:
                raise RuntimeError("Required audio context was discarded")
            audio = self.history[(start - self.history_start) * 2:(window_end - self.history_start) * 2]
            if len(audio) != (window_end - start) * 2:
                raise RuntimeError("Incomplete audio context")
            path = Path(self.capture.output_dir) / f"context_{chunk['index']:04d}_{start}_{window_end}.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(audio)
            entry = dict(chunk, context_path=str(path), buffered={
                "stream_id": self.stream_id, "index": chunk["index"],
                "window_start": start, "window_end": window_end,
                "core_start": self.core_end, "core_end": end,
                "final": final and end == total,
            })
            self.core_end = end
            self.chunks.append(entry)
            ready.append(entry)
            self.pending.pop(0)
        # Retain context for the next core plus every not-yet-emitted sample.
        keep = max(0, self.core_end - self.left) // FRAME_SAMPLES * FRAME_SAMPLES
        del self.history[:(keep - self.history_start) * 2]
        self.history_start = keep
        return ready
