"""Original PCM timeline reconstructed from coordinate-declared upload overlap.

Transport uploads remain durable. Capture boundaries and encoder cores are
internal; insufficient right context returns immediately to the upload queue.
"""
from pathlib import Path
import os
import wave

from app.bark_buffered import FRAME_SAMPLES, LEFT_SAMPLES, RIGHT_SAMPLES


class PauseBoundaryPolicy:
    target, earliest, maximum = 480000, 320000, 640000
    minimum_pause, probe_interval = 12800, 8000

    def choose_pause(self, pauses, observed):
        candidates = []
        margin = self.minimum_pause // 2
        for pause in pauses:
            start, end = max(0, int(pause["start"])), min(observed, self.maximum, int(pause["end"]))
            cut = max(self.earliest, (start + end) // 2)
            if end - start >= self.minimum_pause and start + margin <= cut <= end - margin:
                candidates.append(cut)
        return min(candidates, key=lambda cut: abs(cut - self.target), default=None)


class RecordingAudio:
    def __init__(self, directory, stream_id, pause_detector, log):
        self.directory, self.stream_id = Path(directory), stream_id
        self.pcm_path = self.directory / "original.pcm"
        self.pause_detector, self.log = pause_detector, log
        self.total = self.capture_start = self.core_end = self.index = 0
        self.policy = PauseBoundaryPolicy()
        self.next_probe = self.policy.target
        self.pending = []
        self.closed = False
        # Sessions have unique directories; never truncate recovery audio.
        self.pcm_path.touch(exist_ok=False)

    def read(self, start, end):
        with self.pcm_path.open("rb") as source:
            source.seek(start * 2)
            audio = source.read((end - start) * 2)
        if len(audio) != (end - start) * 2:
            raise RuntimeError("Original audio timeline is incomplete")
        return audio

    def write_wav(self, path, start, end):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            # Bounded copy also supports whole-recording recovery.
            for offset in range(start, end, 160000):
                output.writeframesraw(self.read(offset, min(end, offset + 160000)))
        return str(path)

    def add(self, chunk):
        if self.closed:
            raise RuntimeError("Recording audio is closed")
        start, end = chunk["start_sample"], chunk["end_sample"]
        if start > self.total or end <= self.total or start < 0:
            raise ValueError("Transport chunk has a gap or adds no new audio")
        with wave.open(chunk["path"], "rb") as source:
            audio = source.readframes(source.getnframes())
        if len(audio) != (end - start) * 2:
            raise ValueError("Transport PCM length does not match coordinates")
        overlap = self.total - start
        if overlap and self.read(start, self.total) != audio[:overlap * 2]:
            raise ValueError("Declared transport overlap differs from retained original audio")
        with self.pcm_path.open("ab") as output:
            output.write(audio[overlap * 2:])
            output.flush()
            os.fsync(output.fileno())
        self.total = end
        while self.total - self.capture_start >= self.policy.target:
            available = self.total - self.capture_start
            if available < self.next_probe and available < self.policy.maximum:
                break
            observed = min(available, self.policy.maximum)
            path = self.directory / "pause_probe.wav"
            self.write_wav(path, self.capture_start, self.capture_start + observed)
            try:
                cut = self.policy.choose_pause(self.pause_detector(str(path)), observed)
            except Exception as error:
                self.log("bark_pause_detection_failed", error=error)
                cut = None
            finally:
                path.unlink(missing_ok=True)
            self.next_probe = available + self.policy.probe_interval
            if cut is None and available < self.policy.maximum:
                break
            end = self.capture_start + (cut if cut is not None else self.policy.target)
            self.pending.append((end, "confirmed_pause" if cut is not None else "forced_overlap"))
            self.capture_start = end - (0 if cut is not None else 24000)
            self.next_probe = self.policy.target
        if chunk.get("is_final"):
            return self.finish()
        return self._ready(False)

    def finish(self):
        if self.closed:
            return []
        last_end = self.pending[-1][0] if self.pending else self.core_end
        if last_end < self.total:
            self.pending.append((self.total, "recording_stop"))
        ready = self._ready(True)
        self.closed = True
        return ready

    def _ready(self, final):
        ready = []
        while self.pending:
            end, reason = self.pending[0]
            if not final and self.total < end + RIGHT_SAMPLES:
                break
            start = max(0, self.core_end - LEFT_SAMPLES) // FRAME_SAMPLES * FRAME_SAMPLES
            window_end = min(self.total, end + RIGHT_SAMPLES)
            path = self.directory / f"context_{self.index:04d}_{start}_{window_end}.wav"
            self.write_wav(path, start, window_end)
            ready.append({"index": self.index, "path": str(path), "boundary": reason,
                          "buffered": {"stream_id": self.stream_id, "index": self.index,
                                       "window_start": start, "window_end": window_end,
                                       "core_start": self.core_end, "core_end": end,
                                       "final": final and end == self.total}})
            self.index += 1
            self.core_end = end
            self.pending.pop(0)
        return ready

    def recovery_wav(self):
        return self.write_wav(self.directory / "original.wav", 0, self.total)
