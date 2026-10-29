"""Pause selection and background capture, independent of Tk and model loading."""
from queue import Queue
import threading


class PauseBoundaryPolicy:
    def __init__(self, sample_rate, target_seconds=30, earliest_seconds=20,
                 maximum_seconds=40, pause_seconds=0.8, probe_seconds=0.5):
        self.target = round(target_seconds * sample_rate)
        self.earliest = round(earliest_seconds * sample_rate)
        self.maximum = round(maximum_seconds * sample_rate)
        self.minimum_pause = round(pause_seconds * sample_rate)
        self.probe_interval = round(probe_seconds * sample_rate)
        self.next_probe = self.target
        if not 0 < self.earliest <= self.target < self.maximum:
            raise ValueError("Expected earliest <= target < maximum chunk duration")

    def choose_pause(self, pauses, observed_samples):
        """Choose the pause nearest the target, leaving silence on both sides."""
        candidates = []
        margin = self.minimum_pause // 2
        for pause in pauses:
            start = max(0, int(pause["start"]))
            end = min(observed_samples, self.maximum, int(pause["end"]))
            if end - start < self.minimum_pause:
                continue
            cut = max(self.earliest, (start + end) // 2)
            if start + margin <= cut <= end - margin:
                candidates.append((cut, {"pause_start": start, "pause_end": end}))
        return min(candidates, key=lambda item: abs(item[0] - self.target), default=None)


class BackgroundChunkCapture:
    """One owner for the spooler; the microphone thread only enqueues PCM.

    Stopping skips further pause searches, drains captured frames and flushes the
    remainder. It never waits for the speaker to become silent.
    """
    def __init__(self, spooler, on_chunks):
        self.spooler = spooler
        self.on_chunks = on_chunks
        self.queue = Queue()
        self.stopping = threading.Event()
        self.error = None
        self.spooler.pause_search_enabled = lambda: not self.stopping.is_set()
        self.thread = threading.Thread(target=self._run, name="BarkChunkCapture", daemon=True)
        self.thread.start()

    def add_frame(self, data):
        if self.error:
            raise RuntimeError("Background chunk capture failed") from self.error
        if self.stopping.is_set():
            raise RuntimeError("Chunk capture is already stopping")
        self.queue.put(data)

    def _run(self):
        while True:
            data = self.queue.get()
            try:
                chunks = []
                if data is None:
                    if self.error is None:
                        count = len(self.spooler.chunks)
                        chunks = self.spooler.finalize()[count:]
                elif self.error is None:
                    chunks = self.spooler.add_frame(data)
                if chunks:
                    self.on_chunks(chunks)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()
            if data is None:
                return

    def finalize(self, timeout_seconds=10):
        if not self.stopping.is_set():
            self.stopping.set()
            self.queue.put(None)
        self.thread.join(timeout=timeout_seconds)
        if self.thread.is_alive():
            raise TimeoutError("Background chunk capture did not stop")
        if self.error:
            raise RuntimeError("Background chunk capture failed") from self.error
        return list(self.spooler.chunks)
