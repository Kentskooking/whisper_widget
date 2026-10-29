"""Context lookaround must never duplicate, trim, or wait past recording stop."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import wave

import numpy as np

from app.buffered_audio import BufferedAudioSpooler
from app.chunking import BackgroundChunkCapture
from app.desktop import ChunkCaptureSpooler, WhisperWidget


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        return source.readframes(source.getnframes())


class BufferedAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pcm = np.arange(16000 * 95, dtype=np.int16).tobytes()

    def spooler(self, detector=None):
        if detector is None:
            detector = lambda pcm, rate: [{"start": 28 * rate, "end": 29 * rate}]
        capture = ChunkCaptureSpooler(self.temp.name, 16000, 1, 2, 30, 1.5, Mock(), detector)
        return BufferedAudioSpooler(capture)

    def check_audio(self, chunks, original):
        cores = bytearray()
        for chunk in chunks:
            spec = chunk["buffered"]
            start, end = spec["window_start"], spec["window_end"]
            context = read_pcm(chunk["context_path"])
            self.assertEqual(context, original[start * 2:end * 2])
            self.assertEqual(start % 1280, 0)
            cores.extend(context[(spec["core_start"] - start) * 2:(spec["core_end"] - start) * 2])
            self.assertEqual(read_pcm(chunk["path"]), original[chunk["start_sample"] * 2:chunk["end_sample"] * 2])
        self.assertEqual(cores, original)
        self.assertTrue(chunks[-1]["buffered"]["final"])
        self.assertFalse(any(chunk["buffered"]["final"] for chunk in chunks[:-1]))

    def test_waits_for_lookahead_and_keeps_bounded_history(self):
        spooler = self.spooler()
        self.assertEqual(spooler.add_frame(self.pcm[:30 * 32000]), [])
        self.assertEqual(len(spooler.capture.chunks), 1)  # Raw already safe on disk.
        ready = spooler.add_frame(self.pcm[30 * 32000:int(33.5 * 32000)])
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]["buffered"]["window_end"], round(33.5 * 16000))
        for start in range(int(33.5 * 32000), len(self.pcm), 2048):
            spooler.add_frame(self.pcm[start:start + 2048])
            self.assertLess(len(spooler.history), 60 * 32000)
        chunks = spooler.finalize()
        self.assertEqual(spooler.finalize(), chunks)
        self.check_audio(chunks, self.pcm)
        self.assertEqual(spooler.history, b"")

    def test_stop_flushes_pending_context_without_future_audio(self):
        spooler = self.spooler()
        original = self.pcm[:31 * 32000]
        captured = []
        background = BackgroundChunkCapture(spooler, captured.extend)
        background.add_frame(original)
        background.queue.join()
        self.assertEqual(captured, [])
        chunks = background.finalize()
        self.assertFalse(background.thread.is_alive())
        self.assertEqual(captured, chunks)
        self.check_audio(chunks, original)

    def test_forced_overlap_is_context_not_duplicate_decode(self):
        spooler = self.spooler(lambda pcm, rate: [])
        original = self.pcm[:45 * 32000]
        spooler.add_frame(original)
        chunks = spooler.finalize()
        self.assertEqual(chunks[1]["start_sample"], 456000)
        self.assertEqual(chunks[1]["buffered"]["core_start"], 480000)
        self.check_audio(chunks, original)

    def test_short_recording_and_exact_boundary_stop(self):
        for seconds in (2, 30):
            with self.subTest(seconds=seconds):
                spooler = self.spooler()
                original = self.pcm[:seconds * 32000]
                spooler.add_frame(original)
                self.check_audio(spooler.finalize(), original)

    def test_widget_uses_cumulative_tokens_and_rejects_missing_chunks(self):
        widget = WhisperWidget.__new__(WhisperWidget)
        rows = [{"ok": True, "chunk_index": i, "text": text, "buffered": {
            "stream_id": "one", "index": i, "core_start": i * 100, "core_end": (i + 1) * 100,
            "final": i == 1, "cumulative_text": cumulative,
        }} for i, text, cumulative in ((0, "I agree. I", "I agree. I"),
                                       (1, "agree. XML", "I agree. I agree. XML"))]
        widget.merge_transcript_overlap = Mock(side_effect=AssertionError("Must not merge text overlaps"))
        self.assertEqual(widget.stitch_chunk_transcripts(rows), "I agree. I agree. XML")
        with self.assertRaises(RuntimeError):
            widget.stitch_chunk_transcripts(rows[1:])
        with self.assertRaises(RuntimeError):
            widget.stitch_chunk_transcripts(rows[:1])
        rows[1]["buffered"]["stream_id"] = "another"
        with self.assertRaises(RuntimeError):
            widget.stitch_chunk_transcripts(rows)

    def test_partial_buffered_text_is_available_before_final_flush(self):
        widget = WhisperWidget.__new__(WhisperWidget)
        results = [{"ok": True, "chunk_index": 0, "buffered_request": {},
                    "buffered": {"stream_id": "web", "index": 0, "core_start": 0,
                                 "core_end": 16000, "final": False,
                                 "cumulative_text": "repeat repeat"}}]
        self.assertEqual(widget.stitch_chunk_transcripts(results, require_final=False), "repeat repeat")
        with self.assertRaisesRegex(RuntimeError, "not been finalized"):
            widget.stitch_chunk_transcripts(results)

    def test_buffered_worker_failure_uses_batch_fallback(self):
        widget = WhisperWidget.__new__(WhisperWidget)
        widget.chunk_pipeline_failed = False
        widget.chunk_capture_chunks = []
        widget.chunk_transcribe_results = {0: {"ok": False, "buffered_request": {"index": 0}}}
        import threading
        widget.chunk_transcribe_lock = threading.Lock()
        widget.chunk_transcribe_thread = None
        widget.chunk_transcribe_queue = None
        widget.finish_chunk_transcription_queue = Mock(return_value=list(widget.chunk_transcribe_results.values()))
        for method in ("log_event", "write_chunk_transcript_debug", "cleanup_chunked_temp_dir",
                       "update_recovery_state", "transcribe_audio"):
            setattr(widget, method, Mock())
        widget.finalize_chunked_transcription()
        widget.transcribe_audio.assert_called_once()

    def test_context_files_do_not_replace_raw_recovery_chunks(self):
        spooler = self.spooler()
        spooler.add_frame(self.pcm[:45 * 32000])
        chunks = spooler.finalize()
        widget = WhisperWidget.__new__(WhisperWidget)
        for chunk in chunks:
            self.assertIsNone(widget.parse_chunk_recovery_entry(chunk["context_path"]))
            recovered = widget.parse_chunk_recovery_entry(chunk["path"])
            self.assertEqual(recovered["end_sample"], chunk["end_sample"])
        self.assertEqual(len(list(Path(self.temp.name).glob("chunk_*.wav"))), len(chunks))
