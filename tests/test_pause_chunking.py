"""Audio continuity, forced fallback, stopping and transcript boundary regressions."""
import json
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import wave

from app.chunking import BackgroundChunkCapture, PauseBoundaryPolicy
from app.desktop import ChunkCaptureSpooler, WhisperWidget


class PauseChunkingTests(unittest.TestCase):
    rate = 100  # Small sample-exact fixtures; real Silero is tested separately at 16 kHz.

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def spooler(self, detector=lambda pcm, rate: []):
        return ChunkCaptureSpooler(self.temp.name, self.rate, 1, 2, 30, 1.5, Mock(), detector)

    def audio(self, seconds):
        return b"".join(struct.pack("<h", i % 30000) for i in range(round(seconds * self.rate)))

    def assert_continuity(self, chunks, original):
        recovered = b""
        previous_end = 0
        for chunk in chunks:
            with wave.open(chunk["path"], "rb") as source:
                audio = source.readframes(source.getnframes())
            start, end = chunk["start_sample"], chunk["end_sample"]
            self.assertEqual(audio, original[start * 2:end * 2])
            self.assertLessEqual(start, previous_end)
            overlap = max(0, previous_end - start)
            self.assertEqual(chunk["overlap_previous_samples"], overlap)
            recovered += audio[overlap * 2:]
            previous_end = end
        self.assertEqual(recovered, original)

    def test_pause_at_target_has_no_overlap_and_preserves_every_sample(self):
        spooler = self.spooler(lambda pcm, rate: [{"start": 2800, "end": 2900}])
        original = self.audio(35)
        self.assertEqual(spooler.add_frame(original[:6000])[0]["end_sample"], 2850)
        spooler.add_frame(original[6000:])
        chunks = spooler.finalize()
        self.assertEqual(chunks[1]["start_sample"], 2850)
        self.assertEqual(chunks[0]["boundary"]["reason"], "confirmed_pause")
        self.assert_continuity(chunks, original)

    def test_waits_beyond_target_for_a_confirmed_pause(self):
        def detect(pcm, rate):
            return [{"start": 3300, "end": 3400}] if len(pcm) >= 6800 else []
        spooler = self.spooler(detect)
        original = self.audio(38)
        self.assertEqual(spooler.add_frame(original[:6000]), [])
        self.assertEqual(spooler.add_frame(original[6000:6600]), [])
        chunk = spooler.add_frame(original[6600:6800])[0]
        self.assertEqual(chunk["end_sample"], 3350)
        spooler.add_frame(original[6800:])
        self.assert_continuity(spooler.finalize(), original)

    def test_no_pause_by_40_cuts_at_30_and_keeps_all_buffered_tail(self):
        spooler = self.spooler()
        original = self.audio(45)
        self.assertEqual(spooler.add_frame(original[:7800]), [])
        chunk = spooler.add_frame(original[7800:8000])[0]
        self.assertEqual((chunk["start_sample"], chunk["end_sample"]), (0, 3000))
        self.assertEqual(chunk["boundary"]["reason"], "forced_overlap")
        spooler.add_frame(original[8000:])
        chunks = spooler.finalize()
        self.assertEqual(chunks[1]["start_sample"], 2850)
        self.assert_continuity(chunks, original)

    def test_short_gaps_and_early_pauses_are_not_eligible(self):
        spooler = self.spooler(lambda pcm, rate: [
            {"start": 2500, "end": 2579}, {"start": 1700, "end": 1900}])
        self.assertEqual(spooler.add_frame(self.audio(39)), [])
        chunks = spooler.finalize()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["boundary"]["reason"], "recording_stop")

    def test_silence_only_and_multiple_pauses_choose_valid_interior_cut(self):
        policy = PauseBoundaryPolicy(100)
        self.assertEqual(policy.choose_pause([{"start": 0, "end": 3000}], 3000)[0], 2000)
        self.assertEqual(policy.choose_pause([
            {"start": 2100, "end": 2300}, {"start": 2800, "end": 2900}], 3000)[0], 2850)
        self.assertIsNone(policy.choose_pause([{"start": 2950, "end": 4000}], 3000))

    def test_detection_failure_falls_back_without_losing_audio(self):
        spooler = self.spooler(Mock(side_effect=RuntimeError("VAD unavailable")))
        original = self.audio(75)
        for offset in range(0, len(original), 128):
            spooler.add_frame(original[offset:offset + 128])
        chunks = spooler.finalize()
        self.assertEqual(chunks[0]["boundary"]["reason"], "forced_overlap")
        self.assert_continuity(chunks, original)

    def test_mixed_pause_and_forced_cuts_keep_join_metadata_on_correct_chunk(self):
        calls = 0
        def detect(pcm, rate):
            nonlocal calls
            calls += 1
            return [{"start": 2800, "end": 2900}] if calls in (1, 3) else []
        spooler = self.spooler(detect)
        original = self.audio(95)
        for start, end in ((0, 3000), (3000, 6850), (6850, 8700), (8700, 9500)):
            spooler.add_frame(original[start * 2:end * 2])
        chunks = spooler.finalize()
        self.assertEqual([row["overlap_previous_samples"] for row in chunks], [0, 0, 150, 0])
        self.assert_continuity(chunks, original)

    def test_manual_stop_flushes_without_another_pause_search(self):
        detect = Mock(return_value=[])
        spooler = self.spooler(detect)
        original = self.audio(33)
        spooler.add_frame(original)
        count = detect.call_count
        chunks = spooler.finalize()
        self.assertEqual(detect.call_count, count)
        self.assertEqual(len(chunks), 1)
        self.assert_continuity(chunks, original)

    def test_legacy_fixed_chunk_spooler_still_works_without_pause_detector(self):
        spooler = ChunkCaptureSpooler(self.temp.name, self.rate, 1, 2, 3, 1.5, Mock())
        original = self.audio(8)
        spooler.add_frame(original)
        chunks = spooler.finalize()
        self.assertEqual(chunks[0]["end_sample"], 300)
        self.assertEqual(chunks[1]["start_sample"], 150)
        self.assert_continuity(chunks, original)

    def test_final_probe_at_maximum_can_use_a_newly_confirmed_pause(self):
        def detect(pcm, rate):
            return [{"start": 3920, "end": 4000}]
        spooler = self.spooler(detect)
        original = self.audio(41)
        self.assertEqual(spooler.add_frame(original[:7980]), [])
        chunks = spooler.add_frame(original[7980:8000])
        self.assertEqual(chunks[0]["end_sample"], 3960)
        self.assertEqual(chunks[0]["boundary"]["reason"], "confirmed_pause")
        spooler.add_frame(original[8000:])
        self.assert_continuity(spooler.finalize(), original)

    def test_background_analysis_does_not_block_capture_and_stops_cleanly(self):
        entered, release = threading.Event(), threading.Event()
        def detect(pcm, rate):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test detector was not released")
            return []
        spooler = self.spooler(detect)
        captured = []
        background = BackgroundChunkCapture(spooler, captured.extend)
        original = self.audio(35)
        background.add_frame(original[:6000])
        self.assertTrue(entered.wait(2))
        # These calls must return while the detector is still blocked.
        background.add_frame(original[6000:])
        background.stopping.set()
        background.queue.put(None)
        release.set()
        chunks = background.finalize()
        self.assertFalse(background.thread.is_alive())
        self.assertEqual(captured, chunks)
        self.assert_continuity(chunks, original)
        with self.assertRaises(RuntimeError):
            background.add_frame(b"\0\0")

    def test_background_error_surfaces_and_does_not_leave_thread_alive(self):
        spooler = self.spooler()
        background = BackgroundChunkCapture(spooler, Mock(side_effect=OSError("disk full")))
        background.add_frame(self.audio(2))
        with self.assertRaises(RuntimeError):
            background.finalize()
        self.assertFalse(background.thread.is_alive())

    def test_nonoverlap_preserves_spoken_repetition_but_fallback_deduplicates(self):
        widget = WhisperWidget.__new__(WhisperWidget)
        rows = [{"ok": True, "chunk_index": 0, "text": "I agree."},
                {"ok": True, "chunk_index": 1, "text": "I agree. Let's go.",
                 "overlap_previous_samples": 0}]
        self.assertEqual(widget.stitch_chunk_transcripts(rows), "I agree. I agree. Let's go.")
        rows[1]["overlap_previous_samples"] = 24000
        self.assertEqual(widget.stitch_chunk_transcripts(rows), "I agree. Let's go.")
        del rows[1]["overlap_previous_samples"]
        self.assertEqual(widget.stitch_chunk_transcripts(rows), "I agree. Let's go.")

    def test_recovery_reconstructs_overlap_from_sample_ranges(self):
        root = Path(self.temp.name)
        state = root / "state.json"
        state.write_text(json.dumps({"active": True, "chunk_dir": str(root)}))
        for name in ("chunk_0000_0000000000_0000400000.wav",
                     "chunk_0001_0000400000_0000880000.wav",
                     "chunk_0002_0000856000_0001000000.wav"):
            (root / name).touch()
        widget = SimpleNamespace(runtime_recovery_path=str(state), chunked_temp_dir=str(root))
        widget.parse_chunk_recovery_entry = lambda path: WhisperWidget.parse_chunk_recovery_entry(widget, path)
        recovered = WhisperWidget.load_startup_recovery_candidate(widget)["chunks"]
        self.assertEqual([row["overlap_previous_samples"] for row in recovered], [0, 0, 24000])
