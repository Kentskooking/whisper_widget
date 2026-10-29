"""Sample ownership and queue-independent buffering without model dependencies."""
from pathlib import Path
import tempfile
import unittest
import wave

from app.web.recording_audio import RecordingAudio, PauseBoundaryPolicy
from app.bark_buffered import encoder_windows


def write_audio(path, audio):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(audio)


class AudioTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # Nonconstant samples reveal shifted, dropped and duplicate overlap.
        self.audio = bytes(range(256)) * 10000
        self.pauses = []
        self.events = []
        self.spool = RecordingAudio(self.root, "test", lambda path: self.pauses,
                                    lambda *args, **kw: self.events.append((args, kw)))

    def add(self, start, end, final=False, data=None):
        path = self.root / f"upload_{start}_{end}.wav"
        write_audio(path, self.audio[start * 2:end * 2] if data is None else data)
        return self.spool.add(dict(path=str(path), start_sample=start,
                                   end_sample=end, is_final=final))

    def assert_original(self, windows):
        previous = 0
        for window in windows:
            spec = window["buffered"]
            self.assertEqual(spec["core_start"], previous)
            self.assertEqual(spec["window_start"] % 1280, 0)
            with wave.open(window["path"], "rb") as reader:
                self.assertEqual(reader.readframes(reader.getnframes()),
                                 self.audio[spec["window_start"] * 2:spec["window_end"] * 2])
            previous = spec["core_end"]
            for core_start, core_end, left, right in encoder_windows(spec):
                self.assertLessEqual(core_end - core_start, 160000)
                self.assertLessEqual(right - core_end, 80000)
        self.assertTrue(windows[-1]["buffered"]["final"])
        self.assertEqual(previous, self.spool.total)

    def test_transport_overlap_is_removed_and_forced_cores_partition_once(self):
        self.assertEqual(self.add(0, 480000), [])
        windows = self.add(456000, 936000)
        self.assertEqual(windows[0]["boundary"], "forced_overlap")
        self.assertEqual(windows[0]["buffered"]["core_end"], 480000)
        windows += self.spool.finish()
        self.assert_original(windows)
        self.assertEqual(self.spool.read(0, 936000), self.audio[:936000 * 2])

    def test_pause_waits_for_right_context_without_blocking_then_stop_flushes_tail(self):
        self.pauses = [{"start": 440000, "end": 456000}]
        self.assertEqual(self.add(0, 480000), [])
        windows = self.add(456000, 540000)
        self.assertEqual(windows[0]["buffered"]["core_end"], 448000)
        self.assertEqual(windows[0]["boundary"], "confirmed_pause")
        windows += self.spool.finish()
        self.assert_original(windows)

    def test_finish_exact_transport_boundary_and_short_tail_are_final(self):
        self.assertEqual(self.add(0, 480000), [])
        windows = self.spool.finish()
        self.assert_original(windows)
        self.assertEqual(windows[0]["buffered"]["window_end"], 480000)
        self.assertEqual(self.spool.finish(), [])

    def test_vad_failure_forces_bounded_cut_and_keeps_samples(self):
        def failed(path):
            raise RuntimeError("VAD unavailable")
        self.spool.pause_detector = failed
        self.assertEqual(self.add(0, 480000), [])
        windows = self.add(456000, 640000)
        self.assertEqual(windows[0]["buffered"]["core_end"], 480000)
        windows += self.spool.finish()
        self.assert_original(windows)
        self.assertTrue(self.events)

    def test_long_final_upload_still_applies_available_forced_boundary(self):
        self.assertEqual(self.add(0, 480000), [])
        windows = self.add(456000, 800000, final=True)
        self.assertEqual([item["buffered"]["core_end"] for item in windows], [480000, 800000])
        self.assert_original(windows)

    def test_maximum_wait_overrides_a_later_scheduled_probe(self):
        self.assertEqual(self.add(0, 638400), [])  # 39.9 s; next probe at 40.4 s.
        windows = self.add(638400, 640000)
        self.assertEqual(windows[0]["buffered"]["core_end"], 480000)
        self.assertEqual(windows[0]["boundary"], "forced_overlap")

    def test_reject_gap_changed_overlap_and_repeated_range_without_mutation(self):
        self.add(0, 100)
        for start, end, audio in ((101, 200, None), (0, 100, None), (50, 150, b"\x01\x01" * 100)):
            with self.assertRaises(ValueError):
                self.add(start, end, data=audio)
        self.assertEqual(self.spool.total, 100)
        self.assertEqual(self.spool.read(0, 100), self.audio[:200])

    def test_empty_recording_and_final_short_recording(self):
        self.assertEqual(self.spool.finish(), [])
        other = self.root / "next"
        other.mkdir()
        self.spool = RecordingAudio(other, "next", lambda path: [], lambda *a, **k: None)
        self.assert_original(self.add(0, 100, final=True))

    def test_pause_margin_earliest_long_silence_and_short_word(self):
        policy = PauseBoundaryPolicy()
        self.assertEqual(policy.choose_pause([{"start": 0, "end": 480000}], 480000), 320000)
        self.assertIsNone(policy.choose_pause([{"start": 475000, "end": 480000}], 480000))
        pauses = [{"start": 460000, "end": 466000}, {"start": 467000, "end": 473000}]
        self.assertIsNone(policy.choose_pause(pauses, 480000))


if __name__ == "__main__":
    unittest.main()
