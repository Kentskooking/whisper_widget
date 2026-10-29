"""Original audio and independent file decoder lifetime, without a GPU/model."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.bark_backend import BarkEngine, MODEL_ID


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.engine = BarkEngine.__new__(BarkEngine)
        self.engine.buffered = Mock()

    def test_stateless_file_uses_fresh_bounded_decoder_and_retains_silence(self):
        original = np.array([[0], [0.001], [0.3], [0], [0], [0.2]], dtype=np.float32)
        active = self.engine.buffered
        with patch("soundfile.read", return_value=(original, 16000)), \
                patch("app.bark_buffered.BufferedBark") as decoder:
            decoder.return_value.transcribe.return_value = {"text": "repeat repeat", "buffered": {"cumulative_text": "repeat repeat"}}
            result = self.engine.transcribe("original.wav")
            pcm, rate, spec = decoder.return_value.transcribe.call_args.args
        np.testing.assert_array_equal(pcm, original[:, 0])
        self.assertEqual(rate, 16000)
        self.assertEqual((spec["core_start"], spec["core_end"], spec["final"]), (0, 6, True))
        self.assertIs(self.engine.buffered, active)
        active.transcribe.assert_not_called()
        self.assertEqual(result["text"], "repeat repeat")
        self.assertEqual(result["model"], MODEL_ID)

    def test_buffered_requests_use_existing_state_and_unmodified_input(self):
        original = np.zeros((16000, 1), dtype=np.float32)
        spec = {"stream_id": "ongoing"}
        self.engine.buffered.transcribe.return_value = {"text": ""}
        with patch("soundfile.read", return_value=(original, 16000)):
            self.engine.transcribe("silent-interior.wav", buffered=spec)
        # Interior silence must advance the carried state, never skip the request.
        pcm, rate, received = self.engine.buffered.transcribe.call_args.args
        np.testing.assert_array_equal(pcm, original[:, 0])
        self.assertEqual(received, spec)

    def test_silent_stateless_audio_is_valid_and_empty_or_nonfinite_is_rejected(self):
        with patch("soundfile.read", return_value=(np.zeros((10, 1), np.float32), 16000)):
            self.assertEqual(self.engine.transcribe("silent.wav")["text"], "")
        for audio in (np.zeros((0, 1), np.float32), np.array([[float("nan")]], np.float32)):
            with patch("soundfile.read", return_value=(audio, 16000)), self.assertRaises(ValueError):
                self.engine.transcribe("invalid.wav")

    def test_whisper_options_and_non_english_are_rejected(self):
        for options in ({"language": "de"}, {"condition_on_previous_text": False}):
            with self.assertRaises(ValueError):
                self.engine.transcribe("unused.wav", **options)
