"""The widget must send original samples to Bark, including recovery and retry."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from app.bark_backend import BarkEngine
from app.desktop import WhisperWidget


class UntrimmedAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Quiet edges, an interior pause, and distinct samples on both sides.
        self.pcm = b"\x01\x00" * 4000 + b"\x01\x20" * 4000 + b"\0\0" * 24000 + b"\xff\xff" * 4000
        self.raw = self.root / "chunk_0000_0000000000_0000036000.wav"
        self.write_wav(self.raw, self.pcm)
        self.stale = self.raw.with_name(self.raw.stem + "_speech_only.wav")
        self.write_wav(self.stale, b"\x10\x00" * 512)
        self.widget = WhisperWidget.__new__(WhisperWidget)
        self.widget.log_event = Mock()
        self.widget.vad_client = Mock()
        self.widget.vad_client.run.side_effect = AssertionError("Silence extraction must not run")
        self.widget.is_whisper_ready = Mock(return_value=True)
        self.chunk = {"index": 0, "path": str(self.raw), "start_sample": 0,
                      "end_sample": 36000, "overlap_previous_samples": 0,
                      "boundary": {"reason": "recording_stop"}}

    def write_wav(self, path, pcm):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(pcm)

    def assert_input(self, path):
        with wave.open(str(path), "rb") as source:
            self.assertEqual(source.readframes(source.getnframes()), self.pcm)

    def test_chunk_inference_preserves_pauses_and_quiet_edges(self):
        def transcribe(path, options, **kwargs):
            self.assertEqual(path, str(self.raw))
            self.assert_input(path)
            return {"text": "XML"}
        self.widget.transcribe_with_whisper = Mock(side_effect=transcribe)
        result = self.widget.process_and_transcribe_chunk(self.chunk)
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "XML")
        self.assertEqual(result["transcribed_path"], str(self.raw))
        self.assertIsNone(result["processing"]["speech_secs"])
        self.assertEqual(result["processing"]["audio_seconds"], 2.25)
        self.assertFalse(result["processing"]["silence_trimmed"])
        self.widget.vad_client.run.assert_not_called()

    def test_silent_chunk_keeps_existing_bark_silence_guard(self):
        self.write_wav(self.raw, b"\0\0" * 36000)
        engine = BarkEngine.__new__(BarkEngine)  # No GPU/model needed for its silence guard.
        self.widget.transcribe_with_whisper = lambda path, options, **kwargs: engine.transcribe(path)
        result = self.widget.process_and_transcribe_chunk(self.chunk)
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["transcribed_path"], str(self.raw))
        self.widget.vad_client.run.assert_not_called()

    def test_saved_recording_recovery_and_retry_both_use_original_samples(self):
        widget = self.widget
        widget.temp_filename = str(self.root / "temp.wav")
        widget.temp_speech_filename = str(self.root / "temp_speech_only.wav")
        self.write_wav(widget.temp_speech_filename, b"\0\0" * 512)
        widget.current_raw_backup_path = None
        for name in ("update_recovery_state", "clear_recovery_state", "save_debug_audio_files",
                     "log_transcription", "ui_call", "finish_processing", "play_cue",
                     "remove_raw_audio_backup", "record_btn"):
            setattr(widget, name, Mock())
        widget.copy_transcription_to_clipboard = Mock(return_value=True)
        paths = []
        def transcribe(path, options):
            paths.append(path)
            self.assert_input(path)
            if len(paths) == 1:
                raise RuntimeError("Transient inference failure")
            return {"text": "Recovered XML"}
        widget.transcribe_with_whisper = Mock(side_effect=transcribe)
        with patch("app.desktop.time.sleep"), patch("sys.stdout", new_callable=io.StringIO):
            widget.transcribe_saved_audio(str(self.raw), source_label="startup_recovery")
        self.assertEqual(paths, [widget.temp_filename, widget.temp_filename])
        widget.vad_client.run.assert_not_called()
        widget.copy_transcription_to_clipboard.assert_called_once_with("Recovered XML")
        widget.clear_recovery_state.assert_called_with(reason="transcription_pipeline_complete")
        self.assert_input(self.raw)

    def test_debug_manifest_identifies_raw_input_and_ignores_stale_extraction(self):
        stale_bytes = self.stale.read_bytes()
        self.widget.transcribe_with_whisper = Mock(return_value={"text": "XML"})
        result = self.widget.process_and_transcribe_chunk(self.chunk)
        self.widget.debug_audio_dir = str(self.root / "debug")
        manifest = self.widget.write_chunk_transcript_debug([result], "XML", reason="test")
        saved = json.loads(Path(manifest).read_text(encoding="utf-8"))["chunks"][0]
        self.assertFalse(saved["processing"]["silence_trimmed"])
        self.assertEqual(saved["processing"]["audio_seconds"], 2.25)
        self.assertIsNone(saved["processing"]["speech_secs"])
        self.assertEqual(set(saved["debug_artifacts"]), {"raw"})
        self.assert_input(saved["debug_artifacts"]["raw"])
        self.assertEqual(self.stale.read_bytes(), stale_bytes)
