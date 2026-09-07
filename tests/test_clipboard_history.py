"""Recovery of completed text independently of clipboard availability."""

from pathlib import Path
import tempfile
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.desktop import WhisperWidget


def history_widget(directory):
    widget = SimpleNamespace(
        log_dir=directory,
        last_transcription_path=str(Path(directory) / "last_transcript.txt"),
        last_transcription="",
        last_transcription_lock=threading.Lock(),
        manual_copy_lock=threading.Lock(),
        shutdown_event=threading.Event(),
        log_event=Mock(),
        copy_transcription_to_clipboard=Mock(return_value=False),
    )
    widget.remember_transcription = MethodType(WhisperWidget.remember_transcription, widget)
    return widget


class ClipboardHistoryTests(unittest.TestCase):
    def test_logged_text_survives_clipboard_failure_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            text = "First line.\nSecond line: café."
            WhisperWidget.log_transcription(widget, text)
            self.assertFalse(widget.copy_transcription_to_clipboard(text))
            restarted = history_widget(directory)
            self.assertEqual(WhisperWidget.load_last_transcription(restarted), text)
            widget.remember_transcription("")
            self.assertEqual(widget.last_transcription, text)
            self.assertEqual(WhisperWidget.load_last_transcription(widget), text)

    def test_failed_disk_save_keeps_text_available_for_manual_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            with patch("builtins.open", side_effect=OSError("Disk unavailable")):
                widget.remember_transcription("Still recoverable")
            self.assertEqual(widget.last_transcription, "Still recoverable")

    def test_manual_copy_runs_in_background_and_ignores_repeated_clicks(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            widget.last_transcription = "Previous recording"
            entered = threading.Event()
            release = threading.Event()

            def clipboard(text):
                entered.set()
                release.wait(timeout=5)
                return True

            widget.copy_transcription_to_clipboard.side_effect = clipboard
            try:
                self.assertEqual(WhisperWidget.copy_last_transcription(widget), "break")
                self.assertTrue(entered.wait(timeout=2))
                self.assertEqual(WhisperWidget.copy_last_transcription(widget), "break")
                widget.copy_transcription_to_clipboard.assert_called_once_with("Previous recording")
            finally:
                release.set()
                finished = widget.manual_copy_lock.acquire(timeout=2)
                self.assertTrue(finished)
                if finished:
                    widget.manual_copy_lock.release()


if __name__ == "__main__":
    unittest.main()
