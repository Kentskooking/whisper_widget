"""Recovery of completed text independently of clipboard availability."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from whisper_widget import WhisperWidget
from widget_web_server import RemoteChunkSession, WhisperWidgetWebServer


def history_widget(directory):
    widget = SimpleNamespace(
        last_transcription_path=str(Path(directory) / "last_transcript.txt"),
        last_transcription="",
        last_transcription_lock=threading.Lock(),
        manual_copy_lock=threading.Lock(),
        shutdown_event=threading.Event(),
        log_event=Mock(),
        copy_transcription_to_clipboard=Mock(return_value=False),
        whisper_client=Mock(),
        vad_client=Mock(),
    )
    widget.remember_transcription = MethodType(WhisperWidget.remember_transcription, widget)
    return widget


class ClipboardHistoryTests(unittest.TestCase):
    def latest_response(self, widget, directory):
        server = WhisperWidgetWebServer(widget, "127.0.0.1", 0, directory, None, None,
                                        widget.log_event)
        messages = []

        async def request():
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await server.create_app()(
                {"type": "http", "method": "GET", "path": "/api/transcripts/latest",
                 "root_path": "", "query_string": b"", "headers": [],
                 "scheme": "http", "http_version": "1.1"}, receive, send)

        asyncio.run(request())
        self.assertEqual(messages[0]["status"], 200)
        self.assertIn((b"cache-control", b"no-store"), messages[0]["headers"])
        widget.copy_transcription_to_clipboard.assert_not_called()
        self.assertEqual(widget.whisper_client.mock_calls, [])
        self.assertEqual(widget.vad_client.mock_calls, [])
        return json.loads(b"".join(item.get("body", b"") for item in messages))

    def test_latest_api_returns_completed_history_while_busy_and_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            self.assertEqual(self.latest_response(widget, directory), {"ok": True, "text": ""})
            # All entry points log their finished text through this shared method.
            for text in ("Host microphone.", "Remote microphone: café.\nSecond line."):
                with patch("whisper_widget.LOG_DIR", directory):
                    WhisperWidget.log_transcription(widget, text)
                for recording, processing in ((False, False), (True, False), (False, True)):
                    widget.is_recording = recording
                    widget.is_processing = processing
                    self.assertEqual(self.latest_response(widget, directory), {"ok": True, "text": text})
            widget.remember_transcription("")
            self.assertEqual(self.latest_response(widget, directory)["text"], text)
            restarted = history_widget(directory)
            restarted.last_transcription = WhisperWidget.load_last_transcription(restarted)
            self.assertEqual(self.latest_response(restarted, directory)["text"], text)

    def test_remote_chunk_completion_publishes_history_before_final_status(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            widget.remember_transcription("Previous recording")
            text = "New completed remote recording"
            widget.stitch_chunk_transcripts = Mock(return_value=text)
            widget.log_transcription = MethodType(WhisperWidget.log_transcription, widget)
            manager = SimpleNamespace(widget=widget, log_event=Mock(), finish_session=Mock())
            session = RemoteChunkSession(manager, "history-test", directory, {})
            session.results = {0: {"ok": True, "text": text}}
            session.status = "processing"
            observations = []

            def observe(*_args, **_kwargs):
                observations.append((session.status_payload(), self.latest_response(widget, directory)))

            # A client can poll while debug artifacts are being saved.
            widget.write_chunk_transcript_debug = Mock(side_effect=observe)
            with patch("whisper_widget.LOG_DIR", directory):
                session._complete_success()
            observe()

            for state, latest in observations:
                if state["final"]:
                    self.assertEqual(latest["text"], text)
            self.assertEqual(observations[-1][0]["status"], "complete")
            self.assertTrue(observations[-1][0]["final"])
            manager.finish_session.assert_called_once_with(session, success=True, reason="complete")

    def test_logged_text_survives_clipboard_failure_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            text = "First line.\nSecond line: café."
            with patch("whisper_widget.LOG_DIR", directory):
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

    def test_manual_copy_ignores_empty_history_and_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = history_widget(directory)
            self.assertEqual(WhisperWidget.load_last_transcription(widget), "")
            self.assertEqual(WhisperWidget.copy_last_transcription(widget), "break")
            widget.last_transcription = "Previous recording"
            widget.shutdown_event.set()
            self.assertEqual(WhisperWidget.copy_last_transcription(widget), "break")
            widget.copy_transcription_to_clipboard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
