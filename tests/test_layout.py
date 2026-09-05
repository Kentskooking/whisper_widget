"""Exercise process entry points and paths after source moves."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import desktop, supervisor
from app.paths import REPO_ROOT, VAD_WORKER_MODULE, WHISPER_WORKER_MODULE
from app.transcription_service import TranscriptionService


class LayoutTests(unittest.TestCase):
    def test_worker_module_entry_points(self):
        for module in (VAD_WORKER_MODULE, WHISPER_WORKER_MODULE):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--server", result.stdout)

    def test_whisper_worker_reports_startup_error(self):
        result = subprocess.run(
            [sys.executable, "-m", WHISPER_WORKER_MODULE, "--server",
             "--model", "__invalid_test_model__", "--device", "cpu"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["type"], "ready")
        self.assertFalse(payload["ok"])
        self.assertIn("__invalid_test_model__", payload["error"])

    def test_service_storage_is_independent_of_worker_location(self):
        with tempfile.TemporaryDirectory(prefix="whisper-layout-") as directory:
            service = TranscriptionService(base_dir=directory)
            try:
                self.assertTrue(Path(service.log_dir).is_relative_to(directory))
                self.assertTrue(Path(service.work_dir).is_relative_to(directory))
                self.assertEqual(service.vad_client.workdir, str(REPO_ROOT))
                self.assertEqual(service.whisper_client.workdir, str(REPO_ROOT))
                self.assertEqual(service.vad_client.worker_module, VAD_WORKER_MODULE)
            finally:
                service.close()

    def test_supervisor_launches_desktop_module(self):
        with patch.object(supervisor.subprocess, "Popen") as start, patch.object(supervisor, "log"):
            supervisor.launch_child()
        self.assertEqual(start.call_args.args[0],
                         [sys.executable, "-u", "-m", "app.desktop"])
        self.assertEqual(Path(start.call_args.kwargs["cwd"]), REPO_ROOT)

    def test_clipboard_worker_uses_module_and_preserves_failure(self):
        widget = SimpleNamespace(log_event=lambda *args, **kwargs: None)
        result = SimpleNamespace(returncode=2, stderr="Clipboard is busy", stdout="")
        with patch.object(desktop.subprocess, "run", return_value=result) as run:
            copied = desktop.WhisperWidget.copy_transcription_to_clipboard(widget, "sample")
        self.assertFalse(copied)
        self.assertEqual(run.call_args.args[0][-2:], ["-m", "app.workers.clipboard"])
        self.assertEqual(Path(run.call_args.kwargs["cwd"]), REPO_ROOT)

    def test_web_modules_import_and_render_without_starting_workers(self):
        with tempfile.TemporaryDirectory(prefix="whisper-web-layout-") as directory:
            service = TranscriptionService(base_dir=directory)
            try:
                with patch("app.transcription_service.TranscriptionService", return_value=service):
                    standalone = importlib.import_module("app.web.standalone")
                self.assertIn(b"Whisper Web", standalone.index().body)
                embedded = importlib.import_module("app.web.embedded")
                self.assertIn("Whisper Widget Web", embedded.INDEX_HTML)
                self.assertIsNone(service.vad_client._process)
                self.assertIsNone(service.whisper_client._process)
            finally:
                service.close()
                sys.modules.pop("app.web.standalone", None)

    def test_supervisor_retains_crash_exit_code(self):
        child = SimpleNamespace(pid=424242, returncode=23, poll=lambda: 23)
        with (
            patch.object(supervisor, "launch_child", return_value=child),
            patch.object(supervisor, "heartbeat_snapshot", return_value=None),
            patch.object(supervisor, "remove_file"),
            patch.object(supervisor.os, "makedirs"),
            patch.object(supervisor, "log"),
            patch.object(supervisor, "MAX_CRASHES_IN_WINDOW", 0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(supervisor.main(), 23)


if __name__ == "__main__":
    unittest.main()
