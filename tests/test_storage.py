"""Verify stored output and recovery survive the generated-data reorganization."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import desktop
from app.paths import (
    CHUNKED_TEMP_DIR, LOG_DIR, MIGRATION_MARKER, RAW_AUDIO_BACKUP_DIR, DATA_PLACEHOLDERS,
    RECORDING_WORK_DIR, RUNTIME_STATE_DIR, WIDGET_RECOVERY_FILE, check_data_layout,
)
from app.transcription_service import TranscriptionService
from tools.migrate_data import migrate


class StorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="whisper-storage-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def write(self, relative, content=b"saved audio"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def recovery_state(self, **fields):
        state = {"active": True, **fields}
        self.write("sidecache/runtime/widget_recovery.json", json.dumps(state).encode())
        return state

    def candidate(self):
        widget = SimpleNamespace(
            runtime_recovery_path=str(self.root / RUNTIME_STATE_DIR / WIDGET_RECOVERY_FILE),
            chunked_temp_dir=str(self.root / CHUNKED_TEMP_DIR),
            temp_filename=str(self.root / RECORDING_WORK_DIR / "temp_recording.wav"),
            log_event=lambda *args, **kwargs: None,
            clear_recovery_state=lambda **kwargs: self.fail("Recovery state must not be cleared"),
        )
        widget.parse_chunk_recovery_entry = desktop.WhisperWidget.parse_chunk_recovery_entry.__get__(widget)
        return desktop.WhisperWidget.load_startup_recovery_candidate(widget)

    def test_service_writes_transcripts_and_logs_in_new_directories(self):
        service = TranscriptionService(base_dir=str(self.root))
        try:
            service.log_transcription("storage check")
            transcripts = list((self.root / LOG_DIR).glob("*.txt"))
            self.assertEqual(len(transcripts), 1)
            self.assertIn("storage check", transcripts[0].read_text(encoding="utf-8"))
            log = self.root / "runtime/logs/web_event_log.txt"
            self.assertIn("web_service_initialized", log.read_text(encoding="utf-8"))
            self.assertEqual(Path(service.work_dir), self.root / "runtime/work/web")
            self.assertEqual({path.name for path in self.root.iterdir()}, {"data", "runtime"})
        finally:
            service.close()

    def test_preview_does_not_change_files(self):
        source = self.write("transcriptions/example.txt", b"saved transcript")
        self.assertEqual(migrate(self.root), 1)
        self.assertEqual(source.read_bytes(), b"saved transcript")
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "runtime").exists())
        with self.assertRaisesRegex(RuntimeError, "Legacy generated data"):
            check_data_layout(self.root)
        with self.assertRaisesRegex(RuntimeError, "Legacy generated data"):
            TranscriptionService(base_dir=str(self.root))

    def test_migration_preserves_chunk_recovery(self):
        audio = self.write("chunked_transcription_work/chunk_0000_0_16000.wav")
        self.recovery_state(mode="chunked", chunk_dir=str(audio.parent))
        migrate(self.root, apply=True)
        candidate = self.candidate()
        self.assertEqual(candidate["kind"], "chunked")
        self.assertEqual(Path(candidate["chunk_dir"]), self.root / CHUNKED_TEMP_DIR)
        self.assertEqual(Path(candidate["chunks"][0]["path"]).read_bytes(), b"saved audio")
        self.assertFalse(audio.exists())
        check_data_layout(self.root)
        journals = list((self.root / "runtime").glob("data_migration_completed_*.json"))
        self.assertEqual(len(journals), 1)
        journal = json.loads(journals[0].read_text())
        original = json.loads(journal["original_state"]["sidecache/runtime/widget_recovery.json"])
        self.assertEqual(original["chunk_dir"], str(audio.parent))

    def test_migration_preserves_raw_backup_recovery(self):
        audio = self.write("raw_audio_backups/recording.wav")
        self.recovery_state(mode="single", raw_backup_path=str(audio))
        migrate(self.root, apply=True)
        candidate = self.candidate()
        self.assertEqual(candidate["kind"], "raw_backup")
        self.assertEqual(Path(candidate["raw_path"]), self.root / RAW_AUDIO_BACKUP_DIR / audio.name)
        self.assertEqual(Path(candidate["raw_path"]).read_bytes(), b"saved audio")

    def test_migration_preserves_temporary_recording_recovery(self):
        self.write("temp_recording.wav")
        self.recovery_state(mode="single")
        migrate(self.root, apply=True)
        candidate = self.candidate()
        self.assertEqual(candidate["kind"], "temp_recording")
        self.assertEqual(Path(candidate["raw_path"]).read_bytes(), b"saved audio")

    def test_rotated_logs_and_failed_audio_move_without_content_changes(self):
        self.write("event_log.2.txt", b"old diagnostic")
        self.write("failed_speech_only_example.wav")
        self.write("sidecache/runtime/python_fatal.log", b"fatal diagnostic")
        migrate(self.root, apply=True)
        self.assertEqual((self.root / "runtime/logs/event_log.2.txt").read_bytes(), b"old diagnostic")
        self.assertEqual((self.root / "runtime/logs/python_fatal.log").read_bytes(), b"fatal diagnostic")
        self.assertEqual((self.root / "data/failed_recordings/failed_speech_only_example.wav").read_bytes(),
                         b"saved audio")
        self.assertFalse((self.root / "sidecache").exists())

    def test_migration_preserves_tracked_placeholders_and_recovery(self):
        for directory, filename in DATA_PLACEHOLDERS.items():
            self.write(directory / filename, b"tracked placeholder")
        audio = self.write("raw_audio_backups/recording.wav")
        self.write("transcriptions/example.txt", b"saved transcript")
        self.write("debug_audio/capture/raw.wav", b"saved debug audio")
        self.recovery_state(mode="single", raw_backup_path=str(audio))

        migrate(self.root, apply=True)

        for directory, filename in DATA_PLACEHOLDERS.items():
            self.assertEqual((self.root / directory / filename).read_bytes(), b"tracked placeholder")
        self.assertEqual((self.root / "data/transcriptions/example.txt").read_bytes(), b"saved transcript")
        self.assertEqual((self.root / "data/debug_audio/capture/raw.wav").read_bytes(), b"saved debug audio")
        self.assertEqual(Path(self.candidate()["raw_path"]).read_bytes(), b"saved audio")
        self.assertFalse(audio.parent.exists())
        check_data_layout(self.root)

    def test_empty_legacy_directory_can_migrate_into_placeholder_directory(self):
        placeholder = self.write(LOG_DIR / DATA_PLACEHOLDERS[LOG_DIR], b"tracked placeholder")
        legacy = self.root / "transcriptions"
        legacy.mkdir()
        self.assertEqual(migrate(self.root, apply=True), 1)
        self.assertFalse(legacy.exists())
        self.assertEqual(placeholder.read_bytes(), b"tracked placeholder")
        check_data_layout(self.root)

    def test_collision_is_detected_before_any_moves(self):
        original = self.write("transcriptions/example.txt", b"original")
        self.write("raw_audio_backups/recording.wav")
        destination = self.write("data/raw_audio_backups/recording.wav", b"existing")
        self.write(RAW_AUDIO_BACKUP_DIR / DATA_PLACEHOLDERS[RAW_AUDIO_BACKUP_DIR], b"tracked placeholder")
        with self.assertRaises(FileExistsError):
            migrate(self.root, apply=True)
        self.assertEqual(original.read_bytes(), b"original")
        self.assertEqual(destination.read_bytes(), b"existing")
        self.assertFalse((self.root / MIGRATION_MARKER).exists())

    def test_invalid_recovery_json_prevents_moves(self):
        audio = self.write("raw_audio_backups/recording.wav")
        self.write("sidecache/runtime/widget_recovery.json", b"{bad json")
        with self.assertRaises(ValueError):
            migrate(self.root, apply=True)
        self.assertTrue(audio.exists())
        self.assertFalse((self.root / MIGRATION_MARKER).exists())

    def test_recovery_path_from_another_checkout_prevents_moves(self):
        audio = self.write("raw_audio_backups/recording.wav")
        self.recovery_state(raw_backup_path=str(self.root.parent / "another-checkout/recording.wav"))
        with self.assertRaisesRegex(ValueError, "outside this checkout"):
            migrate(self.root, apply=True)
        self.assertTrue(audio.exists())

    def test_partial_migration_leaves_journal_and_blocks_startup(self):
        self.write("transcriptions/example.txt", b"original")
        self.write("raw_audio_backups/recording.wav")
        rename = Path.rename

        def fail_audio_move(source, target):
            if source.name == "raw_audio_backups":
                raise PermissionError("simulated locked recording")
            return rename(source, target)

        with patch.object(Path, "rename", fail_audio_move):
            with self.assertRaisesRegex(PermissionError, "locked recording"):
                migrate(self.root, apply=True)
        self.assertEqual((self.root / "data/transcriptions/example.txt").read_bytes(), b"original")
        self.assertEqual((self.root / "raw_audio_backups/recording.wav").read_bytes(), b"saved audio")
        self.assertTrue((self.root / MIGRATION_MARKER).exists())
        with self.assertRaisesRegex(RuntimeError, "unfinished data migration"):
            check_data_layout(self.root)


if __name__ == "__main__":
    unittest.main()
