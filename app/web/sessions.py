"""Serialized remote recording sessions and restart-safe status records."""
import json
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import threading
import time
import uuid

from app.state import write_json_atomic
from app.web.recording_audio import RecordingAudio
from app.web.audio import save_chunk_upload, validate_chunk_wav


def datetime_session_id():
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RemoteChunkSession:
    def __init__(self, manager, session_id, config):
        self.manager = manager
        self.backend = manager.backend
        self.session_id = session_id
        self.session_dir = os.path.join(manager.work_dir, "chunked_sessions", session_id)
        self.config = dict(config)
        self.lock = threading.RLock()
        self.queue = Queue()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True,
                                       name=f"web_chunk_session_{session_id}")
        self.created_at = self.updated_at = time.time()
        self.status = "recording"
        self.error = None
        self.next_expected_index = 0
        self.chunks_received = 0
        self.chunks_transcribed = 0
        self.final_chunk_seen = False
        self.final = False
        self.manager_finished = False
        self.cancel_requested = False
        self.partial_text = ""
        self.final_text = ""
        self.results = {}
        self.received_chunks = {}
        self.audio = None
        self.buffered_error = None

    def status_payload(self):
        with self.lock:
            return {
                "ok": self.error is None,
                "session_id": self.session_id,
                "status": self.status,
                "error": self.error,
                "chunks_received": self.chunks_received,
                "chunks_transcribed": self.chunks_transcribed,
                "partial_text": self.partial_text,
                "final_text": self.final_text,
                "final": self.final,
                "config": dict(self.config),
            }

    def persist(self):
        # Call with the session lock held. Audio paths are relative to the data root.
        payload = self.status_payload()
        payload.update(created_at=self.created_at, updated_at=self.updated_at,
                       chunks=[{**chunk, "path": os.path.basename(chunk["path"])}
                               for chunk in self.received_chunks.values()])
        write_json_atomic(os.path.join(self.manager.state_dir, self.session_id + ".json"), payload)

    def start(self):
        os.makedirs(self.session_dir, exist_ok=True)
        self.audio = RecordingAudio(self.session_dir, self.session_id,
                                    self.backend.find_pauses, self.manager.log_event)
        with self.lock:
            self.persist()
        self.thread.start()

    def enqueue_chunk(self, chunk_index, start_sample, end_sample, is_final, upload):
        sample_count = end_sample - start_sample
        if start_sample < 0 or sample_count <= 0:
            raise ValueError("Chunk range must be nonnegative with end_sample greater than start_sample")
        # Keep validation, staging, and acceptance atomic against concurrent uploads/cancel.
        with self.lock:
            if self.final or self.final_chunk_seen or self.cancel_requested:
                raise ValueError("Session is no longer accepting chunks")
            if chunk_index != self.next_expected_index:
                raise ValueError(f"Unexpected chunk index {chunk_index}; expected {self.next_expected_index}")
            previous = self.received_chunks.get(chunk_index - 1)
            if (previous is None and start_sample != 0) or (previous is not None and
                    (start_sample > previous["end_sample"] or end_sample <= previous["end_sample"])):
                raise ValueError("Chunk coordinates must extend the continuous recording timeline")
            filename = f"chunk_{chunk_index:04d}_{start_sample:010d}_{end_sample:010d}.wav"
            chunk_path = os.path.join(self.session_dir, filename)
            staging_path = chunk_path + ".upload"
            try:
                upload_bytes = save_chunk_upload(upload, staging_path)
                validate_chunk_wav(staging_path, int(self.config["sample_rate"]),
                                   int(self.config["channels"]), int(self.config["sample_width"]),
                                   sample_count)
                os.replace(staging_path, chunk_path)
            finally:
                if os.path.exists(staging_path):
                    os.remove(staging_path)
            chunk = {
                "index": chunk_index, "path": chunk_path,
                "start_sample": start_sample, "end_sample": end_sample,
                "seconds": sample_count / float(self.config["sample_rate"]),
                "is_final": bool(is_final),
            }
            self.next_expected_index += 1
            self.chunks_received += 1
            self.received_chunks[chunk_index] = chunk
            self.updated_at = time.time()
            self.status = "processing"
            self.final_chunk_seen = bool(is_final)
            try:
                self.persist()
            except Exception:
                # An unacknowledged chunk is retained, but never queued for inference.
                self.next_expected_index -= 1
                self.chunks_received -= 1
                del self.received_chunks[chunk_index]
                self.final_chunk_seen = False
                raise
            self.queue.put(chunk)
        self.manager.log_event("web_chunk_received", session_id=self.session_id,
                               chunk_index=chunk_index, bytes=upload_bytes, file=chunk_path)
        return self.status_payload()

    def finish(self):
        with self.lock:
            if not self.final and not self.final_chunk_seen and not self.cancel_requested:
                self.final_chunk_seen = True
                self.status = "processing"
                self.updated_at = time.time()
                try:
                    self.persist()
                except Exception:
                    self.final_chunk_seen = False
                    raise
                self.queue.put({"finish": True})
        return self.status_payload()

    def cancel(self, reason="cancelled"):
        with self.lock:
            if not self.final:
                self.cancel_requested = True
                self.status = "cancelled"
                self.error = reason
                self.updated_at = time.time()
                try:
                    self.persist()
                finally:
                    self.queue.put(None)
        # The worker retains the pipeline lock until in-flight inference has stopped.
        return self.status_payload()

    def _worker_loop(self):
        success = False
        try:
            self.backend.whisper_client.ensure_ready()
            while True:
                chunk = self.queue.get()
                try:
                    with self.lock:
                        if self.cancel_requested or chunk is None:
                            self.final = True
                            self.persist()
                            return
                    windows = self.audio.finish() if chunk.get("finish") else self.audio.add(chunk)
                    for window in windows:
                        if self.cancel_requested or self.buffered_error:
                            break
                        try:
                            result = self.backend.process_and_transcribe_chunk(window)
                        except Exception as error:
                            result = {"ok": False, "error": str(error), "chunk_index": window["index"]}
                        if not result.get("ok"):
                            self.buffered_error = result.get("error") or "Buffered inference failed"
                            self.manager.log_event("bark_recording_recovery_required", session_id=self.session_id,
                                                   error=self.buffered_error)
                            break
                        self.results[window["index"]] = result
                    with self.lock:
                        if self.cancel_requested:
                            self.final = True
                            self.persist()
                            return
                        # Public progress counts transport uploads whose entire
                        # range has been decoded, never internal inference cores.
                        decoded_end = max((result.get("buffered", {}).get("core_end", 0)
                                           for result in self.results.values()), default=0)
                        self.chunks_transcribed = sum(item["end_sample"] <= decoded_end
                                                      for item in self.received_chunks.values())
                        self.partial_text = self.backend.stitch_chunk_transcripts(list(self.results.values()), require_final=False)
                        self.updated_at = time.time()
                        self.persist()
                    if chunk.get("finish") or chunk.get("is_final"):
                        success = self._complete_success()
                        return
                finally:
                    self.queue.task_done()
        except Exception as error:
            with self.lock:
                self.status = "cancelled" if self.cancel_requested else "failed"
                self.error = str(error)
                self.final = True
                self.updated_at = time.time()
                try:
                    self.persist()
                except Exception as save_error:
                    self.manager.log_event("web_chunk_state_save_failed", error=save_error)
            self.backend.write_chunk_transcript_debug(list(self.results.values()), "", reason="web_failure")
            self.manager.log_event("web_chunk_session_failed", session_id=self.session_id, error=error)
        finally:
            try:
                # Release abandoned state on cancellation/failure. A failed or
                # timed-out reset terminates the worker rather than leaking state.
                self.backend.whisper_client.reset_stream()
            except Exception as error:
                self.manager.log_event("bark_stream_reset_failed", error=error)
                # Terminate the failed worker without permanently closing the
                # client; a later recording may start a fresh worker.
                self.backend.whisper_client.restart(reason="bark_stream_reset_failed")
            finally:
                self.manager.finish_session(self, success=success)

    def _complete_success(self):
        if self.buffered_error and not self.cancel_requested:
            # A process restart cannot resurrect a decoder cache. Recover once
            # using the full original recording, never a speech-only extract.
            self.backend.whisper_client.reset_stream()
            self.backend.whisper_client.ensure_ready()
            recovered = self.backend.process_and_transcribe_chunk({
                "index": 0, "path": self.audio.recovery_wav(),
            })
            if not recovered.get("ok"):
                raise RuntimeError("Original recording recovery failed: " + str(recovered.get("error")))
            self.results = {0: recovered}
        with self.lock:
            if self.cancel_requested:
                self.final = True
                self.persist()
                return False
            results = [self.results[index] for index in sorted(self.results)]
            failures = sum(not result.get("ok") for result in results)
            if failures:
                raise RuntimeError(f"chunk_transcription_failures={failures}")
            text = self.backend.stitch_chunk_transcripts(results)
            self.backend.write_chunk_transcript_debug(results, text, reason="web_success")
            if text:
                if self.backend.log_transcription(text) is False:
                    raise RuntimeError("Transcript could not be saved; session audio was preserved")
                self.backend.print_transcription_to_terminal(text)
            self.partial_text = self.final_text = text
            self.chunks_transcribed = self.chunks_received
            self.status = "complete"
            self.final = True
            self.updated_at = time.time()
            self.persist()
            return True


class RemoteChunkSessionManager:
    def __init__(self, backend, work_dir, log_fn):
        self.backend = backend
        self.work_dir = work_dir
        self.log_fn = log_fn
        self.lock = threading.Lock()
        self.sessions = {}
        self.active_session_id = None
        self.closing = False
        self.state_dir = os.path.join(work_dir, "session_state")
        os.makedirs(self.state_dir, exist_ok=True)
        self._restore_sessions()

    def log_event(self, event, **fields):
        self.log_fn(event, **fields)

    def chunk_config(self):
        return self.backend.web_chunk_config_payload()

    def _restore_sessions(self):
        for path in Path(self.state_dir).glob("*.json"):
            try:
                if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", path.stem):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                session = RemoteChunkSession(self, path.stem, payload["config"])
                for key in ("status", "error", "chunks_received", "chunks_transcribed",
                            "partial_text", "final_text", "final", "created_at", "updated_at"):
                    setattr(session, key, payload[key])
                session.manager_finished = True
                if not session.final or session.status not in ("complete", "failed", "cancelled"):
                    # Received audio stays on disk. Never imply an interrupted upload is complete.
                    payload.update(status="failed", final=True, ok=False,
                                   error="Service restarted before completion; received audio is preserved. Re-upload the saved recording.",
                                   updated_at=time.time())
                    session.status, session.final, session.error = "failed", True, payload["error"]
                    write_json_atomic(str(path), payload)
                self.sessions[session.session_id] = session
            except Exception as error:
                self.log_event("web_chunk_state_restore_failed", file=str(path), error=error)

    def start_session(self):
        with self.lock:
            if self.closing:
                raise RuntimeError("Service is shutting down")
            if self.active_session_id or self.backend.is_recording or self.backend.is_processing:
                raise RuntimeError("Another recording or transcription is active")
            if not self.backend.web_pipeline_lock.acquire(blocking=False):
                raise RuntimeError("Another audio request is running")
            try:
                session = RemoteChunkSession(self, datetime_session_id(), self.chunk_config())
                self.sessions[session.session_id] = session
                self.active_session_id = session.session_id
                self.backend.processing_started("web_chunk_session_start")
                session.start()
            except Exception:
                self.active_session_id = None
                self.backend.processing_finished()
                self.backend.web_pipeline_lock.release()
                raise
        return session.status_payload()

    def get_session(self, session_id):
        with self.lock:
            return self.sessions[session_id]

    def finish_session(self, session, success):
        with self.lock:
            if session.manager_finished:
                return
            session.manager_finished = True
            if success:
                try:
                    shutil.rmtree(session.session_dir)
                except OSError as error:
                    self.log_event("web_chunk_session_temp_cleanup_failed", error=error)
            if self.active_session_id == session.session_id:
                self.active_session_id = None
            try:
                self.backend.processing_finished()
            finally:
                self.backend.web_pipeline_lock.release()

    def close(self):
        with self.lock:
            self.closing = True
            session = self.sessions.get(self.active_session_id)
        if session:
            try:
                session.cancel("Service shutting down; received audio is preserved")
            finally:
                # Even a failed state write must not close workers underneath inference.
                # Worker requests have finite timeouts; retain ownership until they exit.
                session.thread.join()
