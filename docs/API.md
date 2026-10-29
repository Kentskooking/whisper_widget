# Bark Widget HTTP API

This document describes the full API served by the desktop widget's embedded
server. The standalone diagnostic server offers only the browser/file upload
subset (`/`, `/health`, and `/api/transcribe`).

## Connection details

The default base URL is `http://127.0.0.1:8765`. All examples below target that
local address. Replace it with your own private HTTPS URL for remote clients.
There is no built-in authentication; use an access-controlled proxy or private
network such as Tailscale before allowing remote access. Remote browser
microphone/clipboard use requires a secure context (HTTPS).

Live API descriptions are available at:

- Swagger UI: `<base-url>/docs`
- ReDoc: `<base-url>/redoc`
- OpenAPI JSON: `<base-url>/openapi.json`

## Recommended agent workflow

For a complete audio file, use the one-shot endpoint:

1. Call `GET /health` and wait until `busy` is `false`.
2. Send the file as the `audio` field of a multipart request to `POST /api/transcribe`.
3. Allow several minutes for the request. It remains open until transcription finishes.
4. Require both a successful HTTP status and JSON field `"ok": true`.
5. If the service returns HTTP `409` with `"busy": true`, retry later with bounded backoff. Do not submit concurrent jobs.

Use the chunked API only when audio must be uploaded and transcribed incrementally.

Use `POST /api/vad` when a client needs the service's speech/no-speech decision without running Bark. This uses the separate VAD analysis operation. Transcription preserves all original samples; pause detection only selects chunk boundaries.

## Health and availability

### `GET /health`

Returns the widget and worker state.

```bash
curl --fail-with-body \
  http://127.0.0.1:8765/health
```

Example response:

```json
{
  "ok": true,
  "model_ready": true,
  "vad_ready": true,
  "busy": false,
  "recording": false,
  "processing": false,
  "model": "ConwayResearch/Underdog-Bark-3.0-Pro",
  "device": "cuda:0",
  "worker_pid": 28216,
  "launcher_pid": 9680
}
```

`ok` means the web application is responding. Before submitting work, inspect `busy`. `model_ready` and `vad_ready` report whether the corresponding workers are already warm; a request may start workers when needed.

## Latest completed transcript

### `GET /api/transcripts/latest`

Returns `{"ok": true, "text": "..."}` with the most recent nonempty completed
transcript saved by the desktop widget, a one-shot upload, or a remote chunked
session. All clients of this widget share this history, regardless of which
machine recorded the audio. It survives host restarts through the widget's saved
transcript state.

An empty `text` means the host has no saved transcript yet. The endpoint remains
available while recording or processing; it returns the previous completed text,
never partial chunks. The response uses `Cache-Control: no-store` and does not
change any clipboard, start workers, or run inference. Like the other endpoints,
access follows the host/proxy configuration.

Clients can call this endpoint when their copy button is clicked. A `404` means
the host needs the API update and a widget restart. Network errors should leave
the clipboard untouched rather than silently substitute an older cached copy.

## One-shot transcription

### `POST /api/transcribe`

Send a `multipart/form-data` request with one required field:

| Field | Type | Description |
| --- | --- | --- |
| `audio` | File | Audio or a media container containing audio. |

The upload limit is 200 MiB. The server uses `ffmpeg` to convert the input to 16 kHz, mono, signed 16-bit WAV before Bark processing, preserving silence. Common formats supported by the installed `ffmpeg`, including WAV, MP3, M4A, FLAC, OGG, and WebM/Opus, can be submitted.

The endpoint is synchronous. Use a client timeout of at least 10 minutes unless the caller has stricter requirements.

### curl

```bash
curl --fail-with-body \
  --max-time 600 \
  -F "audio=@recording.wav" \
  http://127.0.0.1:8765/api/transcribe
```

### Python

```python
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8765"
AUDIO_PATH = Path("recording.wav")

health = requests.get(f"{BASE_URL}/health", timeout=10)
health.raise_for_status()
state = health.json()
if state.get("busy"):
    raise RuntimeError("Bark widget is busy; retry later")

with AUDIO_PATH.open("rb") as audio_file:
    response = requests.post(
        f"{BASE_URL}/api/transcribe",
        files={"audio": (AUDIO_PATH.name, audio_file, "application/octet-stream")},
        timeout=600,
    )

try:
    payload = response.json()
except requests.exceptions.JSONDecodeError as exc:
    raise RuntimeError(
        f"Bark returned HTTP {response.status_code} with a non-JSON body"
    ) from exc

if response.status_code == 409 and payload.get("busy"):
    raise RuntimeError("Bark widget became busy; retry later")

response.raise_for_status()
if not payload.get("ok"):
    raise RuntimeError(payload.get("error", "Transcription failed"))

print(payload.get("text", ""))
```

### Success response

```json
{
  "ok": true,
  "text": "The transcription appears here.",
  "speech_secs": null,
  "timings": {
    "whisper_seconds": 2.03,
    "total_seconds": 2.25,
    "convert_seconds": 0.04
  },
  "attempts": [
    {
      "attempt": 1,
      "stage": "original_audio",
      "ok": true,
      "text_chars": 31,
      "seconds": 2.03
    }
  ],
  "request_id": "20260812_143015_123"
}
```

An audio file containing no recognized speech can return `"ok": true` with an empty `text` value. Treat that as a completed transcription with no transcript, not as a transport failure.

Response fields useful to most clients are:

| Field | Meaning |
| --- | --- |
| `ok` | Whether the pipeline completed successfully. |
| `text` | Final transcript; it may be empty. |
| `speech_secs` | `null` for transcription: speech duration is not measured or extracted. `/api/vad` still returns a numeric duration. |
| `timings` | Server-side durations in seconds. `whisper_seconds` is retained as the compatibility name for Bark inference time. |
| `attempts` | Bark retry/fallback attempt summaries. |
| `request_id` | Identifier to correlate the request with host logs. |
| `error` | Failure description, present when applicable. |
| `busy` | Present and true when the shared widget pipeline is occupied. |

Clients should tolerate additional response fields.

## VAD-only analysis

### `POST /api/vad`

This endpoint runs the service's existing Silero VAD without invoking Bark. It is intended for remote preflight gates that need a stronger decision than a cheap local silence check.

Send one `audio` file in `multipart/form-data`, just as for one-shot transcription. The same 200 MiB upload limit and `ffmpeg` input-format support apply.

```bash
curl --fail-with-body \
  --max-time 240 \
  -F "audio=@recording.wav" \
  http://127.0.0.1:8765/api/vad
```

Example speech response:

```json
{
  "ok": true,
  "speech_detected": true,
  "speech_secs": 8.4,
  "segments": [
    {
      "start_seconds": 0.72,
      "end_seconds": 9.12
    }
  ],
  "timings": {
    "vad_seconds": 0.18,
    "total_seconds": 0.19,
    "convert_seconds": 0.04
  },
  "method": "existing_widget_vad",
  "request_id": "20260812_143015_123"
}
```

Example no-speech response:

```json
{
  "ok": true,
  "speech_detected": false,
  "speech_secs": 0.0,
  "segments": [],
  "timings": {
    "vad_seconds": 0.12,
    "total_seconds": 0.13,
    "convert_seconds": 0.03
  },
  "method": "existing_widget_vad",
  "request_id": "20260812_143120_456"
}
```

The optional `segments` array contains the padded and merged ranges selected by the service VAD. Each item has numeric `start_seconds` and `end_seconds` values relative to the uploaded audio. `method` is currently `existing_widget_vad`.

Decision rules for clients:

- Trust `speech_detected` only when the HTTP request succeeds and `ok` is `true`.
- `speech_detected: false` is a successful VAD decision. Bark is not invoked by `/api/vad`, including for no-speech responses.
- `speech_detected: null` accompanies a busy or failed request and is not a no-speech decision.
- `speech_secs` is the selected duration after the service's padding and merging rules, not the original file duration.
- VAD-only failures are returned to the caller without the transcription pipeline's raw-audio fallback, so failure cannot be mistaken for silence.
- The endpoint uses the shared audio pipeline. HTTP `409` means desktop recording, transcription, a chunked session, or another VAD request is active.

Recommended robust gate:

1. Optionally reject obvious digital silence with a cheap local gate.
2. For remaining audio, call `/api/vad`.
3. Skip `/api/transcribe` only after HTTP `200`, `ok: true`, and `speech_detected: false`.
4. If speech is detected, call `/api/transcribe` and retain the caller's post-response transcript trust gate.
5. If VAD is busy or fails, follow explicit caller policy; never silently substitute a weaker local detector or reinterpret the failure as no speech.

## Chunked transcription

The chunked API is intended for long-running or live capture. It accepts sequential PCM WAV chunks, transcribes them in the background, and exposes partial and final stitched text by polling.

Only one chunked session can be active at a time. It also shares the widget pipeline with desktop recording and one-shot API calls.
The server reconstructs a continuous original PCM timeline from declared overlap.
Overlapped samples must match exactly; gaps or changed overlap fail the session.
Transport uploads remain roughly 30 seconds; internal pause boundaries and
10-second decoder cores do not change the upload contract. Audio is never
speech-trimmed. `partial_text` comes from cumulative tokens, without guessed word
deduplication. `chunks_transcribed` counts fully decoded transport ranges, not
internal windows. Right context may arrive with the next upload; it is not a
fixed latency guarantee. Finish flushes the available tail immediately.

Session statuses survive restart. An interrupted recording is marked failed and
its received audio is retained; re-upload a saved original recording rather than
continuing a lost decoder cache. A buffered inference failure can recover from
the reconstructed original audio at completion. Successful final transcripts
continue to update the widget's shared history.

### 1. Read the required audio configuration

`GET /api/chunked/config`

```bash
curl --fail-with-body \
  http://127.0.0.1:8765/api/chunked/config
```

Current response:

```json
{
  "ok": true,
  "config": {
    "sample_rate": 16000,
    "channels": 1,
    "sample_width": 2,
    "chunk_seconds": 30.0,
    "overlap_seconds": 1.5,
    "chunk_samples": 480000,
    "overlap_samples": 24000
  }
}
```

Always consume the returned configuration rather than hard-coding these values.

### 2. Start a session

`POST /api/chunked/start`

The request has no body.

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8765/api/chunked/start
```

The response includes a `session_id`, current status, and the session configuration:

```json
{
  "ok": true,
  "session_id": "20260812_143015_a1b2c3d4",
  "status": "recording",
  "error": null,
  "chunks_received": 0,
  "chunks_transcribed": 0,
  "partial_text": "",
  "final_text": "",
  "final": false,
  "config": {
    "sample_rate": 16000,
    "channels": 1,
    "sample_width": 2,
    "chunk_seconds": 30.0,
    "overlap_seconds": 1.5,
    "chunk_samples": 480000,
    "overlap_samples": 24000
  }
}
```

Save the `session_id` for all remaining calls. HTTP `409` means the widget or chunked pipeline is already busy.

### 3. Upload sequential chunks

`POST /api/chunked/{session_id}/chunks`

Each request is `multipart/form-data`:

| Field | Type | Description |
| --- | --- | --- |
| `chunk_index` | Integer | Zero-based index. Indices must arrive strictly in order. |
| `start_sample` | Integer | Inclusive position of the first sample in the complete recording. |
| `end_sample` | Integer | Exclusive position after the last sample. |
| `is_final` | Boolean | Optional; defaults to `false`. |
| `audio` | File | PCM WAV matching the session configuration exactly. |

The WAV frame count must equal `end_sample - start_sample`. With the current configuration, files must be mono, 16 kHz, 16-bit PCM WAV. These constraints are validated by the server.

For overlapped chunks, advance the next `start_sample` by `chunk_samples - overlap_samples`. With the current values, full chunk ranges are `0..480000`, `456000..936000`, and so on. The last chunk may contain fewer samples.

```bash
curl --fail-with-body \
  -F "chunk_index=0" \
  -F "start_sample=0" \
  -F "end_sample=480000" \
  -F "is_final=false" \
  -F "audio=@chunk_0000.wav;type=audio/wav" \
  http://127.0.0.1:8765/api/chunked/SESSION_ID/chunks
```

The response is the current session status. Upload chunks sequentially; do not issue parallel chunk uploads.

### 4. Finish and poll

After the last upload, request completion:

`POST /api/chunked/{session_id}/finish`

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8765/api/chunked/SESSION_ID/finish
```

Alternatively, upload the last chunk with `is_final=true`. In either case, poll once per second:

`GET /api/chunked/{session_id}`

```bash
curl --fail-with-body \
  http://127.0.0.1:8765/api/chunked/SESSION_ID
```

While work is in progress, `partial_text` contains the current cumulative decoder transcript. Stop polling when `final` becomes `true`. A successful terminal response has `status: "complete"` and the transcript in `final_text`. Terminal failure states are `failed` and `cancelled`; inspect `error`.

### Cancel a session

`DELETE /api/chunked/{session_id}`

```bash
curl --fail-with-body -X DELETE \
  http://127.0.0.1:8765/api/chunked/SESSION_ID
```

Cancel abandoned sessions. Cancellation is requested immediately, but the shared pipeline remains busy until in-flight inference stops and decoder state is released. Poll for `final: true` and wait for `/health` to report `busy: false` before starting another recording.

## HTTP errors and retry policy

| Status | Meaning | Client action |
| --- | --- | --- |
| `200` | Request was accepted or completed. | Still inspect `ok`, `final`, and `status`. |
| `400` | Invalid chunk order, sample range, WAV properties, or session state. | Correct the request; do not retry unchanged. |
| `404` | Unknown chunked session ID. | Start a new session if appropriate. |
| `409` | Widget or shared transcription pipeline is busy. | Retry later with bounded exponential backoff and jitter. |
| `422` | Required path or multipart fields failed validation. | Correct the request; do not retry unchanged. |
| `500` | Conversion, worker, or transcription failure. | Record `error` and `request_id`; retry only according to caller policy. |

Use bounded retries. A suggested busy retry schedule is approximately 2, 4, 8, 16, and 30 seconds with jitter. Avoid parallel retries because the service serializes transcription work.

## Integration checklist

- Configure the host or private HTTPS route and confirm the base URL resolves.
- Verify `GET /health` before sending audio.
- Send multipart form data; do not JSON-encode audio.
- Use `/api/vad` for the service-quality speech gate, `/api/transcribe` for existing files, and `/api/chunked/*` only for incremental capture.
- Set a long client timeout for one-shot requests.
- Treat HTTP `409` as transient busy state.
- Accept an empty successful transcript as valid no-speech output.
- Log the returned `request_id` when diagnosing failures.
- Do not assume the model name, device, or chunk configuration is permanent.
- Do not rely on server-local diagnostic paths returned in optional fields.
