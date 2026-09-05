# Whisper Widget HTTP API

This document is for agents and applications that need transcription without using the browser recorder. The API is served by the running desktop Whisper widget and is reachable from devices permitted by the tailnet policy.

## Connection details

Current tailnet base URL:

```text
https://whisper-host-tail64165-ts-net.tail913055.ts.net
```

Local base URL on the Whisper host:

```text
http://127.0.0.1:8765
```

The tailnet URL is provided by Tailscale Serve and currently proxies to the local widget. A client must be connected to the same tailnet and allowed by its Tailscale ACLs/grants.

The application does not currently implement API keys, bearer tokens, or per-client authorization. Tailscale is the access boundary. Do not expose this service publicly without adding an authentication layer.

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

Use `POST /api/vad` when a client needs the service's speech/no-speech decision without running Whisper. This endpoint uses the same VAD worker and settings as the transcription pipeline.

## Health and availability

### `GET /health`

Returns the widget and worker state.

```bash
curl --fail-with-body \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/health
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
  "model": "large-v3",
  "device": "cuda:1",
  "worker_pid": 28216,
  "launcher_pid": 9680
}
```

`ok` means the web application is responding. Before submitting work, inspect `busy`. `model_ready` and `vad_ready` report whether the corresponding workers are already warm; a request may start workers when needed.

## One-shot transcription

### `POST /api/transcribe`

Send a `multipart/form-data` request with one required field:

| Field | Type | Description |
| --- | --- | --- |
| `audio` | File | Audio or a media container containing audio. |

The upload limit is 200 MiB. The server uses `ffmpeg` to convert the input to 16 kHz, mono, signed 16-bit WAV before VAD and Whisper processing. Common formats supported by the installed `ffmpeg`, including WAV, MP3, M4A, FLAC, OGG, and WebM/Opus, can be submitted.

The endpoint is synchronous. Use a client timeout of at least 10 minutes unless the caller has stricter requirements.

### curl

```bash
curl --fail-with-body \
  --max-time 600 \
  -F "audio=@recording.wav" \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/transcribe
```

### Python

```python
from pathlib import Path

import requests

BASE_URL = "https://whisper-host-tail64165-ts-net.tail913055.ts.net"
AUDIO_PATH = Path("recording.wav")

health = requests.get(f"{BASE_URL}/health", timeout=10)
health.raise_for_status()
state = health.json()
if state.get("busy"):
    raise RuntimeError("Whisper widget is busy; retry later")

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
        f"Whisper returned HTTP {response.status_code} with a non-JSON body"
    ) from exc

if response.status_code == 409 and payload.get("busy"):
    raise RuntimeError("Whisper widget became busy; retry later")

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
  "speech_secs": 8.4,
  "timings": {
    "vad_seconds": 0.18,
    "whisper_seconds": 2.03,
    "total_seconds": 2.25,
    "convert_seconds": 0.04
  },
  "attempts": [
    {
      "attempt": 1,
      "stage": "speech_only",
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
| `speech_secs` | Speech duration detected by VAD. A negative value indicates VAD failed and raw-audio fallback was used. |
| `timings` | Server-side phase durations in seconds. |
| `attempts` | Whisper retry/fallback attempt summaries. |
| `request_id` | Identifier to correlate the request with host logs. |
| `error` | Failure description, present when applicable. |
| `busy` | Present and true when the shared widget pipeline is occupied. |

Clients should tolerate additional response fields.

## VAD-only analysis

### `POST /api/vad`

This endpoint runs the service's existing Silero VAD without invoking Whisper. It is intended for remote preflight gates that need a stronger decision than a cheap local silence check.

Send one `audio` file in `multipart/form-data`, just as for one-shot transcription. The same 200 MiB upload limit and `ffmpeg` input-format support apply.

```bash
curl --fail-with-body \
  --max-time 240 \
  -F "audio=@recording.wav" \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/vad
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
- `speech_detected: false` is a successful VAD decision. Whisper is not invoked by `/api/vad`, including for no-speech responses.
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

### 1. Read the required audio configuration

`GET /api/chunked/config`

```bash
curl --fail-with-body \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/config
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
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/start
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
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/SESSION_ID/chunks
```

The response is the current session status. Upload chunks sequentially; do not issue parallel chunk uploads.

### 4. Finish and poll

After the last upload, request completion:

`POST /api/chunked/{session_id}/finish`

```bash
curl --fail-with-body -X POST \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/SESSION_ID/finish
```

Alternatively, upload the last chunk with `is_final=true`. In either case, poll once per second:

`GET /api/chunked/{session_id}`

```bash
curl --fail-with-body \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/SESSION_ID
```

While work is in progress, `partial_text` contains the best currently stitched transcript. Stop polling when `final` becomes `true`. A successful terminal response has `status: "complete"` and the transcript in `final_text`. Terminal failure states are `failed` and `cancelled`; inspect `error`.

### Cancel a session

`DELETE /api/chunked/{session_id}`

```bash
curl --fail-with-body -X DELETE \
  https://whisper-host-tail64165-ts-net.tail913055.ts.net/api/chunked/SESSION_ID
```

Agents should cancel a session they abandon so the shared pipeline is released promptly.

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

- Connect the agent host to the tailnet and confirm the base URL resolves.
- Verify `GET /health` before sending audio.
- Send multipart form data; do not JSON-encode audio.
- Use `/api/vad` for the service-quality speech gate, `/api/transcribe` for existing files, and `/api/chunked/*` only for incremental capture.
- Set a long client timeout for one-shot requests.
- Treat HTTP `409` as transient busy state.
- Accept an empty successful transcript as valid no-speech output.
- Log the returned `request_id` when diagnosing failures.
- Do not assume the model name, device, or chunk configuration is permanent.
- Do not rely on server-local diagnostic paths returned in optional fields.
