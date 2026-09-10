# Shared transcript copy implementation plan

Status: host endpoint implemented; widget restart is left to the user.
Target checkout: `D:\EDEN\Python Gui\Whisper_web_server\whisper_widget`, branch `simple-nvidia`.
Upstream reference: [baa93655eaa13d608d2a26c05239ba68210f2e19](https://github.com/Kentskooking/whisper_widget/commit/baa93655eaa13d608d2a26c05239ba68210f2e19).

## Intended result

Speak into the microphone attached to this host, finish the transcription, then click Copy on another connected machine to put the host's latest completed text onto that machine's clipboard. Each client reads the same shared latest transcript from this host.

This is a fetch-on-copy workflow. The server provides text; each client performs its own clipboard write. Automatic clipboard synchronization, background broadcast, and transcript-history browsing are separate features.

## Implementation handoff

- Added `GET /api/transcripts/latest` to the existing embedded server, with the upstream response and no-store header.
- The completion-order regression reproduced stale shared text after a remote session reported completion. Its final transcript is now saved before publishing the completed session state, under the existing session lock.
- Extended clipboard-history tests and updated `API.md` and `README.md`.
- The user confirmed that a separate lightweight wrapper is maintained by the remote agent. Client and browser changes below are outside this implementation.
- Restart using `launch_whisper_widget.bat` in this D-drive checkout after the existing widget has exited. This implementation does not restart it automatically.

## Confirmed baseline

- Local HEAD is `5ecdff5`; fetched `origin/main` is `baa9365`, five commits ahead. Preserve the existing uncommitted copy-button changes in `whisper_widget.py`, `README.md`, and `tests/test_clipboard_history.py`.
- The prior button backport already supplies `last_transcription`, its lock, atomic persistence, and startup loading from `sidecache/runtime/last_transcript.txt`. That state file exists.
- Desktop transcription, recovery, one-shot API transcription, and completed remote chunk sessions all call `WhisperWidget.log_transcription()`, which calls `remember_transcription()`. Empty results leave the previous nonempty text available.
- Local `widget_web_server.py` is identical to upstream `app/web/embedded.py` immediately before `baa9365` (Git blob `1916fd37b2dccfb4a4faedec17fa588677f8bc28`).
- Commit `baa9365` adds 11 production lines for one GET route, plus tests and documentation. It does not change any client's Copy button.
- GET-only checks found the live service healthy at `http://127.0.0.1:8765`, with model and VAD ready. Its OpenAPI schema does not yet include the new route.
- The listener runs this D-drive widget through a process launched from this checkout's `.venv\Scripts\python.exe`. The environment is Python 3.10.11.
- Baseline verification passed: `tools/check_all.ps1` and all four existing clipboard-history tests.

## Port map

| Upstream location | Current checkout | Planned change |
| --- | --- | --- |
| `app/web/embedded.py` | `widget_web_server.py` | Add the latest-transcript route in `WhisperWidgetWebServer.create_app()`, beside `/health`. |
| `app/desktop.py` | `whisper_widget.py` | Reuse existing copy-history state; no additional desktop feature is required. |
| `tests/test_clipboard_history.py` | Same path | Adapt upstream API regression tests to local imports and the existing temporary-log-directory fixture. |
| `docs/API.md` | `API.md` | Document the route, empty-history behavior, client errors, and fetch-on-copy flow. |
| `README.md` | Same path | Add the short shared-transcript API description. |

Apply these edits selectively. Keep Python, requirements, folder layout, launchers, model settings, bind address, port, and Tailscale configuration as they are. The standalone `web_server.py` has a separate service object and is not the running widget's embedded server; do not add a disconnected history endpoint there.

## 1. Add the host endpoint

Add `GET /api/transcripts/latest` using the upstream implementation:

1. Acquire `self.widget.last_transcription_lock`.
2. Snapshot `self.widget.last_transcription` and release the lock.
3. Return HTTP 200 JSON `{"ok": true, "text": "..."}` with `Cache-Control: no-store`.

The route returns the latest saved, nonempty completed transcript, regardless of which connected recorder produced it. Before any transcript exists, it returns an empty string with `ok: true`. During another recording or transcription, it returns the previous completed text. It must not require the pipeline to be idle, invoke workers, or write any clipboard.

Use the existing state file and access boundary. No additional dependency, microphone, model process, database, open port, or migration is needed.

## 2. Connect the client copy actions

Remote agent/client integration:

1. On an explicit Copy action, GET `<configured Whisper base URL>/api/transcripts/latest`, bypassing caches and using a bounded request timeout (start with 5 seconds).
2. Require HTTP success, `ok: true`, and a string `text`.
3. For nonempty text, write it using that client's existing clipboard mechanism and show success only after the write succeeds.
4. For empty text, show "No completed transcript yet" and leave the clipboard alone.
5. For HTTP 404, report that the host update/restart is required. For network, malformed-response, or clipboard errors, report the failure and preserve the clipboard; do not silently copy an older cached transcript.
6. Prevent overlapping copy requests so a slower earlier response cannot overwrite a newer clipboard write.

Do not wait for `/health.busy` to clear before reading this endpoint. All clients must target the same Whisper host to share its history. Each host maintains one latest-text value; the contract has no transcript ID, timestamp, push subscription, or queue of older transcripts.

The documented remote URL is `https://whisper-host-tail64165-ts-net.tail913055.ts.net`. Use each client's existing working Whisper base URL and Tailscale access; remote reachability has not been tested from another machine in this planning pass.

### Existing browser recorder

The embedded page's current `copyTranscript()` copies only `transcriptEl.value`, and its Copy button is disabled when that page has no text. Therefore, adding the API alone will not make the existing browser page copy host-microphone transcripts.

If this page is one of the intended clients, add a separate "Copy latest from host" action in `widget_web_server.py`'s `INDEX_HTML`. Keep it usable before a local recording and while the host is busy. Fetch the shared route on click and use the existing browser clipboard mechanism. Preserve the current page-transcript Copy action for copying displayed or edited text. Handle clipboard failure by presenting the fetched text for manual copy, without replacing an active recording's partial-transcript display.

No external client codebases were inspected here. Their exact file edits belong in the remote agent's implementation after identifying which apps own those copy buttons.

## 3. Validate the selective port

Extend `tests/test_clipboard_history.py` with the upstream direct-ASGI request helper; use `from widget_web_server import WhisperWidgetWebServer` and keep `from whisper_widget import WhisperWidget`. Patch `whisper_widget.LOG_DIR` to the test's temporary directory when logging test transcripts so tests do not write real daily logs. No HTTP testing dependency is required.

Cover:

- Empty state returns HTTP 200 with empty text and `Cache-Control: no-store`.
- Latest saved text is returned exactly, including Unicode and newlines.
- A newer completed transcript replaces the older one; an empty result does not clear it.
- Reads during recording/processing return the previous completed transcript without a busy error.
- Restart loading restores the same endpoint result.
- A GET neither calls the clipboard worker nor triggers inference.
- Existing local copy-button behavior, disk-save failure handling, and repeated-click handling continue to pass.

Check one completion-order edge case: `RemoteChunkSession._complete_success()` currently publishes `final=True/status="complete"` before it calls `log_transcription()`. Add a focused test for an immediate latest-transcript read after a client observes completion. If it exposes stale text, move publication of the completed session state after saving its final transcript, retaining the existing lock discipline and failure handling. This is a small hardening beyond the 11-line upstream route; desktop recordings already save before their processing finishes.

Run from the repository root:

```powershell
.\tools\check_all.ps1
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_clipboard_history.py -v
```

The current verifier does not run unittest automatically, so both commands are required. If the browser action is included, also exercise success, empty history, host unreachable, and clipboard-write failure in the browser.

## 4. Roll out and accept

1. Review a diff containing only the intended feature, documentation, tests, and any agreed browser action. Preserve the prior copy-button edits.
2. After verification, wait for an idle host and restart the existing D-drive widget through its established launch flow. A source edit will not add routes to the already-running process.
3. Confirm local `/health`, the new route in `/openapi.json`, and the response/cache header. Repeat from an authorized remote machine using its existing base URL.
4. Complete a short host-microphone transcription. Copy from two client machines and confirm both receive exactly the completed text without recording locally.
5. Start another recording; confirm remote Copy still returns the previous completed transcript. After completion, confirm it returns the new one.
6. Verify completed one-shot and remote chunk recordings update the same shared value. Restart while idle and confirm the latest value survives.
7. Confirm host F8 recording, local copy button, and existing network transcription still work.

Acceptance requires both the host route and functioning copy actions on the intended clients. Publishing the endpoint alone completes the server prerequisite.

If rollback is needed, reverse only this feature's new edits and restart the same D-drive widget; keep the saved transcript and prior copy-button backport. Do not reset the worktree or replace this checkout with remote main.

## Planning boundary

The original plan was produced from the fetched commit, local code, GET-only service discovery, process-path inspection, and existing checks. The approved host implementation is recorded above. Running services, Python packages, and network configuration have not been changed. No client messages were sent. The local-endpoints skill guided use of the existing loopback service and documented Tailscale boundary during planning.
