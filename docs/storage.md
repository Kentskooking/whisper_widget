# Local generated files

All application paths are anchored to the repository, regardless of the terminal's
working directory. `app/paths.py` defines the locations shared by the desktop,
supervisor, and standalone service. The service's optional `base_dir` uses the
same layout under the supplied directory.

| Location | Contents |
| --- | --- |
| `data/transcriptions/` | Daily transcript text files |
| `data/raw_audio_backups/` | Recording backups retained by the existing recovery pipeline |
| `data/debug_audio/` | Debug captures, chunk transcripts, and comparison artifacts |
| `data/failed_recordings/` | Audio saved after a failed transcription |
| `runtime/logs/` | Desktop and web event logs, supervisor log, Python fatal log |
| `runtime/state/` | Widget heartbeat and recovery JSON |
| `runtime/work/recording/` | Current raw and speech-only WAV files |
| `runtime/work/chunks/` | Audio chunks for processing and interrupted-work recovery |
| `runtime/work/web/` | Web upload and processing files |

Generated contents of `data/` and `runtime/` are ignored by Git. Each `data/`
subfolder contains a tracked `*_will_appear_here.txt` placeholder so it exists
in fresh checkouts. Runtime directories are created when needed.
Existing transcript, recording-backup, debug-capture, and temporary-file retention
behavior is unchanged; this reorganization adds no automatic cleanup.
In particular, **do not delete `runtime/` to clear a cache**: its state and work
directories can hold the only copy of an interrupted recording. Back up both
directories together when preserving unfinished work.

Model downloads remain in the libraries' user cache locations. Existing HTTPS
certificates under `sidecache/certs/` and environment settings are unchanged.

## Migrating an older checkout

The app and both launchers stop with a visible error if legacy generated files
are found. They do not silently switch directories or migrate during startup.

1. Stop the widget, its supervisor, and the standalone web server.
2. From this repository root, preview the exact moves:
   ```bat
   .venv\Scripts\python.exe -m tools.migrate_data
   ```
3. Apply them:
   ```bat
   .venv\Scripts\python.exe -m tools.migrate_data --apply
   ```
4. Run either launcher's `--check` before launching normally.

Migration moves the old root-level transcript/audio/work directories, event logs
(including rotated logs), temporary and failed WAV files, and files from
`sidecache/runtime/`. It accepts data destinations containing only their named tracked placeholder,
which it preserves. It refuses other existing destinations, malformed state, links,
junctions, and recovery paths pointing outside this checkout. Resolve those
conditions explicitly before retrying; it never overwrites saved audio.

Live `chunk_dir` and `raw_backup_path` references in heartbeat/recovery JSON are
updated to their new locations. Historical logs and debug manifests are preserved
verbatim, so they can still refer to the old locations.

A migration journal records every source/destination pair and the original state
JSON. Successful migration retains it as `runtime/data_migration_completed_*.json`.
If a move or state write fails, `runtime/data_migration_in_progress.json` remains
and blocks startup. Inspect its move list, finish or reverse the recorded moves,
and restore or update the saved state references before removing that marker.
Do not simply delete the marker and launch: it protects unfinished recovery work.

Legacy ignore rules remain so un-migrated recordings cannot accidentally enter Git.
