"""Preview or explicitly migrate legacy generated files; run with -m tools.migrate_data."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import stat

from app.paths import (
    REPO_ROOT, LEGACY_DIRECTORIES, MIGRATION_MARKER, DATA_PLACEHOLDERS,
    WIDGET_HEARTBEAT_FILE, WIDGET_RECOVERY_FILE, legacy_data_moves,
)


def checked_path(root, path):
    """Refuse paths outside the selected repository, including junctions."""
    path = Path(path)
    if path == root or not path.resolve().is_relative_to(root):
        raise ValueError(f"Migration path is outside the repository: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (
            part.exists()
            and getattr(part.stat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Migration refuses links and junctions: {part}")
    return path


def remap_state(state, root, moves):
    """Update only live recovery references; leave historical diagnostics intact."""
    state = dict(state)
    mappings = [(root / old, root / new) for old, new in LEGACY_DIRECTORIES.items()]
    mappings.extend(moves)
    for key in ("chunk_dir", "raw_backup_path"):
        value = state.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"Recovery field {key} must contain an absolute path: {value}")
        for source, target in mappings:
            if path == source or path.is_relative_to(source):
                state[key] = str(target / path.relative_to(source))
                break
        else:
            if not path.is_relative_to(root):
                raise ValueError(
                    f"Recovery field {key} points outside this checkout: {value}. "
                    "Resolve that recording's location before migration."
                )
    return state


def migrate(root=REPO_ROOT, *, apply=False):
    root = Path(root).resolve()
    marker = checked_path(root, root / MIGRATION_MARKER)
    if marker.exists():
        raise RuntimeError(f"Unfinished migration: inspect {marker} and docs/storage.md before continuing.")
    original_moves = legacy_data_moves(root)
    moves = []
    placeholder_sources = []
    for source, target in original_moves:
        checked_path(root, source)
        checked_path(root, target)
        placeholder = DATA_PLACEHOLDERS.get(target.relative_to(root))
        if source.is_dir() and target.is_dir() and placeholder:
            children = list(target.iterdir())
            if len(children) == 1 and children[0].name == placeholder and children[0].is_file():
                checked_path(root, children[0])
                # Preserve the tracked placeholder; preflight every incoming child.
                moves.extend((child, target / child.name) for child in sorted(source.iterdir()))
                placeholder_sources.append(source)
                continue
        moves.append((source, target))
    state_updates = {}
    originals = {}
    for source, target in moves:
        checked_path(root, source)
        checked_path(root, target)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite or merge existing destination: {target}")
        if source.is_dir():
            for directory, directories, files in os.walk(source, followlinks=False):
                for name in directories + files:
                    checked_path(root, Path(directory) / name)
        if source.parent == root / "sidecache/runtime" and source.name in (
            WIDGET_HEARTBEAT_FILE, WIDGET_RECOVERY_FILE,
        ):
            original = source.read_text(encoding="utf-8")
            state = json.loads(original)
            if not isinstance(state, dict):
                raise ValueError(f"Expected a JSON object in {source}")
            originals[source.relative_to(root).as_posix()] = original
            state_updates[target] = json.dumps(remap_state(state, root, moves), indent=2) + "\n"
        print(f"{source.relative_to(root)} -> {target.relative_to(root)}")

    for source in placeholder_sources:
        print(f"{source.relative_to(root)} -> remove legacy directory after moving its contents")

    if not original_moves:
        print("No legacy generated data found.")
        return 0
    if not apply:
        print("Preview only. Stop the widget, supervisor, and web server, then rerun with --apply.")
        return len(original_moves)

    # Keep a durable journal until every move and recovery rewrite has succeeded.
    # An interrupted run leaves this marker, which prevents application startup.
    completed = marker.with_name(f"data_migration_completed_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    checked_path(root, completed)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({
            "moves": [{"source": source.relative_to(root).as_posix(), "target": target.relative_to(root).as_posix()}
                      for source, target in moves],
            "original_state": originals,
            "remove_empty_sources": [path.relative_to(root).as_posix() for path in placeholder_sources],
        }, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for source, target in moves:
        # Repeat validation immediately before each move; never delete or merge.
        checked_path(root, source)
        checked_path(root, target)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Destination appeared during migration: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for target, content in state_updates.items():
        temporary = checked_path(root, target.with_suffix(".migration.tmp"))
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    for source in placeholder_sources:
        checked_path(root, source)
        source.rmdir()  # Refuse removal if another process added files.
    marker.rename(completed)
    # Remove only empty legacy container directories; certificates stay in place.
    for directory in (root / "sidecache/runtime", root / "sidecache"):
        if directory.exists():
            checked_path(root, directory)
            if not any(directory.iterdir()):
                directory.rmdir()
    print(f"Migration complete. Original state and move list saved in {completed.relative_to(root)}")
    return len(original_moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move files after stopping all Whisper processes")
    args = parser.parse_args()
    try:
        migrate(apply=args.apply)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Migration stopped: {exc}\n")


if __name__ == "__main__":
    main()
