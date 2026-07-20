#!/usr/bin/env python3
"""Materialize the minimal local SQLite snapshot for PLUTO quick tests.

The nuPlan database is mounted from NFS on the server.  Closed-loop val14
uses ``box_observation``, which repeatedly reads its SQLite log during each
simulation.  Keeping the selected logs on local storage avoids NFS hotspots
when multiple scenarios share a large log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PLUTO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = PLUTO_ROOT.parent
DEFAULT_ROOT = WORKSPACE_ROOT / ".local-data" / "benchmark-db"
MANIFEST_NAME = "manifest.json"

BENCHMARKS = {
    "val14": {
        "filter": PLUTO_ROOT / "config/scenario_filter/val14_benchmark.yaml",
        "source_relative": Path("nuplan-v1.1_val/data/cache/val"),
    },
    "test14-hard": {
        "filter": PLUTO_ROOT / "config/scenario_filter/test14-hard.yaml",
        "source_relative": Path("nuplan-v1.1_test/data/cache/test"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help=f"Local snapshot root (default: {DEFAULT_ROOT})"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("NUPLAN_DATA_ROOT", "/mnt/nuplan/database")),
        help="Shared nuPlan database root",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=sorted(BENCHMARKS), default=sorted(BENCHMARKS)
    )
    parser.add_argument("--copy", action="store_true", help="Copy selected DBs and maps after planning")
    parser.add_argument("--skip-maps", action="store_true", help="Do not copy the shared maps directory")
    parser.add_argument("--verify", action="store_true", help="Verify an existing local snapshot without touching NFS")
    return parser.parse_args()


def filter_tokens(path: Path) -> set[bytes]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokens = payload.get("scenario_tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"No scenario_tokens in {path}")
    try:
        return {bytes.fromhex(str(token)) for token in tokens}
    except ValueError as error:
        raise ValueError(f"Invalid hexadecimal scenario token in {path}") from error


def chunks(values: Iterable[bytes], size: int = 900) -> Iterable[list[bytes]]:
    batch: list[bytes] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def tokens_in_database(database: Path, wanted: set[bytes]) -> set[bytes]:
    """Return requested lidar_pc tokens found in one read-only SQLite file."""
    if not wanted:
        return set()
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            found: set[bytes] = set()
            for batch in chunks(wanted):
                placeholders = ",".join("?" for _ in batch)
                query = f"SELECT token FROM lidar_pc WHERE token IN ({placeholders})"
                found.update(row[0] for row in connection.execute(query, batch))
            return found
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise RuntimeError(f"Unable to query {database}: {error}") from error


def resolve_databases(source: Path, wanted: set[bytes]) -> dict[bytes, Path]:
    matches: dict[bytes, Path] = {}
    for index, database in enumerate(sorted(source.glob("*.db")), start=1):
        remaining = wanted.difference(matches)
        if not remaining:
            break
        for token in tokens_in_database(database, remaining):
            existing = matches.setdefault(token, database)
            if existing != database:
                raise RuntimeError(f"Token {token.hex()} appears in multiple DBs")
        if index % 100 == 0:
            print(f"  scanned {index} DBs; matched {len(matches)}/{len(wanted)} tokens", flush=True)
    missing = wanted.difference(matches)
    if missing:
        preview = ", ".join(token.hex() for token in sorted(missing)[:5])
        raise RuntimeError(f"{len(missing)} scenario tokens were not found under {source}: {preview}")
    return matches


def copy_file(source: Path, destination: Path) -> None:
    """Copy a file resumably, publishing it only after its size matches."""
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    source_size = source.stat().st_size
    if offset > source_size:
        partial.unlink()
        offset = 0
    with source.open("rb") as src, partial.open("ab") as dst:
        src.seek(offset)
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    if partial.stat().st_size != source_size:
        raise RuntimeError(f"Incomplete copy for {source}")
    shutil.copystat(source, partial)
    partial.replace(destination)


def copy_maps(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"Maps directory does not exist: {source}")
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            copy_file(path, target)


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def verify(root: Path, benchmark_ids: list[str]) -> bool:
    path = manifest_path(root)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload["benchmarks"]
        for benchmark in benchmark_ids:
            entry = selected[benchmark]
            for item in entry["databases"]:
                local = root / item["relative_path"]
                if not local.is_file() or local.stat().st_size != item["size"]:
                    return False
        return (root / "maps").is_dir()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.verify:
        if verify(root, args.benchmarks):
            print(f"Local benchmark DB snapshot is complete: {root}")
            return 0
        print(f"Local benchmark DB snapshot is incomplete: {root}", file=sys.stderr)
        return 1

    data_root = args.data_root.resolve()
    selected: dict[str, dict[str, object]] = {}
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        source = data_root / spec["source_relative"]
        if not source.is_dir():
            raise RuntimeError(f"Database directory does not exist: {source}")
        tokens = filter_tokens(spec["filter"])
        print(f"Resolving {benchmark}: {len(tokens)} tokens in {source}", flush=True)
        matches = resolve_databases(source, tokens)
        databases = sorted(set(matches.values()))
        print(f"  selected {len(databases)} unique DBs ({sum(path.stat().st_size for path in databases) / 2**30:.2f} GiB)")
        selected[benchmark] = {
            "filter": str(spec["filter"].relative_to(PLUTO_ROOT)),
            "source_relative": str(spec["source_relative"]),
            "token_count": len(tokens),
            "databases": [
                {
                    "relative_path": str(spec["source_relative"] / path.name),
                    "size": path.stat().st_size,
                    "source": str(path),
                }
                for path in databases
            ],
        }

    payload = {"version": 1, "benchmarks": selected}
    if not args.copy:
        print("Plan complete. Re-run with --copy to materialize the local snapshot.")
        return 0

    for benchmark in args.benchmarks:
        print(f"Copying {benchmark} DBs to {root}", flush=True)
        for item in selected[benchmark]["databases"]:
            source = Path(item["source"])
            destination = root / item["relative_path"]
            copy_file(source, destination)
    if not args.skip_maps:
        print("Copying shared maps", flush=True)
        copy_maps(data_root / "maps", root / "maps")

    root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path(root).with_name(MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path(root))
    print(f"Local benchmark DB snapshot is ready: {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
