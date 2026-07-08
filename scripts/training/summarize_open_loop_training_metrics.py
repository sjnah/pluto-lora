#!/usr/bin/env python3
"""Summarize PLUTO training TensorBoard scalars before closed-loop benchmarking.

This is intended as a quick open-loop gate for LoRA curriculum runs. It reads
TensorBoard event files produced during training and reports the losses and
diagnostic scalars that are useful before spending time on a closed-loop
benchmark.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Keep TensorBoard event reading compatible with the local protobuf stack.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - import failure is environment-specific.
    print(
        "Failed to import TensorBoard EventAccumulator. "
        "Run this in the nuplan environment or install tensorboard. "
        f"Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_TAG_PATTERNS = [
    r"^loss/(train|val)_loss_epoch$",
    r"^objectives/(train|val)_(loss|reg_loss|cls_loss|ref_free_reg_loss|collision_loss|prediction_loss|contrastive_loss)$",
    r"^(train|val)/(minADE|minFDE|MR|PredAvgADE|PredAvgFDE)$",
    r"^train/(l2sp_loss|grad_norm|grad_norm_before_clip|grad_clipped|nan_steps_skipped|nan_rate_pct|nan_protection_events|nan_protection_rate_pct)$",
    r"^lr/param_group_\d+$",
    r"^lr-AdamW/.+$",
]

TRAIN_EPOCH_TAG = "loss/train_loss_epoch"

STEP_TAG_PATTERNS = [
    r"^loss/(train|val)_loss_step$",
    r"^lr_step/param_group_\d+$",
]

LOWER_IS_BETTER_PATTERNS = [
    r"^loss/",
    r"^objectives/",
    r"^(train|val)/(minADE|minFDE|MR|PredAvgADE|PredAvgFDE)$",
    r"^train/(grad_norm|grad_norm_before_clip|nan_steps_skipped|nan_rate_pct|nan_protection_events|nan_protection_rate_pct)$",
]

INFO_TAG_PATTERNS = [
    r"^lr/",
    r"^lr_step/",
    r"^lr-AdamW/",
    r"^train/l2sp_loss$",
    r"^epoch$",
]

NONZERO_WARN_TAGS = {
    "log/nan_protection_events",
    "train/nan_protection_events",
    "train/nan_protection_rate_pct",
    "train/nan_steps_skipped",
    "train/nan_rate_pct",
}

TAG_PRIORITY = [
    "loss/val_loss_epoch",
    "loss/train_loss_epoch",
    "objectives/val_loss",
    "objectives/train_loss",
    "objectives/val_reg_loss",
    "objectives/train_reg_loss",
    "objectives/val_cls_loss",
    "objectives/train_cls_loss",
    "objectives/val_prediction_loss",
    "objectives/train_prediction_loss",
    "objectives/val_collision_loss",
    "objectives/train_collision_loss",
    "objectives/val_ref_free_reg_loss",
    "objectives/train_ref_free_reg_loss",
    "val/minADE",
    "train/minADE",
    "val/minFDE",
    "train/minFDE",
    "val/MR",
    "train/MR",
    "train/nan_steps_skipped",
    "train/nan_rate_pct",
    "train/nan_protection_events",
    "train/nan_protection_rate_pct",
    "log/nan_protection_events",
    "train/grad_norm_before_clip",
    "train/grad_norm",
    "train/grad_clipped",
    "lr/param_group_0",
    "lr/param_group_1",
]


@dataclass(frozen=True)
class RunSource:
    name: str
    path: Path
    order: int
    role: str


@dataclass
class ScalarSummary:
    role: str
    run: str
    compare_key: str
    path: str
    tag: str
    count: int
    nonfinite_count: int
    first_step: int | None
    last_step: int | None
    first: float | None
    last: float | None
    minimum: float | None
    maximum: float | None
    tail_mean: float | None
    delta_pct: float | None
    baseline_tail_mean: float | None = None
    vs_baseline_pct: float | None = None
    status: str = "ok"


def has_glob_meta(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def event_dirs_from_path(path: Path) -> list[Path]:
    if path.is_file():
        if path.name.startswith("events.out.tfevents"):
            return [path.parent]
        raise ValueError(f"Not a TensorBoard event file: {path}")

    if not path.is_dir():
        raise ValueError(f"Path does not exist: {path}")

    direct_events = sorted(path.glob("events.out.tfevents*"))
    if direct_events:
        return [path]

    nested_event_dirs = sorted({event.parent for event in path.rglob("events.out.tfevents*")})
    if not nested_event_dirs:
        raise ValueError(f"No TensorBoard event files found under: {path}")

    return nested_event_dirs


def infer_run_name(path: Path) -> str:
    if path.name:
        return path.name
    return str(path)


def infer_compare_key(run_name: str) -> str:
    stage_match = re.search(r"(stage\d+_[A-Za-z0-9_.-]+)$", run_name)
    if stage_match:
        return stage_match.group(1)
    return run_name


def expand_sources(values: Sequence[str], role: str) -> list[RunSource]:
    sources: list[RunSource] = []
    seen: set[Path] = set()

    for raw_value in values:
        matches = [Path(match) for match in glob.glob(raw_value)] if has_glob_meta(raw_value) else [Path(raw_value)]
        if not matches:
            raise ValueError(f"No paths matched: {raw_value}")

        for match in matches:
            for event_dir in event_dirs_from_path(match):
                resolved = event_dir.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                sources.append(
                    RunSource(
                        name=infer_run_name(event_dir),
                        path=event_dir,
                        order=len(sources),
                        role=role,
                    )
                )

    duplicate_counts: dict[str, int] = {}
    deduped: list[RunSource] = []
    for source in sources:
        count = duplicate_counts.get(source.name, 0) + 1
        duplicate_counts[source.name] = count
        name = source.name if count == 1 else f"{source.name}#{count}"
        deduped.append(RunSource(name=name, path=source.path, order=source.order, role=source.role))

    return deduped


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def matches_any(tag: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(tag) for pattern in patterns)


def selected_patterns(args: argparse.Namespace) -> list[re.Pattern[str]]:
    if args.all_tags:
        return [re.compile(r".*")]

    raw_patterns = list(args.tag_regex or DEFAULT_TAG_PATTERNS)
    if args.include_step_tags and not args.tag_regex:
        raw_patterns.extend(STEP_TAG_PATTERNS)
    return compile_patterns(raw_patterns)


def load_scalar_events(source: RunSource) -> dict[str, list]:
    accumulator = EventAccumulator(str(source.path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    return {tag: accumulator.Scalars(tag) for tag in tags}


def first_train_epoch_step(scalars: dict[str, list]) -> int | None:
    events = scalars.get(TRAIN_EPOCH_TAG, [])
    if not events:
        return None
    return min(int(event.step) for event in events)


def filter_sanity_val_events(tag: str, events: Sequence, train_start_step: int | None) -> list:
    if train_start_step is None:
        return list(events)
    if not (tag.startswith("loss/val_") or tag.startswith("objectives/val_") or tag.startswith("val/")):
        return list(events)
    return [event for event in events if int(event.step) >= train_start_step]


def summarize_scalar_events(
    source: RunSource,
    tag: str,
    events: Sequence,
    tail_window: int,
) -> ScalarSummary:
    finite_events = [event for event in events if math.isfinite(float(event.value))]
    nonfinite_count = len(events) - len(finite_events)

    if not finite_events:
        return ScalarSummary(
            role=source.role,
            run=source.name,
            compare_key=infer_compare_key(source.name),
            path=str(source.path),
            tag=tag,
            count=len(events),
            nonfinite_count=nonfinite_count,
            first_step=None,
            last_step=None,
            first=None,
            last=None,
            minimum=None,
            maximum=None,
            tail_mean=None,
            delta_pct=None,
            status="warn_nonfinite",
        )

    values = [float(event.value) for event in finite_events]
    steps = [int(event.step) for event in finite_events]
    first = values[0]
    last = values[-1]
    tail_values = values[-max(1, tail_window) :]
    delta_pct = None
    if first != 0:
        delta_pct = 100.0 * (last - first) / abs(first)

    return ScalarSummary(
        role=source.role,
        run=source.name,
        compare_key=infer_compare_key(source.name),
        path=str(source.path),
        tag=tag,
        count=len(events),
        nonfinite_count=nonfinite_count,
        first_step=steps[0],
        last_step=steps[-1],
        first=first,
        last=last,
        minimum=min(values),
        maximum=max(values),
        tail_mean=sum(tail_values) / len(tail_values),
        delta_pct=delta_pct,
    )


def count_nan_protection_events(source: RunSource) -> int | None:
    log_path = source.path / "log.txt"
    if not log_path.is_file():
        return None
    count = 0
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "[NaN PROTECTION] Detected NaN/Inf" in line:
                count += 1
    return count


def summarize_log_diagnostics(source: RunSource) -> list[ScalarSummary]:
    nan_events = count_nan_protection_events(source)
    if nan_events is None:
        return []

    value = float(nan_events)
    return [
        ScalarSummary(
            role=source.role,
            run=source.name,
            compare_key=infer_compare_key(source.name),
            path=str(source.path),
            tag="log/nan_protection_events",
            count=1,
            nonfinite_count=0,
            first_step=None,
            last_step=None,
            first=value,
            last=value,
            minimum=value,
            maximum=value,
            tail_mean=value,
            delta_pct=None,
        )
    ]


def baseline_lookup_tables(
    baseline_rows: Sequence[ScalarSummary],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    keyed_values: dict[tuple[str, str], list[float]] = {}
    tag_values: dict[str, list[float]] = {}
    for row in baseline_rows:
        if row.tail_mean is None:
            continue
        keyed_values.setdefault((row.compare_key, row.tag), []).append(row.tail_mean)
        tag_values.setdefault(row.tag, []).append(row.tail_mean)

    keyed_baselines = {
        key: sum(values) / len(values) for key, values in keyed_values.items()
    }
    single_tag_baselines = {
        tag: values[0] for tag, values in tag_values.items() if len(values) == 1
    }
    return keyed_baselines, single_tag_baselines


def classify_row(
    row: ScalarSummary,
    baseline_by_key: dict[tuple[str, str], float],
    single_baseline_by_tag: dict[str, float],
    worse_threshold: float,
    better_threshold: float,
    grad_clip_warn_rate: float,
) -> ScalarSummary:
    if row.nonfinite_count:
        row.status = "warn_nonfinite"
        return row

    if matches_any(row.tag, compile_patterns(INFO_TAG_PATTERNS)):
        row.status = "info"

    if row.tag in NONZERO_WARN_TAGS and row.maximum is not None and row.maximum > 0:
        row.status = "warn_nonzero"
        return row

    if row.tag == "train/grad_clipped" and row.tail_mean is not None and row.tail_mean > grad_clip_warn_rate:
        row.status = "warn_clipped"
        return row

    baseline_value = baseline_by_key.get((row.compare_key, row.tag))
    if baseline_value is None:
        baseline_value = single_baseline_by_tag.get(row.tag)
    if baseline_value is None or row.tail_mean is None:
        return row

    row.baseline_tail_mean = baseline_value
    denominator = max(abs(baseline_value), 1e-12)
    row.vs_baseline_pct = 100.0 * (row.tail_mean - baseline_value) / denominator

    if not matches_any(row.tag, compile_patterns(LOWER_IS_BETTER_PATTERNS)):
        return row

    if row.tail_mean > baseline_value * (1.0 + worse_threshold):
        row.status = "worse"
    elif row.tail_mean < baseline_value * (1.0 - better_threshold):
        row.status = "better"
    elif row.status == "info":
        row.status = "info"
    else:
        row.status = "ok"

    return row


def collect_summaries(
    sources: Sequence[RunSource],
    patterns: Sequence[re.Pattern[str]],
    tail_window: int,
    list_tags: bool,
    include_sanity_val: bool,
    include_log_diagnostics: bool,
) -> tuple[list[ScalarSummary], list[dict[str, object]]]:
    summaries: list[ScalarSummary] = []
    tag_reports: list[dict[str, object]] = []

    for source in sources:
        scalars = load_scalar_events(source)
        available_tags = sorted(scalars)
        tag_reports.append(
            {
                "role": source.role,
                "run": source.name,
                "path": str(source.path),
                "tags": available_tags,
            }
        )

        if list_tags:
            continue

        train_start_step = first_train_epoch_step(scalars)
        selected_tags = [tag for tag in available_tags if matches_any(tag, patterns)]
        for tag in selected_tags:
            events = scalars[tag]
            if not include_sanity_val:
                events = filter_sanity_val_events(tag, events, train_start_step)
            if not events:
                continue
            summaries.append(summarize_scalar_events(source, tag, events, tail_window))

        if include_log_diagnostics:
            summaries.extend(summarize_log_diagnostics(source))

    return summaries, tag_reports


def sort_rows(rows: Sequence[ScalarSummary]) -> list[ScalarSummary]:
    priority = {tag: index for index, tag in enumerate(TAG_PRIORITY)}
    return sorted(rows, key=lambda row: (row.role, row.run, priority.get(row.tag, len(priority)), row.tag))


def format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.3e}"
    return f"{number:.4f}"


def rows_as_dicts(rows: Sequence[ScalarSummary]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]


def render_table(rows: Sequence[ScalarSummary]) -> str:
    columns = [
        ("run", "run"),
        ("tag", "tag"),
        ("n", "count"),
        ("last_step", "last_step"),
        ("first", "first"),
        ("last", "last"),
        ("tail", "tail_mean"),
        ("trend%", "delta_pct"),
        ("base_tail", "baseline_tail_mean"),
        ("vs_base%", "vs_baseline_pct"),
        ("status", "status"),
    ]

    table_rows: list[list[str]] = []
    for row in rows:
        raw = asdict(row)
        rendered: list[str] = []
        for _, key in columns:
            value = raw[key]
            if key in {"count", "last_step"}:
                rendered.append("-" if value is None else str(value))
            elif isinstance(value, (float, int)) or value is None:
                rendered.append(format_number(value))
            else:
                rendered.append(str(value))
        table_rows.append(rendered)

    headers = [header for header, _ in columns]
    widths = [len(header) for header in headers]
    for rendered in table_rows:
        for index, cell in enumerate(rendered):
            widths[index] = max(widths[index], len(cell))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for rendered in table_rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(rendered)))

    return "\n".join(lines)


def write_csv(rows: Sequence[ScalarSummary], output_path: Path | None) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ScalarSummary.__annotations__)
    handle = output_path.open("w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    finally:
        if output_path:
            handle.close()


def write_json(rows: Sequence[ScalarSummary], output_path: Path | None) -> None:
    payload = rows_as_dicts(rows)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_path:
        output_path.write_text(text + "\n")
    else:
        print(text)


def write_series_csv(
    sources: Sequence[RunSource],
    patterns: Sequence[re.Pattern[str]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["role", "run", "path", "tag", "step", "wall_time", "value"],
        )
        writer.writeheader()
        for source in sources:
            scalars = load_scalar_events(source)
            for tag in sorted(tag for tag in scalars if matches_any(tag, patterns)):
                for event in scalars[tag]:
                    writer.writerow(
                        {
                            "role": source.role,
                            "run": source.name,
                            "path": source.path,
                            "tag": tag,
                            "step": event.step,
                            "wall_time": event.wall_time,
                            "value": event.value,
                        }
                    )


def print_tag_report(tag_reports: Sequence[dict[str, object]]) -> None:
    for report in tag_reports:
        print(f"[{report['role']}] {report['run']} ({report['path']})")
        for tag in report["tags"]:
            print(f"  {tag}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize PLUTO open-loop training metrics from TensorBoard event files. "
            "Pass run directories, event files, or glob patterns."
        )
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories, event files, or glob patterns to summarize.")
    parser.add_argument("--baseline", nargs="*", default=[], help="Optional baseline run directories/event files/globs for tail-mean comparison.")
    parser.add_argument("--tail-window", type=int, default=5, help="Number of final scalar points used for tail_mean.")
    parser.add_argument("--worse-threshold", type=float, default=0.05, help="Relative worsening threshold for lower-is-better metrics. 0.05 means 5%%.")
    parser.add_argument("--better-threshold", type=float, default=None, help="Relative improvement threshold. Defaults to --worse-threshold.")
    parser.add_argument("--grad-clip-warn-rate", type=float, default=0.5, help="Warn if train/grad_clipped tail_mean exceeds this value.")
    parser.add_argument("--include-step-tags", action="store_true", help="Include high-frequency step loss and lr_step tags.")
    parser.add_argument("--all-tags", action="store_true", help="Summarize every scalar tag.")
    parser.add_argument("--tag-regex", action="append", help="Custom scalar tag regex. Can be repeated. Replaces defaults unless --all-tags is set.")
    parser.add_argument("--list-tags", action="store_true", help="List available scalar tags and exit.")
    parser.add_argument("--include-sanity-val", action="store_true", help="Keep validation scalars logged before the first train epoch. By default these likely Lightning sanity-validation points are excluded.")
    parser.add_argument("--no-log-diagnostics", action="store_true", help="Do not add diagnostics parsed from log.txt, such as NaNProtectionCallback events.")
    parser.add_argument("--show-baseline", action="store_true", help="Include baseline rows in the rendered output.")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Summary output format.")
    parser.add_argument("--output", type=Path, help="Optional summary output path.")
    parser.add_argument("--series-output", type=Path, help="Optional CSV path for raw selected scalar time series.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Exit with code 2 when any selected run has warn_* or worse status.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.tail_window <= 0:
        raise ValueError("--tail-window must be positive.")
    if args.better_threshold is None:
        args.better_threshold = args.worse_threshold

    patterns = selected_patterns(args)
    run_sources = expand_sources(args.runs, role="run")
    baseline_sources = expand_sources(args.baseline, role="baseline") if args.baseline else []

    baseline_rows, baseline_tag_reports = collect_summaries(
        baseline_sources,
        patterns,
        tail_window=args.tail_window,
        list_tags=args.list_tags,
        include_sanity_val=args.include_sanity_val,
        include_log_diagnostics=not args.no_log_diagnostics,
    )
    run_rows, run_tag_reports = collect_summaries(
        run_sources,
        patterns,
        tail_window=args.tail_window,
        list_tags=args.list_tags,
        include_sanity_val=args.include_sanity_val,
        include_log_diagnostics=not args.no_log_diagnostics,
    )

    if args.list_tags:
        print_tag_report([*baseline_tag_reports, *run_tag_reports])
        return 0

    baseline_by_key, single_baseline_by_tag = baseline_lookup_tables(baseline_rows)
    classified_run_rows = [
        classify_row(
            row,
            baseline_by_key=baseline_by_key,
            single_baseline_by_tag=single_baseline_by_tag,
            worse_threshold=args.worse_threshold,
            better_threshold=args.better_threshold,
            grad_clip_warn_rate=args.grad_clip_warn_rate,
        )
        for row in run_rows
    ]

    output_rows = classified_run_rows
    if args.show_baseline:
        output_rows = [*baseline_rows, *classified_run_rows]
    output_rows = sort_rows(output_rows)

    if not output_rows:
        print("No scalar tags matched the selected patterns.", file=sys.stderr)
        return 1

    if args.series_output:
        write_series_csv([*baseline_sources, *run_sources], patterns, args.series_output)

    if args.format == "table":
        text = render_table(output_rows)
        if args.output:
            args.output.write_text(text + "\n")
        else:
            print(text)
    elif args.format == "csv":
        write_csv(output_rows, args.output)
    else:
        write_json(output_rows, args.output)

    if args.fail_on_warn:
        bad_statuses = {"warn_nonfinite", "warn_nonzero", "warn_clipped", "worse"}
        if any(row.role == "run" and row.status in bad_statuses for row in output_rows):
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
