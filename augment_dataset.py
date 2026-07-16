import argparse
import json
import sys
from pathlib import Path

from dataset_builder import samples_from_trace
from trace_io import load_trace


DEFAULT_REPEAT_LEARNED_TRACES = 1
SOURCE_POLICY_LEARNED = "learned"


def load_base_lines(path) -> tuple[list[str], int]:
    lines = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            body = line.rstrip("\r\n")
            if not body.strip():
                continue
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid base dataset JSON at line {line_number}: {exc}") from exc
            lines.append(body)
    return lines, len(lines)


def trace_paths_from_args(args) -> list[Path]:
    if args.trace:
        return [Path(path) for path in args.trace]

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise ValueError(f"trace directory does not exist or is not a directory: {trace_dir}")
    return sorted(trace_dir.glob("*.json"))


def build_augmented_dataset(
    base_dataset_path,
    trace_paths,
    output_path,
    repeat_learned_traces=DEFAULT_REPEAT_LEARNED_TRACES,
    skip_invalid=False,
) -> dict:
    if repeat_learned_traces < 1:
        raise ValueError("repeat_learned_traces must be >= 1")

    base_lines, base_samples = load_base_lines(base_dataset_path)
    summary = {
        "base_samples": base_samples,
        "traces_read": 0,
        "traces_used": 0,
        "invalid_traces": 0,
        "augmented_samples": 0,
        "samples_written": base_samples,
    }
    augmented_rows = []

    for trace_path in trace_paths:
        path = Path(trace_path)
        summary["traces_read"] += 1
        try:
            trace = load_trace(path)
            samples = samples_from_trace(trace, source_trace=path.name)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            summary["invalid_traces"] += 1
            if skip_invalid:
                continue
            raise ValueError(f"invalid trace {path}: {exc}") from exc

        summary["traces_used"] += 1
        for repeat_index in range(repeat_learned_traces):
            for sample in samples:
                row = dict(sample)
                row["source_policy"] = SOURCE_POLICY_LEARNED
                row["augmentation_repeat"] = repeat_index
                augmented_rows.append(row)

    summary["augmented_samples"] = len(augmented_rows)
    summary["samples_written"] = base_samples + len(augmented_rows)
    write_augmented_jsonl(output_path, base_lines, augmented_rows)
    return summary


def write_augmented_jsonl(path, base_lines, augmented_rows) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for line in base_lines:
            handle.write(line + "\n")
        for row in augmented_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Augment a FreeCell policy JSONL dataset with learned-policy traces.")
    parser.add_argument("--base-dataset", required=True)
    trace_group = parser.add_mutually_exclusive_group(required=True)
    trace_group.add_argument("--trace", action="append", help="Solved learned-policy trace JSON path. May repeat.")
    trace_group.add_argument("--trace-dir", help="Directory containing solved learned-policy trace JSON files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-learned-traces", type=int, default=DEFAULT_REPEAT_LEARNED_TRACES)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args(argv)

    if args.repeat_learned_traces < 1:
        parser.error("--repeat-learned-traces must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    base_dataset = Path(args.base_dataset)
    if not base_dataset.exists():
        print(f"base dataset does not exist: {base_dataset}", file=sys.stderr)
        return 1

    try:
        trace_paths = trace_paths_from_args(args)
        summary = build_augmented_dataset(
            base_dataset,
            trace_paths,
            args.output,
            repeat_learned_traces=args.repeat_learned_traces,
            skip_invalid=args.skip_invalid,
        )
    except (OSError, ValueError) as exc:
        print(f"failed to augment dataset: {exc}", file=sys.stderr)
        return 1

    for key in (
        "base_samples",
        "traces_read",
        "traces_used",
        "invalid_traces",
        "augmented_samples",
        "samples_written",
    ):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
