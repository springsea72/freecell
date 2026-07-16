import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


DEGRADATION_COMPARISON_VERSION = 1
DEFAULT_MAX_PAIRS_PER_SAMPLE = 1
SOURCE = "failure_window"
REASON = "avoid_structural_degradation"
POLICY = "learned"


def load_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return rows


def build_degradation_comparison_samples(
    rows,
    max_pairs_per_sample=DEFAULT_MAX_PAIRS_PER_SAMPLE,
) -> tuple[list[dict], dict]:
    if max_pairs_per_sample < 1:
        raise ValueError("max_pairs_per_sample must be >= 1")

    pairs = []
    candidate_rows = 0
    skipped_no_alternative = 0
    by_terminal_reason = defaultdict(lambda: {"candidate_rows": 0, "pairs_written": 0, "skipped_no_alternative": 0})

    for row in rows:
        if not is_degradation_candidate(row):
            continue

        candidate_rows += 1
        terminal_reason = row.get("terminal_reason", "unknown")
        by_terminal_reason[terminal_reason]["candidate_rows"] += 1

        legal_moves = row.get("legal_moves") or []
        model_scores = row.get("model_scores") or []
        rejected_index = row.get("chosen_action_index")
        if not _valid_index(rejected_index, legal_moves):
            skipped_no_alternative += 1
            by_terminal_reason[terminal_reason]["skipped_no_alternative"] += 1
            continue

        preferred_indices = _preferred_indices(legal_moves, model_scores, rejected_index)
        if not preferred_indices:
            skipped_no_alternative += 1
            by_terminal_reason[terminal_reason]["skipped_no_alternative"] += 1
            continue

        for preferred_index in preferred_indices[:max_pairs_per_sample]:
            pairs.append(pair_from_row(row, preferred_index, rejected_index))
            by_terminal_reason[terminal_reason]["pairs_written"] += 1

    return pairs, {
        "rows_read": len(rows),
        "candidate_rows": candidate_rows,
        "pairs_written": len(pairs),
        "skipped_no_alternative": skipped_no_alternative,
        "by_terminal_reason": dict(sorted(by_terminal_reason.items())),
    }


def is_degradation_candidate(row: dict) -> bool:
    return (
        row.get("chosen_reduces_buffer") is True
        and row.get("chosen_releases_low_card") is False
        and row.get("chosen_is_home_move") is False
    )


def pair_from_row(row: dict, preferred_index: int, rejected_index: int) -> dict:
    legal_moves = row["legal_moves"]
    model_scores = row.get("model_scores") or []
    return {
        "version": DEGRADATION_COMPARISON_VERSION,
        "source": SOURCE,
        "seed": row.get("seed"),
        "step_index": row.get("step_index"),
        "terminal_reason": row.get("terminal_reason"),
        "terminal_step": row.get("terminal_step"),
        "state": row.get("state"),
        "preferred_action": legal_moves[preferred_index],
        "rejected_action": legal_moves[rejected_index],
        "preferred_action_index": preferred_index,
        "rejected_action_index": rejected_index,
        "preferred_model_score": _score_at(model_scores, preferred_index),
        "rejected_model_score": _score_at(model_scores, rejected_index),
        "reason": REASON,
        "policy": POLICY,
        "home_cards": row.get("home_cards"),
        "buffer_slots": row.get("buffer_slots"),
        "buried_low_cards": row.get("buried_low_cards"),
        "movable_suffix_total": row.get("movable_suffix_total"),
        "window_index": row.get("window_index"),
    }


def build_degradation_comparison_dataset(
    failure_dataset_path,
    output_path,
    max_pairs_per_sample=DEFAULT_MAX_PAIRS_PER_SAMPLE,
) -> dict:
    rows = load_jsonl(failure_dataset_path)
    pairs, summary = build_degradation_comparison_samples(rows, max_pairs_per_sample=max_pairs_per_sample)
    write_jsonl(output_path, pairs)
    return summary


def write_jsonl(path, rows) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _preferred_indices(legal_moves, model_scores, rejected_index: int) -> list[int]:
    candidates = [index for index in range(len(legal_moves)) if index != rejected_index]
    return sorted(
        candidates,
        key=lambda index: (_move_priority(legal_moves[index]), -_score_at(model_scores, index), index),
    )


def _move_priority(move: dict) -> int:
    move_type = move.get("move_type")
    if move_type in ("FREE_TO_HOME", "COL_TO_HOME"):
        return 0
    if move_type == "FREE_TO_COL":
        return 1
    if move_type == "COL_TO_COL":
        return 2
    if move_type == "COL_TO_FREE":
        return 3
    return 4


def _score_at(model_scores, index: int) -> float:
    if not isinstance(index, int) or index < 0 or index >= len(model_scores):
        return float("-inf")
    return float(model_scores[index])


def _valid_index(index, items) -> bool:
    return isinstance(index, int) and 0 <= index < len(items)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build structural-degradation comparison pairs from failure windows.")
    parser.add_argument("--failure-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs-per-sample", type=int, default=DEFAULT_MAX_PAIRS_PER_SAMPLE)
    args = parser.parse_args(argv)

    if args.max_pairs_per_sample < 1:
        parser.error("--max-pairs-per-sample must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    failure_dataset = Path(args.failure_dataset)
    if not failure_dataset.exists():
        print(f"failure dataset does not exist: {failure_dataset}", file=sys.stderr)
        return 1

    try:
        summary = build_degradation_comparison_dataset(
            failure_dataset,
            args.output,
            max_pairs_per_sample=args.max_pairs_per_sample,
        )
    except (OSError, ValueError) as exc:
        print(f"failed to build degradation comparison dataset: {exc}", file=sys.stderr)
        return 1

    for key in ("rows_read", "candidate_rows", "pairs_written", "skipped_no_alternative"):
        print(f"{key}: {summary[key]}")
    print(f"by_terminal_reason: {json.dumps(summary['by_terminal_reason'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
