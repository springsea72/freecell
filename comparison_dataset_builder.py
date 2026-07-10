import argparse
import json
import random
import sys
from pathlib import Path


COMPARISON_DATASET_VERSION = 1
DEFAULT_RANDOM_SEED = 123
DEFAULT_NEGATIVES_PER_SAMPLE = 1
DEFAULT_NEGATIVE_STRATEGY = "random"
NEGATIVE_STRATEGIES = ("random", "heuristic_bad")


def load_samples(path) -> list[dict]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return samples


def build_comparison_samples(
    samples,
    negatives_per_sample=DEFAULT_NEGATIVES_PER_SAMPLE,
    seed=DEFAULT_RANDOM_SEED,
    negative_strategy=DEFAULT_NEGATIVE_STRATEGY,
):
    if negatives_per_sample < 1:
        raise ValueError("negatives_per_sample must be >= 1")
    if negative_strategy not in NEGATIVE_STRATEGIES:
        raise ValueError(f"unsupported negative_strategy: {negative_strategy}")

    rng = random.Random(seed)
    pairs = []
    skipped_no_negative = 0

    for sample in samples:
        legal_moves = sample.get("legal_moves") or []
        preferred_index = sample.get("action_index")
        if not isinstance(preferred_index, int) or preferred_index < 0 or preferred_index >= len(legal_moves):
            raise ValueError(f"sample has invalid action_index: {preferred_index}")

        negative_indices = [idx for idx in range(len(legal_moves)) if idx != preferred_index]
        if not negative_indices:
            skipped_no_negative += 1
            continue

        ranked_indices = rank_negative_indices(sample, negative_indices, rng, negative_strategy)
        for rejected_rank, rejected_index in enumerate(ranked_indices[:negatives_per_sample]):
            pairs.append(
                comparison_from_sample(
                    sample,
                    preferred_index,
                    rejected_index,
                    negative_strategy=negative_strategy,
                    rejected_rank=rejected_rank,
                )
            )

    return pairs, {
        "samples_read": len(samples),
        "pairs_written": len(pairs),
        "skipped_no_negative": skipped_no_negative,
    }


def rank_negative_indices(sample, negative_indices: list[int], rng: random.Random, negative_strategy: str) -> list[int]:
    ranked_indices = list(negative_indices)
    if negative_strategy == "random":
        rng.shuffle(ranked_indices)
        return ranked_indices

    state = sample["state"]
    legal_moves = sample["legal_moves"]
    ranked = []
    for index in ranked_indices:
        move = legal_moves[index]
        ranked.append((_heuristic_bad_key(state, move, rng.random()), index))
    ranked.sort()
    return [index for _, index in ranked]


def comparison_from_sample(
    sample,
    preferred_index: int,
    rejected_index: int,
    negative_strategy=DEFAULT_NEGATIVE_STRATEGY,
    rejected_rank=0,
) -> dict:
    legal_moves = sample["legal_moves"]
    source_sample = f"{sample.get('source_trace', '')}:{sample.get('step_index', '')}"
    return {
        "version": COMPARISON_DATASET_VERSION,
        "source_sample": source_sample,
        "seed": sample.get("seed"),
        "step_index": sample.get("step_index"),
        "state": sample["state"],
        "preferred_action": sample["action"],
        "rejected_action": legal_moves[rejected_index],
        "preferred_action_index": preferred_index,
        "rejected_action_index": rejected_index,
        "negative_strategy": negative_strategy,
        "rejected_rank": rejected_rank,
        "reason": "trace_action_vs_non_trace",
        "progress": sample.get("progress"),
    }


def write_jsonl(path, rows) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_comparison_dataset(
    dataset_path,
    output_path,
    negatives_per_sample=DEFAULT_NEGATIVES_PER_SAMPLE,
    seed=DEFAULT_RANDOM_SEED,
    negative_strategy=DEFAULT_NEGATIVE_STRATEGY,
) -> dict:
    samples = load_samples(dataset_path)
    pairs, summary = build_comparison_samples(
        samples,
        negatives_per_sample=negatives_per_sample,
        seed=seed,
        negative_strategy=negative_strategy,
    )
    write_jsonl(output_path, pairs)
    return summary


def _heuristic_bad_key(state: dict, move: dict, tie_breaker: float) -> tuple:
    move_type = move.get("move_type")
    is_home_move = move_type in ("FREE_TO_HOME", "COL_TO_HOME")
    releases_low = _move_releases_low_card(state, move)
    empties_source = _move_empties_source_column(state, move)
    reduces_buffer = _move_reduces_buffer_space(state, move)

    if move_type == "COL_TO_FREE" and not releases_low and not empties_source:
        type_priority = 0
    elif move_type == "COL_TO_FREE":
        type_priority = 1
    elif move_type == "COL_TO_COL" and not releases_low:
        type_priority = 2
    elif move_type in ("FREE_TO_COL", "COL_TO_COL"):
        type_priority = 3
    else:
        type_priority = 4

    return (
        1 if is_home_move else 0,
        type_priority,
        0 if reduces_buffer else 1,
        1 if releases_low else 0,
        1 if empties_source else 0,
        tie_breaker,
    )


def _move_empties_source_column(state: dict, move: dict) -> bool:
    if move.get("move_type") not in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL"):
        return False
    count = move.get("count", 1) or 1
    column = _safe_get(state["columns"], move.get("from_idx"), [])
    return bool(column) and len(column) == count


def _move_releases_low_card(state: dict, move: dict) -> bool:
    if move.get("move_type") not in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL"):
        return False
    count = move.get("count", 1) or 1
    column = _safe_get(state["columns"], move.get("from_idx"), [])
    remaining = len(column) - count
    if remaining <= 0:
        return False
    return _is_low_card(column[remaining - 1])


def _move_reduces_buffer_space(state: dict, move: dict) -> bool:
    before = _available_buffer_count(state)
    after = before
    move_type = move.get("move_type")

    if move_type == "COL_TO_FREE":
        after -= 1
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "FREE_TO_COL":
        after += 1
        if _destination_top_card(state, move) is None:
            after -= 1
    elif move_type == "COL_TO_COL":
        if _destination_top_card(state, move) is None:
            after -= 1
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "COL_TO_HOME":
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "FREE_TO_HOME":
        after += 1

    return after < before


def _destination_top_card(state: dict, move: dict):
    if move.get("move_type") not in ("FREE_TO_COL", "COL_TO_COL"):
        return None
    column = _safe_get(state["columns"], move.get("to_idx"), [])
    return column[-1] if column else None


def _available_buffer_count(state: dict) -> int:
    return sum(1 for card in state["free_cells"] if card is None) + sum(1 for column in state["columns"] if not column)


def _is_low_card(card: dict) -> bool:
    value = _card_value(card)
    return 1 <= value <= 3


def _card_value(card: dict) -> int:
    if card is None:
        return 0
    return int(card.get("value", 0))


def _safe_get(items, index, default=None):
    if not isinstance(index, int) or index < 0 or index >= len(items):
        return default
    return items[index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build pairwise FreeCell policy comparison samples from JSONL.")
    parser.add_argument("--dataset", required=True, help="Input JSONL from dataset_builder.py.")
    parser.add_argument("--output", required=True, help="Output comparison JSONL path.")
    parser.add_argument("--negatives-per-sample", type=int, default=DEFAULT_NEGATIVES_PER_SAMPLE)
    parser.add_argument("--negative-strategy", choices=NEGATIVE_STRATEGIES, default=DEFAULT_NEGATIVE_STRATEGY)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args(argv)

    if args.negatives_per_sample < 1:
        parser.error("--negatives-per-sample must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"dataset does not exist: {dataset_path}", file=sys.stderr)
        return 1

    try:
        summary = build_comparison_dataset(
            dataset_path,
            args.output,
            negatives_per_sample=args.negatives_per_sample,
            negative_strategy=args.negative_strategy,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"failed to build comparison dataset: {exc}", file=sys.stderr)
        return 1

    for key in ("samples_read", "pairs_written", "skipped_no_negative"):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
