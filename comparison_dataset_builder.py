import argparse
import json
import random
import sys
from pathlib import Path


COMPARISON_DATASET_VERSION = 1
DEFAULT_RANDOM_SEED = 123
DEFAULT_NEGATIVES_PER_SAMPLE = 1


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


def build_comparison_samples(samples, negatives_per_sample=DEFAULT_NEGATIVES_PER_SAMPLE, seed=DEFAULT_RANDOM_SEED):
    if negatives_per_sample < 1:
        raise ValueError("negatives_per_sample must be >= 1")

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

        rng.shuffle(negative_indices)
        for rejected_index in negative_indices[:negatives_per_sample]:
            pairs.append(comparison_from_sample(sample, preferred_index, rejected_index))

    return pairs, {
        "samples_read": len(samples),
        "pairs_written": len(pairs),
        "skipped_no_negative": skipped_no_negative,
    }


def comparison_from_sample(sample, preferred_index: int, rejected_index: int) -> dict:
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


def build_comparison_dataset(dataset_path, output_path, negatives_per_sample=DEFAULT_NEGATIVES_PER_SAMPLE, seed=DEFAULT_RANDOM_SEED) -> dict:
    samples = load_samples(dataset_path)
    pairs, summary = build_comparison_samples(samples, negatives_per_sample=negatives_per_sample, seed=seed)
    write_jsonl(output_path, pairs)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build pairwise FreeCell policy comparison samples from JSONL.")
    parser.add_argument("--dataset", required=True, help="Input JSONL from dataset_builder.py.")
    parser.add_argument("--output", required=True, help="Output comparison JSONL path.")
    parser.add_argument("--negatives-per-sample", type=int, default=DEFAULT_NEGATIVES_PER_SAMPLE)
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
