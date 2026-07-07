import argparse
import json
import sys
from pathlib import Path


POLICIES = {"first_legal", "heuristic"}


def load_jsonl(path) -> list[dict]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def choose_action(sample, policy="heuristic") -> dict:
    legal_moves = sample.get("legal_moves", [])
    if not legal_moves:
        raise ValueError("sample has no legal moves")

    if policy == "first_legal":
        return legal_moves[0]
    if policy == "heuristic":
        return max(legal_moves, key=_heuristic_score)
    raise ValueError(f"unknown policy: {policy}")


def evaluate_samples(samples, policy="heuristic") -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")

    correct = 0
    for sample in samples:
        if choose_action(sample, policy) == sample.get("action"):
            correct += 1

    total = len(samples)
    return {
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "policy": policy,
    }


def evaluate_file(path, policy="heuristic") -> dict:
    return evaluate_samples(load_jsonl(path), policy=policy)


def _heuristic_score(move) -> tuple:
    move_type = move.get("move_type")
    count = move.get("count", 1) or 1

    if move_type == "FREE_TO_HOME":
        group = 5
    elif move_type == "COL_TO_HOME":
        group = 4
    elif move_type == "FREE_TO_COL":
        group = 3
    elif move_type == "COL_TO_COL":
        group = 2
    elif move_type == "COL_TO_FREE":
        group = 1
    else:
        group = 0

    return (group, count, -move.get("from_idx", 0), -1 if move.get("to_idx") is None else -move["to_idx"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate simple policy baselines on JSONL FreeCell samples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", default="heuristic")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = evaluate_file(args.dataset, policy=args.policy)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to evaluate policy baseline: {exc}", file=sys.stderr)
        return 1

    print(f"policy: {summary['policy']}")
    print(f"samples: {summary['samples']}")
    print(f"correct: {summary['correct']}")
    print(f"accuracy: {summary['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
