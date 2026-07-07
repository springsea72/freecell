import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from game_model import FreeCellGame, Suit
from trace_io import load_trace, move_to_dict, moves_from_trace


DATASET_VERSION = 1


def card_to_dict(card) -> dict:
    return {
        "suit": card.suit.name,
        "value": card.value,
    }


def state_to_dict(game: FreeCellGame) -> dict:
    return {
        "columns": [[card_to_dict(card) for card in column] for column in game.columns],
        "free_cells": [None if card is None else card_to_dict(card) for card in game.free_cells],
        "home_cells": {
            suit.name: [card_to_dict(card) for card in game.home_cells[suit]]
            for suit in Suit
        },
    }


def sample_from_state(game, action, seed, source_trace, step_index, remaining_moves) -> dict:
    legal_moves = game.generate_legal_moves()
    try:
        action_index = legal_moves.index(action)
    except ValueError as exc:
        raise ValueError(f"action is not legal at step {step_index}: {action}") from exc

    return {
        "version": DATASET_VERSION,
        "source_trace": source_trace,
        "seed": seed,
        "step_index": step_index,
        "remaining_moves": remaining_moves,
        "state": state_to_dict(game),
        "legal_moves": [move_to_dict(move) for move in legal_moves],
        "action": move_to_dict(action),
        "action_index": action_index,
    }


def samples_from_trace(trace, source_trace=None) -> list[dict]:
    if not trace.get("solved"):
        raise ValueError("trace is not solved")
    seed = trace.get("seed")
    if seed is None:
        raise ValueError("trace is missing seed")

    moves = moves_from_trace(trace)
    game = FreeCellGame(seed=seed)
    samples = []

    for step_index, action in enumerate(moves):
        samples.append(
            sample_from_state(
                game,
                action,
                seed=seed,
                source_trace=source_trace,
                step_index=step_index,
                remaining_moves=len(moves) - step_index,
            )
        )
        if not game.apply_move(action):
            raise ValueError(f"failed to apply trace action at step {step_index}: {action}")

    if not game.is_won():
        raise ValueError("trace does not replay to a won state")
    return samples


def write_jsonl(path, samples) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(trace_paths: Iterable, output_path, skip_invalid=False) -> dict:
    summary = {
        "traces_read": 0,
        "traces_used": 0,
        "invalid_traces": 0,
        "samples_written": 0,
    }
    all_samples = []

    for trace_path in trace_paths:
        path = Path(trace_path)
        summary["traces_read"] += 1
        try:
            trace = load_trace(path)
            samples = samples_from_trace(trace, source_trace=path.name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["invalid_traces"] += 1
            if skip_invalid:
                continue
            raise ValueError(f"invalid trace {path}: {exc}") from exc

        summary["traces_used"] += 1
        all_samples.extend(samples)

    write_jsonl(output_path, all_samples)
    summary["samples_written"] = len(all_samples)
    return summary


def trace_paths_from_args(args) -> list[Path]:
    if args.trace is not None:
        return [Path(args.trace)]
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise ValueError(f"trace directory does not exist or is not a directory: {trace_dir}")
    return sorted(trace_dir.glob("*.json"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build JSONL policy samples from solved FreeCell traces.")
    trace_group = parser.add_mutually_exclusive_group(required=True)
    trace_group.add_argument("--trace", help="Single trace JSON file.")
    trace_group.add_argument("--trace-dir", help="Directory containing trace JSON files.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--skip-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        trace_paths = trace_paths_from_args(args)
        summary = build_dataset(
            trace_paths,
            args.output,
            skip_invalid=args.skip_invalid,
        )
    except ValueError as exc:
        print(f"failed to build dataset: {exc}", file=sys.stderr)
        return 1

    for key in ("traces_read", "traces_used", "invalid_traces", "samples_written"):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
