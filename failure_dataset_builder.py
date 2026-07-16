import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

import learned_policy
import learned_policy_eval
from dataset_builder import home_card_count, state_to_dict
from game_model import FreeCellGame
from trace_io import move_to_dict


FAILURE_DATASET_VERSION = 1
POLICY_NAME = "learned"
DEFAULT_MAX_STEPS = 500
DEFAULT_WINDOW_SIZE = 10


def seeds_from_args(args) -> list[int]:
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def build_failure_dataset(
    model_path,
    seeds,
    output_path,
    max_steps=DEFAULT_MAX_STEPS,
    device="auto",
    window_size=DEFAULT_WINDOW_SIZE,
) -> dict:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")

    seeds = list(seeds)
    model_bundle = learned_policy.load_model(model_path, device=device)
    rows = []
    reasons = Counter()
    wins = 0
    failures = 0

    for seed in seeds:
        result = collect_failure_window(seed, model_bundle, max_steps=max_steps, device=device, window_size=window_size)
        if result["won"]:
            wins += 1
            continue

        failures += 1
        reasons[result["terminal_reason"]] += 1
        rows.extend(result["samples"])

    write_jsonl(output_path, rows)
    return {
        "games": len(seeds),
        "wins": wins,
        "failures": failures,
        "samples_written": len(rows),
        "reasons": dict(sorted(reasons.items())),
    }


def collect_failure_window(seed, model_bundle, max_steps=DEFAULT_MAX_STEPS, device=None, window_size=DEFAULT_WINDOW_SIZE):
    game = FreeCellGame(seed=seed)
    visited = {game.state_key()}
    records = deque(maxlen=window_size)
    steps = 0

    if game.is_won():
        return _run_result(True, "won", steps, records)

    while steps < max_steps:
        scored_moves = learned_policy.score_legal_moves(game, model_bundle, device=device)
        if not scored_moves:
            reason = "won" if game.is_won() else "no_legal_moves"
            return _run_result(game.is_won(), reason, steps, records)

        selected = _select_non_looping_move(game, scored_moves, visited)
        if selected is None:
            terminal_record = _record_from_scored_moves(
                seed,
                game,
                scored_moves,
                _top_reranked_index(game, scored_moves),
                step_index=steps,
                visited_states=len(visited),
                would_loop=True,
            )
            records.append(terminal_record)
            return _run_result(False, "loop_detected", steps, records)

        chosen_index, chosen_move, _ = selected
        record = _record_from_scored_moves(
            seed,
            game,
            scored_moves,
            chosen_index,
            step_index=steps,
            visited_states=len(visited),
            would_loop=False,
        )

        if not game.apply_move(chosen_move):
            records.append(record)
            return _run_result(False, "invalid_action", steps, records)

        records.append(record)
        steps += 1
        visited.add(game.state_key())

        if game.is_won():
            return _run_result(True, "won", steps, records)

    return _run_result(game.is_won(), "max_steps", steps, records)


def _select_non_looping_move(game, scored_moves, visited):
    ranked = sorted(
        enumerate(scored_moves),
        key=lambda item: learned_policy_eval._reranked_score(game, item[1][0], item[1][1]),
        reverse=True,
    )
    for index, (move, score) in ranked:
        probe = game.clone()
        if not probe.apply_move(move):
            continue
        if probe.state_key() not in visited:
            return index, move, score
    return None


def _top_reranked_index(game, scored_moves) -> int:
    return max(
        range(len(scored_moves)),
        key=lambda index: learned_policy_eval._reranked_score(game, scored_moves[index][0], scored_moves[index][1]),
    )


def _record_from_scored_moves(seed, game, scored_moves, chosen_index, step_index, visited_states, would_loop) -> dict:
    legal_moves = [move for move, _ in scored_moves]
    model_scores = [float(score) for _, score in scored_moves]
    chosen_action = legal_moves[chosen_index]
    return {
        "version": FAILURE_DATASET_VERSION,
        "seed": seed,
        "policy": POLICY_NAME,
        "step_index": step_index,
        "state": state_to_dict(game),
        "legal_moves": [move_to_dict(move) for move in legal_moves],
        "chosen_action": move_to_dict(chosen_action),
        "chosen_action_index": chosen_index,
        "model_scores": model_scores,
        "home_cards": home_card_count(game),
        "visited_states": visited_states,
        "would_loop": would_loop,
    }


def _run_result(won, terminal_reason, terminal_step, records) -> dict:
    samples = list(records)
    for window_index, sample in enumerate(samples):
        sample["terminal_reason"] = terminal_reason
        sample["terminal_step"] = terminal_step
        sample["window_index"] = window_index
        sample["is_terminal_step"] = window_index == len(samples) - 1
    return {
        "won": won,
        "terminal_reason": terminal_reason,
        "terminal_step": terminal_step,
        "samples": samples if not won else [],
    }


def write_jsonl(path, rows) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect failed learned-policy direct-play terminal windows.")
    parser.add_argument("--model", required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    args = parser.parse_args(argv)

    if args.seed_start is not None and args.seed_count is None:
        parser.error("--seed-count is required with --seed-start")
    if args.seed_count is not None and args.seed_start is None:
        parser.error("--seed-count can only be used with --seed-start")
    if args.seed_count is not None and args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model does not exist: {model_path}", file=sys.stderr)
        return 1

    try:
        summary = build_failure_dataset(
            model_path,
            seeds_from_args(args),
            args.output,
            max_steps=args.max_steps,
            device=args.device,
            window_size=args.window_size,
        )
    except (ModuleNotFoundError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"failed to build failure dataset: {exc}", file=sys.stderr)
        return 1

    for key in ("games", "wins", "failures", "samples_written"):
        print(f"{key}: {summary[key]}")
    print(f"reasons: {json.dumps(summary['reasons'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
