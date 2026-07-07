import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import learned_policy
from game_model import FreeCellGame
from policy_baseline import load_jsonl
from policy_features import features_from_sample


@dataclass
class PlayResult:
    seed: int
    policy: str
    won: bool
    steps: int
    reason: str
    home_cards: int
    visited_states: int


def evaluate_samples(samples, model_bundle, device=None) -> dict:
    correct = 0
    for sample in samples:
        features_from_sample(sample)
        scores = score_sample_moves(sample, model_bundle, device=device)
        predicted_index = max(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(predicted_index == sample["action_index"])

    total = len(samples)
    return {
        "mode": "dataset",
        "samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "device": str(_bundle_device(model_bundle, device)),
    }


def evaluate_file(dataset_path, model_path, device="auto") -> dict:
    model_bundle = learned_policy.load_model(model_path, device=device)
    summary = evaluate_samples(load_jsonl(dataset_path), model_bundle, device=device)
    summary["model_path"] = str(model_path)
    return summary


def score_sample_moves(sample, model_bundle, device=None) -> list[float]:
    torch = learned_policy._import_torch()
    resolved_device = _bundle_device(model_bundle, device)
    state_features, move_features, _ = features_from_sample(sample)
    if not move_features:
        return []

    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=resolved_device)
    move_tensor = torch.tensor(move_features, dtype=torch.float32, device=resolved_device)
    state_batch = state_tensor.unsqueeze(0).repeat(len(move_features), 1)
    model_input = torch.cat([state_batch, move_tensor], dim=1)

    model = model_bundle.model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        return [float(score) for score in model(model_input).view(-1).detach().cpu().tolist()]


def play_game(seed, model_bundle, max_steps=500, device=None) -> PlayResult:
    game = FreeCellGame(seed=seed)
    visited = {game.state_key()}
    steps = 0

    if game.is_won():
        return _result(seed, True, steps, "won", game, visited)

    while steps < max_steps:
        legal_moves = game.generate_legal_moves()
        if not legal_moves:
            reason = "won" if game.is_won() else "no_legal_moves"
            return _result(seed, game.is_won(), steps, reason, game, visited)

        selected_move = _choose_non_looping_move(game, model_bundle, visited, device=device)
        if selected_move is None:
            return _result(seed, False, steps, "loop_detected", game, visited)

        if not game.apply_move(selected_move):
            return _result(seed, False, steps, "invalid_action", game, visited)

        steps += 1
        visited.add(game.state_key())

        if game.is_won():
            return _result(seed, True, steps, "won", game, visited)

    return _result(seed, game.is_won(), steps, "max_steps", game, visited)


def evaluate_seeds(seeds, model_path, max_steps=500, device="auto") -> dict:
    model_bundle = learned_policy.load_model(model_path, device=device)
    results = [play_game(seed, model_bundle, max_steps=max_steps, device=device) for seed in seeds]
    games = len(results)
    won = sum(1 for result in results if result.won)
    return {
        "mode": "play",
        "games": games,
        "won": won,
        "win_rate": won / games if games else 0.0,
        "average_steps": sum(result.steps for result in results) / games if games else 0.0,
        "average_home_cards": sum(result.home_cards for result in results) / games if games else 0.0,
        "device": str(model_bundle.device),
        "model_path": str(model_path),
        "results": results,
    }


def _choose_non_looping_move(game, model_bundle, visited, device=None):
    scored_moves = learned_policy.score_legal_moves(game, model_bundle, device=device)
    for move, _ in sorted(scored_moves, key=lambda item: item[1], reverse=True):
        probe = game.clone()
        if not probe.apply_move(move):
            continue
        if probe.state_key() not in visited:
            return move
    return None


def _result(seed, won, steps, reason, game, visited):
    return PlayResult(
        seed=seed,
        policy="learned",
        won=won,
        steps=steps,
        reason=reason,
        home_cards=sum(len(stack) for stack in game.home_cells.values()),
        visited_states=len(visited),
    )


def _bundle_device(model_bundle, device=None):
    if device is None:
        return model_bundle.device
    return learned_policy.resolve_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a learned FreeCell policy.")
    parser.add_argument("--model", required=True)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dataset")
    mode_group.add_argument("--seed", type=int)
    mode_group.add_argument("--seeds", type=int, nargs="+")
    mode_group.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if args.seed_start is not None and args.seed_count is None:
        parser.error("--seed-count is required with --seed-start")
    if args.seed_count is not None and args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.seed_count is not None and args.seed_start is None:
        parser.error("--seed-count can only be used with --seed-start")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    return args


def seeds_from_args(args):
    if args.seed is not None:
        return [args.seed]
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.model).exists():
        print(f"model does not exist: {args.model}", file=sys.stderr)
        return 1
    if args.dataset is not None and not Path(args.dataset).exists():
        print(f"dataset does not exist: {args.dataset}", file=sys.stderr)
        return 1

    try:
        if args.dataset is not None:
            summary = evaluate_file(args.dataset, args.model, device=args.device)
            print_dataset_summary(summary)
        else:
            summary = evaluate_seeds(seeds_from_args(args), args.model, max_steps=args.max_steps, device=args.device)
            print_play_summary(summary)
    except (ModuleNotFoundError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"failed to evaluate learned policy: {exc}", file=sys.stderr)
        return 1
    return 0


def print_dataset_summary(summary):
    print(f"mode: {summary['mode']}")
    print(f"samples: {summary['samples']}")
    print(f"correct: {summary['correct']}")
    print(f"accuracy: {summary['accuracy']:.6f}")
    print(f"device: {summary['device']}")
    print(f"model_path: {summary['model_path']}")


def print_play_summary(summary):
    print("seed policy won reason steps home_cards visited_states")
    for result in summary["results"]:
        print(
            f"{result.seed} {result.policy} {result.won} {result.reason} "
            f"{result.steps} {result.home_cards} {result.visited_states}"
        )
    print()
    print(f"mode: {summary['mode']}")
    print(f"games: {summary['games']}")
    print(f"won: {summary['won']}")
    print(f"win_rate: {summary['win_rate']:.6f}")
    print(f"average_steps: {summary['average_steps']:.2f}")
    print(f"average_home_cards: {summary['average_home_cards']:.2f}")
    print(f"device: {summary['device']}")
    print(f"model_path: {summary['model_path']}")


if __name__ == "__main__":
    sys.exit(main())
