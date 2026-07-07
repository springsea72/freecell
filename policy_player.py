import argparse
import sys
from dataclasses import dataclass

from dataset_builder import state_to_dict
from game_model import FreeCellGame
from policy_baseline import choose_action
from trace_io import move_from_dict, move_to_dict


@dataclass
class PlayResult:
    seed: int
    policy: str
    won: bool
    steps: int
    reason: str
    home_cards: int
    visited_states: int


def play_game(seed, policy="heuristic", max_steps=500) -> PlayResult:
    game = FreeCellGame(seed=seed)
    visited = {game.state_key()}
    steps = 0

    if game.is_won():
        return _result(seed, policy, True, steps, "won", game, visited)

    while steps < max_steps:
        legal_moves = game.generate_legal_moves()
        if not legal_moves:
            reason = "won" if game.is_won() else "no_legal_moves"
            return _result(seed, policy, game.is_won(), steps, reason, game, visited)

        selected_move = _choose_non_looping_move(game, legal_moves, visited, policy)
        if selected_move is None:
            return _result(seed, policy, False, steps, "loop_detected", game, visited)

        if not game.apply_move(selected_move):
            return _result(seed, policy, False, steps, "invalid_action", game, visited)

        steps += 1
        visited.add(game.state_key())

        if game.is_won():
            return _result(seed, policy, True, steps, "won", game, visited)

    return _result(seed, policy, game.is_won(), steps, "max_steps", game, visited)


def evaluate_seeds(seeds, policy="heuristic", max_steps=500) -> dict:
    results = [play_game(seed, policy=policy, max_steps=max_steps) for seed in seeds]
    games = len(results)
    won = sum(1 for result in results if result.won)
    return {
        "games": games,
        "won": won,
        "win_rate": won / games if games else 0.0,
        "average_steps": sum(result.steps for result in results) / games if games else 0.0,
        "average_home_cards": sum(result.home_cards for result in results) / games if games else 0.0,
        "policy": policy,
        "results": results,
    }


def _choose_non_looping_move(game, legal_moves, visited, policy):
    candidates = [move_to_dict(move) for move in legal_moves]
    while candidates:
        sample = {
            "state": state_to_dict(game),
            "legal_moves": candidates,
            "action": None,
        }
        action = choose_action(sample, policy=policy)
        move = move_from_dict(action)

        probe = game.clone()
        if not probe.apply_move(move):
            candidates.remove(action)
            continue
        if probe.state_key() not in visited:
            return move
        candidates.remove(action)
    return None


def _result(seed, policy, won, steps, reason, game, visited):
    return PlayResult(
        seed=seed,
        policy=policy,
        won=won,
        steps=steps,
        reason=reason,
        home_cards=_home_card_count(game),
        visited_states=len(visited),
    )


def _home_card_count(game) -> int:
    return sum(len(stack) for stack in game.home_cells.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate simple policies by directly playing FreeCell.")
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--policy", default="heuristic")
    parser.add_argument("--max-steps", type=int, default=500)
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
    try:
        summary = evaluate_seeds(seeds_from_args(args), policy=args.policy, max_steps=args.max_steps)
    except ValueError as exc:
        print(f"failed to play policy: {exc}", file=sys.stderr)
        return 1

    print("seed policy won reason steps home_cards visited_states")
    for result in summary["results"]:
        print(
            f"{result.seed} {result.policy} {result.won} {result.reason} "
            f"{result.steps} {result.home_cards} {result.visited_states}"
        )
    print()
    print("summary")
    print(f"games: {summary['games']}")
    print(f"won: {summary['won']}")
    print(f"win_rate: {summary['win_rate']:.6f}")
    print(f"average_steps: {summary['average_steps']:.2f}")
    print(f"average_home_cards: {summary['average_home_cards']:.2f}")
    print(f"policy: {summary['policy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
