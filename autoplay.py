import argparse
import sys

from game_model import FreeCellGame
from solver import solve
from trace_io import save_trace


def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def verify_solution(game, moves):
    verification = game.clone()
    for move in moves:
        if not verification.apply_move(move):
            return False
    return verification.is_won()


def main(argv=None):
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Run an offline FreeCell search.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--save-trace", default=None)
    args = parser.parse_args(argv)

    if args.save_trace is not None and args.seed is None:
        print("--save-trace requires --seed", file=sys.stderr)
        return 1

    game = FreeCellGame(seed=args.seed)
    result = solve(game, max_nodes=args.max_nodes, max_depth=args.max_depth)

    print(f"solved: {result.solved}")
    print(f"explored_nodes: {result.explored_nodes}")
    print(f"generated_nodes: {result.generated_nodes}")
    print(f"max_frontier: {result.max_frontier}")
    print(f"path_length: {len(result.moves)}")
    print(f"reason: {result.reason}")

    if result.solved:
        for index, move in enumerate(result.moves, start=1):
            print(f"{index}: {move}")
        verified_won = verify_solution(game, result.moves)
        print(f"verified_won: {verified_won}")
        if not verified_won:
            return 1
        if args.save_trace is not None:
            try:
                save_trace(args.save_trace, args.seed, args.max_nodes, args.max_depth, result)
            except (OSError, ValueError) as exc:
                print(f"failed to save trace: {exc}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
