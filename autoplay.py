import argparse
import sys

from game_model import FreeCellGame
from solver import solve


def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Run an offline FreeCell search.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=200)
    args = parser.parse_args()

    game = FreeCellGame(seed=args.seed)
    result = solve(game, max_nodes=args.max_nodes, max_depth=args.max_depth)

    print(f"solved: {result.solved}")
    print(f"explored_nodes: {result.explored_nodes}")
    print(f"generated_nodes: {result.generated_nodes}")
    print(f"max_frontier: {result.max_frontier}")
    print(f"path_length: {len(result.moves)}")
    print(f"reason: {result.reason}")

    if result.solved:
        verification = game.clone()
        valid_path = True
        for index, move in enumerate(result.moves, start=1):
            print(f"{index}: {move}")
            if not verification.apply_move(move):
                valid_path = False
                break
        print(f"verified_won: {valid_path and verification.is_won()}")


if __name__ == "__main__":
    main()
