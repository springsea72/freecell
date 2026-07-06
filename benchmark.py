import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List

from game_model import FreeCellGame
from solver import solve


@dataclass
class BenchmarkRecord:
    seed: int
    solved: bool
    reason: str
    explored_nodes: int
    generated_nodes: int
    max_frontier: int
    path_length: int
    elapsed_seconds: float


@dataclass
class BenchmarkSummary:
    total_games: int
    solved_games: int
    solve_rate: float
    average_elapsed_seconds: float
    average_explored_nodes: float
    average_path_length_for_solved: float


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark the offline FreeCell solver.")
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list.")
    seed_group.add_argument("--seed-start", type=int, help="First seed in a generated range.")
    parser.add_argument("--seed-count", type=int, help="Number of seeds when using --seed-start.")
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--format", choices=("text", "csv"), default="text")
    args = parser.parse_args(argv)

    if args.seeds is not None and args.seed_count is not None:
        parser.error("--seed-count can only be used with --seed-start")
    if args.seed_start is not None and args.seed_count is None:
        parser.error("--seed-count is required with --seed-start")
    if args.seed_count is not None and args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.max_nodes <= 0:
        parser.error("--max-nodes must be positive")
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")
    return args


def seeds_from_args(args) -> List[int]:
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def run_benchmark(seeds: Iterable[int], max_nodes: int, max_depth: int) -> List[BenchmarkRecord]:
    records = []
    for seed in seeds:
        game = FreeCellGame(seed=seed)
        started = time.perf_counter()
        result = solve(game, max_nodes=max_nodes, max_depth=max_depth)
        elapsed_seconds = time.perf_counter() - started
        records.append(
            BenchmarkRecord(
                seed=seed,
                solved=result.solved,
                reason=result.reason,
                explored_nodes=result.explored_nodes,
                generated_nodes=result.generated_nodes,
                max_frontier=result.max_frontier,
                path_length=len(result.moves),
                elapsed_seconds=elapsed_seconds,
            )
        )
    return records


def summarize(records: List[BenchmarkRecord]) -> BenchmarkSummary:
    total_games = len(records)
    solved_records = [record for record in records if record.solved]
    solved_games = len(solved_records)

    if total_games == 0:
        return BenchmarkSummary(0, 0, 0.0, 0.0, 0.0, 0.0)

    average_elapsed_seconds = sum(record.elapsed_seconds for record in records) / total_games
    average_explored_nodes = sum(record.explored_nodes for record in records) / total_games
    average_path_length_for_solved = (
        sum(record.path_length for record in solved_records) / solved_games if solved_games else 0.0
    )
    return BenchmarkSummary(
        total_games=total_games,
        solved_games=solved_games,
        solve_rate=solved_games / total_games,
        average_elapsed_seconds=average_elapsed_seconds,
        average_explored_nodes=average_explored_nodes,
        average_path_length_for_solved=average_path_length_for_solved,
    )


def format_text(records: List[BenchmarkRecord], summary: BenchmarkSummary) -> str:
    lines = [
        "seed solved reason explored_nodes generated_nodes max_frontier path_length elapsed_seconds",
    ]
    for record in records:
        lines.append(
            " ".join(
                [
                    str(record.seed),
                    str(record.solved),
                    record.reason,
                    str(record.explored_nodes),
                    str(record.generated_nodes),
                    str(record.max_frontier),
                    str(record.path_length),
                    f"{record.elapsed_seconds:.6f}",
                ]
            )
        )

    lines.extend(
        [
            "",
            "summary",
            f"total_games: {summary.total_games}",
            f"solved_games: {summary.solved_games}",
            f"solve_rate: {summary.solve_rate:.6f}",
            f"average_elapsed_seconds: {summary.average_elapsed_seconds:.6f}",
            f"average_explored_nodes: {summary.average_explored_nodes:.2f}",
            f"average_path_length_for_solved: {summary.average_path_length_for_solved:.2f}",
        ]
    )
    return "\n".join(lines)


def format_csv(records: List[BenchmarkRecord], summary: BenchmarkSummary) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "seed",
            "solved",
            "reason",
            "explored_nodes",
            "generated_nodes",
            "max_frontier",
            "path_length",
            "elapsed_seconds",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.seed,
                record.solved,
                record.reason,
                record.explored_nodes,
                record.generated_nodes,
                record.max_frontier,
                record.path_length,
                f"{record.elapsed_seconds:.6f}",
            ]
        )

    writer.writerow([])
    writer.writerow(["metric", "value"])
    writer.writerow(["total_games", summary.total_games])
    writer.writerow(["solved_games", summary.solved_games])
    writer.writerow(["solve_rate", f"{summary.solve_rate:.6f}"])
    writer.writerow(["average_elapsed_seconds", f"{summary.average_elapsed_seconds:.6f}"])
    writer.writerow(["average_explored_nodes", f"{summary.average_explored_nodes:.2f}"])
    writer.writerow(["average_path_length_for_solved", f"{summary.average_path_length_for_solved:.2f}"])
    return output.getvalue().rstrip("\n")


def render(records: List[BenchmarkRecord], output_format: str) -> str:
    summary = summarize(records)
    if output_format == "csv":
        return format_csv(records, summary)
    return format_text(records, summary)


def main(argv=None):
    args = parse_args(argv)
    records = run_benchmark(
        seeds_from_args(args),
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
    )
    print(render(records, args.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
