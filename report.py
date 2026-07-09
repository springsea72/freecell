import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import benchmark
import policy_baseline
import policy_player


def build_report(
    seeds,
    max_nodes,
    max_depth,
    dataset=None,
    baseline_policy="heuristic",
    player_policy="heuristic",
    max_steps=500,
    model_path=None,
    model_device="auto",
    model_max_steps=500,
) -> dict:
    seeds = list(seeds)
    solver_records = benchmark.run_benchmark(seeds, max_nodes=max_nodes, max_depth=max_depth)
    solver_summary = benchmark.summarize(solver_records)
    player_summary = policy_player.evaluate_seeds(seeds, policy=player_policy, max_steps=max_steps)

    if dataset is None:
        baseline_summary = {
            "skipped": True,
            "reason": "no dataset provided",
        }
    else:
        baseline_summary = policy_baseline.evaluate_file(dataset, policy=baseline_policy)
        baseline_summary["skipped"] = False

    result = {
        "solver": _solver_summary_to_dict(solver_summary),
        "policy_baseline": baseline_summary,
        "policy_player": _player_summary_to_dict(player_summary),
    }

    if model_path is not None:
        result["learned_policy"] = _build_learned_policy_summary(
            seeds,
            model_path=model_path,
            dataset=dataset,
            device=model_device,
            max_steps=model_max_steps,
        )

    return result


def render_text(report) -> str:
    lines = [
        "solver",
        f"games: {report['solver']['games']}",
        f"solved: {report['solver']['solved']}",
        f"solve_rate: {report['solver']['solve_rate']:.6f}",
        f"average_elapsed_seconds: {report['solver']['average_elapsed_seconds']:.6f}",
        f"average_explored_nodes: {report['solver']['average_explored_nodes']:.2f}",
        f"average_path_length_for_solved: {report['solver']['average_path_length_for_solved']:.2f}",
        "",
        "policy_baseline",
    ]

    baseline = report["policy_baseline"]
    if baseline.get("skipped"):
        lines.append(f"skipped: {baseline['reason']}")
    else:
        lines.extend(
            [
                f"policy: {baseline['policy']}",
                f"samples: {baseline['samples']}",
                f"correct: {baseline['correct']}",
                f"accuracy: {baseline['accuracy']:.6f}",
            ]
        )

    player = report["policy_player"]
    lines.extend(
        [
            "",
            "policy_player",
            f"policy: {player['policy']}",
            f"games: {player['games']}",
            f"won: {player['won']}",
            f"win_rate: {player['win_rate']:.6f}",
            f"average_steps: {player['average_steps']:.2f}",
            f"average_home_cards: {player['average_home_cards']:.2f}",
        ]
    )

    learned = report.get("learned_policy")
    if learned is not None:
        dataset = learned["dataset"]
        play = learned["play"]
        lines.extend(
            [
                "",
                "learned_policy",
                f"model_path: {learned['model_path']}",
                f"device: {learned['device']}",
                "dataset",
            ]
        )
        if dataset.get("skipped"):
            lines.append(f"skipped: {dataset['reason']}")
        else:
            lines.extend(
                [
                    f"samples: {dataset['samples']}",
                    f"correct: {dataset['correct']}",
                    f"accuracy: {dataset['accuracy']:.6f}",
                ]
            )
        lines.extend(
            [
                "play",
                f"games: {play['games']}",
                f"won: {play['won']}",
                f"win_rate: {play['win_rate']:.6f}",
                f"average_steps: {play['average_steps']:.2f}",
                f"average_home_cards: {play['average_home_cards']:.2f}",
            ]
        )
    return "\n".join(lines)


def render_json(report) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a combined FreeCell evaluation report.")
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--dataset")
    parser.add_argument("--baseline-policy", choices=sorted(policy_baseline.POLICIES), default="heuristic")
    parser.add_argument("--player-policy", choices=sorted(policy_baseline.POLICIES), default="heuristic")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--model")
    parser.add_argument("--model-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-max-steps", type=int, default=500)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.seed_start is not None and args.seed_count is None:
        parser.error("--seed-count is required with --seed-start")
    if args.seed_count is not None and args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.seed_count is not None and args.seed_start is None:
        parser.error("--seed-count can only be used with --seed-start")
    if args.max_nodes <= 0:
        parser.error("--max-nodes must be positive")
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.model_max_steps < 0:
        parser.error("--model-max-steps must be non-negative")
    return args


def seeds_from_args(args):
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def main(argv=None):
    args = parse_args(argv)
    if args.dataset is not None and not Path(args.dataset).exists():
        print(f"dataset does not exist: {args.dataset}", file=sys.stderr)
        return 1
    if args.model is not None and not Path(args.model).exists():
        print(f"model does not exist: {args.model}", file=sys.stderr)
        return 1

    try:
        report = build_report(
            seeds_from_args(args),
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            dataset=args.dataset,
            baseline_policy=args.baseline_policy,
            player_policy=args.player_policy,
            max_steps=args.max_steps,
            model_path=args.model,
            model_device=args.model_device,
            model_max_steps=args.model_max_steps,
        )
    except (ModuleNotFoundError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"failed to build report: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(render_json(report))
    else:
        print(render_text(report))
    return 0


def _solver_summary_to_dict(summary) -> dict:
    return {
        "games": summary.total_games,
        "solved": summary.solved_games,
        "solve_rate": summary.solve_rate,
        "average_elapsed_seconds": summary.average_elapsed_seconds,
        "average_explored_nodes": summary.average_explored_nodes,
        "average_path_length_for_solved": summary.average_path_length_for_solved,
    }


def _player_summary_to_dict(summary) -> dict:
    return {
        "games": summary["games"],
        "won": summary["won"],
        "win_rate": summary["win_rate"],
        "average_steps": summary["average_steps"],
        "average_home_cards": summary["average_home_cards"],
        "policy": summary["policy"],
        "results": [_to_plain_dict(result) for result in summary.get("results", [])],
    }


def _build_learned_policy_summary(seeds, model_path, dataset=None, device="auto", max_steps=500) -> dict:
    import learned_policy_eval

    play_summary = learned_policy_eval.evaluate_seeds(
        seeds,
        model_path=model_path,
        max_steps=max_steps,
        device=device,
    )

    if dataset is None:
        dataset_summary = {
            "skipped": True,
            "reason": "no dataset provided",
        }
        model_device = play_summary["device"]
    else:
        dataset_result = learned_policy_eval.evaluate_file(dataset, model_path, device=device)
        dataset_summary = {
            "skipped": False,
            "samples": dataset_result["samples"],
            "correct": dataset_result["correct"],
            "accuracy": dataset_result["accuracy"],
        }
        model_device = dataset_result["device"]

    return {
        "model_path": str(model_path),
        "device": model_device,
        "dataset": dataset_summary,
        "play": {
            "games": play_summary["games"],
            "won": play_summary["won"],
            "win_rate": play_summary["win_rate"],
            "average_steps": play_summary["average_steps"],
            "average_home_cards": play_summary["average_home_cards"],
            "results": [_to_plain_dict(result) for result in play_summary.get("results", [])],
        },
    }


def _to_plain_dict(value):
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


if __name__ == "__main__":
    sys.exit(main())
