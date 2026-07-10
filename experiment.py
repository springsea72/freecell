import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import comparison_dataset_builder
from dataset_builder import build_dataset
from game_model import FreeCellGame
import report
from solver import solve
import train_policy
from trace_io import save_trace


DEFAULT_TRAINING_SEED = 123
MANIFEST_VERSION = 1
MANIFEST_SCHEMA = "freecell_experiment_manifest"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a small end-to-end FreeCell learned-policy experiment.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=train_policy.learned_policy.DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--progress-loss-weight",
        type=float,
        default=train_policy.learned_policy.DEFAULT_PROGRESS_LOSS_WEIGHT,
    )
    parser.add_argument("--comparison-negatives-per-sample", type=int, default=0)
    parser.add_argument(
        "--comparison-negative-strategy",
        choices=comparison_dataset_builder.NEGATIVE_STRATEGIES,
        default=comparison_dataset_builder.DEFAULT_NEGATIVE_STRATEGY,
    )
    parser.add_argument("--comparison-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-format", choices=("text", "json", "both"), default="both")
    parser.add_argument("--player-max-steps", type=int, default=500)
    parser.add_argument("--model-max-steps", type=int, default=500)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    args = parser.parse_args(argv)

    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.max_nodes <= 0:
        parser.error("--max-nodes must be positive")
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.progress_loss_weight < 0:
        parser.error("--progress-loss-weight must be non-negative")
    if args.comparison_negatives_per_sample < 0:
        parser.error("--comparison-negatives-per-sample must be non-negative")
    if args.comparison_loss_weight < 0:
        parser.error("--comparison-loss-weight must be non-negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.validation_split < 0 or args.validation_split >= 1:
        parser.error("--validation-split must be >= 0 and < 1")
    if args.player_max_steps < 0:
        parser.error("--player-max-steps must be non-negative")
    if args.model_max_steps < 0:
        parser.error("--model-max-steps must be non-negative")
    return args


def run_experiment(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = output_dir / "traces"
    model_dir = output_dir / "models"
    dataset_path = output_dir / "dataset.jsonl"
    comparison_dataset_path = output_dir / "comparison_dataset.jsonl"
    model_path = model_dir / "policy.pt"
    report_text_path = output_dir / "report.txt"
    report_json_path = output_dir / "report.json"
    manifest_path = output_dir / "manifest.json"

    for path in (trace_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    _assert_under_output(
        output_dir,
        trace_dir,
        dataset_path,
        comparison_dataset_path,
        model_path,
        report_text_path,
        report_json_path,
        manifest_path,
    )

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    solved_trace_paths = _solve_and_save_traces(
        seeds,
        trace_dir,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
    )
    if not solved_trace_paths:
        raise RuntimeError("solve stage failed: no solved traces")

    dataset_summary = build_dataset(solved_trace_paths, dataset_path, skip_invalid=False)
    samples = dataset_summary["samples_written"]
    if samples <= 0:
        raise RuntimeError("dataset stage failed: no samples written")

    comparison_summary = {
        "samples_read": 0,
        "pairs_written": 0,
        "skipped_no_negative": 0,
    }
    train_comparison_dataset = None
    if args.comparison_negatives_per_sample > 0:
        comparison_summary = comparison_dataset_builder.build_comparison_dataset(
            dataset_path,
            comparison_dataset_path,
            negatives_per_sample=args.comparison_negatives_per_sample,
            negative_strategy=args.comparison_negative_strategy,
            seed=args.training_seed,
        )
        train_comparison_dataset = comparison_dataset_path

    train_summary = train_policy.train(
        SimpleNamespace(
            dataset=dataset_path,
            comparison_dataset=train_comparison_dataset,
            output=model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            progress_loss_weight=args.progress_loss_weight,
            comparison_loss_weight=args.comparison_loss_weight,
            lr=args.lr,
            seed=args.training_seed,
            device=args.device,
            validation_split=args.validation_split,
        )
    )

    report_data = report.build_report(
        seeds,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        dataset=dataset_path,
        baseline_policy="heuristic",
        player_policy="heuristic",
        max_steps=args.player_max_steps,
        model_path=model_path,
        model_device=args.device,
        model_max_steps=args.model_max_steps,
    )

    report_paths = _write_reports(report_data, args.report_format, report_text_path, report_json_path)

    summary = {
        "seeds": len(seeds),
        "solved_traces": len(solved_trace_paths),
        "samples": samples,
        "model_path": str(model_path),
        "report_path": _summary_report_path(report_paths),
        "manifest_path": str(manifest_path),
        "device": train_summary["device"],
        "train_accuracy": train_summary["train_accuracy"],
        "validation_accuracy": train_summary["validation_accuracy"],
        "comparison_samples": comparison_summary["pairs_written"],
        "comparison_loss": train_summary.get("comparison_loss", 0.0),
        "comparison_accuracy": train_summary.get("comparison_accuracy", 0.0),
        "comparison_dataset_path": None if train_comparison_dataset is None else str(train_comparison_dataset),
        "trace_dir": str(trace_dir),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
    }
    _write_manifest(
        manifest_path,
        args=args,
        seeds=seeds,
        solved_trace_paths=solved_trace_paths,
        dataset_path=dataset_path,
        comparison_dataset_path=train_comparison_dataset,
        comparison_summary=comparison_summary,
        model_path=model_path,
        report_paths=report_paths,
        train_summary=train_summary,
        summary=summary,
    )
    return summary


def _solve_and_save_traces(seeds, trace_dir: Path, max_nodes: int, max_depth: int) -> list[Path]:
    paths = []
    for seed in seeds:
        result = solve(FreeCellGame(seed=seed), max_nodes=max_nodes, max_depth=max_depth)
        if not result.solved:
            continue
        path = trace_dir / f"seed_{seed:06d}.json"
        save_trace(path, seed=seed, max_nodes=max_nodes, max_depth=max_depth, result=result)
        paths.append(path)
    return paths


def _write_reports(report_data, report_format, text_path: Path, json_path: Path) -> list[Path]:
    paths = []
    if report_format in ("text", "both"):
        text_path.write_text(report.render_text(report_data) + "\n", encoding="utf-8")
        paths.append(text_path)
    if report_format in ("json", "both"):
        json_path.write_text(report.render_json(report_data) + "\n", encoding="utf-8")
        paths.append(json_path)
    return paths


def _write_manifest(
    manifest_path: Path,
    *,
    args,
    seeds: list[int],
    solved_trace_paths: list[Path],
    dataset_path: Path,
    comparison_dataset_path,
    comparison_summary: dict,
    model_path: Path,
    report_paths: list[Path],
    train_summary: dict,
    summary: dict,
) -> None:
    manifest = {
        "version": MANIFEST_VERSION,
        "schema": MANIFEST_SCHEMA,
        "parameters": {
            "max_nodes": args.max_nodes,
            "max_depth": args.max_depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "progress_loss_weight": args.progress_loss_weight,
            "comparison_negatives_per_sample": args.comparison_negatives_per_sample,
            "comparison_negative_strategy": args.comparison_negative_strategy,
            "comparison_loss_weight": args.comparison_loss_weight,
            "lr": args.lr,
            "device": args.device,
            "validation_split": args.validation_split,
            "report_format": args.report_format,
            "player_max_steps": args.player_max_steps,
            "model_max_steps": args.model_max_steps,
            "training_seed": args.training_seed,
        },
        "seed_range": {
            "start": args.seed_start,
            "count": args.seed_count,
            "seeds": seeds,
        },
        "summary": {
            "seeds": summary["seeds"],
            "solved_traces": summary["solved_traces"],
            "samples": summary["samples"],
            "model_path": summary["model_path"],
            "report_path": summary["report_path"],
            "manifest_path": summary["manifest_path"],
            "device": summary["device"],
            "train_accuracy": summary["train_accuracy"],
            "validation_accuracy": summary["validation_accuracy"],
            "comparison_samples": summary["comparison_samples"],
            "comparison_loss": summary["comparison_loss"],
            "comparison_accuracy": summary["comparison_accuracy"],
        },
        "paths": {
            "dataset": str(dataset_path),
            "comparison_dataset": None if comparison_dataset_path is None else str(comparison_dataset_path),
            "model": str(model_path),
            "reports": [str(path) for path in report_paths],
            "traces": [str(path) for path in solved_trace_paths],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "progress_loss_weight": args.progress_loss_weight,
            "comparison_negatives_per_sample": args.comparison_negatives_per_sample,
            "comparison_negative_strategy": args.comparison_negative_strategy,
            "comparison_loss_weight": args.comparison_loss_weight,
            "comparison_dataset": None if comparison_dataset_path is None else str(comparison_dataset_path),
            "comparison_samples": comparison_summary["pairs_written"],
            "lr": args.lr,
            "validation_split": args.validation_split,
            "device": summary["device"],
            "requested_device": args.device,
            "seed": args.training_seed,
        },
        "train_summary": train_summary,
        "comparison_summary": comparison_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_report_path(paths: list[Path]) -> str:
    if len(paths) == 1:
        return str(paths[0])
    return ",".join(str(path) for path in paths)


def _assert_under_output(output_dir: Path, *paths: Path) -> None:
    root = output_dir.resolve()
    for path in paths:
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"output path escapes output directory: {path}")


def print_summary(summary) -> None:
    for key in (
        "seeds",
        "solved_traces",
        "samples",
        "model_path",
        "report_path",
        "manifest_path",
        "device",
        "train_accuracy",
        "validation_accuracy",
        "comparison_samples",
        "comparison_loss",
        "comparison_accuracy",
    ):
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main(argv=None):
    try:
        args = parse_args(argv)
        summary = run_experiment(args)
    except SystemExit:
        raise
    except (ModuleNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
