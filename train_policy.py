import argparse
import json
import random
import sys
from pathlib import Path

import learned_policy
from policy_features import features_from_sample, features_from_state_and_moves


COMPARISON_TARGETS = ("trace_action_preferred_over_sampled_legal_move",)


def load_samples(path) -> list[dict]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                features_from_sample(sample)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid sample at line {line_number}: {exc}") from exc
            samples.append(sample)
    if not samples:
        raise ValueError("dataset contains no samples")
    return samples


def load_comparison_samples(path) -> list[dict]:
    if path is None:
        return []
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                comparison_features_from_pair(pair)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid comparison sample at line {line_number}: {exc}") from exc
            samples.append(pair)
    return samples


def split_samples(samples, validation_split: float, seed: int):
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError("validation_split must be >= 0 and < 1")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(len(shuffled) * validation_split)
    if validation_count >= len(shuffled) and shuffled:
        validation_count = len(shuffled) - 1
    validation_samples = shuffled[:validation_count]
    train_samples = shuffled[validation_count:]
    if not train_samples:
        raise ValueError("training split contains no samples")
    return train_samples, validation_samples


def train(args) -> dict:
    torch = learned_policy._import_torch()
    device = learned_policy.resolve_device(args.device, torch=torch)
    _set_seed(args.seed, torch)

    samples = load_samples(args.dataset)
    train_samples, validation_samples = split_samples(samples, args.validation_split, args.seed)

    hidden_size = getattr(args, "hidden_size", learned_policy.DEFAULT_HIDDEN_SIZE)
    batch_size = getattr(args, "batch_size", 1)
    progress_loss_weight = getattr(
        args,
        "progress_loss_weight",
        learned_policy.DEFAULT_PROGRESS_LOSS_WEIGHT,
    )
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if progress_loss_weight < 0:
        raise ValueError("progress_loss_weight must be >= 0")
    comparison_dataset = getattr(args, "comparison_dataset", None)
    comparison_loss_weight = getattr(args, "comparison_loss_weight", 0.0)
    if comparison_loss_weight < 0:
        raise ValueError("comparison_loss_weight must be >= 0")
    comparison_samples = load_comparison_samples(comparison_dataset) if comparison_dataset else []

    model = learned_policy.create_model(hidden_size=hidden_size, torch=torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss = 0.0
    train_accuracy = 0.0
    progress_loss = 0.0
    comparison_loss = 0.0
    comparison_accuracy = 0.0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_samples)
        if comparison_samples:
            random.Random(args.seed + epoch).shuffle(comparison_samples)
        train_loss, train_accuracy, progress_loss = _run_epoch(
            model,
            train_samples,
            optimizer,
            device,
            torch,
            batch_size=batch_size,
            progress_loss_weight=progress_loss_weight,
            comparison_samples=comparison_samples,
            comparison_loss_weight=comparison_loss_weight,
        )

    validation_accuracy = _evaluate(model, validation_samples, device, torch) if validation_samples else 0.0
    if comparison_samples:
        comparison_loss, comparison_accuracy = _evaluate_comparison_samples(model, comparison_samples, device, torch)

    metadata = learned_policy.base_metadata(
        seed=args.seed,
        hidden_size=hidden_size,
        progress_loss_weight=progress_loss_weight,
        training_args={
            "epochs": args.epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "lr": args.lr,
            "progress_loss_weight": progress_loss_weight,
            "comparison_dataset": None if comparison_dataset is None else str(comparison_dataset),
            "comparison_loss_weight": comparison_loss_weight,
            "comparison_targets": list(COMPARISON_TARGETS),
            "validation_split": args.validation_split,
            "device": str(device),
            "batch_accumulation": True,
        },
    )
    metadata["comparison_loss_weight"] = comparison_loss_weight
    metadata["comparison_targets"] = list(COMPARISON_TARGETS)
    learned_policy.save_model(args.output, model, metadata)

    return {
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "lr": args.lr,
        "device": str(device),
        "train_loss": train_loss,
        "progress_loss": progress_loss,
        "progress_loss_weight": progress_loss_weight,
        "comparison_samples": len(comparison_samples),
        "comparison_loss": comparison_loss,
        "comparison_accuracy": comparison_accuracy,
        "comparison_loss_weight": comparison_loss_weight,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "model_path": str(args.output),
    }


def _run_epoch(
    model,
    samples,
    optimizer,
    device,
    torch,
    batch_size=1,
    progress_loss_weight=0.0,
    comparison_samples=None,
    comparison_loss_weight=0.0,
):
    model.train()
    total_loss = 0.0
    total_progress_loss = 0.0
    progress_count = 0
    correct = 0
    batch_size = max(1, int(batch_size))
    optimizer.zero_grad()
    pending_losses = []

    for sample_index, sample in enumerate(samples, start=1):
        loss, predicted, target, progress_loss = _sample_loss(
            model,
            sample,
            device,
            torch,
            progress_loss_weight=progress_loss_weight,
        )
        if comparison_samples and comparison_loss_weight > 0:
            pair = comparison_samples[(sample_index - 1) % len(comparison_samples)]
            pair_loss, _ = _comparison_loss(model, pair, device, torch)
            loss = loss + comparison_loss_weight * pair_loss
        pending_losses.append(loss)
        total_loss += float(loss.detach().cpu())
        if progress_loss is not None:
            total_progress_loss += progress_loss
            progress_count += 1
        correct += int(predicted == target)

        if len(pending_losses) >= batch_size or sample_index == len(samples):
            batch_loss = torch.stack(pending_losses).mean()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pending_losses = []
    average_progress_loss = total_progress_loss / progress_count if progress_count else 0.0
    return total_loss / len(samples), correct / len(samples), average_progress_loss


def _evaluate(model, samples, device, torch) -> float:
    if not samples:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in samples:
            _, predicted, target, _ = _sample_loss(model, sample, device, torch, compute_loss=False)
            correct += int(predicted == target)
    return correct / len(samples)


def _evaluate_comparison_samples(model, samples, device, torch) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for sample in samples:
            loss, is_correct = _comparison_loss(model, sample, device, torch)
            total_loss += float(loss.detach().cpu())
            correct += int(is_correct)
    return total_loss / len(samples), correct / len(samples)


def _sample_loss(model, sample, device, torch, compute_loss=True, progress_loss_weight=0.0):
    state_features, move_features, action_index = features_from_sample(sample)
    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
    move_tensor = torch.tensor(move_features, dtype=torch.float32, device=device)
    state_batch = state_tensor.unsqueeze(0).repeat(len(move_features), 1)
    model_input = torch.cat([state_batch, move_tensor], dim=1)
    outputs = model(model_input)
    action_scores = learned_policy.action_scores_from_output(outputs).view(1, -1)
    target = torch.tensor([action_index], dtype=torch.long, device=device)
    predicted = int(action_scores.argmax(dim=1).item())
    progress_loss = _progress_loss(outputs, sample, action_index, device, torch)
    if compute_loss:
        action_loss = torch.nn.functional.cross_entropy(action_scores, target)
        loss = action_loss
        if progress_loss is not None and progress_loss_weight > 0:
            loss = loss + progress_loss_weight * progress_loss
    else:
        loss = torch.tensor(0.0, device=device)
    progress_loss_value = None if progress_loss is None else float(progress_loss.detach().cpu())
    return loss, predicted, action_index, progress_loss_value


def _comparison_loss(model, pair, device, torch):
    state_features, preferred_features, rejected_features = comparison_features_from_pair(pair)
    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
    move_tensor = torch.tensor([preferred_features, rejected_features], dtype=torch.float32, device=device)
    state_batch = state_tensor.unsqueeze(0).repeat(2, 1)
    model_input = torch.cat([state_batch, move_tensor], dim=1)
    outputs = model(model_input)
    action_scores = learned_policy.action_scores_from_output(outputs)
    score_diff = action_scores[0] - action_scores[1]
    target = torch.ones((), dtype=torch.float32, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score_diff, target)
    return loss, bool(score_diff.detach().cpu().item() > 0)


def comparison_features_from_pair(pair):
    state_features, move_features = features_from_state_and_moves(
        pair["state"],
        [pair["preferred_action"], pair["rejected_action"]],
    )
    return state_features, move_features[0], move_features[1]


def _progress_loss(outputs, sample, action_index, device, torch):
    target_value = _progress_target(sample)
    if target_value is None:
        return None
    progress_scores = learned_policy.progress_scores_from_output(outputs)
    if progress_scores is None:
        return None
    predicted_value = progress_scores[action_index].view(1)
    target = torch.tensor([target_value], dtype=torch.float32, device=device)
    return torch.nn.functional.mse_loss(predicted_value, target)


def _progress_target(sample):
    progress = sample.get("progress")
    if not isinstance(progress, dict):
        return None
    remaining_after = progress.get("remaining_moves_after")
    remaining = progress.get("remaining_moves", sample.get("remaining_moves"))
    if remaining_after is None or remaining is None:
        return None
    path_length = int(sample.get("step_index", 0)) + max(int(remaining), 0)
    if path_length <= 0:
        return 0.0
    value = 1.0 - max(int(remaining_after), 0) / path_length
    return min(max(value, 0.0), 1.0)


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a minimal imitation policy on FreeCell JSONL samples.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to accumulate before each optimizer step.",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=learned_policy.DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--progress-loss-weight", type=float, default=learned_policy.DEFAULT_PROGRESS_LOSS_WEIGHT)
    parser.add_argument("--comparison-dataset")
    parser.add_argument("--comparison-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-split", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"dataset does not exist: {dataset_path}", file=sys.stderr)
        return 1
    if args.epochs < 1:
        print("epochs must be >= 1", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("batch-size must be >= 1", file=sys.stderr)
        return 1
    if args.hidden_size < 1:
        print("hidden-size must be >= 1", file=sys.stderr)
        return 1
    if args.progress_loss_weight < 0:
        print("progress-loss-weight must be >= 0", file=sys.stderr)
        return 1
    if args.comparison_loss_weight < 0:
        print("comparison-loss-weight must be >= 0", file=sys.stderr)
        return 1
    if args.comparison_dataset is not None and not Path(args.comparison_dataset).exists():
        print(f"comparison dataset does not exist: {args.comparison_dataset}", file=sys.stderr)
        return 1

    try:
        summary = train(args)
    except (ModuleNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"failed to train policy: {exc}", file=sys.stderr)
        return 1

    for key in (
        "samples",
        "train_samples",
        "validation_samples",
        "epochs",
        "batch_size",
        "hidden_size",
        "lr",
        "device",
        "train_loss",
        "progress_loss",
        "progress_loss_weight",
        "comparison_samples",
        "comparison_loss",
        "comparison_accuracy",
        "comparison_loss_weight",
        "train_accuracy",
        "validation_accuracy",
        "model_path",
    ):
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
