import argparse
import json
import random
import sys
from pathlib import Path

import learned_policy
from policy_features import features_from_sample


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
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model = learned_policy.create_model(hidden_size=hidden_size, torch=torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss = 0.0
    train_accuracy = 0.0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_samples)
        train_loss, train_accuracy = _run_epoch(
            model,
            train_samples,
            optimizer,
            device,
            torch,
            batch_size=batch_size,
        )

    validation_accuracy = _evaluate(model, validation_samples, device, torch) if validation_samples else 0.0

    metadata = learned_policy.base_metadata(
        seed=args.seed,
        hidden_size=hidden_size,
        training_args={
            "epochs": args.epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "lr": args.lr,
            "validation_split": args.validation_split,
            "device": str(device),
            "batch_accumulation": True,
        },
    )
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
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "model_path": str(args.output),
    }


def _run_epoch(model, samples, optimizer, device, torch, batch_size=1):
    model.train()
    total_loss = 0.0
    correct = 0
    batch_size = max(1, int(batch_size))
    optimizer.zero_grad()
    pending_losses = []

    for sample_index, sample in enumerate(samples, start=1):
        loss, predicted, target = _sample_loss(model, sample, device, torch)
        pending_losses.append(loss)
        total_loss += float(loss.detach().cpu())
        correct += int(predicted == target)

        if len(pending_losses) >= batch_size or sample_index == len(samples):
            batch_loss = torch.stack(pending_losses).mean()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pending_losses = []
    return total_loss / len(samples), correct / len(samples)


def _evaluate(model, samples, device, torch) -> float:
    if not samples:
        return 0.0
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in samples:
            _, predicted, target = _sample_loss(model, sample, device, torch, compute_loss=False)
            correct += int(predicted == target)
    return correct / len(samples)


def _sample_loss(model, sample, device, torch, compute_loss=True):
    state_features, move_features, action_index = features_from_sample(sample)
    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
    move_tensor = torch.tensor(move_features, dtype=torch.float32, device=device)
    state_batch = state_tensor.unsqueeze(0).repeat(len(move_features), 1)
    model_input = torch.cat([state_batch, move_tensor], dim=1)
    scores = model(model_input).view(1, -1)
    target = torch.tensor([action_index], dtype=torch.long, device=device)
    predicted = int(scores.argmax(dim=1).item())
    if compute_loss:
        loss = torch.nn.functional.cross_entropy(scores, target)
    else:
        loss = torch.tensor(0.0, device=device)
    return loss, predicted, action_index


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
