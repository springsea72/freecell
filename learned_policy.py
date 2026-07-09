from dataclasses import dataclass
from pathlib import Path

from dataset_builder import state_to_dict
from game_model import Move
from policy_features import (
    FEATURE_VERSION,
    MOVE_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    features_from_state_and_moves,
)
from trace_io import move_to_dict


MODEL_TYPE = "candidate_mlp_v2"
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_PROGRESS_LOSS_WEIGHT = 0.1
AUXILIARY_TARGETS = ("progress_value",)


@dataclass
class ModelBundle:
    model: object
    metadata: dict
    device: object


def load_model(path, device=None) -> ModelBundle:
    torch = _import_torch()
    resolved_device = resolve_device(device, torch=torch)
    checkpoint = _torch_load(path, resolved_device, torch=torch)
    metadata = checkpoint["metadata"]
    _validate_metadata(metadata)

    model = create_model(
        hidden_size=metadata.get("hidden_size", DEFAULT_HIDDEN_SIZE),
        torch=torch,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return ModelBundle(model=model, metadata=metadata, device=resolved_device)


def save_model(path, model, metadata: dict) -> None:
    torch = _import_torch()
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "model_state": model.state_dict()}, target)


def create_model_bundle(seed=None, device=None, hidden_size=DEFAULT_HIDDEN_SIZE) -> ModelBundle:
    torch = _import_torch()
    resolved_device = resolve_device(device, torch=torch)
    if seed is not None:
        torch.manual_seed(seed)
    metadata = base_metadata(seed=seed, hidden_size=hidden_size, training_args={})
    model = create_model(hidden_size=hidden_size, torch=torch).to(resolved_device)
    model.eval()
    return ModelBundle(model=model, metadata=metadata, device=resolved_device)


def create_model(hidden_size=DEFAULT_HIDDEN_SIZE, torch=None):
    if torch is None:
        torch = _import_torch()
    input_size = STATE_FEATURE_SIZE + MOVE_FEATURE_SIZE
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 2),
    )


def base_metadata(
    seed=None,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    training_args=None,
    progress_loss_weight=DEFAULT_PROGRESS_LOSS_WEIGHT,
) -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "state_feature_size": STATE_FEATURE_SIZE,
        "move_feature_size": MOVE_FEATURE_SIZE,
        "model_type": MODEL_TYPE,
        "hidden_size": hidden_size,
        "auxiliary_targets": list(AUXILIARY_TARGETS),
        "progress_loss_weight": progress_loss_weight,
        "seed": seed,
        "training_args": training_args or {},
    }


def score_legal_moves(game, model_bundle: ModelBundle, device=None) -> list[tuple[Move, float]]:
    torch = _import_torch()
    resolved_device = resolve_device(device, torch=torch) if device is not None else model_bundle.device
    legal_moves = game.generate_legal_moves()
    if not legal_moves:
        return []

    state = state_to_dict(game)
    legal_move_dicts = [move_to_dict(move) for move in legal_moves]
    state_features, move_features = features_from_state_and_moves(state, legal_move_dicts)

    state_tensor = torch.tensor(state_features, dtype=torch.float32, device=resolved_device)
    move_tensor = torch.tensor(move_features, dtype=torch.float32, device=resolved_device)
    state_batch = state_tensor.unsqueeze(0).repeat(len(move_features), 1)
    model_input = torch.cat([state_batch, move_tensor], dim=1)

    model = model_bundle.model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        outputs = model(model_input)
        scores = action_scores_from_output(outputs).view(-1).detach().cpu().tolist()
    return list(zip(legal_moves, [float(score) for score in scores]))


def action_scores_from_output(outputs):
    if outputs.dim() == 1:
        return outputs
    return outputs[:, 0]


def progress_scores_from_output(outputs):
    if outputs.dim() == 1:
        return None
    if outputs.size(-1) < 2:
        return None
    return outputs[:, 1]


def choose_action(game, model_bundle: ModelBundle, device=None) -> Move:
    scored_moves = score_legal_moves(game, model_bundle, device=device)
    if not scored_moves:
        raise ValueError("game has no legal moves")
    return max(scored_moves, key=lambda item: item[1])[0]


def resolve_device(device=None, torch=None):
    if torch is None:
        torch = _import_torch()
    requested = "auto" if device is None else str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda device requested but not available")
    if requested == "cpu" or requested == "cuda":
        return torch.device(requested)
    raise ValueError(f"unsupported device: {device}")


def _validate_metadata(metadata: dict) -> None:
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("model feature version does not match current features")
    if metadata.get("state_feature_size") != STATE_FEATURE_SIZE:
        raise ValueError("model state feature size does not match current features")
    if metadata.get("move_feature_size") != MOVE_FEATURE_SIZE:
        raise ValueError("model move feature size does not match current features")
    if metadata.get("model_type") != MODEL_TYPE:
        raise ValueError(f"unsupported model type: {metadata.get('model_type')}")


def _torch_load(path, device, torch):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for learned policy training/inference. "
            "Install optional dependencies from requirements-ml.txt."
        ) from exc
    return torch
