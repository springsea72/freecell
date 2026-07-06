import json
from pathlib import Path

from game_model import FreeCellGame, Move, MoveType


TRACE_VERSION = 1


def move_to_dict(move: Move) -> dict:
    return {
        "move_type": move.move_type.name,
        "from_idx": move.from_idx,
        "to_idx": move.to_idx,
        "count": move.count,
    }


def move_from_dict(data) -> Move:
    return Move(
        MoveType[data["move_type"]],
        data["from_idx"],
        data.get("to_idx"),
        data.get("count", 1),
    )


def save_trace(path, seed, max_nodes, max_depth, result) -> None:
    if not result.solved:
        raise ValueError("only solved traces can be saved")

    trace = {
        "version": TRACE_VERSION,
        "seed": seed,
        "max_nodes": max_nodes,
        "max_depth": max_depth,
        "solved": result.solved,
        "reason": result.reason,
        "moves": [move_to_dict(move) for move in result.moves],
        "stats": {
            "explored_nodes": result.explored_nodes,
            "generated_nodes": result.generated_nodes,
            "max_frontier": result.max_frontier,
            "path_length": len(result.moves),
        },
    }

    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_trace(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moves_from_trace(trace) -> list[Move]:
    return [move_from_dict(move_data) for move_data in trace.get("moves", [])]


def verify_trace(trace) -> bool:
    if trace.get("version") != TRACE_VERSION:
        return False
    if not trace.get("solved"):
        return False
    if trace.get("seed") is None:
        return False

    game = FreeCellGame(seed=trace["seed"])
    try:
        moves = moves_from_trace(trace)
    except (KeyError, TypeError, ValueError):
        return False

    for move in moves:
        if not game.apply_move(move):
            return False
    return game.is_won()
