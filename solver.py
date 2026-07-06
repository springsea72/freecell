import heapq
from dataclasses import dataclass
from itertools import count
from typing import List

from game_model import FreeCellGame, Move, MoveType


@dataclass
class SolveResult:
    solved: bool
    moves: List[Move]
    explored_nodes: int
    generated_nodes: int
    max_frontier: int
    reason: str


def solve(game: FreeCellGame, max_nodes=50000, max_depth=200) -> SolveResult:
    start = game.clone()

    if start.is_won():
        return SolveResult(True, [], 0, 0, 0, "won")

    if max_nodes <= 0:
        return SolveResult(False, [], 0, 0, 0, "max_nodes_exceeded")

    frontier = []
    sequence = count()
    seen = {start.state_key()}
    heapq.heappush(frontier, (_state_score(start, 0), 0, next(sequence), start, []))

    explored_nodes = 0
    generated_nodes = 0
    max_frontier = 1
    reached_node_limit = False
    reached_depth_limit = False

    while frontier:
        _, depth, _, current, path = heapq.heappop(frontier)
        explored_nodes += 1

        if current.is_won():
            return SolveResult(True, path, explored_nodes, generated_nodes, max_frontier, "won")

        if depth >= max_depth:
            reached_depth_limit = True
            continue

        for move in _ordered_moves(current.generate_legal_moves()):
            child = current.clone()
            if not child.apply_move(move):
                continue

            key = child.state_key()
            if key in seen:
                continue

            if generated_nodes + 1 >= max_nodes:
                reached_node_limit = True
                continue

            seen.add(key)
            generated_nodes += 1
            child_path = path + [move]

            if child.is_won():
                return SolveResult(
                    True,
                    child_path,
                    explored_nodes,
                    generated_nodes,
                    max_frontier,
                    "won",
                )

            heapq.heappush(
                frontier,
                (_state_score(child, len(child_path)), len(child_path), next(sequence), child, child_path),
            )
            max_frontier = max(max_frontier, len(frontier))

    if reached_node_limit:
        reason = "max_nodes_exceeded"
    elif reached_depth_limit:
        reason = "max_depth_exceeded"
    else:
        reason = "frontier_exhausted"
    return SolveResult(False, [], explored_nodes, generated_nodes, max_frontier, reason)


def _ordered_moves(moves: List[Move]) -> List[Move]:
    return sorted(moves, key=_move_priority)


def _move_priority(move: Move):
    if move.move_type in (MoveType.COL_TO_HOME, MoveType.FREE_TO_HOME):
        group = 0
    elif move.move_type == MoveType.FREE_TO_COL:
        group = 1
    elif move.move_type == MoveType.COL_TO_COL:
        group = 2
    else:
        group = 3
    return (group, -move.count, move.from_idx, -1 if move.to_idx is None else move.to_idx)


def _state_score(game: FreeCellGame, depth: int):
    home_cards = sum(len(stack) for stack in game.home_cells.values())
    free_used = sum(1 for cell in game.free_cells if cell is not None)
    empty_columns = sum(1 for column in game.columns if not column)
    return (52 - home_cards, free_used, -empty_columns, depth)
