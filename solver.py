import heapq
from dataclasses import dataclass
from itertools import count
from typing import List, Optional

from game_model import FreeCellGame, Move, MoveType


@dataclass
class SolveResult:
    solved: bool
    moves: List[Move]
    explored_nodes: int
    generated_nodes: int
    max_frontier: int
    reason: str


@dataclass
class _SearchNode:
    game: FreeCellGame
    parent_idx: Optional[int]
    move: Optional[Move]
    depth: int


def solve(game: FreeCellGame, max_nodes=50000, max_depth=200) -> SolveResult:
    start = game.clone()

    if start.is_won():
        return SolveResult(True, [], 0, 0, 0, "won")

    if max_nodes <= 0:
        return SolveResult(False, [], 0, 0, 0, "max_nodes_exceeded")

    frontier = []
    sequence = count()
    start_key = start.state_key()
    seen = {start_key}
    nodes = [_SearchNode(start, None, None, 0)]
    heapq.heappush(frontier, (_state_score(start_key, 0), 0, next(sequence), 0))

    explored_nodes = 0
    generated_nodes = 0
    max_frontier = 1
    reached_node_limit = False
    reached_depth_limit = False

    while frontier:
        _, _, _, node_idx = heapq.heappop(frontier)
        node = nodes[node_idx]
        current = node.game
        explored_nodes += 1

        if current.is_won():
            return SolveResult(
                True,
                _rebuild_path(nodes, node_idx),
                explored_nodes,
                generated_nodes,
                max_frontier,
                "won",
            )

        if node.depth >= max_depth:
            reached_depth_limit = True
            continue

        if len(nodes) >= max_nodes:
            reached_node_limit = True
            continue

        for move in _ordered_moves(current.generate_legal_moves()):
            if len(nodes) >= max_nodes:
                reached_node_limit = True
                break

            child = current.clone()
            if not child.apply_move(move):
                continue

            key = child.state_key()
            if key in seen:
                continue

            seen.add(key)
            generated_nodes += 1
            child_idx = len(nodes)
            child_depth = node.depth + 1
            nodes.append(_SearchNode(child, node_idx, move, child_depth))

            if child.is_won():
                return SolveResult(
                    True,
                    _rebuild_path(nodes, child_idx),
                    explored_nodes,
                    generated_nodes,
                    max_frontier,
                    "won",
                )

            heapq.heappush(
                frontier,
                (_state_score(key, child_depth), child_depth, next(sequence), child_idx),
            )
            max_frontier = max(max_frontier, len(frontier))

    if reached_node_limit:
        reason = "max_nodes_exceeded"
    elif reached_depth_limit:
        reason = "max_depth_exceeded"
    else:
        reason = "frontier_exhausted"
    return SolveResult(False, [], explored_nodes, generated_nodes, max_frontier, reason)


def _rebuild_path(nodes: List[_SearchNode], node_idx: int) -> List[Move]:
    moves = []
    while node_idx is not None:
        node = nodes[node_idx]
        if node.move is not None:
            moves.append(node.move)
        node_idx = node.parent_idx
    moves.reverse()
    return moves


def _ordered_moves(moves: List[Move]) -> List[Move]:
    return sorted(moves, key=_move_priority)


def _move_priority(move: Move):
    if move.move_type == MoveType.FREE_TO_HOME:
        group = 0
    elif move.move_type == MoveType.COL_TO_HOME:
        group = 1
    elif move.move_type == MoveType.FREE_TO_COL:
        group = 2
    elif move.move_type == MoveType.COL_TO_COL:
        group = 3
    else:
        group = 4
    return (group, -move.count, move.from_idx, -1 if move.to_idx is None else move.to_idx)


def _state_score(state_key, depth: int):
    columns_key, free_cells_key, home_cells_key = state_key
    home_cards = sum(len(stack) for stack in home_cells_key)
    free_used = sum(1 for cell in free_cells_key if cell is not None)
    empty_columns = sum(1 for column in columns_key if not column)
    return (52 - home_cards, free_used, -empty_columns, depth)
