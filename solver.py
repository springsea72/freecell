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
    state_key: tuple
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
    nodes = [_SearchNode(start, start_key, None, None, 0)]
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

        for move in _ordered_moves(current.generate_legal_moves(), node.state_key, node.move):
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
            nodes.append(_SearchNode(child, key, node_idx, move, child_depth))

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


def _ordered_moves(moves: List[Move], state_key=None, previous_move: Optional[Move] = None) -> List[Move]:
    return sorted(moves, key=lambda move: _move_priority(move, state_key, previous_move))


def _move_priority(move: Move, state_key=None, previous_move: Optional[Move] = None):
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

    source_emptied = 0
    target_empty = 0
    reveal_score = 0
    free_release = 0
    reverse_penalty = _reverse_move_penalty(move, previous_move)

    if state_key is not None:
        columns_key, free_cells_key, _ = state_key
        if move.move_type == MoveType.COL_TO_COL:
            source = columns_key[move.from_idx]
            target = columns_key[move.to_idx]
            source_emptied = int(len(source) == move.count)
            target_empty = int(len(target) == 0)
            reveal_score = _revealed_card_score(source, move.count)
        elif move.move_type == MoveType.COL_TO_FREE:
            source = columns_key[move.from_idx]
            source_emptied = int(len(source) == 1)
            reveal_score = _revealed_card_score(source, 1)
        elif move.move_type == MoveType.FREE_TO_COL:
            target = columns_key[move.to_idx]
            target_empty = int(len(target) == 0)
            free_release = int(free_cells_key[move.from_idx] is not None)

    target_empty_penalty = target_empty if move.move_type in (MoveType.COL_TO_COL, MoveType.FREE_TO_COL) else 0
    return (
        group,
        reverse_penalty,
        target_empty_penalty,
        -free_release,
        -source_emptied,
        -move.count,
        -reveal_score,
        move.from_idx,
        -1 if move.to_idx is None else move.to_idx,
    )


def _state_score(state_key, depth: int):
    columns_key, free_cells_key, home_cells_key = state_key
    home_cards = sum(len(stack) for stack in home_cells_key)
    free_used = sum(1 for cell in free_cells_key if cell is not None)
    empty_columns = sum(1 for column in columns_key if not column)
    sequence_score, max_sequence = _sequence_metrics(columns_key)
    buried_low_penalty = _buried_low_card_penalty(columns_key)
    next_home_blockers = _next_home_blocker_penalty(columns_key, free_cells_key, home_cells_key)
    return (
        52 - home_cards,
        next_home_blockers,
        buried_low_penalty,
        free_used,
        -empty_columns,
        -sequence_score,
        -max_sequence,
        depth,
    )


def _reverse_move_penalty(move: Move, previous_move: Optional[Move]) -> int:
    if previous_move is None:
        return 0
    if (
        previous_move.move_type == MoveType.COL_TO_FREE
        and move.move_type == MoveType.FREE_TO_COL
        and previous_move.from_idx == move.to_idx
        and previous_move.to_idx == move.from_idx
    ):
        return 1
    if (
        previous_move.move_type == MoveType.FREE_TO_COL
        and move.move_type == MoveType.COL_TO_FREE
        and previous_move.to_idx == move.from_idx
        and previous_move.from_idx == move.to_idx
    ):
        return 1
    if (
        previous_move.move_type == MoveType.COL_TO_COL
        and move.move_type == MoveType.COL_TO_COL
        and previous_move.from_idx == move.to_idx
        and previous_move.to_idx == move.from_idx
        and previous_move.count == move.count
    ):
        return 1
    return 0


def _revealed_card_score(source_column, count: int) -> int:
    if len(source_column) <= count:
        return 20
    card = source_column[-count - 1]
    return 15 - card[1]


def _sequence_metrics(columns_key):
    total = 0
    max_sequence = 0
    for column in columns_key:
        length = _movable_suffix_length(column)
        total += length * length
        max_sequence = max(max_sequence, length)
    return total, max_sequence


def _movable_suffix_length(column) -> int:
    if not column:
        return 0
    length = 1
    for idx in range(len(column) - 2, -1, -1):
        if not _is_valid_sequence_pair(column[idx], column[idx + 1]):
            break
        length += 1
    return length


def _buried_low_card_penalty(columns_key) -> int:
    penalty = 0
    for column in columns_key:
        for idx, card in enumerate(column[:-1]):
            cards_above = len(column) - idx - 1
            penalty += (15 - card[1]) * cards_above
    return penalty


def _next_home_blocker_penalty(columns_key, free_cells_key, home_cells_key) -> int:
    penalty = 0
    next_home_cards = {
        suit_name: len(home_stack) + 1
        for suit_name, home_stack in zip(("SPADES", "HEARTS", "CLUBS", "DIAMONDS"), home_cells_key)
        if len(home_stack) < 13
    }
    free_cards = {card for card in free_cells_key if card is not None}

    for suit_name, value in next_home_cards.items():
        needed = (suit_name, value)
        if needed in free_cards:
            continue
        for column in columns_key:
            if needed in column:
                idx = column.index(needed)
                penalty += len(column) - idx - 1
                break
    return penalty


def _is_valid_sequence_pair(upper, lower) -> bool:
    return _card_color(upper) != _card_color(lower) and upper[1] == lower[1] + 1


def _card_color(card) -> str:
    return "red" if card[0] in ("HEARTS", "DIAMONDS") else "black"
