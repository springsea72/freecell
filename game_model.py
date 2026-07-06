import copy
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class MoveType(Enum):
    COL_TO_COL = auto()
    COL_TO_FREE = auto()
    FREE_TO_COL = auto()
    COL_TO_HOME = auto()
    FREE_TO_HOME = auto()


@dataclass(frozen=True)
class Move:
    move_type: MoveType
    from_idx: int
    to_idx: Optional[int] = None
    count: int = 1

    def __str__(self):
        count_suffix = f" x{self.count}" if self.count > 1 else ""
        if self.move_type == MoveType.COL_TO_COL:
            return f"列 {self.from_idx} → 列 {self.to_idx}{count_suffix}"
        if self.move_type == MoveType.COL_TO_FREE:
            if self.to_idx is None:
                return f"列 {self.from_idx} → 空当"
            return f"列 {self.from_idx} → 空当 {self.to_idx}"
        if self.move_type == MoveType.FREE_TO_COL:
            return f"空当 {self.from_idx} → 列 {self.to_idx}"
        if self.move_type == MoveType.COL_TO_HOME:
            return f"列 {self.from_idx} → 目标堆"
        if self.move_type == MoveType.FREE_TO_HOME:
            return f"空当 {self.from_idx} → 目标堆"
        return "未知动作"


class Suit(Enum):
    SPADES = "♠"
    HEARTS = "♥"
    CLUBS = "♣"
    DIAMONDS = "♦"


class Card:
    def __init__(self, suit: Suit, value: int):
        self.suit = suit
        self.value = value  # 1~13，对应 A ~ K

    def color(self):
        return "red" if self.suit in (Suit.HEARTS, Suit.DIAMONDS) else "black"

    def __str__(self):
        value_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
        v = value_map.get(self.value, str(self.value))
        return f"{v}{self.suit.value}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.value == other.value

    def __hash__(self):
        return hash((self.suit, self.value))


class FreeCellGame:
    def __init__(self, seed=None, deal=None):
        self.columns: List[List[Card]] = [[] for _ in range(8)]  # 8 列牌堆
        self.free_cells: List[Optional[Card]] = [None] * 4       # 4 个空位
        self.home_cells: dict[Suit, List[Card]] = {suit: [] for suit in Suit}

        if deal is None:
            self.init_deck(seed=seed)
        else:
            self.load_deal(deal)

    def init_deck(self, seed=None):
        self.columns = [[] for _ in range(8)]
        deck = [Card(suit, value) for suit in Suit for value in range(1, 14)]
        rng = random.Random(seed)
        rng.shuffle(deck)

        for i, card in enumerate(deck):
            self.columns[i % 8].append(card)

    def load_deal(self, deal):
        columns = [list(column) for column in deal]
        if len(columns) != 8:
            raise ValueError("deal must contain exactly 8 columns")
        self.columns = columns

    def display(self):
        print("Free Cells: ", self.free_cells)
        print("Home Cells:")
        for suit, stack in self.home_cells.items():
            print(f"  {suit.value}: {stack}")

        print("\nTableau (8 Columns):")
        max_height = max((len(col) for col in self.columns), default=0)
        for row in range(max_height):
            line = []
            for col in self.columns:
                if row < len(col):
                    line.append(str(col[row]))
                else:
                    line.append("  ")
            print(" | ".join(line))

    def can_move_to_column(self, from_card: Card, to_column: List[Card]) -> bool:
        if from_card is None:
            return False
        if not to_column:
            return True
        top_card = to_column[-1]
        return from_card.color() != top_card.color() and from_card.value == top_card.value - 1

    def can_move_to_home(self, card: Card) -> bool:
        if card is None:
            return False
        home_stack = self.home_cells[card.suit]
        expected_value = len(home_stack) + 1
        return card.value == expected_value

    @staticmethod
    def is_valid_sequence(sequence: List[Card]) -> bool:
        if not sequence:
            return False
        for upper, lower in zip(sequence, sequence[1:]):
            if upper.color() == lower.color() or upper.value != lower.value + 1:
                return False
        return True

    def get_valid_sequence_from_column(self, col_idx: int, start_idx: int) -> List[Card]:
        if not self._valid_column_index(col_idx):
            return []
        column = self.columns[col_idx]
        if start_idx < 0 or start_idx >= len(column):
            return []

        sequence = column[start_idx:]
        for i in range(1, len(sequence) + 1):
            prefix = sequence[:i]
            if not self.is_valid_sequence(prefix):
                return sequence[:i - 1]
        return sequence

    def max_movable_sequence_length(self, to_col_idx=None) -> int:
        empty_free_cells = sum(1 for cell in self.free_cells if cell is None)
        empty_columns = sum(1 for col in self.columns if not col)
        effective_empty_columns = empty_columns
        if self._valid_column_index(to_col_idx) and not self.columns[to_col_idx]:
            effective_empty_columns -= 1
        return (empty_free_cells + 1) * (2 ** effective_empty_columns)

    def can_move_sequence_between_columns(self, from_col_idx: int, to_col_idx: int, count: int = 1) -> bool:
        if not self._valid_column_index(from_col_idx) or not self._valid_column_index(to_col_idx):
            return False
        if from_col_idx == to_col_idx or count < 1:
            return False

        from_col = self.columns[from_col_idx]
        if len(from_col) < count:
            return False

        sequence = from_col[-count:]
        if not self.is_valid_sequence(sequence):
            return False
        if count > self.max_movable_sequence_length(to_col_idx):
            return False
        return self.can_move_to_column(sequence[0], self.columns[to_col_idx])

    def can_apply_move(self, move: Move) -> bool:
        if not isinstance(move, Move) or move.count < 1:
            return False

        if move.move_type == MoveType.COL_TO_COL:
            if move.to_idx is None:
                return False
            return self.can_move_sequence_between_columns(move.from_idx, move.to_idx, move.count)

        if move.move_type == MoveType.COL_TO_FREE:
            if move.count != 1 or not self._valid_column_index(move.from_idx):
                return False
            if not self.columns[move.from_idx]:
                return False
            if move.to_idx is None:
                return any(cell is None for cell in self.free_cells)
            return self._valid_free_index(move.to_idx) and self.free_cells[move.to_idx] is None

        if move.move_type == MoveType.FREE_TO_COL:
            if move.count != 1 or move.to_idx is None:
                return False
            if not self._valid_free_index(move.from_idx) or not self._valid_column_index(move.to_idx):
                return False
            card = self.free_cells[move.from_idx]
            return card is not None and self.can_move_to_column(card, self.columns[move.to_idx])

        if move.move_type == MoveType.COL_TO_HOME:
            if move.count != 1 or not self._valid_column_index(move.from_idx):
                return False
            if not self.columns[move.from_idx]:
                return False
            return self.can_move_to_home(self.columns[move.from_idx][-1])

        if move.move_type == MoveType.FREE_TO_HOME:
            if move.count != 1 or not self._valid_free_index(move.from_idx):
                return False
            return self.can_move_to_home(self.free_cells[move.from_idx])

        return False

    def apply_move(self, move: Move) -> bool:
        if not self.can_apply_move(move):
            return False

        if move.move_type == MoveType.COL_TO_COL:
            moving = self.columns[move.from_idx][-move.count:]
            del self.columns[move.from_idx][-move.count:]
            self.columns[move.to_idx].extend(moving)
            return True

        if move.move_type == MoveType.COL_TO_FREE:
            free_idx = move.to_idx
            if free_idx is None:
                free_idx = next(i for i, cell in enumerate(self.free_cells) if cell is None)
            self.free_cells[free_idx] = self.columns[move.from_idx].pop()
            return True

        if move.move_type == MoveType.FREE_TO_COL:
            card = self.free_cells[move.from_idx]
            self.columns[move.to_idx].append(card)
            self.free_cells[move.from_idx] = None
            return True

        if move.move_type == MoveType.COL_TO_HOME:
            card = self.columns[move.from_idx].pop()
            self.home_cells[card.suit].append(card)
            return True

        if move.move_type == MoveType.FREE_TO_HOME:
            card = self.free_cells[move.from_idx]
            self.home_cells[card.suit].append(card)
            self.free_cells[move.from_idx] = None
            return True

        return False

    def auto_move_to_home(self) -> bool:
        changed = False
        while True:
            moved_this_round = False

            for col_idx in range(len(self.columns)):
                if self.apply_move(Move(MoveType.COL_TO_HOME, col_idx)):
                    changed = True
                    moved_this_round = True

            for free_idx in range(len(self.free_cells)):
                if self.apply_move(Move(MoveType.FREE_TO_HOME, free_idx)):
                    changed = True
                    moved_this_round = True

            if not moved_this_round:
                return changed

    def generate_legal_moves(self) -> List[Move]:
        moves = []
        max_count = self.max_movable_sequence_length()

        for from_idx, from_col in enumerate(self.columns):
            if not from_col:
                continue

            for count in range(1, min(len(from_col), max_count) + 1):
                sequence = from_col[-count:]
                if not self.is_valid_sequence(sequence):
                    break
                for to_idx in range(len(self.columns)):
                    move = Move(MoveType.COL_TO_COL, from_idx, to_idx, count=count)
                    if self.can_apply_move(move):
                        moves.append(move)

            for free_idx, cell in enumerate(self.free_cells):
                if cell is None:
                    moves.append(Move(MoveType.COL_TO_FREE, from_idx, free_idx))

            if self.can_move_to_home(from_col[-1]):
                moves.append(Move(MoveType.COL_TO_HOME, from_idx))

        for free_idx, card in enumerate(self.free_cells):
            if card is None:
                continue
            for to_idx in range(len(self.columns)):
                move = Move(MoveType.FREE_TO_COL, free_idx, to_idx)
                if self.can_apply_move(move):
                    moves.append(move)
            if self.can_move_to_home(card):
                moves.append(Move(MoveType.FREE_TO_HOME, free_idx))

        return moves

    def move_card_between_columns(self, from_col_idx: int, to_col_idx: int, count: int = 1) -> bool:
        return self.apply_move(Move(MoveType.COL_TO_COL, from_col_idx, to_col_idx, count=count))

    def move_card_to_free_cell(self, from_col_idx: int, free_cell_idx: Optional[int] = None) -> bool:
        return self.apply_move(Move(MoveType.COL_TO_FREE, from_col_idx, free_cell_idx))

    def move_card_from_free_cell(self, free_cell_idx: int, to_col_idx: int) -> bool:
        return self.apply_move(Move(MoveType.FREE_TO_COL, free_cell_idx, to_col_idx))

    def move_card_to_home(self, from_col_idx: int) -> bool:
        return self.apply_move(Move(MoveType.COL_TO_HOME, from_col_idx))

    def move_card_from_free_cell_to_home(self, free_cell_idx: int) -> bool:
        return self.apply_move(Move(MoveType.FREE_TO_HOME, free_cell_idx))

    def state_key(self) -> tuple:
        return (
            tuple(tuple(self._card_key(card) for card in column) for column in self.columns),
            tuple(None if card is None else self._card_key(card) for card in self.free_cells),
            tuple(tuple(self._card_key(card) for card in self.home_cells[suit]) for suit in Suit),
        )

    def is_won(self) -> bool:
        return all(len(self.home_cells[suit]) == 13 for suit in Suit)

    def is_terminal(self) -> bool:
        return self.is_won() or not self.generate_legal_moves()

    def clone(self):
        return copy.deepcopy(self)

    def __eq__(self, other):
        return isinstance(other, FreeCellGame) and self.state_key() == other.state_key()

    @staticmethod
    def _card_key(card: Card) -> tuple:
        return (card.suit.name, card.value)

    @staticmethod
    def _valid_column_index(idx: int) -> bool:
        return isinstance(idx, int) and 0 <= idx < 8

    @staticmethod
    def _valid_free_index(idx: int) -> bool:
        return isinstance(idx, int) and 0 <= idx < 4
