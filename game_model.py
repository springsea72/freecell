import random
import copy
from enum import Enum
from typing import List, Optional
from enum import Enum, auto

class MoveType(Enum):
    COL_TO_COL = auto()
    COL_TO_FREE = auto()
    FREE_TO_COL = auto()
    COL_TO_HOME = auto()

class Move:
    def __init__(self, move_type: MoveType, from_idx: int, to_idx: Optional[int] = None):
        self.move_type = move_type
        self.from_idx = from_idx
        self.to_idx = to_idx  # 有些动作不需要 to_idx

    def __str__(self):
        if self.move_type == MoveType.COL_TO_COL:
            return f"列 {self.from_idx} → 列 {self.to_idx}"
        elif self.move_type == MoveType.COL_TO_FREE:
            return f"列 {self.from_idx} → 空当"
        elif self.move_type == MoveType.FREE_TO_COL:
            return f"空当 {self.from_idx} → 列 {self.to_idx}"
        elif self.move_type == MoveType.COL_TO_HOME:
            return f"列 {self.from_idx} → 目标堆"
        else:
            return "未知动作"



class Suit(Enum):
    SPADES = '♠'
    HEARTS = '♥'
    CLUBS = '♣'
    DIAMONDS = '♦'


class Card:
    def __init__(self, suit: Suit, value: int):
        self.suit = suit
        self.value = value  # 1~13，对应 A ~ K

    def color(self):
        return 'red' if self.suit in (Suit.HEARTS, Suit.DIAMONDS) else 'black'

    def __str__(self):
        value_map = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        v = value_map.get(self.value, str(self.value))
        return f"{v}{self.suit.value}"

    def __repr__(self):
        return str(self)


class FreeCellGame:
    def __init__(self):
        self.columns: List[List[Card]] = [[] for _ in range(8)]  # 8 列牌堆
        self.free_cells: List[Optional[Card]] = [None] * 4       # 4 个空位
        self.home_cells: dict[Suit, List[Card]] = {
            suit: [] for suit in Suit
        }  # 每个花色一个目标堆

        self.init_deck()

    def init_deck(self):
        # 生成一副洗好的牌
        deck = [Card(suit, value) for suit in Suit for value in range(1, 14)]
        random.shuffle(deck)

        # 平均分发到8列
        for i, card in enumerate(deck):
            self.columns[i % 8].append(card)

    def display(self):
        print("Free Cells: ", self.free_cells)
        print("Home Cells:")
        for suit, stack in self.home_cells.items():
            print(f"  {suit.value}: {stack}")

        print("\nTableau (8 Columns):")
        max_height = max(len(col) for col in self.columns)
        for row in range(max_height):
            line = []
            for col in self.columns:
                if row < len(col):
                    line.append(str(col[row]))
                else:
                    line.append("  ")
            print(" | ".join(line))

    def can_move_to_column(self, from_card: Card, to_column: List[Card]) -> bool:
        """
        判断 from_card 能否移动到目标列 to_column 顶部
        """
        if not to_column:
            return True  # 空列允许任何牌放入
        top_card = to_column[-1]
        return from_card.color() != top_card.color() and from_card.value == top_card.value - 1

    def can_move_to_home(self, card: Card) -> bool:
        """
        判断 card 能否进入其对应花色的 home cell（目标堆）
        """
        home_stack = self.home_cells[card.suit]
        expected_value = len(home_stack) + 1
        return card.value == expected_value

    def move_card_between_columns(self, from_col_idx: int, to_col_idx: int) -> bool:
        """
        从一列移动一张牌到另一列，合法才执行
        """
        from_col = self.columns[from_col_idx]
        to_col = self.columns[to_col_idx]

        if not from_col:
            return False  # 源列为空
        card = from_col[-1]
        if self.can_move_to_column(card, to_col):
            to_col.append(from_col.pop())
            return True
        return False

    def move_card_to_free_cell(self, from_col_idx: int) -> bool:
        """
        从一列移一张牌到空当位
        """
        from_col = self.columns[from_col_idx]
        if not from_col:
            return False

        for i in range(4):
            if self.free_cells[i] is None:
                self.free_cells[i] = from_col.pop()
                return True
        return False  # 没有空位

    def move_card_from_free_cell(self, free_cell_idx: int, to_col_idx: int) -> bool:
        """
        从空当位移回某列
        """
        card = self.free_cells[free_cell_idx]
        if card is None:
            return False

        to_col = self.columns[to_col_idx]
        if self.can_move_to_column(card, to_col):
            to_col.append(card)
            self.free_cells[free_cell_idx] = None
            return True
        return False

    def move_card_to_home(self, from_col_idx: int) -> bool:
        """
        将一列顶部牌移入 home cell（如果合法）
        """
        from_col = self.columns[from_col_idx]
        if not from_col:
            return False
        card = from_col[-1]
        if self.can_move_to_home(card):
            self.home_cells[card.suit].append(from_col.pop())
            return True
        return False

    def generate_legal_moves(self) -> List[Move]:
        moves = []

        # 1. 列 → 列
        for i, from_col in enumerate(self.columns):
            if not from_col:
                continue
            card = from_col[-1]
            for j, to_col in enumerate(self.columns):
                if i == j:
                    continue
                if self.can_move_to_column(card, to_col):
                    moves.append(Move(MoveType.COL_TO_COL, i, j))

        # 2. 列 → 空当
        for i, from_col in enumerate(self.columns):
            if not from_col:
                continue
            if any(cell is None for cell in self.free_cells):
                moves.append(Move(MoveType.COL_TO_FREE, i))

        # 3. 空当 → 列
        for i, cell_card in enumerate(self.free_cells):
            if cell_card is None:
                continue
            for j, to_col in enumerate(self.columns):
                if self.can_move_to_column(cell_card, to_col):
                    moves.append(Move(MoveType.FREE_TO_COL, i, j))

        # 4. 列 → home cell
        for i, from_col in enumerate(self.columns):
            if not from_col:
                continue
            card = from_col[-1]
            if self.can_move_to_home(card):
                moves.append(Move(MoveType.COL_TO_HOME, i))

        return moves

    def clone(self):
        return copy.deepcopy(self)