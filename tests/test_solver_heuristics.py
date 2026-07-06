import unittest

from game_model import Card, FreeCellGame, Move, MoveType, Suit
from solver import _ordered_moves, solve


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


def won_game():
    game = FreeCellGame(deal=empty_deal())
    for suit in Suit:
        game.home_cells[suit] = [card(suit, value) for value in range(1, 14)]
    return game


def two_step_home_game():
    game = won_game()
    game.home_cells[Suit.SPADES] = [card(Suit.SPADES, value) for value in range(1, 12)]
    game.columns[0] = [card(Suit.SPADES, 13), card(Suit.SPADES, 12)]
    return game


class SolverHeuristicTests(unittest.TestCase):
    def test_ordered_moves_prioritizes_home_moves(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 1)],
                [card(Suit.CLUBS, 13)],
                [card(Suit.DIAMONDS, 12)],
                [],
                [],
                [],
                [],
                [],
            ]
        )
        game.free_cells[0] = card(Suit.HEARTS, 1)
        moves = [
            Move(MoveType.COL_TO_FREE, 1, 1),
            Move(MoveType.COL_TO_COL, 2, 1),
            Move(MoveType.COL_TO_HOME, 0),
            Move(MoveType.FREE_TO_HOME, 0),
            Move(MoveType.FREE_TO_COL, 0, 3),
        ]

        ordered = _ordered_moves(moves, game.state_key())

        self.assertEqual(Move(MoveType.FREE_TO_HOME, 0), ordered[0])
        self.assertEqual(Move(MoveType.COL_TO_HOME, 0), ordered[1])

    def test_reverse_move_is_deprioritized_but_kept(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 9)],
                [card(Suit.HEARTS, 8)],
                [card(Suit.CLUBS, 7)],
                [card(Suit.DIAMONDS, 6)],
                [],
                [],
                [],
                [],
            ]
        )
        reverse = Move(MoveType.COL_TO_COL, 1, 0)
        non_reverse = Move(MoveType.COL_TO_COL, 2, 3)
        moves = [reverse, non_reverse]

        ordered = _ordered_moves(moves, game.state_key(), Move(MoveType.COL_TO_COL, 0, 1))

        self.assertEqual(non_reverse, ordered[0])
        self.assertIn(reverse, ordered)

    def test_optimized_solver_path_replays_to_win(self):
        game = two_step_home_game()
        result = solve(game, max_nodes=20, max_depth=5)
        clone = game.clone()

        self.assertTrue(result.solved)
        for move in result.moves:
            self.assertTrue(clone.apply_move(move), msg=str(move))
        self.assertTrue(clone.is_won())


if __name__ == "__main__":
    unittest.main()
