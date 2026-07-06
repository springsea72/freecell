import unittest

from game_model import Card, FreeCellGame, Suit
from solver import SolveResult, solve


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


class SolverTests(unittest.TestCase):
    def test_solver_solves_already_won_state(self):
        result = solve(won_game())

        self.assertIsInstance(result, SolveResult)
        self.assertTrue(result.solved)
        self.assertEqual([], result.moves)
        self.assertEqual("won", result.reason)

    def test_solver_solves_simple_home_position(self):
        result = solve(two_step_home_game(), max_nodes=20, max_depth=5)

        self.assertTrue(result.solved)
        self.assertEqual(2, len(result.moves))
        self.assertEqual("won", result.reason)

    def test_solver_does_not_modify_input_game(self):
        game = two_step_home_game()
        before = game.state_key()

        result = solve(game, max_nodes=20, max_depth=5)

        self.assertTrue(result.solved)
        self.assertEqual(before, game.state_key())

    def test_solver_returned_moves_are_legal_on_clone(self):
        game = two_step_home_game()
        result = solve(game, max_nodes=20, max_depth=5)
        clone = game.clone()

        self.assertTrue(result.solved)
        for move in result.moves:
            self.assertTrue(clone.apply_move(move), msg=str(move))
        self.assertTrue(clone.is_won())

    def test_solver_node_limit_returns_clear_failure(self):
        result = solve(two_step_home_game(), max_nodes=1, max_depth=5)

        self.assertFalse(result.solved)
        self.assertEqual([], result.moves)
        self.assertEqual("max_nodes_exceeded", result.reason)
        self.assertLessEqual(result.generated_nodes + 1, 1)


if __name__ == "__main__":
    unittest.main()
