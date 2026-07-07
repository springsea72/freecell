import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import policy_player
import solver
from game_model import Card, FreeCellGame, Move, MoveType, Suit


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


class LoopGame:
    def __init__(self):
        self.columns = [[] for _ in range(8)]
        self.free_cells = [None] * 4
        self.home_cells = {suit: [] for suit in Suit}

    def state_key(self):
        return ("loop",)

    def is_won(self):
        return False

    def generate_legal_moves(self):
        return [Move(MoveType.COL_TO_FREE, 0, 0)]

    def clone(self):
        return LoopGame()

    def apply_move(self, move):
        return True


class SlowProgressGame:
    def __init__(self, steps=0):
        self.steps = steps
        self.columns = [[] for _ in range(8)]
        self.free_cells = [None] * 4
        self.home_cells = {suit: [] for suit in Suit}

    def state_key(self):
        return ("slow", self.steps)

    def is_won(self):
        return False

    def generate_legal_moves(self):
        return [Move(MoveType.COL_TO_FREE, 0, 0)]

    def clone(self):
        return SlowProgressGame(self.steps)

    def apply_move(self, move):
        self.steps += 1
        return True


class PolicyPlayerTests(unittest.TestCase):
    def test_policy_player_does_not_call_solver(self):
        with patch("policy_player.FreeCellGame", return_value=won_game()), patch.object(solver, "solve") as solve:
            result = policy_player.play_game(seed=1, policy="heuristic", max_steps=10)

        self.assertTrue(result.won)
        solve.assert_not_called()

    def test_already_won_state_returns_won(self):
        with patch("policy_player.FreeCellGame", return_value=won_game()):
            result = policy_player.play_game(seed=1, policy="heuristic", max_steps=10)

        self.assertTrue(result.won)
        self.assertEqual("won", result.reason)
        self.assertEqual(0, result.steps)

    def test_simple_winning_state_returns_won(self):
        with patch("policy_player.FreeCellGame", return_value=two_step_home_game()):
            result = policy_player.play_game(seed=1, policy="heuristic", max_steps=10)

        self.assertTrue(result.won)
        self.assertEqual("won", result.reason)
        self.assertEqual(2, result.steps)

    def test_loop_detection(self):
        with patch("policy_player.FreeCellGame", return_value=LoopGame()):
            result = policy_player.play_game(seed=1, policy="first_legal", max_steps=10)

        self.assertFalse(result.won)
        self.assertEqual("loop_detected", result.reason)
        self.assertEqual(0, result.steps)

    def test_max_steps_limit(self):
        with patch("policy_player.FreeCellGame", return_value=SlowProgressGame()):
            result = policy_player.play_game(seed=1, policy="first_legal", max_steps=3)

        self.assertFalse(result.won)
        self.assertEqual("max_steps", result.reason)
        self.assertEqual(3, result.steps)

    def test_evaluate_seeds_summary(self):
        results = [
            policy_player.PlayResult(1, "heuristic", True, 2, "won", 52, 3),
            policy_player.PlayResult(2, "heuristic", False, 5, "max_steps", 10, 6),
        ]

        with patch("policy_player.play_game", side_effect=results):
            summary = policy_player.evaluate_seeds([1, 2], policy="heuristic", max_steps=5)

        self.assertEqual(2, summary["games"])
        self.assertEqual(1, summary["won"])
        self.assertEqual(0.5, summary["win_rate"])
        self.assertEqual(3.5, summary["average_steps"])
        self.assertEqual(31.0, summary["average_home_cards"])
        self.assertEqual("heuristic", summary["policy"])

    def test_cli_valid_args_returns_zero(self):
        result = policy_player.PlayResult(1, "heuristic", False, 3, "max_steps", 5, 4)

        with patch("policy_player.play_game", return_value=result):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = policy_player.main(["--seed", "1", "--policy", "heuristic", "--max-steps", "3"])

        self.assertEqual(0, exit_code)
        self.assertIn("summary", stdout.getvalue())
        self.assertIn("policy: heuristic", stdout.getvalue())

    def test_cli_unknown_policy_returns_nonzero(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = policy_player.main(["--seed", "1", "--policy", "unknown", "--max-steps", "3"])

        self.assertEqual(1, exit_code)
        self.assertIn("unknown policy", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
