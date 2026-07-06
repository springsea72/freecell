import io
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import autoplay
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from solver import SolveResult


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


class AutoplayTests(unittest.TestCase):
    def test_autoplay_solved_path_verification_success_exits_zero(self):
        game = two_step_home_game()
        moves = [Move(MoveType.COL_TO_HOME, 0), Move(MoveType.COL_TO_HOME, 0)]
        result = SolveResult(True, moves, 2, 2, 1, "won")

        with patch("autoplay.FreeCellGame", return_value=game), patch("autoplay.solve", return_value=result):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = autoplay.main(["--seed", "1"])

        self.assertEqual(0, exit_code)
        self.assertIn("verified_won: True", output.getvalue())

    def test_autoplay_solved_path_verification_failure_exits_nonzero(self):
        game = two_step_home_game()
        result = SolveResult(True, [], 1, 0, 0, "won")

        with patch("autoplay.FreeCellGame", return_value=game), patch("autoplay.solve", return_value=result):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = autoplay.main(["--seed", "1"])

        self.assertEqual(1, exit_code)
        self.assertIn("verified_won: False", output.getvalue())

    def test_autoplay_save_trace_only_when_solved_and_verified(self):
        game = two_step_home_game()
        moves = [Move(MoveType.COL_TO_HOME, 0), Move(MoveType.COL_TO_HOME, 0)]
        result = SolveResult(True, moves, 2, 2, 1, "won")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}\\trace.json"
            with patch("autoplay.FreeCellGame", return_value=game), patch(
                "autoplay.solve", return_value=result
            ), patch("autoplay.save_trace") as save_trace:
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = autoplay.main(["--seed", "1", "--save-trace", path])

        self.assertEqual(0, exit_code)
        save_trace.assert_called_once_with(path, 1, 10000, 200, result)

    def test_autoplay_save_trace_skips_unverified_solution(self):
        game = two_step_home_game()
        result = SolveResult(True, [], 1, 0, 0, "won")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}\\trace.json"
            with patch("autoplay.FreeCellGame", return_value=game), patch(
                "autoplay.solve", return_value=result
            ), patch("autoplay.save_trace") as save_trace:
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = autoplay.main(["--seed", "1", "--save-trace", path])

        self.assertEqual(1, exit_code)
        save_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
