import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from game_model import Card, FreeCellGame, Move, MoveType, Suit
from solver import SolveResult
from trace_io import load_trace, move_from_dict, move_to_dict, save_trace, verify_trace


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


def two_step_home_game():
    game = FreeCellGame(deal=empty_deal())
    for suit in Suit:
        game.home_cells[suit] = [card(suit, value) for value in range(1, 14)]
    game.home_cells[Suit.SPADES] = [card(Suit.SPADES, value) for value in range(1, 12)]
    game.columns[0] = [card(Suit.SPADES, 13), card(Suit.SPADES, 12)]
    return game


class TraceIoTests(unittest.TestCase):
    def test_move_dict_roundtrip(self):
        move = Move(MoveType.COL_TO_COL, 1, 2, count=3)

        self.assertEqual(move, move_from_dict(move_to_dict(move)))

    def test_solved_trace_save_load_and_verify(self):
        result = SolveResult(
            True,
            [Move(MoveType.COL_TO_HOME, 0), Move(MoveType.COL_TO_HOME, 0)],
            2,
            2,
            1,
            "won",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            save_trace(path, seed=123, max_nodes=20, max_depth=5, result=result)
            trace = load_trace(path)

        self.assertEqual(1, trace["version"])
        self.assertEqual(123, trace["seed"])
        self.assertEqual(2, trace["stats"]["path_length"])
        with patch("trace_io.FreeCellGame", return_value=two_step_home_game()):
            self.assertTrue(verify_trace(trace))

    def test_tampered_move_fails_verify(self):
        result = SolveResult(
            True,
            [Move(MoveType.COL_TO_HOME, 0), Move(MoveType.COL_TO_HOME, 0)],
            2,
            2,
            1,
            "won",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            save_trace(path, seed=123, max_nodes=20, max_depth=5, result=result)
            trace = load_trace(path)

        trace["moves"][0]["from_idx"] = 7
        with patch("trace_io.FreeCellGame", return_value=two_step_home_game()):
            self.assertFalse(verify_trace(trace))


if __name__ == "__main__":
    unittest.main()
