import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import failure_dataset_builder
import solver
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from trace_io import move_to_dict


def card(suit, value):
    return Card(suit, value)


MOVE_FREE = Move(MoveType.COL_TO_FREE, 0, 0)
MOVE_HOME = Move(MoveType.COL_TO_HOME, 0)


class FakeGameBase:
    def __init__(self, seed=None, steps=0):
        self.seed = seed
        self.steps = steps
        self.columns = [[card(Suit.SPADES, 1)], [], [], [], [], [], [], []]
        self.free_cells = [None, None, None, None]
        self.home_cells = {suit: [] for suit in Suit}

    def state_key(self):
        return (type(self).__name__, self.seed, self.steps)

    def is_won(self):
        return False

    def clone(self):
        return type(self)(self.seed, self.steps)

    def apply_move(self, move):
        if move not in self.generate_legal_moves():
            return False
        self.steps += 1
        return True


class WonGame(FakeGameBase):
    def is_won(self):
        return True

    def generate_legal_moves(self):
        return []


class NoLegalAfterOneGame(FakeGameBase):
    def generate_legal_moves(self):
        if self.steps == 0:
            return [MOVE_FREE, MOVE_HOME]
        return []


class LoopGame(FakeGameBase):
    def state_key(self):
        return ("loop",)

    def generate_legal_moves(self):
        return [MOVE_FREE]


def score_game(game, model_bundle, device=None):
    scores = []
    for move in game.generate_legal_moves():
        score = 1.0 if move == MOVE_HOME else 0.2
        scores.append((move, score))
    return scores


class FailureDatasetBuilderTests(unittest.TestCase):
    def test_seed_list_args_parse(self):
        args = failure_dataset_builder.parse_args(
            ["--model", "model.pt", "--seeds", "1", "2", "3", "--output", "failures.jsonl"]
        )
        self.assertEqual([1, 2, 3], failure_dataset_builder.seeds_from_args(args))

        args = failure_dataset_builder.parse_args(
            [
                "--model",
                "model.pt",
                "--seed-start",
                "5",
                "--seed-count",
                "3",
                "--output",
                "failures.jsonl",
            ]
        )
        self.assertEqual([5, 6, 7], failure_dataset_builder.seeds_from_args(args))

    def test_wins_do_not_write_samples_but_count_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", WonGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                summary = failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(1, summary["games"])
        self.assertEqual(1, summary["wins"])
        self.assertEqual(0, summary["failures"])
        self.assertEqual(0, summary["samples_written"])
        self.assertEqual([], rows)

    def test_no_legal_failure_writes_last_window_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                summary = failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual({"no_legal_moves": 1}, summary["reasons"])
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual("no_legal_moves", row["terminal_reason"])
        self.assertEqual(1, row["terminal_step"])
        self.assertTrue(row["is_terminal_step"])
        self.assertFalse(row["would_loop"])
        self.assertEqual(4, row["empty_free_cells"])
        self.assertEqual(7, row["empty_columns"])
        self.assertEqual(11, row["buffer_slots"])
        self.assertEqual(0, row["buried_low_cards"])
        self.assertEqual(1, row["movable_suffix_total"])
        self.assertEqual(1, row["top_cards_to_home"])
        self.assertFalse(row["chosen_reduces_buffer"])
        self.assertFalse(row["chosen_releases_low_card"])
        self.assertTrue(row["chosen_is_home_move"])

    def test_loop_failure_writes_terminal_candidate_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", LoopGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                summary = failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual({"loop_detected": 1}, summary["reasons"])
        self.assertEqual(1, len(rows))
        self.assertEqual("loop_detected", rows[0]["terminal_reason"])
        self.assertTrue(rows[0]["would_loop"])

    def test_model_scores_align_with_legal_moves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(len(row["legal_moves"]), len(row["model_scores"]))

    def test_chosen_action_index_points_to_chosen_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["chosen_action"], row["legal_moves"][row["chosen_action_index"]])
        self.assertEqual(move_to_dict(MOVE_HOME), row["chosen_action"])

    def test_diagnostics_detect_buffer_reduction_without_mutating_game(self):
        game = FreeCellGame(deal=[[] for _ in range(8)])
        game.columns[0] = [card(Suit.CLUBS, 6), card(Suit.SPADES, 5)]
        move = Move(MoveType.COL_TO_COL, 0, 1)
        before = game.state_key()

        diagnostics = failure_dataset_builder._diagnostics_from_game(game, move)

        self.assertTrue(diagnostics["chosen_reduces_buffer"])
        self.assertEqual(before, game.state_key())

    def test_diagnostics_detect_released_low_card(self):
        game = FreeCellGame(deal=[[] for _ in range(8)])
        game.columns[0] = [card(Suit.SPADES, 1), card(Suit.HEARTS, 5)]
        move = Move(MoveType.COL_TO_COL, 0, 1)

        diagnostics = failure_dataset_builder._diagnostics_from_game(game, move)

        self.assertEqual(1, diagnostics["buried_low_cards"])
        self.assertTrue(diagnostics["chosen_releases_low_card"])

    def test_builder_does_not_call_solver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game), patch.object(
                solver, "solve"
            ) as solve:
                failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

        solve.assert_not_called()

    def test_output_jsonl_is_parseable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "failures.jsonl"
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game):
                failure_dataset_builder.build_failure_dataset("model.pt", [1], output, device="cpu")

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, len(rows))
        self.assertEqual(failure_dataset_builder.FAILURE_DATASET_VERSION, rows[0]["version"])

    def test_cli_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.pt"
            model.write_text("fake", encoding="utf-8")
            output = Path(tmpdir) / "failures.jsonl"
            stdout = io.StringIO()
            with patch("failure_dataset_builder.FreeCellGame", NoLegalAfterOneGame), patch(
                "failure_dataset_builder.learned_policy.load_model", return_value=object()
            ), patch("failure_dataset_builder.learned_policy.score_legal_moves", side_effect=score_game), redirect_stdout(
                stdout
            ):
                exit_code = failure_dataset_builder.main(
                    ["--model", str(model), "--seeds", "1", "--output", str(output), "--device", "cpu"]
                )

        self.assertEqual(0, exit_code)
        self.assertIn("games: 1", stdout.getvalue())
        self.assertIn('"no_legal_moves": 1', stdout.getvalue())

    def test_cli_missing_model_returns_nonzero(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = failure_dataset_builder.main(
                ["--model", "missing.pt", "--seeds", "1", "--output", "failures.jsonl"]
            )

        self.assertEqual(1, exit_code)
        self.assertIn("model does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
