import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import dataset_builder
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from policy_features import features_from_sample
from trace_io import move_to_dict


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


def solved_trace():
    moves = [Move(MoveType.COL_TO_HOME, 0), Move(MoveType.COL_TO_HOME, 0)]
    return {
        "version": 1,
        "seed": 123,
        "solved": True,
        "reason": "won",
        "moves": [move_to_dict(move) for move in moves],
        "stats": {
            "explored_nodes": 2,
            "generated_nodes": 2,
            "max_frontier": 1,
            "path_length": 2,
        },
    }


def invalid_trace():
    trace = solved_trace()
    trace["moves"][0]["from_idx"] = 7
    return trace


class DatasetBuilderTests(unittest.TestCase):
    def test_card_to_dict_is_stable(self):
        self.assertEqual(
            {"suit": "SPADES", "value": 7},
            dataset_builder.card_to_dict(card(Suit.SPADES, 7)),
        )

    def test_state_to_dict_shape(self):
        state = dataset_builder.state_to_dict(two_step_home_game())

        self.assertEqual(8, len(state["columns"]))
        self.assertEqual(4, len(state["free_cells"]))
        self.assertEqual(["SPADES", "HEARTS", "CLUBS", "DIAMONDS"], list(state["home_cells"].keys()))

    def test_samples_from_trace_action_index_points_to_correct_action(self):
        with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
            samples = dataset_builder.samples_from_trace(solved_trace(), source_trace="seed_000123.json")

        self.assertEqual(2, len(samples))
        for sample in samples:
            self.assertEqual(sample["action"], sample["legal_moves"][sample["action_index"]])

    def test_sample_from_state_includes_progress_for_home_move(self):
        game = two_step_home_game()
        sample = dataset_builder.sample_from_state(
            game,
            Move(MoveType.COL_TO_HOME, 0),
            seed=123,
            source_trace="seed_000123.json",
            step_index=0,
            remaining_moves=2,
        )

        self.assertEqual(
            {
                "home_cards": 50,
                "home_cards_after": 51,
                "home_delta": 1,
                "remaining_moves": 2,
                "remaining_moves_after": 1,
                "won_after": False,
            },
            sample["progress"],
        )

    def test_progress_for_non_home_move_has_zero_home_delta(self):
        game = FreeCellGame(deal=empty_deal())
        game.columns[0] = [card(Suit.SPADES, 5)]
        game.columns[1] = [card(Suit.HEARTS, 6)]
        sample = dataset_builder.sample_from_state(
            game,
            Move(MoveType.COL_TO_COL, 0, 1),
            seed=1,
            source_trace="trace.json",
            step_index=0,
            remaining_moves=3,
        )

        self.assertEqual(0, sample["progress"]["home_cards"])
        self.assertEqual(0, sample["progress"]["home_cards_after"])
        self.assertEqual(0, sample["progress"]["home_delta"])
        self.assertEqual(2, sample["progress"]["remaining_moves_after"])

    def test_final_step_progress_marks_won_after(self):
        with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
            samples = dataset_builder.samples_from_trace(solved_trace(), source_trace="seed_000123.json")

        progress = samples[-1]["progress"]
        self.assertEqual(51, progress["home_cards"])
        self.assertEqual(52, progress["home_cards_after"])
        self.assertEqual(1, progress["home_delta"])
        self.assertEqual(1, progress["remaining_moves"])
        self.assertEqual(0, progress["remaining_moves_after"])
        self.assertTrue(progress["won_after"])

    def test_progress_probe_does_not_mutate_original_game(self):
        game = two_step_home_game()
        before = game.state_key()
        sample = dataset_builder.sample_from_state(
            game,
            Move(MoveType.COL_TO_HOME, 0),
            seed=123,
            source_trace="seed_000123.json",
            step_index=0,
            remaining_moves=2,
        )

        self.assertEqual(before, game.state_key())
        self.assertEqual(50, dataset_builder.home_card_count(game))
        self.assertEqual(51, sample["progress"]["home_cards_after"])

    def test_features_from_sample_accepts_progress_field(self):
        sample = dataset_builder.sample_from_state(
            two_step_home_game(),
            Move(MoveType.COL_TO_HOME, 0),
            seed=123,
            source_trace="seed_000123.json",
            step_index=0,
            remaining_moves=2,
        )

        state_features, move_features, action_index = features_from_sample(sample)

        self.assertTrue(state_features)
        self.assertTrue(move_features)
        self.assertEqual(sample["action_index"], action_index)

    def test_sample_state_is_before_action(self):
        with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
            samples = dataset_builder.samples_from_trace(solved_trace(), source_trace="seed_000123.json")

        first_state = samples[0]["state"]
        second_state = samples[1]["state"]
        self.assertEqual(11, len(first_state["home_cells"]["SPADES"]))
        self.assertEqual([{"suit": "SPADES", "value": 13}, {"suit": "SPADES", "value": 12}], first_state["columns"][0])
        self.assertEqual(12, len(second_state["home_cells"]["SPADES"]))

    def test_valid_trace_writes_jsonl_with_path_length_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
                samples = dataset_builder.samples_from_trace(solved_trace(), source_trace="seed_000123.json")
            dataset_builder.write_jsonl(path, samples)

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(2, len(lines))
        self.assertEqual(0, json.loads(lines[0])["step_index"])

    def test_tampered_trace_fails_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "bad.json"
            output_path = Path(tmpdir) / "dataset.jsonl"
            trace_path.write_text(json.dumps(invalid_trace()), encoding="utf-8")

            with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
                with self.assertRaises(ValueError):
                    dataset_builder.build_dataset([trace_path], output_path)

    def test_skip_invalid_skips_bad_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "bad.json"
            output_path = Path(tmpdir) / "dataset.jsonl"
            trace_path.write_text(json.dumps(invalid_trace()), encoding="utf-8")

            with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
                summary = dataset_builder.build_dataset([trace_path], output_path, skip_invalid=True)

            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            {"traces_read": 1, "traces_used": 0, "invalid_traces": 1, "samples_written": 0},
            summary,
        )
        self.assertEqual([], lines)

    def test_cli_writes_parseable_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "seed_000123.json"
            output_path = Path(tmpdir) / "dataset.jsonl"
            trace_path.write_text(json.dumps(solved_trace()), encoding="utf-8")

            with patch("dataset_builder.FreeCellGame", side_effect=lambda seed=None: two_step_home_game()):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = dataset_builder.main(["--trace", str(trace_path), "--output", str(output_path)])

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(0, exit_code)
        self.assertEqual(2, len(records))
        self.assertIn("samples_written: 2", stdout.getvalue())

    def test_trace_paths_from_args_rejects_missing_trace_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = Path(tmpdir) / "missing"
            args = dataset_builder.parse_args(
                ["--trace-dir", str(missing_dir), "--output", str(Path(tmpdir) / "dataset.jsonl")]
            )

            with self.assertRaisesRegex(ValueError, "trace directory"):
                dataset_builder.trace_paths_from_args(args)

    def test_cli_missing_trace_dir_returns_nonzero_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_dir = Path(tmpdir) / "missing"
            output_path = Path(tmpdir) / "dataset.jsonl"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = dataset_builder.main(
                    ["--trace-dir", str(missing_dir), "--output", str(output_path), "--skip-invalid"]
                )

            self.assertEqual(1, exit_code)
            self.assertFalse(output_path.exists())
            self.assertIn("trace directory", stderr.getvalue())

    def test_cli_empty_existing_trace_dir_writes_empty_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            output_path = Path(tmpdir) / "dataset.jsonl"
            trace_dir.mkdir()
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = dataset_builder.main(["--trace-dir", str(trace_dir), "--output", str(output_path)])

            self.assertEqual(0, exit_code)
            self.assertEqual("", output_path.read_text(encoding="utf-8"))
            self.assertIn("samples_written: 0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
