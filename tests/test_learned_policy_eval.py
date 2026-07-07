import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import learned_policy_eval
import solver
from game_model import FreeCellGame, Move, MoveType


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def sample(action_index=1):
    legal_moves = [
        {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
    ]
    return {
        "version": 1,
        "seed": 1,
        "step_index": 0,
        "remaining_moves": 1,
        "state": {
            "columns": [[{"suit": "SPADES", "value": 1}], [], [], [], [], [], [], []],
            "free_cells": [None, None, None, None],
            "home_cells": {"SPADES": [], "HEARTS": [], "CLUBS": [], "DIAMONDS": []},
        },
        "legal_moves": legal_moves,
        "action": legal_moves[action_index],
        "action_index": action_index,
    }


class RecordingGame:
    instance = None
    total_generate_calls = 0
    total_apply_calls = 0

    def __init__(self, seed=None, steps=0):
        self.seed = seed
        self.steps = steps
        self.generate_calls = 0
        self.apply_calls = 0
        self.columns = [[] for _ in range(8)]
        self.free_cells = [None] * 4
        self.home_cells = {}
        RecordingGame.instance = self

    def state_key(self):
        return ("recording", self.steps)

    def is_won(self):
        return self.steps >= 1

    def generate_legal_moves(self):
        self.generate_calls += 1
        type(self).total_generate_calls += 1
        return [Move(MoveType.COL_TO_HOME, 0)]

    def clone(self):
        return RecordingGame(self.seed, self.steps)

    def apply_move(self, move):
        self.apply_calls += 1
        type(self).total_apply_calls += 1
        self.steps += 1
        return True


class LoopGame(RecordingGame):
    instance = None

    def __init__(self, seed=None, steps=0):
        super().__init__(seed, steps)
        LoopGame.instance = self

    def state_key(self):
        return ("loop",)

    def is_won(self):
        return False

    def clone(self):
        return LoopGame(self.seed, self.steps)


class SlowGame(RecordingGame):
    instance = None

    def __init__(self, seed=None, steps=0):
        super().__init__(seed, steps)
        SlowGame.instance = self

    def state_key(self):
        return ("slow", self.steps)

    def is_won(self):
        return False

    def clone(self):
        return SlowGame(self.seed, self.steps)


class LearnedPolicyEvalTests(unittest.TestCase):
    def setUp(self):
        for cls in (RecordingGame, LoopGame, SlowGame):
            cls.total_generate_calls = 0
            cls.total_apply_calls = 0

    def test_dataset_accuracy_calculation(self):
        samples = [sample(action_index=1), sample(action_index=1)]
        bundle = type("Bundle", (), {"device": "cpu"})()

        with patch("learned_policy_eval.score_sample_moves", side_effect=[[0.1, 0.9], [0.2, 0.1]]):
            summary = learned_policy_eval.evaluate_samples(samples, model_bundle=bundle)

        self.assertEqual("dataset", summary["mode"])
        self.assertEqual(2, summary["samples"])
        self.assertEqual(1, summary["correct"])
        self.assertEqual(0.5, summary["accuracy"])

    def test_play_uses_generate_legal_moves_and_apply_move(self):
        with patch("learned_policy_eval.FreeCellGame", RecordingGame), patch(
            "learned_policy.score_legal_moves",
            side_effect=lambda game, bundle, device=None: [(game.generate_legal_moves()[0], 1.0)],
        ):
            result = learned_policy_eval.play_game(1, model_bundle=object(), max_steps=5)

        self.assertTrue(result.won)
        self.assertGreaterEqual(RecordingGame.total_generate_calls, 1)
        self.assertGreaterEqual(RecordingGame.total_apply_calls, 1)

    def test_loop_detection(self):
        with patch("learned_policy_eval.FreeCellGame", LoopGame), patch(
            "learned_policy.score_legal_moves",
            side_effect=lambda game, bundle, device=None: [(game.generate_legal_moves()[0], 1.0)],
        ):
            result = learned_policy_eval.play_game(1, model_bundle=object(), max_steps=5)

        self.assertFalse(result.won)
        self.assertEqual("loop_detected", result.reason)

    def test_max_steps_limit(self):
        with patch("learned_policy_eval.FreeCellGame", SlowGame), patch(
            "learned_policy.score_legal_moves",
            side_effect=lambda game, bundle, device=None: [(game.generate_legal_moves()[0], 1.0)],
        ):
            result = learned_policy_eval.play_game(1, model_bundle=object(), max_steps=2)

        self.assertFalse(result.won)
        self.assertEqual("max_steps", result.reason)
        self.assertEqual(2, result.steps)

    def test_model_eval_does_not_call_solver(self):
        with patch("learned_policy_eval.FreeCellGame", RecordingGame), patch(
            "learned_policy.score_legal_moves",
            side_effect=lambda game, bundle, device=None: [(game.generate_legal_moves()[0], 1.0)],
        ), patch.object(solver, "solve") as solve:
            learned_policy_eval.play_game(1, model_bundle=object(), max_steps=5)

        solve.assert_not_called()

    def test_non_looping_selection_does_not_mutate_game(self):
        game = FreeCellGame(seed=1)
        legal_move = game.generate_legal_moves()[0]
        before = game.state_key()

        with patch("learned_policy.score_legal_moves", return_value=[(legal_move, 1.0)]):
            selected = learned_policy_eval._choose_non_looping_move(game, object(), visited={before})

        self.assertEqual(legal_move, selected)
        self.assertEqual(before, game.state_key())

    def test_cli_missing_model_returns_nonzero(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = learned_policy_eval.main(["--model", "missing.pt", "--seed", "1"])

        self.assertEqual(1, exit_code)
        self.assertIn("model does not exist", stderr.getvalue())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_evaluate_file_loads_model_and_reports_accuracy(self):
        with patch("learned_policy_eval.load_jsonl", return_value=[sample(action_index=1)]), patch(
            "learned_policy.load_model", return_value=type("Bundle", (), {"device": "cpu"})()
        ), patch("learned_policy_eval.score_sample_moves", return_value=[0.1, 0.9]):
            summary = learned_policy_eval.evaluate_file("dataset.jsonl", "model.pt", device="cpu")

        self.assertEqual("model.pt", summary["model_path"])
        self.assertEqual(1.0, summary["accuracy"])


if __name__ == "__main__":
    unittest.main()
