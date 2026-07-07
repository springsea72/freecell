import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dataset_builder import sample_from_state, write_jsonl
from game_model import Card, FreeCellGame, Move, MoveType, Suit
import learned_policy
import train_policy


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


def tiny_sample():
    game = FreeCellGame(deal=empty_deal())
    game.columns[0] = [card(Suit.SPADES, 1)]
    return sample_from_state(
        game,
        Move(MoveType.COL_TO_HOME, 0),
        seed=1,
        source_trace="trace.json",
        step_index=0,
        remaining_moves=1,
    )


class TrainPolicyTests(unittest.TestCase):
    def test_cli_missing_dataset_returns_nonzero(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = train_policy.main(
                [
                    "--dataset",
                    "missing.jsonl",
                    "--output",
                    "models/policy.pt",
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(1, exit_code)
        self.assertIn("dataset does not exist", stderr.getvalue())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_tiny_dataset_trains_saves_and_loads_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny.jsonl"
            output = Path(tmpdir) / "policy.pt"
            write_jsonl(dataset, [tiny_sample()])

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = train_policy.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                        "--epochs",
                        "1",
                        "--seed",
                        "123",
                        "--device",
                        "cpu",
                        "--validation-split",
                        "0",
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertTrue(output.exists())
            self.assertIn("samples: 1", stdout.getvalue())
            self.assertIn("device: cpu", stdout.getvalue())

            bundle = learned_policy.load_model(output, device="cpu")
            game = FreeCellGame(deal=empty_deal())
            game.columns[0] = [card(Suit.SPADES, 1)]
            self.assertIn(learned_policy.choose_action(game, bundle, device="cpu"), game.generate_legal_moves())


if __name__ == "__main__":
    unittest.main()
