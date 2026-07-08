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

    def test_cli_rejects_invalid_hidden_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny.jsonl"
            output = Path(tmpdir) / "policy.pt"
            write_jsonl(dataset, [tiny_sample()])

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = train_policy.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                        "--epochs",
                        "1",
                        "--hidden-size",
                        "0",
                        "--device",
                        "cpu",
                    ]
                )

            self.assertEqual(1, exit_code)
            self.assertIn("hidden-size must be >= 1", stderr.getvalue())
            self.assertFalse(output.exists())

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
                        "--batch-size",
                        "1",
                        "--hidden-size",
                        "32",
                        "--validation-split",
                        "0",
                    ]
                )

            self.assertEqual(0, exit_code)
            self.assertTrue(output.exists())
            self.assertIn("samples: 1", stdout.getvalue())
            self.assertIn("batch_size: 1", stdout.getvalue())
            self.assertIn("hidden_size: 32", stdout.getvalue())
            self.assertIn("lr: 0.001000", stdout.getvalue())
            self.assertIn("device: cpu", stdout.getvalue())

            bundle = learned_policy.load_model(output, device="cpu")
            self.assertEqual(32, bundle.metadata["hidden_size"])
            self.assertEqual(32, bundle.metadata["training_args"]["hidden_size"])
            self.assertEqual(32, bundle.model[0].out_features)
            game = FreeCellGame(deal=empty_deal())
            game.columns[0] = [card(Suit.SPADES, 1)]
            self.assertIn(learned_policy.choose_action(game, bundle, device="cpu"), game.generate_legal_moves())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_run_epoch_uses_batch_size_for_optimizer_steps(self):
        class CountingOptimizer:
            def __init__(self, model):
                self.model = model
                self.steps = 0
                self.zeroes = 0

            def zero_grad(self):
                self.zeroes += 1
                self.model.zero_grad(set_to_none=True)

            def step(self):
                self.steps += 1

        model = learned_policy.create_model(hidden_size=16, torch=torch)
        optimizer = CountingOptimizer(model)
        samples = [tiny_sample(), tiny_sample(), tiny_sample(), tiny_sample()]

        loss, accuracy = train_policy._run_epoch(
            model,
            samples,
            optimizer,
            torch.device("cpu"),
            torch,
            batch_size=2,
        )

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertEqual(2, optimizer.steps)
        self.assertEqual(3, optimizer.zeroes)


if __name__ == "__main__":
    unittest.main()
