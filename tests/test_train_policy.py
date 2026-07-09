import io
import copy
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


def tiny_sample_without_progress():
    sample = copy.deepcopy(tiny_sample())
    sample.pop("progress", None)
    return sample


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
            self.assertIn("progress_loss:", stdout.getvalue())
            self.assertIn("progress_loss_weight: 0.100000", stdout.getvalue())
            self.assertIn("device: cpu", stdout.getvalue())

            bundle = learned_policy.load_model(output, device="cpu")
            self.assertEqual(learned_policy.MODEL_TYPE, bundle.metadata["model_type"])
            self.assertEqual(["progress_value"], bundle.metadata["auxiliary_targets"])
            self.assertEqual(0.1, bundle.metadata["progress_loss_weight"])
            self.assertEqual(32, bundle.metadata["hidden_size"])
            self.assertEqual(32, bundle.metadata["training_args"]["hidden_size"])
            self.assertEqual(0.1, bundle.metadata["training_args"]["progress_loss_weight"])
            self.assertEqual(32, bundle.model[0].out_features)
            self.assertEqual(2, bundle.model[-1].out_features)
            game = FreeCellGame(deal=empty_deal())
            game.columns[0] = [card(Suit.SPADES, 1)]
            self.assertIn(learned_policy.choose_action(game, bundle, device="cpu"), game.generate_legal_moves())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_progress_sample_reports_progress_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "tiny.jsonl"
            output = Path(tmpdir) / "policy.pt"
            write_jsonl(dataset, [tiny_sample()])
            args = type(
                "Args",
                (),
                {
                    "dataset": dataset,
                    "output": output,
                    "epochs": 1,
                    "batch_size": 1,
                    "hidden_size": 16,
                    "lr": 0.001,
                    "seed": 123,
                    "device": "cpu",
                    "validation_split": 0,
                    "progress_loss_weight": 0.1,
                },
            )()

            summary = train_policy.train(args)

        self.assertIn("progress_loss", summary)
        self.assertGreaterEqual(summary["progress_loss"], 0.0)
        self.assertEqual(0.1, summary["progress_loss_weight"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_progress_loss_weight_zero_is_recorded_in_metadata(self):
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
                        "--progress-loss-weight",
                        "0",
                    ]
                )

            bundle = learned_policy.load_model(output, device="cpu")

        self.assertEqual(0, exit_code)
        self.assertEqual(0.0, bundle.metadata["progress_loss_weight"])
        self.assertEqual(0.0, bundle.metadata["training_args"]["progress_loss_weight"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; optional ML tests skipped")
    def test_old_sample_without_progress_still_trains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "old.jsonl"
            output = Path(tmpdir) / "policy.pt"
            write_jsonl(dataset, [tiny_sample_without_progress()])
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
            output_exists = output.exists()

        self.assertEqual(0, exit_code)
        self.assertTrue(output_exists)
        self.assertIn("progress_loss: 0.000000", stdout.getvalue())

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

        loss, accuracy, progress_loss = train_policy._run_epoch(
            model,
            samples,
            optimizer,
            torch.device("cpu"),
            torch,
            batch_size=2,
            progress_loss_weight=0.1,
        )

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertGreaterEqual(progress_loss, 0.0)
        self.assertEqual(2, optimizer.steps)
        self.assertEqual(3, optimizer.zeroes)


if __name__ == "__main__":
    unittest.main()
