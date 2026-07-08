import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import experiment
from solver import SolveResult


def args(output_dir, **overrides):
    values = {
        "seed_start": 1,
        "seed_count": 2,
        "max_nodes": 100,
        "max_depth": 20,
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "device": "cpu",
        "validation_split": 0.0,
        "output_dir": output_dir,
        "report_format": "json",
        "player_max_steps": 50,
        "model_max_steps": 50,
        "training_seed": 123,
    }
    values.update(overrides)
    return type("Args", (), values)()


class ExperimentTests(unittest.TestCase):
    def test_parse_rejects_invalid_seed_count(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(["--seed-start", "1", "--seed-count", "0", "--output-dir", "out"])

    def test_parse_rejects_invalid_max_nodes(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(["--seed-start", "1", "--seed-count", "1", "--max-nodes", "0", "--output-dir", "out"])

    def test_parse_rejects_invalid_epochs(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(["--seed-start", "1", "--seed-count", "1", "--epochs", "0", "--output-dir", "out"])

    def test_parse_requires_output_dir(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(["--seed-start", "1", "--seed-count", "1"])

    def test_no_solved_trace_returns_error_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "experiment.solve", return_value=SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded")
        ), patch("experiment.train_policy.train") as train:
            with self.assertRaisesRegex(RuntimeError, "no solved traces"):
                experiment.run_experiment(args(tmpdir))

        train.assert_not_called()

    def test_solved_trace_flow_calls_stages_in_order_and_paths_stay_under_output(self):
        calls = []
        solved = SolveResult(True, [], 1, 0, 0, "won")

        def fake_save_trace(path, **kwargs):
            calls.append("trace")
            self.assert_path_under(path, output_dir)

        def fake_build_dataset(trace_paths, output_path, skip_invalid=False):
            calls.append("dataset")
            self.assertTrue(trace_paths)
            self.assert_path_under(output_path, output_dir)
            return {"samples_written": 3}

        def fake_train(train_args):
            calls.append("train")
            self.assert_path_under(train_args.dataset, output_dir)
            self.assert_path_under(train_args.output, output_dir)
            return {
                "device": "cpu",
                "train_accuracy": 0.25,
                "validation_accuracy": 0.5,
            }

        def fake_build_report(seeds, **kwargs):
            calls.append("report")
            self.assertEqual([1, 2], seeds)
            self.assert_path_under(kwargs["dataset"], output_dir)
            self.assert_path_under(kwargs["model_path"], output_dir)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "experiment"
            with patch("experiment.solve", return_value=solved), patch(
                "experiment.save_trace", side_effect=fake_save_trace
            ), patch("experiment.build_dataset", side_effect=fake_build_dataset), patch(
                "experiment.train_policy.train", side_effect=fake_train
            ), patch(
                "experiment.report.build_report", side_effect=fake_build_report
            ), patch(
                "experiment.report.render_json", return_value="{}"
            ):
                summary = experiment.run_experiment(args(output_dir))

        self.assertEqual(["trace", "trace", "dataset", "train", "report"], calls)
        self.assertEqual(2, summary["seeds"])
        self.assertEqual(2, summary["solved_traces"])
        self.assertEqual(3, summary["samples"])
        self.assert_path_under(summary["model_path"], output_dir)
        self.assert_path_under(summary["report_path"], output_dir)
        self.assertEqual("cpu", summary["device"])
        self.assertEqual(0.25, summary["train_accuracy"])
        self.assertEqual(0.5, summary["validation_accuracy"])

    def test_summary_fields_are_stable(self):
        summary = {
            "seeds": 2,
            "solved_traces": 1,
            "samples": 3,
            "model_path": "out/models/policy.pt",
            "report_path": "out/report.json",
            "device": "cpu",
            "train_accuracy": 0.25,
            "validation_accuracy": 0.5,
        }
        expected = [
            "seeds",
            "solved_traces",
            "samples",
            "model_path",
            "report_path",
            "device",
            "train_accuracy",
            "validation_accuracy",
        ]

        self.assertEqual(expected, list(summary.keys()))

    def assert_path_under(self, path, output_dir):
        root = Path(output_dir).resolve()
        resolved = Path(path).resolve()
        self.assertTrue(resolved == root or root in resolved.parents, f"{resolved} is outside {root}")


if __name__ == "__main__":
    unittest.main()
