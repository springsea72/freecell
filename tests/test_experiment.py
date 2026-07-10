import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import experiment
import train_policy
from solver import SolveResult


def args(output_dir, **overrides):
    values = {
        "seed_start": 1,
        "seed_count": 2,
        "max_nodes": 100,
        "max_depth": 20,
        "epochs": 1,
        "batch_size": 1,
        "hidden_size": 64,
        "progress_loss_weight": train_policy.learned_policy.DEFAULT_PROGRESS_LOSS_WEIGHT,
        "comparison_negatives_per_sample": 0,
        "comparison_loss_weight": 0.0,
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

    def test_parse_rejects_invalid_hidden_size(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(
                ["--seed-start", "1", "--seed-count", "1", "--hidden-size", "0", "--output-dir", "out"]
            )

    def test_parse_accepts_progress_loss_weight(self):
        parsed = experiment.parse_args(
            [
                "--seed-start",
                "1",
                "--seed-count",
                "1",
                "--progress-loss-weight",
                "0.2",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(0.2, parsed.progress_loss_weight)

    def test_parse_default_progress_loss_weight_matches_train_default(self):
        parsed = experiment.parse_args(["--seed-start", "1", "--seed-count", "1", "--output-dir", "out"])

        self.assertEqual(train_policy.learned_policy.DEFAULT_PROGRESS_LOSS_WEIGHT, parsed.progress_loss_weight)

    def test_parse_rejects_negative_progress_loss_weight(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(
                [
                    "--seed-start",
                    "1",
                    "--seed-count",
                    "1",
                    "--progress-loss-weight",
                    "-0.1",
                    "--output-dir",
                    "out",
                ]
            )

    def test_parse_rejects_negative_comparison_negatives(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(
                [
                    "--seed-start",
                    "1",
                    "--seed-count",
                    "1",
                    "--comparison-negatives-per-sample",
                    "-1",
                    "--output-dir",
                    "out",
                ]
            )

    def test_parse_rejects_negative_comparison_loss_weight(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(
                [
                    "--seed-start",
                    "1",
                    "--seed-count",
                    "1",
                    "--comparison-loss-weight",
                    "-0.1",
                    "--output-dir",
                    "out",
                ]
            )

    def test_parse_requires_output_dir(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.parse_args(["--seed-start", "1", "--seed-count", "1"])

    def test_no_solved_trace_returns_error_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "experiment"
            with patch(
                "experiment.solve", return_value=SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded")
            ), patch("experiment.train_policy.train") as train:
                with self.assertRaisesRegex(RuntimeError, "no solved traces"):
                    experiment.run_experiment(args(output_dir))

            self.assertFalse((output_dir / "manifest.json").exists())

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
            self.assertEqual(64, train_args.hidden_size)
            self.assertEqual(1, train_args.batch_size)
            self.assertEqual(0.1, train_args.progress_loss_weight)
            self.assertIsNone(train_args.comparison_dataset)
            self.assertEqual(0.0, train_args.comparison_loss_weight)
            self.assertEqual(0.001, train_args.lr)
            self.assertEqual(0.0, train_args.validation_split)
            return {
                "samples": 3,
                "train_samples": 3,
                "validation_samples": 0,
                "epochs": 1,
                "batch_size": train_args.batch_size,
                "hidden_size": train_args.hidden_size,
                "progress_loss_weight": train_args.progress_loss_weight,
                "comparison_samples": 0,
                "comparison_loss": 0.0,
                "comparison_accuracy": 0.0,
                "comparison_loss_weight": train_args.comparison_loss_weight,
                "lr": train_args.lr,
                "device": "cpu",
                "train_accuracy": 0.25,
                "validation_accuracy": 0.5,
                "model_path": str(train_args.output),
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
            ), patch("experiment.comparison_dataset_builder.build_comparison_dataset") as build_comparison:
                summary = experiment.run_experiment(args(output_dir))

            self.assertEqual(["trace", "trace", "dataset", "train", "report"], calls)
            build_comparison.assert_not_called()
            self.assertEqual(2, summary["seeds"])
            self.assertEqual(2, summary["solved_traces"])
            self.assertEqual(3, summary["samples"])
            self.assert_path_under(summary["model_path"], output_dir)
            self.assert_path_under(summary["report_path"], output_dir)
            self.assert_path_under(summary["manifest_path"], output_dir)
            self.assertEqual("cpu", summary["device"])
            self.assertEqual(0.25, summary["train_accuracy"])
            self.assertEqual(0.5, summary["validation_accuracy"])

            manifest_path = Path(summary["manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(experiment.MANIFEST_VERSION, manifest["version"])
            self.assertEqual(experiment.MANIFEST_SCHEMA, manifest["schema"])
            self.assertEqual(3, manifest["summary"]["samples"])
            self.assertEqual(2, manifest["summary"]["solved_traces"])
            self.assertEqual(64, manifest["training"]["hidden_size"])
            self.assertEqual(1, manifest["training"]["batch_size"])
            self.assertEqual(0.1, manifest["parameters"]["progress_loss_weight"])
            self.assertEqual(0.1, manifest["training"]["progress_loss_weight"])
            self.assertEqual(0.1, manifest["train_summary"]["progress_loss_weight"])
            self.assertEqual(0, manifest["parameters"]["comparison_negatives_per_sample"])
            self.assertEqual(0.0, manifest["parameters"]["comparison_loss_weight"])
            self.assertIsNone(manifest["paths"]["comparison_dataset"])
            self.assertEqual(0, manifest["summary"]["comparison_samples"])
            self.assertEqual(0, manifest["training"]["comparison_samples"])
            self.assertEqual(0.0, manifest["training"]["comparison_loss_weight"])
            self.assertEqual(0.001, manifest["training"]["lr"])
            self.assert_path_under(manifest["paths"]["dataset"], output_dir)
            self.assert_path_under(manifest["paths"]["model"], output_dir)
            for report_path in manifest["paths"]["reports"]:
                self.assert_path_under(report_path, output_dir)
            for trace_path in manifest["paths"]["traces"]:
                self.assert_path_under(trace_path, output_dir)

    def test_comparison_dataset_is_generated_and_passed_to_train(self):
        solved = SolveResult(True, [], 1, 0, 0, "won")

        def fake_train(train_args):
            self.assert_path_under(train_args.comparison_dataset, output_dir)
            self.assertEqual(0.25, train_args.comparison_loss_weight)
            return {
                "device": "cpu",
                "train_accuracy": 0.25,
                "validation_accuracy": 0.5,
                "comparison_loss": 0.7,
                "comparison_accuracy": 0.6,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "experiment"
            with patch("experiment.solve", return_value=solved), patch("experiment.save_trace"), patch(
                "experiment.build_dataset", return_value={"samples_written": 3}
            ), patch(
                "experiment.comparison_dataset_builder.build_comparison_dataset",
                return_value={"samples_read": 3, "pairs_written": 2, "skipped_no_negative": 1},
            ) as build_comparison, patch(
                "experiment.train_policy.train", side_effect=fake_train
            ), patch(
                "experiment.report.build_report", return_value={"ok": True}
            ), patch(
                "experiment.report.render_json", return_value="{}"
            ):
                summary = experiment.run_experiment(
                    args(output_dir, comparison_negatives_per_sample=1, comparison_loss_weight=0.25)
                )

            build_comparison.assert_called_once()
            manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(2, summary["comparison_samples"])
        self.assertEqual(0.7, summary["comparison_loss"])
        self.assertEqual(0.6, summary["comparison_accuracy"])
        self.assert_path_under(summary["comparison_dataset_path"], output_dir)
        self.assertEqual(1, manifest["parameters"]["comparison_negatives_per_sample"])
        self.assertEqual(0.25, manifest["parameters"]["comparison_loss_weight"])
        self.assertEqual(2, manifest["comparison_summary"]["pairs_written"])
        self.assertEqual(2, manifest["training"]["comparison_samples"])
        self.assert_path_under(manifest["paths"]["comparison_dataset"], output_dir)

    def test_summary_fields_are_stable(self):
        summary = {
            "seeds": 2,
            "solved_traces": 1,
            "samples": 3,
            "model_path": "out/models/policy.pt",
            "report_path": "out/report.json",
            "manifest_path": "out/manifest.json",
            "device": "cpu",
            "train_accuracy": 0.25,
            "validation_accuracy": 0.5,
            "comparison_samples": 0,
            "comparison_loss": 0.0,
            "comparison_accuracy": 0.0,
        }
        expected = [
            "seeds",
            "solved_traces",
            "samples",
            "model_path",
            "report_path",
            "manifest_path",
            "device",
            "train_accuracy",
            "validation_accuracy",
            "comparison_samples",
            "comparison_loss",
            "comparison_accuracy",
        ]

        self.assertEqual(expected, list(summary.keys()))

    def assert_path_under(self, path, output_dir):
        root = Path(output_dir).resolve()
        resolved = Path(path).resolve()
        self.assertTrue(resolved == root or root in resolved.parents, f"{resolved} is outside {root}")


if __name__ == "__main__":
    unittest.main()
