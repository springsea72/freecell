import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import benchmark
import learned_policy_eval
import report


class ReportTests(unittest.TestCase):
    def test_without_dataset_baseline_is_skipped(self):
        with patch("report.benchmark.run_benchmark", return_value=[]), patch(
            "report.benchmark.summarize",
            return_value=benchmark.BenchmarkSummary(1, 0, 0.0, 0.2, 10.0, 0.0),
        ), patch(
            "report.policy_player.evaluate_seeds",
            return_value={
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        ):
            result = report.build_report([1], 1000, 100, dataset=None)

        self.assertTrue(result["policy_baseline"]["skipped"])
        self.assertEqual("no dataset provided", result["policy_baseline"]["reason"])
        self.assertNotIn("learned_policy", result)

    def test_with_dataset_includes_baseline_summary(self):
        baseline = {"samples": 2, "correct": 1, "accuracy": 0.5, "policy": "heuristic"}

        with patch("report.benchmark.run_benchmark", return_value=[]), patch(
            "report.benchmark.summarize",
            return_value=benchmark.BenchmarkSummary(1, 1, 1.0, 0.2, 10.0, 3.0),
        ), patch(
            "report.policy_baseline.evaluate_file", return_value=baseline
        ), patch(
            "report.policy_player.evaluate_seeds",
            return_value={
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        ):
            result = report.build_report([1], 1000, 100, dataset="dataset.jsonl")

        self.assertFalse(result["policy_baseline"]["skipped"])
        self.assertEqual(2, result["policy_baseline"]["samples"])
        self.assertEqual(0.5, result["policy_baseline"]["accuracy"])

    def test_solver_summary_comes_from_benchmark_summarize(self):
        records = [benchmark.BenchmarkRecord(1, True, "won", 4, 5, 2, 3, 0.1)]
        summary = benchmark.BenchmarkSummary(1, 1, 1.0, 0.1, 4.0, 3.0)

        with patch("report.benchmark.run_benchmark", return_value=records), patch(
            "report.benchmark.summarize", return_value=summary
        ) as summarize, patch(
            "report.policy_player.evaluate_seeds",
            return_value={
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        ):
            result = report.build_report([1], 1000, 100)

        summarize.assert_called_once_with(records)
        self.assertEqual(1, result["solver"]["games"])
        self.assertEqual(1, result["solver"]["solved"])
        self.assertEqual(4.0, result["solver"]["average_explored_nodes"])

    def test_player_summary_comes_from_policy_player(self):
        player_summary = {
            "games": 2,
            "won": 1,
            "win_rate": 0.5,
            "average_steps": 12.0,
            "average_home_cards": 20.0,
            "policy": "first_legal",
            "results": [],
        }

        with patch("report.benchmark.run_benchmark", return_value=[]), patch(
            "report.benchmark.summarize",
            return_value=benchmark.BenchmarkSummary(2, 0, 0.0, 0.1, 4.0, 0.0),
        ), patch("report.policy_player.evaluate_seeds", return_value=player_summary) as evaluate:
            result = report.build_report([1, 2], 1000, 100, player_policy="first_legal", max_steps=300)

        evaluate.assert_called_once_with([1, 2], policy="first_legal", max_steps=300)
        self.assertEqual(2, result["policy_player"]["games"])
        self.assertEqual(0.5, result["policy_player"]["win_rate"])

    def test_json_output_is_parseable(self):
        rendered = report.render_json(
            {
                "solver": {"games": 1, "solved": 0, "solve_rate": 0.0},
                "policy_baseline": {"skipped": True, "reason": "no dataset provided"},
                "policy_player": {"games": 1, "won": 0, "win_rate": 0.0, "policy": "heuristic"},
            }
        )

        self.assertEqual("heuristic", json.loads(rendered)["policy_player"]["policy"])

    def test_cli_valid_args_returns_zero(self):
        fake_report = {
            "solver": {
                "games": 1,
                "solved": 0,
                "solve_rate": 0.0,
                "average_elapsed_seconds": 0.1,
                "average_explored_nodes": 4.0,
                "average_path_length_for_solved": 0.0,
            },
            "policy_baseline": {"skipped": True, "reason": "no dataset provided"},
            "policy_player": {
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        }

        with patch("report.build_report", return_value=fake_report):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = report.main(["--seeds", "1", "--max-nodes", "10", "--max-depth", "5"])

        self.assertEqual(0, exit_code)
        self.assertIn("solver", stdout.getvalue())
        self.assertIn("policy_baseline", stdout.getvalue())

    def test_cli_missing_dataset_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.jsonl"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = report.main(["--seeds", "1", "--dataset", str(missing)])

        self.assertEqual(1, exit_code)
        self.assertIn("dataset does not exist", stderr.getvalue())

    def test_cli_missing_model_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pt"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = report.main(["--seeds", "1", "--model", str(missing)])

        self.assertEqual(1, exit_code)
        self.assertIn("model does not exist", stderr.getvalue())

    def test_build_report_with_model_and_dataset_includes_learned_policy(self):
        learned = {
            "model_path": "model.pt",
            "device": "cpu",
            "dataset": {"skipped": False, "samples": 2, "correct": 1, "accuracy": 0.5},
            "play": {
                "games": 2,
                "won": 1,
                "win_rate": 0.5,
                "average_steps": 12.5,
                "average_home_cards": 7.0,
            },
        }

        with patch("report.benchmark.run_benchmark", return_value=[]), patch(
            "report.benchmark.summarize",
            return_value=benchmark.BenchmarkSummary(2, 0, 0.0, 0.1, 4.0, 0.0),
        ), patch(
            "report.policy_baseline.evaluate_file",
            return_value={"samples": 2, "correct": 1, "accuracy": 0.5, "policy": "heuristic"},
        ), patch(
            "report.policy_player.evaluate_seeds",
            return_value={
                "games": 2,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        ), patch("report._build_learned_policy_summary", return_value=learned) as learned_summary:
            result = report.build_report(
                [1, 2],
                1000,
                100,
                dataset="dataset.jsonl",
                model_path="model.pt",
                model_device="cpu",
                model_max_steps=200,
            )

        learned_summary.assert_called_once_with(
            [1, 2],
            model_path="model.pt",
            dataset="dataset.jsonl",
            device="cpu",
            max_steps=200,
        )
        self.assertEqual(0.5, result["learned_policy"]["dataset"]["accuracy"])
        self.assertEqual(0.5, result["learned_policy"]["play"]["win_rate"])

    def test_build_report_with_model_preserves_learned_play_results(self):
        play_result = learned_policy_eval.PlayResult(
            seed=1,
            policy="learned",
            won=False,
            steps=12,
            reason="loop_detected",
            home_cards=4,
            visited_states=13,
        )

        with patch("report.benchmark.run_benchmark", return_value=[]), patch(
            "report.benchmark.summarize",
            return_value=benchmark.BenchmarkSummary(1, 0, 0.0, 0.1, 4.0, 0.0),
        ), patch(
            "report.policy_player.evaluate_seeds",
            return_value={
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
        ), patch(
            "learned_policy_eval.evaluate_seeds",
            return_value={
                "mode": "play",
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 12.0,
                "average_home_cards": 4.0,
                "device": "cpu",
                "model_path": "model.pt",
                "results": [play_result],
            },
        ):
            result = report.build_report([1], 1000, 100, model_path="model.pt", model_device="cpu")

        learned_results = result["learned_policy"]["play"]["results"]
        self.assertEqual(1, len(learned_results))
        self.assertEqual(
            {
                "seed",
                "policy",
                "won",
                "reason",
                "steps",
                "home_cards",
                "visited_states",
            },
            set(learned_results[0]),
        )
        self.assertEqual("loop_detected", learned_results[0]["reason"])

    def test_render_text_and_json_include_learned_policy_metrics(self):
        data = {
            "solver": {
                "games": 1,
                "solved": 0,
                "solve_rate": 0.0,
                "average_elapsed_seconds": 0.1,
                "average_explored_nodes": 4.0,
                "average_path_length_for_solved": 0.0,
            },
            "policy_baseline": {"skipped": True, "reason": "no dataset provided"},
            "policy_player": {
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 5.0,
                "average_home_cards": 2.0,
                "policy": "heuristic",
                "results": [],
            },
            "learned_policy": {
                "model_path": "model.pt",
                "device": "cpu",
                "dataset": {"skipped": False, "samples": 2, "correct": 1, "accuracy": 0.5},
                "play": {
                    "games": 3,
                    "won": 1,
                    "win_rate": 1 / 3,
                    "average_steps": 11.0,
                    "average_home_cards": 6.0,
                    "results": [
                        {
                            "seed": 1,
                            "policy": "learned",
                            "won": False,
                            "reason": "loop_detected",
                            "steps": 10,
                            "home_cards": 4,
                            "visited_states": 11,
                        }
                    ],
                },
            },
        }

        text = report.render_text(data)
        rendered_json = json.loads(report.render_json(data))

        self.assertIn("learned_policy", text)
        self.assertIn("accuracy: 0.500000", text)
        self.assertIn("win_rate: 0.333333", text)
        self.assertNotIn("loop_detected", text)
        self.assertNotIn("visited_states", text)
        self.assertEqual(2, rendered_json["learned_policy"]["dataset"]["samples"])
        self.assertEqual(3, rendered_json["learned_policy"]["play"]["games"])
        self.assertEqual(1, len(rendered_json["learned_policy"]["play"]["results"]))
        self.assertEqual("loop_detected", rendered_json["learned_policy"]["play"]["results"][0]["reason"])

    def test_learned_policy_dataset_skipped_without_dataset(self):
        learned = {
            "model_path": "model.pt",
            "device": "cpu",
            "dataset": {"skipped": True, "reason": "no dataset provided"},
            "play": {
                "games": 1,
                "won": 0,
                "win_rate": 0.0,
                "average_steps": 4.0,
                "average_home_cards": 1.0,
            },
        }

        text = report.render_text(
            {
                "solver": {
                    "games": 1,
                    "solved": 0,
                    "solve_rate": 0.0,
                    "average_elapsed_seconds": 0.1,
                    "average_explored_nodes": 4.0,
                    "average_path_length_for_solved": 0.0,
                },
                "policy_baseline": {"skipped": True, "reason": "no dataset provided"},
                "policy_player": {
                    "games": 1,
                    "won": 0,
                    "win_rate": 0.0,
                    "average_steps": 5.0,
                    "average_home_cards": 2.0,
                    "policy": "heuristic",
                    "results": [],
                },
                "learned_policy": learned,
            }
        )

        self.assertIn("skipped: no dataset provided", text)
        self.assertIn("play", text)

    def test_report_has_no_top_level_torch_import(self):
        source = Path("report.py").read_text(encoding="utf-8")

        self.assertNotIn("import torch", source)
        self.assertNotIn("from torch", source)

    def test_cli_with_model_text_includes_learned_policy(self):
        fake_report = _fake_report_with_learned(dataset_skipped=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.pt"
            dataset = Path(tmpdir) / "dataset.jsonl"
            model.write_text("model", encoding="utf-8")
            dataset.write_text("{}", encoding="utf-8")
            with patch("report.build_report", return_value=fake_report):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = report.main(
                        [
                            "--seeds",
                            "1",
                            "--dataset",
                            str(dataset),
                            "--model",
                            str(model),
                            "--model-device",
                            "cpu",
                        ]
                    )

        self.assertEqual(0, exit_code)
        self.assertIn("learned_policy", stdout.getvalue())
        self.assertIn("accuracy: 0.500000", stdout.getvalue())
        self.assertIn("average_home_cards: 6.00", stdout.getvalue())

    def test_cli_with_model_json_includes_learned_policy(self):
        fake_report = _fake_report_with_learned(dataset_skipped=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.pt"
            dataset = Path(tmpdir) / "dataset.jsonl"
            model.write_text("model", encoding="utf-8")
            dataset.write_text("{}", encoding="utf-8")
            with patch("report.build_report", return_value=fake_report):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = report.main(
                        [
                            "--seeds",
                            "1",
                            "--dataset",
                            str(dataset),
                            "--model",
                            str(model),
                            "--model-device",
                            "cpu",
                            "--format",
                            "json",
                        ]
                    )

        self.assertEqual(0, exit_code)
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(0.5, rendered["learned_policy"]["dataset"]["accuracy"])
        self.assertEqual(3, rendered["learned_policy"]["play"]["games"])

    def test_cli_with_model_without_dataset_marks_learned_dataset_skipped(self):
        fake_report = _fake_report_with_learned(dataset_skipped=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.pt"
            model.write_text("model", encoding="utf-8")
            with patch("report.build_report", return_value=fake_report):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = report.main(["--seeds", "1", "--model", str(model), "--model-device", "cpu"])

        self.assertEqual(0, exit_code)
        self.assertIn("skipped: no dataset provided", stdout.getvalue())
        self.assertIn("play", stdout.getvalue())

def _fake_report_with_learned(dataset_skipped=False):
    dataset = (
        {"skipped": True, "reason": "no dataset provided"}
        if dataset_skipped
        else {"skipped": False, "samples": 2, "correct": 1, "accuracy": 0.5}
    )
    return {
        "solver": {
            "games": 1,
            "solved": 0,
            "solve_rate": 0.0,
            "average_elapsed_seconds": 0.1,
            "average_explored_nodes": 4.0,
            "average_path_length_for_solved": 0.0,
        },
        "policy_baseline": {"skipped": True, "reason": "no dataset provided"},
        "policy_player": {
            "games": 1,
            "won": 0,
            "win_rate": 0.0,
            "average_steps": 5.0,
            "average_home_cards": 2.0,
            "policy": "heuristic",
            "results": [],
        },
        "learned_policy": {
            "model_path": "model.pt",
            "device": "cpu",
            "dataset": dataset,
            "play": {
                "games": 3,
                "won": 1,
                "win_rate": 1 / 3,
                "average_steps": 11.0,
                "average_home_cards": 6.0,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
