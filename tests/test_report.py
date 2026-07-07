import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import benchmark
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


if __name__ == "__main__":
    unittest.main()
