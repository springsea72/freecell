import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import benchmark
from solver import SolveResult


class BenchmarkTests(unittest.TestCase):
    def test_seed_argument_parsing_explicit_and_range(self):
        explicit = benchmark.parse_args(["--seeds", "1", "2", "3", "--max-nodes", "1000"])
        ranged = benchmark.parse_args(["--seed-start", "4", "--seed-count", "3"])

        self.assertEqual([1, 2, 3], benchmark.seeds_from_args(explicit))
        self.assertEqual([4, 5, 6], benchmark.seeds_from_args(ranged))

    def test_summary_metrics(self):
        records = [
            benchmark.BenchmarkRecord(1, True, "won", 10, 20, 5, 3, 0.5),
            benchmark.BenchmarkRecord(2, False, "max_nodes_exceeded", 30, 40, 7, 0, 1.5),
        ]

        summary = benchmark.summarize(records)

        self.assertEqual(2, summary.total_games)
        self.assertEqual(1, summary.solved_games)
        self.assertEqual(0.5, summary.solve_rate)
        self.assertEqual(1.0, summary.average_elapsed_seconds)
        self.assertEqual(20.0, summary.average_explored_nodes)
        self.assertEqual(3.0, summary.average_path_length_for_solved)

    def test_csv_output_contains_header_and_records(self):
        records = [
            benchmark.BenchmarkRecord(1, True, "won", 10, 20, 5, 3, 0.5),
            benchmark.BenchmarkRecord(2, False, "max_nodes_exceeded", 30, 40, 7, 0, 1.5),
        ]

        output = benchmark.render(records, "csv")

        self.assertIn(
            "seed,solved,reason,explored_nodes,generated_nodes,max_frontier,path_length,elapsed_seconds",
            output,
        )
        self.assertIn("1,True,won,10,20,5,3,0.500000", output)
        self.assertIn("2,False,max_nodes_exceeded,30,40,7,0,1.500000", output)
        self.assertIn("metric,value", output)

    def test_unsolved_game_returns_normal_result(self):
        result = SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded")

        with patch("benchmark.solve", return_value=result), patch(
            "benchmark.time.perf_counter", side_effect=[10.0, 10.25]
        ):
            records = benchmark.run_benchmark([1], max_nodes=1, max_depth=1)

        self.assertEqual(1, len(records))
        self.assertFalse(records[0].solved)
        self.assertEqual("max_nodes_exceeded", records[0].reason)
        self.assertEqual(0.25, records[0].elapsed_seconds)

    def test_main_returns_zero_for_unsolved_benchmark(self):
        result = SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded")

        with patch("benchmark.solve", return_value=result), patch(
            "benchmark.time.perf_counter", side_effect=[10.0, 10.25]
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = benchmark.main(["--seeds", "1", "--max-nodes", "1", "--max-depth", "1"])

        self.assertEqual(0, exit_code)
        self.assertIn("max_nodes_exceeded", output.getvalue())

    def test_save_solved_traces_only_saves_solved_results(self):
        solved = SolveResult(True, [], 1, 1, 1, "won")
        unsolved = SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            with patch("benchmark.solve", side_effect=[solved, unsolved]), patch(
                "benchmark.time.perf_counter", side_effect=[1.0, 1.1, 2.0, 2.2]
            ), patch("benchmark.save_trace") as save_trace, patch(
                "benchmark.load_trace", return_value={}
            ), patch(
                "benchmark.verify_trace", return_value=True
            ):
                records = benchmark.run_benchmark(
                    [1, 2],
                    max_nodes=10,
                    max_depth=5,
                    save_solved_traces=trace_dir,
                )

            self.assertTrue(trace_dir.exists())

        self.assertEqual([True, False], [record.solved for record in records])
        save_trace.assert_called_once_with(trace_dir / "seed_000001.json", 1, 10, 5, solved)


if __name__ == "__main__":
    unittest.main()
