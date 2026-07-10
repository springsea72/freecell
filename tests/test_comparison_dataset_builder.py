import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import comparison_dataset_builder


def sample(action_index=1, legal_count=3, step_index=12):
    legal_moves = [
        {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        {"move_type": "COL_TO_COL", "from_idx": 0, "to_idx": 1, "count": 1},
    ][:legal_count]
    return {
        "version": 1,
        "source_trace": "seed_000001.json",
        "seed": 1,
        "step_index": step_index,
        "remaining_moves": 42,
        "state": {
            "columns": [[{"suit": "SPADES", "value": 1}], [], [], [], [], [], [], []],
            "free_cells": [None, None, None, None],
            "home_cells": {"SPADES": [], "HEARTS": [], "CLUBS": [], "DIAMONDS": []},
        },
        "legal_moves": legal_moves,
        "action": legal_moves[action_index],
        "action_index": action_index,
        "progress": {
            "home_cards": 12,
            "home_cards_after": 13,
            "home_delta": 1,
            "remaining_moves": 42,
            "remaining_moves_after": 41,
            "won_after": False,
        },
    }


class ComparisonDatasetBuilderTests(unittest.TestCase):
    def test_single_sample_with_multiple_legal_moves_generates_pair(self):
        pairs, summary = comparison_dataset_builder.build_comparison_samples([sample()], seed=123)

        self.assertEqual({"samples_read": 1, "pairs_written": 1, "skipped_no_negative": 0}, summary)
        self.assertEqual(1, len(pairs))
        pair = pairs[0]
        self.assertEqual(1, pair["seed"])
        self.assertEqual(12, pair["step_index"])
        self.assertEqual("seed_000001.json:12", pair["source_sample"])
        self.assertEqual("trace_action_vs_non_trace", pair["reason"])
        self.assertEqual(sample()["action"], pair["preferred_action"])

    def test_rejected_action_index_differs_from_preferred_index(self):
        pairs, _ = comparison_dataset_builder.build_comparison_samples([sample()], seed=123)
        pair = pairs[0]

        self.assertNotEqual(pair["preferred_action_index"], pair["rejected_action_index"])
        self.assertNotEqual(pair["preferred_action"], pair["rejected_action"])

    def test_sample_with_one_legal_move_is_skipped(self):
        pairs, summary = comparison_dataset_builder.build_comparison_samples([sample(action_index=0, legal_count=1)])

        self.assertEqual([], pairs)
        self.assertEqual({"samples_read": 1, "pairs_written": 0, "skipped_no_negative": 1}, summary)

    def test_seed_makes_output_stable(self):
        samples = [sample(step_index=0), sample(step_index=1), sample(step_index=2)]

        first, _ = comparison_dataset_builder.build_comparison_samples(samples, negatives_per_sample=1, seed=99)
        second, _ = comparison_dataset_builder.build_comparison_samples(samples, negatives_per_sample=1, seed=99)

        self.assertEqual(first, second)

    def test_negatives_per_sample_is_capped_by_available_negatives(self):
        pairs, summary = comparison_dataset_builder.build_comparison_samples(
            [sample()],
            negatives_per_sample=5,
            seed=123,
        )

        self.assertEqual(2, len(pairs))
        self.assertEqual(2, summary["pairs_written"])
        self.assertEqual({0, 2}, {pair["rejected_action_index"] for pair in pairs})

    def test_output_jsonl_is_parseable_and_fields_are_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pairs.jsonl"
            pairs, _ = comparison_dataset_builder.build_comparison_samples([sample()], seed=123)
            comparison_dataset_builder.write_jsonl(output, pairs)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, len(rows))
        self.assertEqual(
            {
                "version",
                "source_sample",
                "seed",
                "step_index",
                "state",
                "preferred_action",
                "rejected_action",
                "preferred_action_index",
                "rejected_action_index",
                "reason",
                "progress",
            },
            set(rows[0]),
        )

    def test_progress_field_is_preserved(self):
        pairs, _ = comparison_dataset_builder.build_comparison_samples([sample()], seed=123)

        self.assertEqual(sample()["progress"], pairs[0]["progress"])

    def test_cli_missing_dataset_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            output = Path(tmpdir) / "pairs.jsonl"

            with redirect_stderr(stderr):
                exit_code = comparison_dataset_builder.main(
                    ["--dataset", str(Path(tmpdir) / "missing.jsonl"), "--output", str(output)]
                )

        self.assertEqual(1, exit_code)
        self.assertIn("dataset does not exist", stderr.getvalue())

    def test_cli_writes_summary_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset.jsonl"
            output = Path(tmpdir) / "pairs.jsonl"
            dataset.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = comparison_dataset_builder.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                        "--negatives-per-sample",
                        "2",
                        "--seed",
                        "123",
                    ]
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(0, exit_code)
        self.assertEqual(2, len(rows))
        self.assertIn("samples_read: 1", stdout.getvalue())
        self.assertIn("pairs_written: 2", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
