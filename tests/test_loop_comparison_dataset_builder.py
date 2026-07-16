import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import loop_comparison_dataset_builder


def row(would_loop=True, chosen_index=1, scores=None, legal_count=3):
    legal_moves = [
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        {"move_type": "COL_TO_COL", "from_idx": 0, "to_idx": 1, "count": 1},
        {"move_type": "FREE_TO_COL", "from_idx": 0, "to_idx": 2, "count": 1},
    ][:legal_count]
    scores = scores if scores is not None else [0.8, 0.95, 0.7][:legal_count]
    return {
        "version": 1,
        "seed": 32,
        "policy": "learned",
        "terminal_reason": "loop_detected",
        "terminal_step": 17,
        "window_index": 9,
        "step_index": 17,
        "state": {"columns": [], "free_cells": [], "home_cells": {}},
        "legal_moves": legal_moves,
        "chosen_action": legal_moves[chosen_index],
        "chosen_action_index": chosen_index,
        "model_scores": scores,
        "home_cards": 5,
        "visited_states": 18,
        "is_terminal_step": True,
        "would_loop": would_loop,
    }


class LoopComparisonDatasetBuilderTests(unittest.TestCase):
    def test_only_processes_would_loop_rows(self):
        pairs, summary = loop_comparison_dataset_builder.build_loop_comparison_samples(
            [row(would_loop=False), row(would_loop=True)]
        )

        self.assertEqual(1, summary["loop_rows"])
        self.assertEqual(1, summary["pairs_written"])
        self.assertEqual(2, summary["rows_read"])
        self.assertEqual(1, len(pairs))

    def test_rejected_action_is_chosen_action(self):
        sample = row(chosen_index=1)
        pairs, _ = loop_comparison_dataset_builder.build_loop_comparison_samples([sample])

        pair = pairs[0]
        self.assertEqual(sample["chosen_action"], pair["rejected_action"])
        self.assertEqual(sample["chosen_action_index"], pair["rejected_action_index"])

    def test_preferred_action_comes_from_same_legal_moves_and_is_not_rejected(self):
        sample = row(chosen_index=1)
        pairs, _ = loop_comparison_dataset_builder.build_loop_comparison_samples([sample])
        pair = pairs[0]

        self.assertIn(pair["preferred_action"], sample["legal_moves"])
        self.assertNotEqual(pair["preferred_action_index"], pair["rejected_action_index"])

    def test_preferred_defaults_to_highest_scoring_non_rejected_action(self):
        sample = row(chosen_index=1, scores=[0.4, 0.95, 0.9])
        pairs, _ = loop_comparison_dataset_builder.build_loop_comparison_samples([sample])

        self.assertEqual(2, pairs[0]["preferred_action_index"])
        self.assertEqual(0.9, pairs[0]["preferred_model_score"])
        self.assertEqual(0.95, pairs[0]["rejected_model_score"])

    def test_single_legal_action_is_skipped(self):
        pairs, summary = loop_comparison_dataset_builder.build_loop_comparison_samples(
            [row(chosen_index=0, legal_count=1, scores=[1.0])]
        )

        self.assertEqual([], pairs)
        self.assertEqual(1, summary["loop_rows"])
        self.assertEqual(1, summary["skipped_no_alternative"])

    def test_max_pairs_per_sample_can_emit_multiple_preferred_candidates(self):
        pairs, summary = loop_comparison_dataset_builder.build_loop_comparison_samples(
            [row(chosen_index=1, scores=[0.4, 0.95, 0.9])],
            max_pairs_per_sample=2,
        )

        self.assertEqual(2, summary["pairs_written"])
        self.assertEqual([2, 0], [pair["preferred_action_index"] for pair in pairs])

    def test_output_jsonl_is_parseable_and_fields_are_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pairs.jsonl"
            pairs, _ = loop_comparison_dataset_builder.build_loop_comparison_samples([row()])
            loop_comparison_dataset_builder.write_jsonl(output, pairs)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, len(rows))
        self.assertEqual(
            {
                "version",
                "source",
                "seed",
                "step_index",
                "terminal_reason",
                "terminal_step",
                "state",
                "preferred_action",
                "rejected_action",
                "preferred_action_index",
                "rejected_action_index",
                "preferred_model_score",
                "rejected_model_score",
                "reason",
                "policy",
                "home_cards",
                "window_index",
            },
            set(rows[0]),
        )
        self.assertEqual("failure_window", rows[0]["source"])
        self.assertEqual("avoid_loop_action", rows[0]["reason"])
        self.assertEqual("learned", rows[0]["policy"])

    def test_cli_missing_input_returns_nonzero(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = loop_comparison_dataset_builder.main(
                ["--failure-dataset", "missing.jsonl", "--output", "pairs.jsonl"]
            )

        self.assertEqual(1, exit_code)
        self.assertIn("failure dataset does not exist", stderr.getvalue())

    def test_cli_writes_summary_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            failure_dataset = Path(tmpdir) / "failure.jsonl"
            output = Path(tmpdir) / "pairs.jsonl"
            failure_dataset.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = loop_comparison_dataset_builder.main(
                    ["--failure-dataset", str(failure_dataset), "--output", str(output)]
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(rows))
        self.assertIn("rows_read: 1", stdout.getvalue())
        self.assertIn("pairs_written: 1", stdout.getvalue())

    def test_source_does_not_import_model_game_solver_or_torch(self):
        source = Path("loop_comparison_dataset_builder.py").read_text(encoding="utf-8")

        forbidden = (
            "import torch",
            "from torch",
            "import solver",
            "from solver",
            "import learned_policy",
            "from learned_policy",
            "import game_model",
            "from game_model",
        )
        for pattern in forbidden:
            self.assertNotIn(pattern, source)


if __name__ == "__main__":
    unittest.main()
