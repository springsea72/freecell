import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import degradation_comparison_dataset_builder


def row(
    chosen_index=3,
    scores=None,
    legal_moves=None,
    terminal_reason="no_legal_moves",
    reduces_buffer=True,
    releases_low=False,
    is_home=False,
):
    legal_moves = legal_moves or [
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        {"move_type": "FREE_TO_COL", "from_idx": 0, "to_idx": 2, "count": 1},
        {"move_type": "COL_TO_COL", "from_idx": 1, "to_idx": 2, "count": 1},
        {"move_type": "COL_TO_FREE", "from_idx": 1, "to_idx": 0, "count": 1},
    ]
    scores = scores if scores is not None else [0.1, 0.2, 0.9, 1.0][: len(legal_moves)]
    return {
        "version": 1,
        "seed": 7,
        "policy": "learned",
        "terminal_reason": terminal_reason,
        "terminal_step": 12,
        "window_index": 4,
        "step_index": 8,
        "state": {"columns": [], "free_cells": [], "home_cells": {}},
        "legal_moves": legal_moves,
        "chosen_action": legal_moves[chosen_index],
        "chosen_action_index": chosen_index,
        "model_scores": scores,
        "home_cards": 3,
        "buffer_slots": 1,
        "buried_low_cards": 6,
        "movable_suffix_total": 11,
        "chosen_reduces_buffer": reduces_buffer,
        "chosen_releases_low_card": releases_low,
        "chosen_is_home_move": is_home,
    }


class DegradationComparisonDatasetBuilderTests(unittest.TestCase):
    def test_only_processes_structural_degradation_rows(self):
        pairs, summary = degradation_comparison_dataset_builder.build_degradation_comparison_samples(
            [
                row(reduces_buffer=False),
                row(releases_low=True),
                row(is_home=True),
                row(),
            ]
        )

        self.assertEqual(1, summary["candidate_rows"])
        self.assertEqual(1, summary["pairs_written"])
        self.assertEqual(4, summary["rows_read"])

    def test_rejected_action_is_chosen_action(self):
        sample = row(chosen_index=3)
        pairs, _ = degradation_comparison_dataset_builder.build_degradation_comparison_samples([sample])
        pair = pairs[0]

        self.assertEqual(sample["chosen_action"], pair["rejected_action"])
        self.assertEqual(3, pair["rejected_action_index"])

    def test_preferred_action_comes_from_same_legal_moves_and_is_not_rejected(self):
        sample = row(chosen_index=3)
        pairs, _ = degradation_comparison_dataset_builder.build_degradation_comparison_samples([sample])
        pair = pairs[0]

        self.assertIn(pair["preferred_action"], sample["legal_moves"])
        self.assertNotEqual(pair["preferred_action_index"], pair["rejected_action_index"])

    def test_home_move_is_preferred_over_other_candidates(self):
        sample = row(chosen_index=3, scores=[0.1, 0.9, 0.8, 1.0])
        pairs, _ = degradation_comparison_dataset_builder.build_degradation_comparison_samples([sample])

        self.assertEqual("COL_TO_HOME", pairs[0]["preferred_action"]["move_type"])

    def test_col_to_free_is_not_preferred_over_free_to_col_or_col_to_col(self):
        legal_moves = [
            {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
            {"move_type": "FREE_TO_COL", "from_idx": 0, "to_idx": 2, "count": 1},
            {"move_type": "COL_TO_COL", "from_idx": 1, "to_idx": 2, "count": 1},
            {"move_type": "COL_TO_FREE", "from_idx": 1, "to_idx": 1, "count": 1},
        ]
        sample = row(chosen_index=3, legal_moves=legal_moves, scores=[100.0, 0.2, 0.1, 1.0])
        pairs, _ = degradation_comparison_dataset_builder.build_degradation_comparison_samples([sample])

        self.assertEqual("FREE_TO_COL", pairs[0]["preferred_action"]["move_type"])

    def test_single_legal_action_is_skipped(self):
        legal_moves = [{"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1}]
        pairs, summary = degradation_comparison_dataset_builder.build_degradation_comparison_samples(
            [row(chosen_index=0, legal_moves=legal_moves, scores=[1.0])]
        )

        self.assertEqual([], pairs)
        self.assertEqual(1, summary["candidate_rows"])
        self.assertEqual(1, summary["skipped_no_alternative"])

    def test_by_terminal_reason_summary_is_recorded(self):
        _, summary = degradation_comparison_dataset_builder.build_degradation_comparison_samples(
            [row(terminal_reason="no_legal_moves"), row(terminal_reason="loop_detected")]
        )

        self.assertEqual(1, summary["by_terminal_reason"]["no_legal_moves"]["pairs_written"])
        self.assertEqual(1, summary["by_terminal_reason"]["loop_detected"]["pairs_written"])

    def test_output_jsonl_is_parseable_and_fields_are_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "pairs.jsonl"
            pairs, _ = degradation_comparison_dataset_builder.build_degradation_comparison_samples([row()])
            degradation_comparison_dataset_builder.write_jsonl(output, pairs)
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
                "buffer_slots",
                "buried_low_cards",
                "movable_suffix_total",
                "window_index",
            },
            set(rows[0]),
        )
        self.assertEqual("failure_window", rows[0]["source"])
        self.assertEqual("avoid_structural_degradation", rows[0]["reason"])
        self.assertEqual("learned", rows[0]["policy"])

    def test_cli_missing_input_returns_nonzero(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = degradation_comparison_dataset_builder.main(
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
                exit_code = degradation_comparison_dataset_builder.main(
                    ["--failure-dataset", str(failure_dataset), "--output", str(output)]
                )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(rows))
        self.assertIn("rows_read: 1", stdout.getvalue())
        self.assertIn("pairs_written: 1", stdout.getvalue())
        self.assertIn("by_terminal_reason:", stdout.getvalue())

    def test_source_does_not_import_model_game_solver_or_torch(self):
        source = Path("degradation_comparison_dataset_builder.py").read_text(encoding="utf-8")

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
