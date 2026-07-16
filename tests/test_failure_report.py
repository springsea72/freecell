import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import failure_report


def row(reason="loop_detected", terminal=True, move_type="COL_TO_FREE", scores=None, chosen_index=1):
    scores = scores if scores is not None else [0.1, 0.9, 0.5]
    legal_moves = [
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        {"move_type": move_type, "from_idx": 0, "to_idx": 0, "count": 1},
        {"move_type": "COL_TO_COL", "from_idx": 1, "to_idx": 2, "count": 1},
    ]
    return {
        "version": 1,
        "seed": 1,
        "policy": "learned",
        "terminal_reason": reason,
        "terminal_step": 10,
        "window_index": 0,
        "step_index": 9,
        "state": {},
        "legal_moves": legal_moves,
        "chosen_action": legal_moves[chosen_index],
        "chosen_action_index": chosen_index,
        "model_scores": scores,
        "home_cards": 4,
        "visited_states": 10,
        "is_terminal_step": terminal,
        "would_loop": reason == "loop_detected",
        "empty_free_cells": 2,
        "empty_columns": 1,
        "buffer_slots": 3,
        "buried_low_cards": 4,
        "movable_suffix_total": 12,
        "top_cards_to_home": 2,
        "chosen_reduces_buffer": True,
        "chosen_releases_low_card": False,
        "chosen_is_home_move": move_type in ("FREE_TO_HOME", "COL_TO_HOME"),
    }


class FailureReportTests(unittest.TestCase):
    def test_chosen_action_rank_uses_descending_model_scores(self):
        sample = row(scores=[0.8, 0.2, 0.5], chosen_index=2)

        self.assertEqual(1, failure_report.chosen_action_rank(sample))

    def test_terminal_reason_grouping(self):
        report = failure_report.build_report(
            [
                row(reason="loop_detected", terminal=True, move_type="COL_TO_FREE"),
                row(reason="loop_detected", terminal=False, move_type="COL_TO_COL"),
                row(reason="no_legal_moves", terminal=True, move_type="COL_TO_HOME"),
            ]
        )

        self.assertEqual({"loop_detected": 1, "no_legal_moves": 1}, report["terminal_reasons"])
        self.assertEqual(2, report["by_terminal_reason"]["loop_detected"]["samples"])
        self.assertEqual(1, report["by_terminal_reason"]["no_legal_moves"]["samples"])

    def test_json_output_is_parseable(self):
        report = failure_report.build_report([row()])
        parsed = json.loads(failure_report.render_json(report))

        self.assertEqual(1, parsed["samples"])
        self.assertEqual(1, parsed["terminal_samples"])

    def test_text_output_contains_core_metrics(self):
        text = failure_report.render_text(failure_report.build_report([row()]))

        self.assertIn("terminal_reasons:", text)
        self.assertIn("average_chosen_action_rank:", text)
        self.assertIn("average_buffer_slots:", text)
        self.assertIn("top_bad_patterns:", text)

    def test_empty_input_returns_zero_metrics(self):
        report = failure_report.build_report([])

        self.assertEqual(0, report["samples"])
        self.assertEqual(0, report["terminal_samples"])
        self.assertEqual(0.0, report["would_loop_ratio"])
        self.assertEqual(0.0, report["average_chosen_action_rank"])
        self.assertEqual(0.0, report["average_buffer_slots"])

    def test_diagnostic_averages_are_calculated(self):
        first = row(move_type="COL_TO_FREE")
        second = row(reason="no_legal_moves", move_type="COL_TO_HOME")
        second["buffer_slots"] = 7
        second["buried_low_cards"] = 0
        second["movable_suffix_total"] = 4
        second["top_cards_to_home"] = 6

        report = failure_report.build_report([first, second])

        self.assertEqual(5.0, report["average_buffer_slots"])
        self.assertEqual(2.0, report["average_buried_low_cards"])
        self.assertEqual(8.0, report["average_movable_suffix_total"])
        self.assertEqual(4.0, report["average_top_cards_to_home"])
        self.assertEqual(3.0, report["by_terminal_reason"]["loop_detected"]["average_buffer_slots"])
        self.assertEqual(7.0, report["by_terminal_reason"]["no_legal_moves"]["average_buffer_slots"])
        self.assertEqual(2, report["top_bad_patterns"]["chosen_reduces_buffer_count"])
        self.assertEqual(1, report["top_bad_patterns"]["chosen_is_home_move_count"])
        self.assertEqual(0, report["top_bad_patterns"]["chosen_releases_low_card_count"])

    def test_missing_diagnostic_fields_default_to_zero(self):
        legacy = row()
        for field in (
            "buffer_slots",
            "buried_low_cards",
            "movable_suffix_total",
            "top_cards_to_home",
            "chosen_reduces_buffer",
            "chosen_releases_low_card",
            "chosen_is_home_move",
        ):
            legacy.pop(field, None)

        report = failure_report.build_report([legacy])

        self.assertEqual(0.0, report["average_buffer_slots"])
        self.assertEqual(0.0, report["by_terminal_reason"]["loop_detected"]["average_buried_low_cards"])
        self.assertEqual(0, report["top_bad_patterns"]["chosen_reduces_buffer_count"])

    def test_missing_file_returns_nonzero(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = failure_report.main(["--failure-dataset", "missing.jsonl"])

        self.assertEqual(1, exit_code)
        self.assertIn("failure dataset does not exist", stderr.getvalue())

    def test_cli_json_output_is_parseable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "failure.jsonl"
            path.write_text(json.dumps(row()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = failure_report.main(["--failure-dataset", str(path), "--format", "json"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, json.loads(stdout.getvalue())["samples"])

    def test_cli_empty_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = failure_report.main(["--failure-dataset", str(path), "--format", "json"])

        self.assertEqual(0, exit_code)
        self.assertEqual(0, json.loads(stdout.getvalue())["samples"])

    def test_source_does_not_import_solver_or_torch(self):
        source = Path("failure_report.py").read_text(encoding="utf-8")

        self.assertNotIn("import solver", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("from torch", source)


if __name__ == "__main__":
    unittest.main()
