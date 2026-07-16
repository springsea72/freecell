import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import augment_dataset
from policy_features import features_from_sample


def sample(step_index=0):
    legal_moves = [
        {"move_type": "COL_TO_HOME", "from_idx": 0, "to_idx": None, "count": 1},
        {"move_type": "COL_TO_FREE", "from_idx": 0, "to_idx": 0, "count": 1},
    ]
    return {
        "version": 1,
        "source_trace": "seed_000001.json",
        "seed": 1,
        "step_index": step_index,
        "remaining_moves": 1,
        "state": {
            "columns": [[{"suit": "SPADES", "value": 1}], [], [], [], [], [], [], []],
            "free_cells": [None, None, None, None],
            "home_cells": {"SPADES": [], "HEARTS": [], "CLUBS": [], "DIAMONDS": []},
        },
        "legal_moves": legal_moves,
        "action": legal_moves[0],
        "action_index": 0,
        "progress": {
            "home_cards": 0,
            "home_cards_after": 1,
            "home_delta": 1,
            "remaining_moves": 1,
            "remaining_moves_after": 0,
            "won_after": False,
        },
    }


class AugmentDatasetTests(unittest.TestCase):
    def write_base(self, tmpdir, line=None):
        base = Path(tmpdir) / "base.jsonl"
        base.write_text((line or json.dumps(sample(), separators=(",", ":"))) + "\n", encoding="utf-8")
        return base

    def test_base_rows_are_preserved_verbatim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_line = '{"z":1, "a":2}'
            base = self.write_base(tmpdir, base_line)
            output = Path(tmpdir) / "aug.jsonl"

            with patch("augment_dataset.load_trace", return_value={}), patch(
                "augment_dataset.samples_from_trace", return_value=[]
            ):
                summary = augment_dataset.build_augmented_dataset(base, [Path(tmpdir) / "trace.json"], output)

            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(base_line, rows[0])
        self.assertEqual(1, summary["base_samples"])
        self.assertEqual(1, summary["samples_written"])

    def test_learned_trace_samples_are_appended_with_augmentation_fields(self):
        learned_sample = sample(step_index=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"

            with patch("augment_dataset.load_trace", return_value={}), patch(
                "augment_dataset.samples_from_trace", return_value=[learned_sample]
            ):
                summary = augment_dataset.build_augmented_dataset(base, [Path(tmpdir) / "seed_000032.json"], output)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, len(rows))
        self.assertNotIn("source_policy", rows[0])
        self.assertEqual("learned", rows[1]["source_policy"])
        self.assertEqual(0, rows[1]["augmentation_repeat"])
        self.assertEqual(1, summary["augmented_samples"])
        self.assertEqual(2, summary["samples_written"])

    def test_repeat_learned_traces_increases_sample_count(self):
        learned_samples = [sample(step_index=0), sample(step_index=1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"

            with patch("augment_dataset.load_trace", return_value={}), patch(
                "augment_dataset.samples_from_trace", return_value=learned_samples
            ):
                summary = augment_dataset.build_augmented_dataset(
                    base,
                    [Path(tmpdir) / "trace.json"],
                    output,
                    repeat_learned_traces=3,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(6, summary["augmented_samples"])
        self.assertEqual(7, summary["samples_written"])
        self.assertEqual([0, 0, 1, 1, 2, 2], [row["augmentation_repeat"] for row in rows[1:]])

    def test_invalid_trace_fails_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"
            trace = Path(tmpdir) / "bad.json"
            trace.write_text("{bad", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid trace"):
                augment_dataset.build_augmented_dataset(base, [trace], output)

            self.assertFalse(output.exists())

    def test_skip_invalid_trace_writes_base_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"
            trace = Path(tmpdir) / "bad.json"
            trace.write_text("{bad", encoding="utf-8")

            summary = augment_dataset.build_augmented_dataset(base, [trace], output, skip_invalid=True)
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(1, summary["invalid_traces"])
        self.assertEqual(0, summary["traces_used"])
        self.assertEqual(1, summary["samples_written"])
        self.assertEqual(1, len(rows))

    def test_trace_dir_missing_returns_nonzero_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = augment_dataset.main(
                    [
                        "--base-dataset",
                        str(base),
                        "--trace-dir",
                        str(Path(tmpdir) / "missing"),
                        "--output",
                        str(output),
                    ]
                )

        self.assertEqual(1, exit_code)
        self.assertIn("trace directory does not exist", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_output_jsonl_is_accepted_by_policy_features(self):
        learned_sample = sample(step_index=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"

            with patch("augment_dataset.load_trace", return_value={}), patch(
                "augment_dataset.samples_from_trace", return_value=[learned_sample]
            ):
                augment_dataset.build_augmented_dataset(base, [Path(tmpdir) / "trace.json"], output)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        features_from_sample(rows[1])

    def test_cli_writes_summary_and_augmented_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = self.write_base(tmpdir)
            output = Path(tmpdir) / "aug.jsonl"
            trace = Path(tmpdir) / "trace.json"
            trace.write_text("{}", encoding="utf-8")
            stdout = io.StringIO()

            with patch("augment_dataset.load_trace", return_value={}), patch(
                "augment_dataset.samples_from_trace", return_value=[sample()]
            ), redirect_stdout(stdout):
                exit_code = augment_dataset.main(
                    [
                        "--base-dataset",
                        str(base),
                        "--trace",
                        str(trace),
                        "--output",
                        str(output),
                        "--repeat-learned-traces",
                        "2",
                    ]
                )

            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(0, exit_code)
        self.assertEqual(3, len(rows))
        self.assertIn("base_samples: 1", stdout.getvalue())
        self.assertIn("augmented_samples: 2", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
