import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import policy_baseline


def move(move_type, from_idx=0, to_idx=None, count=1):
    return {
        "move_type": move_type,
        "from_idx": from_idx,
        "to_idx": to_idx,
        "count": count,
    }


def sample(legal_moves, action):
    return {
        "version": 1,
        "source_trace": "seed_000001.json",
        "seed": 1,
        "step_index": 0,
        "remaining_moves": 1,
        "state": {},
        "legal_moves": legal_moves,
        "action": action,
        "action_index": legal_moves.index(action),
    }


class PolicyBaselineTests(unittest.TestCase):
    def test_load_jsonl_parses_lines(self):
        records = [
            sample([move("COL_TO_FREE")], move("COL_TO_FREE")),
            sample([move("COL_TO_HOME")], move("COL_TO_HOME")),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            loaded = policy_baseline.load_jsonl(path)

        self.assertEqual(records, loaded)

    def test_first_legal_returns_first_legal_move(self):
        first = move("COL_TO_FREE", 1, 0)
        second = move("COL_TO_HOME", 2)
        record = sample([first, second], second)

        self.assertEqual(first, policy_baseline.choose_action(record, policy="first_legal"))

    def test_heuristic_prefers_home_move(self):
        to_free = move("COL_TO_FREE", 1, 0)
        to_col = move("COL_TO_COL", 2, 3, count=3)
        to_home = move("COL_TO_HOME", 4)
        record = sample([to_free, to_col, to_home], to_home)

        self.assertEqual(to_home, policy_baseline.choose_action(record, policy="heuristic"))

    def test_evaluate_samples_accuracy(self):
        first = move("COL_TO_FREE", 1, 0)
        home = move("COL_TO_HOME", 2)
        records = [
            sample([first, home], first),
            sample([first, home], home),
        ]

        summary = policy_baseline.evaluate_samples(records, policy="first_legal")

        self.assertEqual(
            {"samples": 2, "correct": 1, "accuracy": 0.5, "policy": "first_legal"},
            summary,
        )

    def test_empty_dataset_accuracy_zero(self):
        summary = policy_baseline.evaluate_samples([], policy="heuristic")

        self.assertEqual(
            {"samples": 0, "correct": 0, "accuracy": 0.0, "policy": "heuristic"},
            summary,
        )

    def test_cli_valid_dataset_returns_zero(self):
        record = sample([move("COL_TO_HOME")], move("COL_TO_HOME"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = policy_baseline.main(["--dataset", str(path), "--policy", "heuristic"])

        self.assertEqual(0, exit_code)
        self.assertIn("policy: heuristic", stdout.getvalue())
        self.assertIn("accuracy: 1.000000", stdout.getvalue())

    def test_cli_unknown_policy_returns_nonzero(self):
        record = sample([move("COL_TO_HOME")], move("COL_TO_HOME"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = policy_baseline.main(["--dataset", str(path), "--policy", "unknown"])

        self.assertEqual(1, exit_code)
        self.assertIn("unknown policy", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
