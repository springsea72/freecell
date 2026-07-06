import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import replay


class ReplayTests(unittest.TestCase):
    def test_replay_cli_valid_trace_returns_zero(self):
        trace = {"version": 1, "seed": 1, "moves": [], "stats": {"path_length": 0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with patch("replay.verify_trace", return_value=True):
                with redirect_stdout(StringIO()):
                    exit_code = replay.main([str(path)])

        self.assertEqual(0, exit_code)

    def test_replay_cli_invalid_trace_returns_nonzero(self):
        trace = {"version": 1, "seed": 1, "moves": [], "stats": {"path_length": 0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with patch("replay.verify_trace", return_value=False):
                with redirect_stdout(StringIO()):
                    exit_code = replay.main([str(path)])

        self.assertEqual(1, exit_code)


if __name__ == "__main__":
    unittest.main()
