import argparse
import sys
from pathlib import Path

from trace_io import load_trace, verify_trace


def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv=None):
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Replay and verify a saved FreeCell trace.")
    parser.add_argument("trace_path")
    args = parser.parse_args(argv)

    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        print(f"trace not found: {trace_path}", file=sys.stderr)
        return 1

    try:
        trace = load_trace(trace_path)
    except (OSError, ValueError) as exc:
        print(f"failed to load trace: {exc}", file=sys.stderr)
        return 1

    verified = verify_trace(trace)
    path_length = len(trace.get("moves", []))
    stats = trace.get("stats", {})
    if "path_length" in stats:
        path_length = stats["path_length"]

    print(f"seed: {trace.get('seed')}")
    print(f"path_length: {path_length}")
    print(f"verified: {verified}")
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
