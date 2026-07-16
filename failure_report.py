import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DIAGNOSTIC_AVERAGE_FIELDS = (
    "buffer_slots",
    "buried_low_cards",
    "movable_suffix_total",
    "top_cards_to_home",
)


def load_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return rows


def chosen_action_rank(row: dict) -> int:
    scores = row.get("model_scores") or []
    chosen_index = row.get("chosen_action_index")
    if not isinstance(chosen_index, int) or chosen_index < 0 or chosen_index >= len(scores):
        return -1
    ranked_indices = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return ranked_indices.index(chosen_index)


def chosen_action_score(row: dict):
    scores = row.get("model_scores") or []
    chosen_index = row.get("chosen_action_index")
    if not isinstance(chosen_index, int) or chosen_index < 0 or chosen_index >= len(scores):
        return None
    return float(scores[chosen_index])


def build_report(rows: list[dict]) -> dict:
    samples = len(rows)
    terminal_rows = [row for row in rows if row.get("is_terminal_step")]
    would_loop_count = sum(1 for row in rows if row.get("would_loop"))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("terminal_reason", "unknown")].append(row)

    return {
        "samples": samples,
        "terminal_samples": len(terminal_rows),
        "terminal_reasons": _counter_dict(row.get("terminal_reason", "unknown") for row in terminal_rows),
        "chosen_move_types": _counter_dict(_chosen_move_type(row) for row in rows),
        "terminal_step_chosen_move_types": _counter_dict(_chosen_move_type(row) for row in terminal_rows),
        "would_loop_count": would_loop_count,
        "would_loop_ratio": would_loop_count / samples if samples else 0.0,
        "average_chosen_action_score": _average(chosen_action_score(row) for row in rows),
        "average_chosen_action_rank": _average_rank(rows),
        "average_buffer_slots": _average_metric(rows, "buffer_slots"),
        "average_buried_low_cards": _average_metric(rows, "buried_low_cards"),
        "average_movable_suffix_total": _average_metric(rows, "movable_suffix_total"),
        "average_top_cards_to_home": _average_metric(rows, "top_cards_to_home"),
        "by_terminal_reason": {
            reason: _summary_for_rows(reason_rows)
            for reason, reason_rows in sorted(grouped.items())
        },
        "top_bad_patterns": {
            "loop_terminal_chosen_move_types": _counter_dict(
                _chosen_move_type(row)
                for row in terminal_rows
                if row.get("terminal_reason") == "loop_detected"
            ),
            "no_legal_previous_window_chosen_move_types": _counter_dict(
                _chosen_move_type(row)
                for row in terminal_rows
                if row.get("terminal_reason") == "no_legal_moves"
            ),
            "col_to_free_chosen_count": sum(1 for row in rows if _chosen_move_type(row) == "COL_TO_FREE"),
            "chosen_reduces_buffer_count": sum(1 for row in rows if row.get("chosen_reduces_buffer")),
            "chosen_is_home_move_count": sum(1 for row in rows if row.get("chosen_is_home_move")),
            "chosen_releases_low_card_count": sum(1 for row in rows if row.get("chosen_releases_low_card")),
        },
    }


def render_text(report: dict) -> str:
    lines = [
        f"samples: {report['samples']}",
        f"terminal_samples: {report['terminal_samples']}",
        f"terminal_reasons: {json.dumps(report['terminal_reasons'], sort_keys=True)}",
        f"chosen_move_types: {json.dumps(report['chosen_move_types'], sort_keys=True)}",
        f"terminal_step_chosen_move_types: {json.dumps(report['terminal_step_chosen_move_types'], sort_keys=True)}",
        f"would_loop: {report['would_loop_count']} ({report['would_loop_ratio']:.6f})",
        f"average_chosen_action_score: {report['average_chosen_action_score']:.6f}",
        f"average_chosen_action_rank: {report['average_chosen_action_rank']:.6f}",
        f"average_buffer_slots: {report['average_buffer_slots']:.6f}",
        f"average_buried_low_cards: {report['average_buried_low_cards']:.6f}",
        f"average_movable_suffix_total: {report['average_movable_suffix_total']:.6f}",
        f"average_top_cards_to_home: {report['average_top_cards_to_home']:.6f}",
        "by_terminal_reason:",
    ]
    for reason, summary in report["by_terminal_reason"].items():
        lines.append(
            "  "
            f"{reason}: samples={summary['samples']} "
            f"avg_home={summary['average_home_cards']:.2f} "
            f"avg_legal_moves={summary['average_legal_moves']:.2f} "
            f"avg_score={summary['average_chosen_action_score']:.6f} "
            f"avg_rank={summary['average_chosen_action_rank']:.6f} "
            f"avg_buffer={summary['average_buffer_slots']:.2f} "
            f"avg_buried_low={summary['average_buried_low_cards']:.2f} "
            f"avg_suffix={summary['average_movable_suffix_total']:.2f} "
            f"avg_top_home={summary['average_top_cards_to_home']:.2f}"
        )
    patterns = report["top_bad_patterns"]
    lines.extend(
        [
            "top_bad_patterns:",
            f"  loop_terminal_chosen_move_types: {json.dumps(patterns['loop_terminal_chosen_move_types'], sort_keys=True)}",
            f"  no_legal_previous_window_chosen_move_types: {json.dumps(patterns['no_legal_previous_window_chosen_move_types'], sort_keys=True)}",
            f"  col_to_free_chosen_count: {patterns['col_to_free_chosen_count']}",
            f"  chosen_reduces_buffer_count: {patterns['chosen_reduces_buffer_count']}",
            f"  chosen_is_home_move_count: {patterns['chosen_is_home_move_count']}",
            f"  chosen_releases_low_card_count: {patterns['chosen_releases_low_card_count']}",
        ]
    )
    return "\n".join(lines)


def render_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def _summary_for_rows(rows: list[dict]) -> dict:
    summary = {
        "samples": len(rows),
        "average_home_cards": _average(row.get("home_cards", 0) for row in rows),
        "average_legal_moves": _average(len(row.get("legal_moves") or []) for row in rows),
        "average_chosen_action_score": _average(chosen_action_score(row) for row in rows),
        "average_chosen_action_rank": _average_rank(rows),
    }
    for field in DIAGNOSTIC_AVERAGE_FIELDS:
        summary[f"average_{field}"] = _average_metric(rows, field)
    return summary


def _average_metric(rows: list[dict], field: str) -> float:
    return _average(row.get(field, 0) for row in rows)


def _average_rank(rows: list[dict]) -> float:
    ranks = [chosen_action_rank(row) for row in rows]
    return _average(rank for rank in ranks if rank >= 0)


def _average(values) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _counter_dict(values) -> dict:
    return dict(sorted(Counter(value for value in values if value is not None).items()))


def _chosen_move_type(row: dict):
    action = row.get("chosen_action") or {}
    return action.get("move_type")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize learned-policy failure-window JSONL.")
    parser.add_argument("--failure-dataset", required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.failure_dataset)
    if not path.exists():
        print(f"failure dataset does not exist: {path}", file=sys.stderr)
        return 1

    try:
        report = build_report(load_jsonl(path))
    except (OSError, ValueError) as exc:
        print(f"failed to read failure dataset: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(render_json(report))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
