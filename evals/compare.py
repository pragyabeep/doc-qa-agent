"""Compare two eval runs:  python -m evals.compare baseline improved"""
import json
import sys
from pathlib import Path

R = Path(__file__).parent / "results"


def load(label: str) -> dict:
    return json.loads((R / f"{label}.json").read_text())


def main(a_label: str, b_label: str) -> None:
    a, b = load(a_label), load(b_label)
    sa, sb = a["summary"], b["summary"]
    print(f"{'metric':<16}{a_label:>14}{b_label:>14}{'delta':>10}")
    print(f"{'pass_rate':<16}{sa['pass_rate']:>14}{sb['pass_rate']:>14}{sb['pass_rate'] - sa['pass_rate']:>+10.3f}")
    for m in sorted(set(sa["metrics"]) | set(sb["metrics"])):
        x, y = sa["metrics"].get(m), sb["metrics"].get(m)
        d = f"{y - x:+.3f}" if x is not None and y is not None else "n/a"
        print(f"{m:<16}{x if x is not None else '-':>14}{y if y is not None else '-':>14}{d:>10}")
    for k in ("avg_latency_ms", "avg_tokens"):
        print(f"{k:<16}{sa[k]:>14}{sb[k]:>14}{sb[k] - sa[k]:>+10}")
    pa = {r["id"]: r["passed"] for r in a["rows"]}
    print("\nFixed:    ", [r["id"] for r in b["rows"] if r["passed"] and not pa.get(r["id"], False)])
    print("Regressed:", [r["id"] for r in b["rows"] if not r["passed"] and pa.get(r["id"], False)])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m evals.compare <label_a> <label_b>")
    main(sys.argv[1], sys.argv[2])
