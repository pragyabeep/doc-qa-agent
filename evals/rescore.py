"""Re-score a SAVED eval run with the current checks (no model calls, no waiting).
Use it after you fix a grader bug:   python -m evals.rescore prompt-v1
The old file is kept as <label>.before-rescore.json."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from evals.run_evals import check, summarize

HERE = Path(__file__).parent


def main(label: str) -> None:
    path = HERE / "results" / f"{label}.json"
    data = json.loads(path.read_text())
    items = {i["id"]: i for i in json.loads((HERE / "dataset.json").read_text())}
    shutil.copy(path, path.with_name(f"{label}.before-rescore.json"))
    for row in data["rows"]:
        if "answer" not in row:   # the run errored for this question; nothing to re-score
            continue
        a = row["answer"]
        r = NS(completed=row["scores"].get("completed", 1.0) == 1.0,
               answer=NS(answer=a["answer"], found_in_documents=a["found_in_documents"],
                         citations=[NS(**c) for c in a["citations"]]),
               steps=[NS(tool=t) for t in row["tools"]])
        faithful = row["scores"].get("faithful")
        row["scores"] = check(items[row["id"]], r)
        if faithful is not None:
            row["scores"]["faithful"] = faithful
        row["passed"] = all(v == 1.0 for v in row["scores"].values())
    base = {k: data["summary"][k] for k in ("label", "model", "prompt_version", "retrieval_mode", "top_k")}
    data["summary"] = summarize(data["rows"], base)
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m evals.rescore <label>")
    main(sys.argv[1])
