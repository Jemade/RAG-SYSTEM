"""Export observed failures for human-reviewed regression case creation."""

import json
import sqlite3
from pathlib import Path


def export_failures(root, report_path, output):
    report = json.loads(Path(report_path).read_text())
    cases = []
    with sqlite3.connect(
        f"file:{(Path(root) / 'traces.sqlite3').resolve()}?mode=ro", uri=True
    ) as db:
        for row in report["rows"]:
            if not row["failures"]:
                continue
            found = db.execute(
                "SELECT payload FROM traces WHERE id=?", (row.get("trace_id"),)
            ).fetchone()
            trace = json.loads(found[0]) if found else {}
            cases.append(
                {
                    "origin_case": row["id"],
                    "trace_id": row.get("trace_id"),
                    "question": row["question"],
                    "failures": row["failures"],
                    "answer": row.get("answer"),
                    "retrieved_sources": [
                        c["source"] for c in trace.get("retrieved", [])
                    ],
                    "review_status": "needs_human_reference",
                    "proposed_regression": {
                        "id": "",
                        "question": "",
                        "answers": [],
                        "sources": [],
                        "category": row["category"],
                        "kind": row["kind"],
                        "split": "dev",
                        "rubric": "",
                    },
                }
            )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    return {"observed_failures": len(cases), "output": str(path)}
