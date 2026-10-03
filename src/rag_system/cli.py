import argparse
import json

from .core import Pipeline
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="var")
    parser.add_argument("--mode", choices=["demo", "semantic"], default="demo")
    parser.add_argument("--top-k", type=int, default=5)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("directory", nargs="?", default="data/corpus")
    query = commands.add_parser("query")
    query.add_argument("question")
    query.add_argument("--model", default="demo")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--dataset", default="data/evaluation.jsonl")
    ev.add_argument("--output", default="var/reports/latest.json")
    ev.add_argument("--model", default="demo")
    ev.add_argument("--judge")
    ev.add_argument("--split", choices=["dev", "holdout"])
    ev.add_argument("--min-correctness", type=float)
    ev.add_argument("--require-judge", action="store_true")
    triage = commands.add_parser("triage")
    triage.add_argument("report")
    triage.add_argument("--output", default="var/reports/failure-review.jsonl")
    commands.add_parser("serve")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(args.root), host="127.0.0.1", port=8000)
        return
    if args.command == "triage":
        from .triage import export_failures

        print(
            json.dumps(export_failures(args.root, args.report, args.output), indent=2)
        )
        return
    pipeline = Pipeline(args.root, args.mode, args.top_k)
    try:
        if args.command == "ingest":
            result = pipeline.ingest(args.directory)
        elif args.command == "query":
            result = pipeline.query(args.question, args.model)
        else:
            if args.require_judge and not args.judge:
                parser.error("--require-judge requires --judge")
            result = evaluate(
                pipeline, args.dataset, args.output, args.model, args.judge, args.split
            )
            failed = any(
                "pipeline_error" in r["failures"] or "judge_error" in r["failures"]
                for r in result["rows"]
            )
            if args.min_correctness is not None:
                if not 0 <= args.min_correctness <= 1:
                    parser.error("--min-correctness must be between 0 and 1")
                failed |= (
                    result["correctness_scored"] != result["count"]
                    or (result["correctness_mean"] or 0) < args.min_correctness
                )
            print(
                json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)
            )
            if failed:
                raise SystemExit(1)
            return
        print(json.dumps(result, indent=2))
    finally:
        pipeline.close()
