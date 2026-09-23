#!/usr/bin/env python3
"""
Run one model over the benchmark sets and save per-example predictions.

    python run.py --model needle
    python run.py --model openjev --tasks triage toxicity
    python run.py --model jev --limit 20        # quick check first

Results go to results/<model>/<task>.jsonl. Re-running resumes: examples
already saved are skipped (failed ones are retried), so an interrupted run
can simply be restarted.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from models import MODELS

ROOT = Path(__file__).parent
TASKS = ["triage", "intent_oos", "toxicity", "tool_select"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--limit", type=int, help="first N rows per task")
    args = parser.parse_args()

    model = MODELS[args.model]()
    try:
        for task in args.tasks:
            rows = [json.loads(l) for l in open(ROOT / "data" / f"{task}.jsonl")][:args.limit]
            out_path = ROOT / "results" / args.model / f"{task}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Errored examples are retried; the report keeps the last record per id.
            saved = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
            done = {r["id"] for r in saved if "error" not in r}
            todo = [r for r in rows if r["id"] not in done]
            print(f"[{args.model}] {task}: {len(done)} done, {len(todo)} to run", flush=True)

            with open(out_path, "a") as out:
                for i, row in enumerate(todo, 1):
                    record = {"id": row["id"], "gold": row["gold"]}
                    t0 = time.perf_counter()
                    try:
                        record["pred"], record["extra"] = model.predict(row)
                        record["latency_ms"] = (time.perf_counter() - t0) * 1000
                    except Exception as e:
                        record["error"] = f"{type(e).__name__}: {e}"
                        traceback.print_exc(limit=1)
                    out.write(json.dumps(record, default=str) + "\n")
                    out.flush()
                    if i % 20 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()
