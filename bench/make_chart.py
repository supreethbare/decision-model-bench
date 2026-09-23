#!/usr/bin/env python3
"""Fill chart_template.html with results/summary.csv -> results/comparison.html."""

import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).parent
NUMERIC = ["n", "errors", "declined", "accuracy", "macro_f1", "auto@95", "ece", "brier", "p50_ms", "p95_ms"]


def main():
    rows = []
    for r in csv.DictReader(open(ROOT / "results" / "summary.csv")):
        for k in NUMERIC:
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        rows.append(r)
    meta = {
        "Run": datetime.date.today().isoformat(),
        "Cases": f"{int(sum(r['n'] for r in rows if r['model'] == 'jev'))} per model",
        "Jev": "jev-1.13.0",
        "Open-Jev": "2B · Qwen3.5-2B + LoRA",
        "Needle": "cactus-needle 3.0.4",
    }
    html = (ROOT / "chart_template.html").read_text()
    html = html.replace("/*DATA*/[]", json.dumps(rows)).replace("/*META*/{}", json.dumps(meta))
    out = ROOT / "results" / "comparison.html"
    out.write_text(html)
    print(out)


if __name__ == "__main__":
    main()
