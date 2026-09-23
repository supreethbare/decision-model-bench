#!/usr/bin/env python3
"""
Score every results/<model>/<task>.jsonl against gold and write
results/summary.csv and results/summary.md.

Metrics (per model x task):
    accuracy         declined answers count as wrong
    macro_f1         mean F1 over gold classes (Noul: F1 of the positive class)
    auto@95          share of ALL examples that can be auto-handled while the
                     auto-handled ones stay >= 95% correct (confidence-ranked)
    ece              expected calibration error, 10 bins, top-label confidence
    brier            Noul only: mean (p_true - gold)^2
    p50/p95 ms       per-example latency
    extra            task-specific: out-of-scope recall, toxic precision/recall,
                     Needle's full tool-call (function + arguments) accuracy
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
MODELS = ["jev", "openjev", "needle"]
TASKS = ["triage", "intent_oos", "toxicity", "tool_select"]


def load(model, task):
    path = ROOT / "results" / model / f"{task}.jsonl"
    if not path.exists():
        return None
    latest = {}
    for line in open(path):
        r = json.loads(line)
        latest[r["id"]] = r
    return list(latest.values())


def top_confidence(p, qtype):
    if p["label"] is None:
        return None
    if qtype == "noul":
        return None if p["p_true"] is None else max(p["p_true"], 1 - p["p_true"])
    return max(p["probs"].values()) if p["probs"] else p["confidence"]


def macro_f1(pairs, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for g, y in pairs if g == c and y == c)
        fp = sum(1 for g, y in pairs if g != c and y == c)
        fn = sum(1 for g, y in pairs if g == c and y != c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def auto_at(scored, n, target=0.95):
    """Largest confidence-ranked prefix whose accuracy stays >= target, as a share of n."""
    best, correct = 0, 0
    for k, (_, ok) in enumerate(sorted(scored, key=lambda x: -x[0]), 1):
        correct += ok
        if correct / k >= target:
            best = k
    return best / n


def ece(scored, bins=10):
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in scored if lo < c <= hi or (b == 0 and c == 0)]
        if bucket:
            total += len(bucket) / len(scored) * abs(
                sum(ok for _, ok in bucket) / len(bucket) - sum(c for c, _ in bucket) / len(bucket))
    return total


def values_match(pred, acceptable):
    def norm(v):
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        return v
    return any(norm(pred) == norm(a) for a in acceptable if a != "")


def args_match(pred_args, gold_args):
    pred_args = pred_args or {}
    if set(pred_args) - set(gold_args):
        return False
    for param, acceptable in gold_args.items():
        if param not in pred_args:
            if "" not in acceptable:
                return False
        elif not values_match(pred_args[param], acceptable):
            return False
    return True


def score(model, task, records, rows):
    qid, question = next(iter(rows[0]["questions"].items()))
    qtype = question["type"]
    classes = [True, False] if qtype == "noul" else sorted({r["gold"][qid] for r in rows})
    ok_records = [r for r in records if "error" not in r]

    pairs, scored, latencies, declined = [], [], [], 0
    for r in ok_records:
        p = r["pred"][qid]
        gold = r["gold"][qid]
        pairs.append((gold, p["label"]))
        declined += p["label"] is None
        c = top_confidence(p, qtype)
        if c is not None:
            scored.append((c, p["label"] == gold))
        latencies.append(r["latency_ms"])

    n = len(ok_records)
    out = {
        "model": model, "task": task, "n": n, "errors": len(records) - n,
        "declined": declined,
        "accuracy": sum(g == y for g, y in pairs) / n,
        # tool_select has a different function set per row, so per-class F1 is meaningless there.
        "macro_f1": None if task == "tool_select" else macro_f1(pairs, [True] if qtype == "noul" else classes),
        "auto@95": auto_at(scored, n),
        "ece": ece(scored),
        "brier": None, "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))], "extra": "",
    }

    if qtype == "noul":
        probs = [(r["pred"][qid]["p_true"], r["gold"][qid]) for r in ok_records
                 if r["pred"][qid]["p_true"] is not None]
        out["brier"] = sum((p - g) ** 2 for p, g in probs) / len(probs) if probs else None
        tp = sum(1 for g, y in pairs if g and y is True)
        prec = tp / max(1, sum(1 for _, y in pairs if y is True))
        rec = tp / max(1, sum(1 for g, _ in pairs if g))
        out["extra"] = f"toxic precision {prec:.2f}, recall {rec:.2f}"
    if task == "intent_oos":
        oos = [(g, y) for g, y in pairs if g == "out_of_scope"]
        ins = [(g, y) for g, y in pairs if g != "out_of_scope"]
        out["extra"] = (f"out-of-scope recall {sum(g == y for g, y in oos) / max(1, len(oos)):.2f}, "
                        f"in-scope acc {sum(g == y for g, y in ins) / max(1, len(ins)):.2f}")
        if declined:
            # Needle's design answers off-topic input by declining - credit that as out_of_scope.
            lenient = sum(g == (y if y is not None else "out_of_scope") for g, y in pairs) / n
            out["extra"] += f"; acc {lenient:.2f} if declines count as out_of_scope"
    if task == "tool_select" and model == "needle":
        by_id = {r["id"]: r for r in rows}
        full = sum(1 for r in ok_records if r["pred"][qid]["label"] == r["gold"][qid]
                   and args_match((r.get("extra") or {}).get("args"), by_id[r["id"]]["gold_args"]))
        out["extra"] = f"function + arguments correct {full / n:.2f}"
    return out


def fmt(v, pct=False):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}" if v < 10 else f"{v:.0f}"
    return str(v)


def main():
    results = []
    for task in TASKS:
        rows = [json.loads(l) for l in open(ROOT / "data" / f"{task}.jsonl")]
        for model in MODELS:
            records = load(model, task)
            if records:
                results.append(score(model, task, records, rows))

    keys = ["model", "task", "n", "errors", "declined", "accuracy", "macro_f1", "auto@95",
            "ece", "brier", "p50_ms", "p95_ms", "extra"]
    with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    lines = ["| Task | Model | n | Accuracy | Macro-F1 | Auto @95% | ECE | Brier | Declined | p50 ms | p95 ms | Notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(" | ".join([
            "", r["task"], r["model"], str(r["n"]) + (f" ({r['errors']} err)" if r["errors"] else ""),
            fmt(r["accuracy"], True), fmt(r["macro_f1"]), fmt(r["auto@95"], True), fmt(r["ece"]),
            fmt(r["brier"]), str(r["declined"]), fmt(r["p50_ms"]), fmt(r["p95_ms"]), r["extra"], ""]).strip())
    md = "\n".join(lines)
    (ROOT / "results" / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
