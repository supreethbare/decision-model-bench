#!/usr/bin/env python3
"""
Build the four benchmark sets from public datasets (raw files in ./raw).

Every row has the same shape, so all three models read the same input:

    {"id", "task", "state", "questions": {qid: {type, instructions, criteria}},
     "gold": {qid: label}, ...task extras}

Tasks (fixed seed, balanced sampling):

    triage      Bitext customer support   Choice over 11 support categories
    intent_oos  CLINC150 (plus, test)     Choice over 15 banking intents + out_of_scope
    toxicity    Civil Comments (test)     Noul: is the comment toxic?
    tool_select BFCL v3 "multiple"        Choice of function; Needle also fills arguments
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SEED = 42
RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "data"

TRIAGE_CATEGORIES = {
    "account": "Creating, deleting, editing or switching an account; password recovery; registration problems.",
    "cancel": "Questions about cancellation fees.",
    "contact": "Wants to reach customer service or a human agent.",
    "delivery": "Delivery options and delivery times.",
    "feedback": "Complaints or reviews about the company or service.",
    "invoice": "Viewing or obtaining an invoice.",
    "order": "Placing, changing, cancelling or tracking an order.",
    "payment": "Payment methods or problems making a payment.",
    "refund": "Refund policy, requesting a refund or tracking a refund.",
    "shipping": "Setting up or changing a shipping address.",
    "subscription": "Newsletter subscription.",
}

BANKING_INTENTS = {
    "transfer": "Move money between accounts or send it to someone.",
    "transactions": "See recent transactions on an account.",
    "balance": "Check an account balance.",
    "freeze_account": "Freeze or lock a bank account.",
    "pay_bill": "Pay a bill.",
    "bill_balance": "How much is owed on a bill.",
    "bill_due": "When a bill is due.",
    "interest_rate": "The interest rate on an account.",
    "routing": "The bank's routing number.",
    "min_payment": "The minimum payment due.",
    "order_checks": "Order new checks.",
    "pin_change": "Change a PIN.",
    "report_fraud": "Report fraudulent activity.",
    "account_blocked": "An account is blocked or locked and the user wants to know why.",
    "spending_history": "How much was spent over a period or on something.",
    "out_of_scope": "None of the banking requests above.",
}
# Near-domain hard negatives: real credit-card intents that a banking bot must NOT claim.
NEAR_DOMAIN_OOS = ["report_lost_card", "credit_limit", "card_declined", "apr", "new_card"]


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name:12} {len(rows)} rows")


def build_triage(rng, per_class=18):
    by_cat = defaultdict(list)
    for r in pq.read_table(RAW / "bitext.parquet", columns=["instruction", "category"]).to_pylist():
        by_cat[r["category"].lower()].append(r["instruction"])
    rows = []
    for cat in TRIAGE_CATEGORIES:
        for i, text in enumerate(rng.sample(by_cat[cat], per_class)):
            rows.append({
                "id": f"triage_{cat}_{i}", "task": "triage",
                "state": {"customer_message": text},
                "questions": {"route": {
                    "type": "choice",
                    "instructions": "Which support category does this customer message belong to?",
                    "criteria": TRIAGE_CATEGORIES}},
                "gold": {"route": cat},
            })
    return rows


def build_intent(rng, per_intent=10, n_oos=25, per_near=5):
    table = pq.read_table(RAW / "clinc_plus_test.parquet")
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    by_intent = defaultdict(list)
    for r in table.to_pylist():
        by_intent[names[r["intent"]]].append(r["text"])
    picks = [(i, t, i) for i in BANKING_INTENTS if i != "out_of_scope"
             for t in rng.sample(by_intent[i], per_intent)]
    picks += [("oos", t, "out_of_scope") for t in rng.sample(by_intent["oos"], n_oos)]
    picks += [(i, t, "out_of_scope") for i in NEAR_DOMAIN_OOS for t in rng.sample(by_intent[i], per_near)]
    return [{
        "id": f"intent_{src}_{n}", "task": "intent_oos", "source_intent": src,
        "state": {"user_utterance": text},
        "questions": {"intent": {
            "type": "choice",
            "instructions": "Which banking request is the user making? Use out_of_scope if none applies.",
            "criteria": BANKING_INTENTS}},
        "gold": {"intent": gold},
    } for n, (src, text, gold) in enumerate(picks)]


def build_toxicity(rng, per_class=100):
    t = pq.read_table(RAW / "civil_test.parquet", columns=["text", "toxicity"]).to_pylist()
    toxic = [r for r in t if r["toxicity"] >= 0.5 and len(r["text"]) < 1500]
    clean = [r for r in t if r["toxicity"] < 0.5 and len(r["text"]) < 1500]
    rows = []
    for label, pool in ((True, toxic), (False, clean)):
        for i, r in enumerate(rng.sample(pool, per_class)):
            rows.append({
                "id": f"toxicity_{'pos' if label else 'neg'}_{i}", "task": "toxicity",
                "annotator_toxicity": round(r["toxicity"], 3),
                "state": {"comment": r["text"]},
                "questions": {"toxic": {
                    "type": "noul",
                    "instructions": ("Is this comment toxic - rude, disrespectful or unreasonable "
                                     "enough to make someone leave a discussion?")}},
                "gold": {"toxic": label},
            })
    return rows


def safe_key(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def build_tool_select():
    questions = [json.loads(l) for l in open(RAW / "bfcl_multiple.json")]
    answers = {a["id"]: a["ground_truth"][0] for a in map(json.loads, open(RAW / "bfcl_multiple_answers.json"))}
    rows = []
    for q in questions:
        funcs = q["function"]
        keys = {f["name"]: safe_key(f["name"]) for f in funcs}
        assert len(set(keys.values())) == len(keys), q["id"]
        (gold_name, gold_args), = answers[q["id"]].items()
        rows.append({
            "id": q["id"], "task": "tool_select",
            "state": {"user_request": q["question"][0][0]["content"]},
            "questions": {"function": {
                "type": "choice",
                "instructions": "Which function should be called to fulfil the user's request?",
                "criteria": {keys[f["name"]]: f["description"] for f in funcs}}},
            "gold": {"function": keys[gold_name]},
            "functions": funcs,                 # original schemas, for Needle's real tool calling
            "key_to_name": {v: k for k, v in keys.items()},
            "gold_args": gold_args,             # BFCL possible answers: {param: [acceptable values]}
        })
    return rows


def main():
    write("triage", build_triage(random.Random(SEED)))
    write("intent_oos", build_intent(random.Random(SEED)))
    write("toxicity", build_toxicity(random.Random(SEED)))
    write("tool_select", build_tool_select())


if __name__ == "__main__":
    main()
