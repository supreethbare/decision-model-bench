#!/usr/bin/env python3
"""
Jev smoke test - support ticket triage.

Minimal check that your TypeSafe API key works and that Jev returns typed,
calibrated decisions. One call, three question types, no parsing.

Setup (conda):

    conda activate <your-env>
    pip install typesafe-sdk
    export TYPESAFE_API_KEY="sk-..."

Run:

    python jev_smoke_test.py
"""

import getpass
import os
import sys
import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient


# --------------------------------------------------------------------------
# The state. Whatever your program already has in hand - a dict, not a prompt.
# --------------------------------------------------------------------------
STATE = {
    "ticket": (
        "Hi - I've been charged twice for the September invoice and the duplicate "
        "still hasn't been reversed. This is the third time I'm writing in. We have "
        "payroll running Friday and I need this sorted before then."
    ),
    "account_tier": "business",
    "prior_tickets_this_month": 3,
}

# --------------------------------------------------------------------------
# The typed decisions we want back. All answered in one round trip, in parallel.
#
#   Noul   - a single probability (0 to 1) that a statement is true.
#   Choice - pick one of N labelled options, with per-option probabilities.
#   Score  - a rubric rating, returned as a probability-weighted average.
# --------------------------------------------------------------------------
QUESTIONS = {
    "route": Choice(
        instructions="Which team should own this ticket?",
        criteria={
            "billing": "Charges, refunds, invoices, payment methods.",
            "technical": "Bugs, outages, API or integration problems.",
            "account": "Access, permissions, plan changes, cancellations.",
            "other": "None of the above fits clearly.",
        },
    ),
    "is_time_sensitive": Noul(
        instructions="Does the ticket state an explicit deadline or time pressure?",
    ),
    "frustration": Score(
        instructions="How frustrated does the customer sound?",
        criteria=[
            "Calm and neutral",
            "Mildly annoyed but polite",
            "Clearly frustrated, patience running out",
            "Angry, threatening to escalate or churn",
        ],
    ),
}


def load_api_key():
    """Read TYPESAFE_API_KEY from the environment, prompting if it is missing."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        if not sys.stdin.isatty():
            sys.exit("TYPESAFE_API_KEY is not set and stdin is not a terminal.")
        key = getpass.getpass("TYPESAFE_API_KEY: ")
        os.environ["TYPESAFE_API_KEY"] = key
    return key


def answer(response, key):
    """Fetch one answer, tolerating either SDK accessor style.

    The SDK exposes a unified `response.answers` dict, and some versions also
    expose type-narrowed `response.choices` / `.nouls` / `.scores`.
    """
    answers = getattr(response, "answers", None)
    if answers and key in answers:
        return answers[key]
    for bucket in ("choices", "nouls", "scores"):
        d = getattr(response, bucket, None)
        if d and key in d:
            return d[key]
    raise KeyError(key)


def main():
    load_api_key()

    with TypeSafeClient() as client:
        t0 = time.perf_counter()
        response = client.system_one(state=STATE, questions=QUESTIONS)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"model: {response.model}")
    print(f"round trip: {elapsed_ms:.0f} ms")

    route = answer(response, "route")
    urgent = answer(response, "is_time_sensitive")
    frustration = answer(response, "frustration")

    # No JSON parsing and no validation step - the fields are guaranteed to
    # exist with the types we asked for.
    print("\nROUTE")
    print(f"  -> {route.choice}  (confidence {route.confidence:.2f})")
    for option, p in sorted(route.probabilities.items(), key=lambda kv: -kv[1]):
        print(f"     {option:<12} {p:.3f}")

    print("\nTIME SENSITIVE")
    print(f"  -> p(yes) = {urgent.noul:.3f}")

    levels = len(QUESTIONS["frustration"].criteria) - 1
    print("\nFRUSTRATION")
    print(f"  -> {frustration.score:.2f} / {levels}"
          f"  (confidence {frustration.confidence:.2f})")

    # ----------------------------------------------------------------------
    # The point: a smart if-statement. The probabilities are calibrated, so
    # thresholds mean something - and the low-confidence branch is where a
    # human gets pulled in.
    # ----------------------------------------------------------------------
    if route.confidence < 0.7:
        queue = "manual_triage"
    elif urgent.noul > 0.6 and frustration.score >= 2.0:
        queue = f"{route.choice}_priority"
    else:
        queue = route.choice

    print("\nqueue:", queue)


if __name__ == "__main__":
    main()
