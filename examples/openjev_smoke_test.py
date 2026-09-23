#!/usr/bin/env python3
"""
Open-Jev smoke test - support ticket triage (local, Open-Jev-2B).

Counterpart to jev_smoke_test.py and needle_smoke_test.py: same ticket, same
three typed decisions, same routing logic. Open-Jev is an open reproduction of
the Jev idea (Qwen3.5-2B + LoRA + a calibrated decision head) that runs on
this machine behind a small local HTTP server.

One-time setup (already done in ./Open-Jev):

    cd Open-Jev
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e '.[train]'
    hf download ZefanCai/Open-Jev-2B \
      --revision 0c7aa498b1627be8da4acf34c863ff0ee0a92785 --local-dir models/Open-Jev-2B

Start the server (terminal 1, leave it running; first load ~30 s):

    cd Open-Jev
    HF_HUB_OFFLINE=1 ./.venv/bin/python -m jev.server \
      --checkpoint models/Open-Jev-2B/package/checkpoint \
      --device mps --max-length 4096 --batch-size 16 --no-prefix-cache

Run (terminal 2, any Python 3 - standard library only):

    python openjev_smoke_test.py            # 1 warmup + 5 timed requests
    python openjev_smoke_test.py --runs 20
"""

import argparse
import json
import statistics
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ENDPOINT = "http://127.0.0.1:8791/v1/systemone"

# --------------------------------------------------------------------------
# The state - identical to jev_smoke_test.py.
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
# The same three questions as the Jev test, in Open-Jev's JSON form.
# --------------------------------------------------------------------------
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
            "billing": "Charges, refunds, invoices, payment methods.",
            "technical": "Bugs, outages, API or integration problems.",
            "account": "Access, permissions, plan changes, cancellations.",
            "other": "None of the above fits clearly.",
        },
    },
    "is_time_sensitive": {
        "type": "noul",
        "instructions": "Does the ticket state an explicit deadline or time pressure?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": [
            "Calm and neutral",
            "Mildly annoyed but polite",
            "Clearly frustrated, patience running out",
            "Angry, threatening to escalate or churn",
        ],
    },
}


def ask(state, questions):
    body = json.dumps({"model": "open-jev", "state": state, "questions": questions}).encode()
    request = Request(ENDPOINT, body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    return result, (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="timed requests after one warmup")
    runs = parser.parse_args().runs

    try:
        ask(STATE, QUESTIONS)  # warmup: first request pays one-off MPS kernel setup
    except URLError:
        sys.exit(f"Open-Jev server not reachable at {ENDPOINT} - start it first (see docstring).")

    timings = []
    for _ in range(runs):
        response, ms = ask(STATE, QUESTIONS)
        timings.append(ms)

    print(f"model: {response.get('model', 'open-jev')}")
    print(f"round trip over {runs} runs: median {statistics.median(timings):.0f} ms"
          f"  (min {min(timings):.0f}, max {max(timings):.0f})")

    answers = response["answers"]
    route, urgent, frustration = (answers["route"], answers["is_time_sensitive"],
                                  answers["frustration"])

    print("\nROUTE")
    print(f"  -> {route['choice']}  (confidence {route['confidence']:.2f})")
    for option, p in sorted(route["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"     {option:<12} {p:.3f}")

    print("\nTIME SENSITIVE")
    print(f"  -> p(yes) = {urgent['noul']:.3f}")

    levels = len(QUESTIONS["frustration"]["criteria"]) - 1
    print("\nFRUSTRATION")
    print(f"  -> {frustration['score']:.2f} / {levels}"
          f"  (confidence {frustration['confidence']:.2f})")

    # Same routing logic as the Jev test.
    if route["confidence"] < 0.7:
        queue = "manual_triage"
    elif urgent["noul"] > 0.6 and frustration["score"] >= 2.0:
        queue = f"{route['choice']}_priority"
    else:
        queue = route["choice"]

    print("\nqueue:", queue)


if __name__ == "__main__":
    main()
