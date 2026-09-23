#!/usr/bin/env python3
"""
Needle 3 smoke test - support ticket triage (local, on-device).

Counterpart to jev_smoke_test.py: same ticket, same three decisions, so the
two outputs can be compared side by side. Needle runs entirely on this
machine - no API key, no network after the first download.

How the Jev questions map onto Needle:

    Jev Choice  (route)             -> tool with an enum argument
    Jev Noul    (is_time_sensitive) -> tool with a boolean argument
    Jev Score   (frustration)       -> tool with an enum over the rubric levels

Needle is a tool-calling / extraction model, so each question is declared as
a single "record" tool and the model fills its argument. One agent per
question gives a calibrated confidence per decision (Needle reports one
confidence per call, not per-option probabilities like Jev).

Setup (conda):

    conda activate <your-env>
    pip install cactus-needle
    # optional, turns off anonymous usage counts:
    export NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1

Run:

    python needle_smoke_test.py

First run downloads the engine (<1 MB) and needle3.cact weights (~35 MB) from
Hugging Face into ~/.cache/cactus-needle/v3/. Later runs are offline.
"""

import time

import needle


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

# Needle takes text, not a dict, so the state is rendered into the request.
REQUEST = (
    f"Support ticket from a {STATE['account_tier']} account "
    f"({STATE['prior_tickets_this_month']} prior tickets this month): "
    f"\"{STATE['ticket']}\""
)

FRUSTRATION_LEVELS = [
    "calm_neutral",          # 0  Calm and neutral
    "mildly_annoyed",        # 1  Mildly annoyed but polite
    "clearly_frustrated",    # 2  Clearly frustrated, patience running out
    "angry_escalating",      # 3  Angry, threatening to escalate or churn
]

# --------------------------------------------------------------------------
# One raw JSON-schema tool per decision. The decode grammar guarantees the
# argument is one of the allowed values - no parsing, no validation step.
# --------------------------------------------------------------------------
QUESTIONS = {
    "route": {
        "name": "route_ticket",
        "description": (
            "Record which team should own this support ticket. "
            "billing: charges, refunds, invoices, payment methods. "
            "technical: bugs, outages, API or integration problems. "
            "account: access, permissions, plan changes, cancellations. "
            "other: none of the above fits clearly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "enum": ["billing", "technical", "account", "other"],
                },
            },
            "required": ["team"],
        },
    },
    "is_time_sensitive": {
        "name": "record_time_sensitivity",
        "description": (
            "Record whether the ticket states an explicit deadline or time pressure."
        ),
        "parameters": {
            "type": "object",
            "properties": {"time_sensitive": {"type": "boolean"}},
            "required": ["time_sensitive"],
        },
    },
    "frustration": {
        "name": "record_frustration",
        "description": (
            "Record how frustrated the customer sounds. "
            "calm_neutral: calm and neutral. "
            "mildly_annoyed: mildly annoyed but polite. "
            "clearly_frustrated: clearly frustrated, patience running out. "
            "angry_escalating: angry, threatening to escalate or churn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": FRUSTRATION_LEVELS},
            },
            "required": ["level"],
        },
    },
}


def ask(agent, text):
    """One turn. Returns (arguments or None, confidence, response, ms).

    Needle may refuse (empty function_calls) or withhold a low-confidence call
    into suppressed_calls - both are surfaced rather than hidden.
    """
    t0 = time.perf_counter()
    r = agent.complete(text)
    ms = (time.perf_counter() - t0) * 1000
    calls = r.get("function_calls") or r.get("suppressed_calls") or []
    args = calls[0]["arguments"] if calls else None
    return args, r.get("confidence"), r, ms


def fmt_conf(c):
    return "n/a" if c is None else f"{c:.2f}"


def main():
    # Construction loads the engine + weights (downloads on first run), so it
    # is timed separately from inference.
    t0 = time.perf_counter()
    agents = {k: needle.Needle(tools=[schema]) for k, schema in QUESTIONS.items()}
    load_ms = (time.perf_counter() - t0) * 1000

    results = {}
    total_ms = 0.0
    for key, agent in agents.items():
        results[key] = ask(agent, REQUEST)
        total_ms += results[key][3]

    print("model: needle3 (local)")
    print(f"load: {load_ms:.0f} ms   inference (3 calls, sequential): {total_ms:.0f} ms")

    route_args, route_conf, route_r, route_ms = results["route"]
    urgent_args, urgent_conf, urgent_r, urgent_ms = results["is_time_sensitive"]
    frus_args, frus_conf, frus_r, frus_ms = results["frustration"]

    route = route_args["team"] if route_args else None
    urgent = urgent_args["time_sensitive"] if urgent_args else None
    level = frus_args["level"] if frus_args else None
    frustration = FRUSTRATION_LEVELS.index(level) if level in FRUSTRATION_LEVELS else None

    print("\nROUTE")
    print(f"  -> {route}  (confidence {fmt_conf(route_conf)}, {route_ms:.0f} ms)")
    print(f"     reasoning: {route_r.get('reasoning')}")

    print("\nTIME SENSITIVE")
    print(f"  -> {urgent}  (confidence {fmt_conf(urgent_conf)}, {urgent_ms:.0f} ms)")
    print(f"     reasoning: {urgent_r.get('reasoning')}")

    print("\nFRUSTRATION")
    print(f"  -> {level} = {frustration} / {len(FRUSTRATION_LEVELS) - 1}"
          f"  (confidence {fmt_conf(frus_conf)}, {frus_ms:.0f} ms)")
    print(f"     reasoning: {frus_r.get('reasoning')}")

    for key, (_, _, r, _) in results.items():
        if r.get("suppressed_calls"):
            print(f"\nnote: {key} call was withheld by the engine (low confidence)")
        if (r.get("validation") or {}).get("ungrounded"):
            print(f"\nnote: {key} ungrounded fields: {r['validation']['ungrounded']}")

    # ----------------------------------------------------------------------
    # Same routing logic as the Jev test. Needle gives a boolean rather than
    # p(yes), so "urgent" gates on the decision plus its confidence.
    # ----------------------------------------------------------------------
    if route is None or route_conf is None or route_conf < 0.7:
        queue = "manual_triage"
    elif urgent and (urgent_conf or 0) > 0.6 and frustration is not None and frustration >= 2:
        queue = f"{route}_priority"
    else:
        queue = route

    print("\nqueue:", queue)


if __name__ == "__main__":
    main()
