"""
One adapter per model. Each takes a benchmark row and returns

    {qid: {"label": str | bool | None, "probs": dict | None,
           "confidence": float | None, "p_true": float | None}}

plus optional extras (Needle's tool-call arguments). label None = the model
declined to answer (Needle refusals); it counts as wrong and never automated.
"""

import getpass
import json
import os
import sys
from urllib.request import Request, urlopen


# --------------------------------------------------------------------------
# Jev (TypeSafe API)
# --------------------------------------------------------------------------
class Jev:
    name = "jev"

    def __init__(self):
        from typesafe_sdk import TypeSafeClient
        key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
        if not key:
            if not sys.stdin.isatty():
                sys.exit("TYPESAFE_API_KEY is not set and stdin is not a terminal.")
            key = getpass.getpass("TYPESAFE_API_KEY: ").strip()
        os.environ["TYPESAFE_API_KEY"] = key
        self.client = TypeSafeClient().__enter__()

    def _questions(self, questions):
        from typesafe_sdk import Choice, Noul, Score
        out = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                out[qid] = Choice(instructions=q["instructions"], criteria=q["criteria"])
            elif q["type"] == "noul":
                out[qid] = Noul(instructions=q["instructions"])
            else:
                out[qid] = Score(instructions=q["instructions"], criteria=q["criteria"])
        return out

    def predict(self, row):
        response = self.client.system_one(state=row["state"], questions=self._questions(row["questions"]))
        answers = getattr(response, "answers", None) or {}
        out = {}
        for qid, q in row["questions"].items():
            a = answers[qid] if qid in answers else _bucket(response, qid)
            if q["type"] == "noul":
                out[qid] = {"label": a.noul >= 0.5, "p_true": a.noul, "probs": None,
                            "confidence": max(a.noul, 1 - a.noul)}
            else:
                out[qid] = {"label": a.choice, "probs": dict(a.probabilities),
                            "confidence": a.confidence, "p_true": None}
        return out, {"model_version": getattr(response, "model", None)}

    def close(self):
        self.client.__exit__(None, None, None)


def _bucket(response, qid):
    for bucket in ("choices", "nouls", "scores"):
        d = getattr(response, bucket, None)
        if d and qid in d:
            return d[qid]
    raise KeyError(qid)


# --------------------------------------------------------------------------
# Open-Jev-2B (local HTTP server, see ../openjev_smoke_test.py for startup)
# --------------------------------------------------------------------------
class OpenJev:
    name = "openjev"
    endpoint = os.environ.get("OPENJEV_ENDPOINT", "http://127.0.0.1:8791/v1/systemone")

    def predict(self, row):
        body = json.dumps({"model": "open-jev", "state": row["state"], "questions": row["questions"]}).encode()
        with urlopen(Request(self.endpoint, body, headers={"Content-Type": "application/json"}), timeout=600) as r:
            answers = json.load(r)["answers"]
        out = {}
        for qid, q in row["questions"].items():
            a = answers[qid]
            if q["type"] == "noul":
                out[qid] = {"label": a["noul"] >= 0.5, "p_true": a["noul"], "probs": None,
                            "confidence": max(a["noul"], 1 - a["noul"])}
            else:
                out[qid] = {"label": a["choice"], "probs": a["probabilities"],
                            "confidence": a["confidence"], "p_true": None}
        return out, {}

    def close(self):
        pass


# --------------------------------------------------------------------------
# Needle 3 (local, in-process)
# --------------------------------------------------------------------------
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def to_json_schema(p):
    """BFCL uses Python-ish types (dict, float, tuple, any); Needle wants JSON Schema."""
    p = {k: v for k, v in p.items() if k != "optional"}
    if "type" in p:
        p["type"] = _TYPE_MAP.get(p["type"], p["type"])
    if "properties" in p:
        p["properties"] = {k: to_json_schema(v) for k, v in p["properties"].items()}
    if isinstance(p.get("items"), dict):
        p["items"] = to_json_schema(p["items"])
    return p


class Needle:
    name = "needle"

    def __init__(self):
        import needle
        self.needle = needle
        self.agents = {}

    def _classifier_tool(self, qid, q):
        if q["type"] == "noul":
            return {"name": f"record_{qid}", "description": q["instructions"],
                    "parameters": {"type": "object", "properties": {"answer": {"type": "boolean"}},
                                   "required": ["answer"]}}
        options = " ".join(f"{k}: {v}" for k, v in q["criteria"].items())
        return {"name": f"record_{qid}", "description": f"{q['instructions']} {options}",
                "parameters": {"type": "object",
                               "properties": {"answer": {"type": "string", "enum": list(q["criteria"])}},
                               "required": ["answer"]}}

    def _complete(self, cache_key, tools, text):
        agent = self.agents.get(cache_key)
        if agent is None:
            agent = self.needle.Needle(tools=tools)
            if cache_key is not None:
                self.agents[cache_key] = agent
        agent.reset()
        r = agent.complete(text)
        calls = r.get("function_calls") or []
        held = r.get("suppressed_calls") or []
        return r, (calls or held), bool(held and not calls)

    def predict(self, row):
        text = " ".join(str(v) for v in row["state"].values())
        out, extra = {}, {}
        for qid, q in row["questions"].items():
            if row["task"] == "tool_select":
                tools = [dict(f, parameters=to_json_schema(f["parameters"])) for f in row["functions"]]
                r, calls, suppressed = self._complete(None, tools, text)
                name_to_key = {v: k for k, v in row["key_to_name"].items()}
                label = name_to_key.get(calls[0]["name"]) if calls else None
                extra = {"args": calls[0]["arguments"] if calls else None}
            else:
                r, calls, suppressed = self._complete((row["task"], qid), [self._classifier_tool(qid, q)], text)
                label = calls[0]["arguments"].get("answer") if calls else None
            conf = r.get("confidence")
            if q["type"] == "noul":
                p_true = None if label is None or conf is None else (conf if label else 1 - conf)
                out[qid] = {"label": label, "p_true": p_true, "probs": None, "confidence": conf}
            else:
                out[qid] = {"label": label, "probs": None, "confidence": conf, "p_true": None}
            extra.update({"suppressed": suppressed, "reasoning": r.get("reasoning")})
        return out, extra

    def close(self):
        pass


MODELS = {"jev": Jev, "openjev": OpenJev, "needle": Needle}
