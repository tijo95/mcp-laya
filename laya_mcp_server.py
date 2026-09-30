"""Laya MCP — local System 1 decision model (open Jev reproduction).

Mirrors the 6 tools of the JEV MCP, but runs entirely locally on
convaiinnovations/laya-multilingual (mmBERT-base, 322M, 100+ languages).

State + typed questions (choice / noul / score) -> calibrated probabilities
in a single encoder pass. No token generation, no API key.

Run:
    python laya_mcp_server.py            # stdio (for MCP clients)

Env:
    LAYA_CHECKPOINT  default: convaiinnovations/laya-multilingual
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")  # avoid TF/abseil import deadlock

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "laya",
    instructions=(
        "Local Laya decision model (open Jev reproduction). Give it a state and "
        "typed questions; it returns calibrated probabilities in one encoder pass. "
        "Use laya_decide for raw typed questions; the other tools are opinionated "
        "presets (guard, routing, research, completion)."
    ),
)

CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya-multilingual")
_agent = None


def get_agent():
    """Lazy singleton — first tool call pays the ~30 s model download/load."""
    global _agent
    if _agent is None:
        import laya
        _agent = laya.load(CHECKPOINT)
    return _agent


def predict(state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
    res = get_agent().predict(state, questions)
    answers = {}
    for name, a in res["answers"].items():
        t = a["type"]
        if t == "choice":
            answers[name] = {
                "choice": a["choice"],
                "probabilities": a["probabilities"],
                "confidence": round(a["confidence"], 4),
            }
        elif t == "noul":
            answers[name] = {"noul": round(a["noul"], 4), "confidence": round(a["confidence"], 4)}
        elif t == "score":
            answers[name] = {
                "score": round(a["score"], 4),
                "legend": a.get("legend"),
                "probabilities": a["probabilities"],
                "confidence": round(a["confidence"], 4),
            }
        else:
            answers[name] = a
    return {
        "model": CHECKPOINT,
        "answers": answers,
        "usage": {"input_tokens": res["usage"]["input_tokens"], "truncated": res["usage"]["truncated"]},
    }


# ---------------------------------------------------------------------------
# 1. Guard
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_guard_tool_call(
    tool: str,
    action: str,
    arguments_summary: Optional[List[str]] = None,
    side_effects: Optional[List[str]] = None,
    safeguards: Optional[List[str]] = None,
    policy: Optional[List[str]] = None,
    reversibility: str = "unknown",
) -> Dict[str, Any]:
    """Immediately before a consequential tool call. Returns allow, confirm, review, or deny. Does not execute the tool."""
    state = {
        "tool": tool,
        "action": action,
        "arguments_summary": "; ".join(arguments_summary or []),
        "side_effects": "; ".join(side_effects or []),
        "safeguards": "; ".join(safeguards or []),
        "policy": "; ".join(policy or []),
        "reversibility": reversibility,
    }
    # Calibrated preset (r4): the 322M model answers short positive keyword
    # questions reliably (destr 0.99/0.01, prod 0.91/0.30, sends 0.78/0.00)
    # but fails at negation and multi-step reasoning, so the decision rule
    # combines the signals in python instead of asking for a verdict.
    # Battery: 5/6, both errors over-confirm (safe side).
    questions = {
        "destr": {
            "type": "noul",
            "instructions": "Does this tool call delete, drop, or overwrite data?",
        },
        "prod": {
            "type": "noul",
            "instructions": "Does the state mention a production system, customer data, or a real user account?",
        },
        "sends": {
            "type": "noul",
            "instructions": "Does this tool call send an email, message, or notification to someone, or restart a service?",
        },
    }
    out = predict(state, questions)
    d = out["answers"]["destr"]["noul"]
    p = out["answers"]["prod"]["noul"]
    s = out["answers"]["sends"]["noul"]
    if d > 0.7 and p > 0.7:
        out["decision"] = "deny"
    elif d > 0.7:
        out["decision"] = "confirm"
    elif s > 0.5:
        out["decision"] = "confirm"
    elif p > 0.6:
        out["decision"] = "confirm"
    else:
        out["decision"] = "allow"
    return out


# ---------------------------------------------------------------------------
# 2. Route model
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_route_model(
    task: str,
    candidates: List[Dict[str, Any]],
    priorities: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    stakes: str = "normal",
) -> Dict[str, Any]:
    """Choose among models the current environment can actually invoke based on task stakes, quality, cost, latency, context, and tool-use needs.

    candidates: list of objects, each with an "id" (required; "name" is
    accepted as an alias) plus optional "description", "cost" (low/medium/high),
    "latency" (fast/medium/slow), "context_window" (int), "tool_use" (bool).
    Example: [{"id": "small-fast-model", "cost": "low", "latency": "fast"}]"""
    # Accept "name" as an alias for "id"; fail with a clear message otherwise.
    for c in candidates:
        if "id" not in c and "name" in c:
            c["id"] = c["name"]
    if not candidates or any("id" not in c for c in candidates):
        return {
            "ok": False,
            "error": "each candidate needs an 'id' field, e.g. {\"id\": \"gpt-4o-mini\", \"cost\": \"low\"} — 'name' is accepted as an alias",
        }
    cand_lines = []
    for c in candidates:
        parts = [f"{c['id']}: {c.get('description', '') or c['id']}"]
        for k in ("cost", "latency"):
            if c.get(k):
                parts.append(f"{k}={c[k]}")
        if c.get("context_window") is not None:
            parts.append(f"context_window={c['context_window']} tokens")
        if c.get("tool_use") is not None:
            parts.append(f"tool_use={'yes' if c['tool_use'] else 'no'}")
        cand_lines.append(" | ".join(parts))
    state = {
        "task": task,
        "candidates": "\n".join(cand_lines),
        "priorities": "; ".join(priorities or []),
        "constraints": "; ".join(constraints or []),
        "stakes": stakes,
    }
    questions = {
        "best_model": {
            "type": "choice",
            "instructions": "Which candidate model is the best fit for this task given stakes, priorities and constraints?",
            "criteria": {c["id"]: c.get("description", c["id"]) for c in candidates},
        },
        "needs_human_review": {
            "type": "noul",
            "instructions": "Is the choice between candidates close enough that a human should double-check it?",
        },
    }
    out = predict(state, questions)
    out["best_model"] = out["answers"]["best_model"]["choice"]
    return out


# ---------------------------------------------------------------------------
# 3. Route task
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_route_task(
    task: str,
    evidence: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Choose proceed_fast, deep_review, split_task, or block for an ambiguous or risky task."""
    state = {
        "task": task,
        "evidence": "; ".join(evidence or []),
        "constraints": "; ".join(constraints or []),
    }
    # Calibrated preset (r3/r4): choice collapses to argmax on this model and
    # over-fires on one option, so routing is done from three short positive
    # noul signals: dangerous 0.93 on unclear-scope tasks, highrisk 0.88 on
    # production/root tasks, trivial 0.54-0.96 on small local changes.
    # Battery: 5/6, the miss is a soft one (sudo install -> split_task).
    questions = {
        "trivial": {
            "type": "noul",
            "instructions": "Is the task a small change to one local file?",
        },
        "dangerous": {
            "type": "noul",
            "instructions": "Is the scope of what to delete, change, or do unclear, or is information missing?",
        },
        "highrisk": {
            "type": "noul",
            "instructions": "Does the task involve a production system, real customer data, money, security, or root privileges?",
        },
    }
    out = predict(state, questions)
    t = out["answers"]["trivial"]["noul"]
    dg = out["answers"]["dangerous"]["noul"]
    hr = out["answers"]["highrisk"]["noul"]
    if dg > 0.7:
        out["routing"] = "block"
    elif hr > 0.7:
        out["routing"] = "deep_review"
    elif t > 0.5:
        out["routing"] = "proceed_fast"
    else:
        out["routing"] = "split_task"
    return out


# ---------------------------------------------------------------------------
# 4. Check research
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_check_research(
    claim: str,
    evidence: Optional[List[str]] = None,
    source_quality: str = "",
    stakes: str = "normal",
) -> Dict[str, Any]:
    """Judge whether supplied evidence is enough to accept a precise claim, needs more verification, or should be rejected. Does not browse."""
    state = {
        "claim": claim,
        "evidence": "; ".join(evidence or []),
        "source_quality": source_quality,
        "stakes": stakes,
    }
    # Calibrated preset (r4): the model cannot compare numbers (322M vs 1B
    # -> contra 0.13) and a bare choice over-fires "supported" on anecdotes.
    # Accepting requires BOTH the choice argmax AND a low weak-evidence
    # signal (the "cures cancer" anecdote has weak=0.28 > 0.25 and is
    # caught). Battery: 5/6, zero unsafe accepts, one safe miss.
    questions = {
        "verdict": {
            "type": "choice",
            "instructions": "Is the claim in the state directly supported, weakly supported, or contradicted by the evidence?",
            "criteria": {
                "supported": "the evidence directly supports the claim with the same facts",
                "weak": "the evidence is thin, anecdotal, missing, or from an unverified source",
                "contradicts": "the evidence says the opposite of the claim",
            },
        },
        "_mapping_note": None,  # verdict mapping below (calibrated 2026-09-30)
        "weak": {
            "type": "noul",
            "instructions": "Is the evidence thin, anecdotal, missing, or from a single unverified source?",
        },
        "contra": {
            "type": "noul",
            "instructions": "Does the evidence state something different from the claim, such as a different number, an opposite fact, or an opposite conclusion?",
        },
    }
    out = predict(state, {k: v for k, v in questions.items() if v is not None})
    w = out["answers"]["weak"]["noul"]
    c = out["answers"]["contra"]["noul"]
    v = out["answers"]["verdict"]
    choice = v["choice"]
    probs = v.get("probabilities", {})
    # Calibrated (2026-09-30 live battery): the contra noul alone misses
    # contradictions the choice sees (contra 0.23 / choice contradicts 0.65),
    # and a bare choice argmax over-accepts anecdotes. So: reject on the
    # contra signal OR a confident contradicts choice; accept only on a
    # confident supported choice with a low weak-evidence signal.
    if c > 0.6 or (choice == "contradicts" and probs.get("contradicts", 0) > 0.45):
        out["verdict"] = "reject"
    elif choice == "supported" and probs.get("supported", 0) > 0.5 and w < 0.25:
        out["verdict"] = "accept"
    else:
        out["verdict"] = "needs_more_verification"
    return out


# ---------------------------------------------------------------------------
# 5. Review completion
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_review_completion(
    objective: str,
    completed_work: Optional[List[str]] = None,
    verification: Optional[List[str]] = None,
    known_gaps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare the objective, completed work, verification, and known gaps before reporting a task complete."""
    state = {
        "objective": objective,
        "completed_work": "; ".join(completed_work or []),
        "verification": "; ".join(verification or []),
        "known_gaps": "; ".join(known_gaps or []),
    }
    questions = {
        "verdict": {
            "type": "choice",
            "instructions": "Given the objective, what was completed, how it was verified, and known gaps, can the task be reported complete?",
            "criteria": {
                "complete": "objective fully met and verified",
                "complete_with_notes": "substantially met; minor gaps that do not block delivery",
                "incomplete": "objective not met; significant gaps remain",
            },
        },
        "fully_complete": {
            "type": "noul",
            "instructions": "Is the objective fully met with no blocking gaps?",
        },
    }
    out = predict(state, questions)
    out["verdict"] = out["answers"]["verdict"]["choice"]
    return out


# ---------------------------------------------------------------------------
# 6. Raw decide
# ---------------------------------------------------------------------------
@mcp.tool()
def laya_decide(
    state: Dict[str, Any],
    questions: Dict[str, Any],
    model: str = "",
) -> Dict[str, Any]:
    """Send custom state and choice/noul/score questions. Returns calibrated probabilities. questions: {name: {type: choice|noul|score, instructions, criteria}} — choice criteria is a dict of option->description, score criteria is a list of level descriptions (index 0 first), noul takes no criteria."""
    out = predict(state, questions)
    out["model"] = model or CHECKPOINT
    return out


def preload() -> str:
    """Load the model on the calling (main) thread. Called at startup so the
    first tool call is just a forward pass, and so a bad checkpoint fails fast
    instead of hanging a worker thread mid-session."""
    get_agent()
    return CHECKPOINT


if __name__ == "__main__":
    import sys
    t = time.time()
    print(f"laya-mcp: loading {CHECKPOINT} ...", file=sys.stderr, flush=True)
    preload()
    print(f"laya-mcp: ready in {time.time()-t:.1f}s", file=sys.stderr, flush=True)
    mcp.run()
