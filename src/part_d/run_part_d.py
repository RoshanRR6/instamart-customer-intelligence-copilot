"""Part D: offline policy RAG assistant and resumable refund ReAct agent."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402

POLICY_DIR = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "instamart.db"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

POLICY_FILES = [
    "refund_policy.txt", "delivery_policy.txt", "escalation_policy.txt",
    "payment_policy.txt", "privacy_policy.txt",
]


def chunk_documents(overlap: int = 10) -> list[dict[str, str]]:
    """Split each policy into 40-word chunks with the requested overlap."""
    if not 0 <= overlap < 40:
        raise ValueError("overlap must be between 0 and 39")
    chunks: list[dict[str, str]] = []
    step = 40 - overlap
    for filename in POLICY_FILES:
        words = (POLICY_DIR / filename).read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({"source": filename, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


class PolicyRetriever:
    def __init__(self, overlap: int = 10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            {"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]}
            for i in ranked
        ]


RETRIEVER = PolicyRetriever(overlap=10)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    return RETRIEVER.retrieve(query, k)


POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy context. "
    "If the context does not contain the answer, say: I could not find that information in the policy documents. "
    "Do not use outside knowledge or infer missing policy rules."
)


def answer(question: str) -> dict[str, Any]:
    results = retrieve(question, k=3)
    context_lines: list[str] = []
    for item in results:
        context_lines.extend([f"[{item['source']}]"]) 
        context_lines.append(item["text"])
    user_content = "CONTEXT:\n" + "\n".join(context_lines) + "\nQUESTION:\n" + question
    response = mock_llm.chat([
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return {"answer": response.choices[0].message.content, "sources": list(dict.fromkeys(x["source"] for x in results))}


def evaluate_retrieval(overlap: int) -> dict[str, Any]:
    retriever = PolicyRetriever(overlap=overlap)
    eval_df = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
    answerable_hit = answerable_fact = fallback = 0
    failures: list[dict[str, str]] = []
    for row in eval_df.to_dict("records"):
        results = retriever.retrieve(row["question"], 3)
        if row["expected_doc"] == "NONE":
            # Reuse the same answer prompt/format, but with this retriever instance.
            context = "CONTEXT:\n" + "\n".join(
                line for x in results for line in (f"[{x['source']}]", x["text"])
            ) + "\nQUESTION:\n" + row["question"]
            resp = mock_llm.chat([
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ]).choices[0].message.content
            if resp.startswith("I could not find"):
                fallback += 1
            else:
                failures.append({"question": row["question"], "reason": "unanswerable question did not use fallback"})
        else:
            hit = row["expected_doc"] in [x["source"] for x in results]
            answerable_hit += int(hit)
            context = "CONTEXT:\n" + "\n".join(
                line for x in results for line in (f"[{x['source']}]", x["text"])
            ) + "\nQUESTION:\n" + row["question"]
            resp = mock_llm.chat([
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ]).choices[0].message.content
            fact_hit = row["expected_fact"] in resp
            answerable_fact += int(fact_hit)
            if not hit or not fact_hit:
                failures.append({"question": row["question"], "reason": f"retrieval_hit={hit}; expected_fact_hit={fact_hit}; answer={resp}"})
    return {
        "overlap": overlap, "chunk_count": len(retriever.chunks),
        "answerable_retrieval_hits": answerable_hit,
        "answerable_expected_facts": answerable_fact,
        "unanswerable_fallbacks": fallback,
        "failures": failures,
    }


# Tool contracts are deliberately explicit and kept separate from Python dispatch.
TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order and return its status, amount and delivery duration.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the most relevant policy chunks for a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate a late-delivery refund for a completed order amount and delivery duration.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an automatically approved refund of INR 1000 or less.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a refund above INR 1000 for human approval without issuing it.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ? LIMIT 1",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if row[3] is None else float(row[3])}


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount: float, delivery_minutes: float | None) -> dict[str, float]:
    refund = float(order_amount) if delivery_minutes is not None and float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def _record_refund(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool_name": tool_name,
           "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with (OUTPUT_DIR / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"refund_inr": row["refund_inr"]}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    if float(refund_inr) > 1000:
        raise ValueError("refund above INR 1000 must be escalated")
    return _record_refund("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    if float(refund_inr) <= 1000:
        raise ValueError("only refunds above INR 1000 require human escalation")
    return _record_refund("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

AGENT_SYSTEM_PROMPT = (
    "You are an AGENT for refund requests. Use the available tools in sequence. "
    "Only completed orders are refundable. A completed order delivered in more than 45 minutes gets a full refund; "
    "45 minutes or less gets zero. Refunds above INR 1000 must be escalated to a human and never issued automatically. "
    "Use search_policy to ground the decision. Do not claim an action occurred unless its tool observation confirms it."
)


class SimulatedCrash(RuntimeError):
    pass


def _status_from_answer(final_answer: str) -> str:
    return final_answer.split(":", 1)[0].strip() if ":" in final_answer else "COMPLETED"


def _save_checkpoint(run_id: str, scenario_id: str, messages: list[dict[str, Any]],
                     tokens_used: int, step_number: int, tools_executed: list[dict[str, Any]], llm_calls: int = 0) -> None:
    checkpoint = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages,
                  "tokens_used": tokens_used, "step_number": step_number,
                  "tools_executed": tools_executed, "llm_calls": llm_calls}
    path = STATE_DIR / f"{run_id}.json"
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _continue_agent(run_id: str, checkpoint: dict[str, Any], max_iterations: int,
                    token_limit: int, time_limit_seconds: float,
                    crash_after_tools: int | None = None) -> dict[str, Any]:
    messages = checkpoint["messages"]
    scenario_id = checkpoint["scenario_id"]
    tokens_used = int(checkpoint["tokens_used"])
    steps = checkpoint.get("tools_executed", [])
    llm_calls = int(checkpoint.get("llm_calls", 0))
    started = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "FALLBACK_TIMEOUT: Time limit reached; hand this request to a human agent with the completed steps.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        used = int(response.usage.total_tokens)
        tokens_used += used
        if tokens_used > token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "FALLBACK_TOKEN_LIMIT: Token limit reached; hand this request to a human agent with the completed steps.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        msg = response.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            final = msg.content or "FALLBACK: The model returned no final answer; human review required."
            return {"status": _status_from_answer(final), "final_answer": final,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        # The supplied mock issues one call per turn; process every returned call in order.
        assistant_calls = []
        for call in calls:
            assistant_calls.append({"id": call.id, "name": call.name, "arguments": call.arguments})
        messages.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
        for call in calls:
            name = call.name
            args = call.arguments if isinstance(call.arguments, dict) else json.loads(call.arguments)
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                observation = TOOL_MAP[name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name,
                             "content": json.dumps(observation, sort_keys=True)})
            step = {"tool_name": name, "arguments": args, "observation": observation}
            steps.append(step)
            checkpoint.update({"messages": messages, "tokens_used": tokens_used,
                               "step_number": len(steps), "tools_executed": steps,
                               "llm_calls": llm_calls})
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), steps, llm_calls)
            if crash_after_tools is not None and len(steps) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool execution {crash_after_tools}")
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "FALLBACK_MAX_ITERATIONS: Maximum model-call iterations reached; hand this request to a human agent with the completed steps.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6,
              token_limit: int = 6000, time_limit_seconds: float = 10) -> dict[str, Any]:
    run_id = f"{scenario_id}_{uuid.uuid4().hex[:10]}"
    checkpoint = {"run_id": run_id, "scenario_id": scenario_id,
                  "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                               {"role": "user", "content": user_message}],
                  "tokens_used": 0, "step_number": 0, "tools_executed": [], "llm_calls": 0}
    _save_checkpoint(run_id, scenario_id, checkpoint["messages"], 0, 0, [])
    return _continue_agent(run_id, checkpoint, max_iterations, token_limit, time_limit_seconds)


def resume_agent(run_id: str, max_iterations: int = 6, token_limit: int = 6000,
                 time_limit_seconds: float = 10) -> dict[str, Any]:
    path = STATE_DIR / f"{run_id}.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    return _continue_agent(run_id, checkpoint, max_iterations, token_limit, time_limit_seconds)


def _run_with_crash(user_message: str, scenario_id: str, crash_after_tools: int) -> tuple[str, Exception]:
    run_id = f"{scenario_id}_resume_demo"
    checkpoint = {"run_id": run_id, "scenario_id": scenario_id,
                  "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                               {"role": "user", "content": user_message}],
                  "tokens_used": 0, "step_number": 0, "tools_executed": [], "llm_calls": 0}
    _save_checkpoint(run_id, scenario_id, checkpoint["messages"], 0, 0, [])
    try:
        _continue_agent(run_id, checkpoint, 6, 6000, 10, crash_after_tools=crash_after_tools)
    except SimulatedCrash as exc:
        return run_id, exc
    raise AssertionError("crash injection did not occur")


def run_all() -> dict[str, Any]:
    # Clear prior simulation artifacts so the run is reproducible and ledger counts are meaningful.
    for name in ("agent_runs.jsonl", "refund_ledger.jsonl"):
        p = OUTPUT_DIR / name
        if p.exists():
            p.unlink()
    # A fresh full run starts with a clean checkpoint directory.
    for checkpoint_file in STATE_DIR.glob("*.json"):
        checkpoint_file.unlink()
    mock_llm.reset()
    # RAG chunking, retrieval and evaluation at 10-word and zero overlap.
    chunks = RETRIEVER.chunks
    retrieval_demo = retrieve("refund policy late delivery", 3)
    eval_overlap10 = evaluate_retrieval(10)
    eval_overlap0 = evaluate_retrieval(0)
    # Tool policy tests.
    assert len(chunks) == 15, f"expected 15 chunks, got {len(chunks)}"
    assert all(x["source"] == "refund_policy.txt" for x in retrieval_demo)
    assert eval_overlap10["answerable_retrieval_hits"] == 12
    assert eval_overlap10["answerable_expected_facts"] == 12
    assert eval_overlap10["unanswerable_fallbacks"] == 2

    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    scenario_results = []
    for row in scenarios.to_dict("records"):
        result = run_agent(row["message"], row["scenario_id"], max_iterations=6,
                           token_limit=6000, time_limit_seconds=10)
        expected = row["expected_outcome"]
        if expected == "FALLBACK":
            assert result["status"] == "FALLBACK_MAX_ITERATIONS", (row["scenario_id"], result)
        else:
            assert result["status"] == expected, (row["scenario_id"], result)
        refund_amount = next((s["observation"].get("refund_inr") for s in result["steps"]
                              if s["tool_name"] == "calculate_refund"), 0.0)
        assert abs(float(refund_amount) - float(row["expected_refund_inr"])) <= 0.01
        scenario_results.append({"scenario_id": row["scenario_id"], "status": result["status"],
                                 "final_answer": result["final_answer"], "refund_amount": float(refund_amount),
                                 "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                                 "steps": result["steps"]})
        with (OUTPUT_DIR / "agent_runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"scenario_id": row["scenario_id"], "status": result["status"],
                                "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
                                "tokens_used": result["tokens_used"], "steps": result["steps"]}) + "\n")

    # Guardrail demonstrations.
    s1 = scenarios.loc[scenarios.scenario_id == "S1"].iloc[0]
    token_demo = run_agent(s1.message, "S1_TOKEN_DEMO", token_limit=200)
    timeout_demo = run_agent(s1.message, "S1_TIMEOUT_DEMO", time_limit_seconds=0)
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"

    # Crash after the second tool observation is checkpointed, then resume.
    s1_run_id, _ = _run_with_crash(s1.message, "S1_CRASH_RESUME", crash_after_tools=2)
    resumed = resume_agent(s1_run_id)
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    actual_tools = [x["tool_name"] for x in resumed["steps"]]
    assert actual_tools == expected_tools, actual_tools
    assert resumed["status"] == "REFUND" and "128.25" in resumed["final_answer"]

    # Compact report for README and validator.
    report = {
        "chunk_count": len(chunks),
        "chunk_counts_by_source": {name: sum(c["source"] == name for c in chunks) for name in POLICY_FILES},
        "retrieval_demo_sources": [x["source"] for x in retrieval_demo],
        "retrieval_evaluation_overlap10": eval_overlap10,
        "retrieval_evaluation_overlap0": eval_overlap0,
        "scenario_results": [{k: x[k] for k in ("scenario_id", "status", "refund_amount", "llm_calls", "tokens_used")} for x in scenario_results],
        "token_demo_status": token_demo["status"], "token_demo_tokens": token_demo["tokens_used"],
        "timeout_demo_status": timeout_demo["status"],
        "crash_resume": {"run_id": s1_run_id, "status": resumed["status"], "final_answer": resumed["final_answer"],
                         "tools_executed": actual_tools, "llm_calls": resumed["llm_calls"], "tokens_used": resumed["tokens_used"]},
        "react_vs_plan_execute": (
            "In this mock, ReAct takes 5 model calls for S1, 3 for S5 and 2 for S6 because each observation is returned to the model before the next action. "
            "That adds prompt/observation tokens compared with a plan-and-execute design that asks for a complete tool sequence once and replans only after an error. "
            "However, ReAct can inspect NOT_FOUND, cancelled status, delivery time and calculated refund before choosing the next tool, whereas a fixed plan risks attempting an invalid action or needs a replan. "
            "For refunds I would choose ReAct: the extra calls are justified by conditional decisions and the ability to stop safely before an unauthorized or incorrect payment."
        ),
    }
    (OUTPUT_DIR / "part_d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_all()
    print("PART D SUMMARY")
    print(json.dumps(result, indent=2))
    print("ALL PART D CHECKPOINTS PASSED")
