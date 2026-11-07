"""Acceptance checks for Part D outputs and guardrails."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
report = json.loads((ROOT / "outputs" / "part_d_report.json").read_text(encoding="utf-8"))
assert report["chunk_count"] == 15
assert report["chunk_counts_by_source"] == {
    "refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
    "payment_policy.txt": 3, "privacy_policy.txt": 2,
}
assert report["retrieval_demo_sources"] == ["refund_policy.txt"] * 3
r10 = report["retrieval_evaluation_overlap10"]
assert (r10["answerable_retrieval_hits"], r10["answerable_expected_facts"], r10["unanswerable_fallbacks"]) == (12, 12, 2)
r0 = report["retrieval_evaluation_overlap0"]
assert r0["answerable_expected_facts"] < 12
assert report["token_demo_status"] == "FALLBACK_TOKEN_LIMIT"
assert report["timeout_demo_status"] == "FALLBACK_TIMEOUT"
assert report["crash_resume"]["status"] == "REFUND"
assert report["crash_resume"]["tools_executed"] == [
    "get_order", "search_policy", "calculate_refund", "issue_refund"
]
assert report["crash_resume"]["llm_calls"] == 5
expected = {
    "S1": ("REFUND", 128.25, 5), "S2": ("REFUND", 135.00, 5),
    "S3": ("ESCALATE", 1999.00, 5), "S4": ("DENIED", 0.0, 4),
    "S5": ("DENIED", 0.0, 3), "S6": ("NOT_FOUND", 0.0, 2),
    "S7": ("FALLBACK_MAX_ITERATIONS", 0.0, 6),
}
for row in report["scenario_results"]:
    status, amount, calls = expected[row["scenario_id"]]
    assert row["status"] == status
    assert abs(row["refund_amount"] - amount) <= 0.01
    assert row["llm_calls"] == calls
runs = [json.loads(x) for x in (ROOT / "outputs" / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(runs) == 7
assert all(set(("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")).issubset(x) for x in runs)
ledger = [json.loads(x) for x in (ROOT / "outputs" / "refund_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
assert any(x["tool_name"] == "issue_refund" for x in ledger)
assert any(x["tool_name"] == "escalate_to_human" for x in ledger)
assert len(list((ROOT / "state").glob("*.json"))) >= 7
print("ALL PART D CHECKPOINTS PASSED")
