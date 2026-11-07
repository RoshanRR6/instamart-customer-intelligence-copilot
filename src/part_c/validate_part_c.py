"""Acceptance checks for Part C outputs and privacy/routing invariants."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
report = json.loads((OUT / "part_c_report.json").read_text(encoding="utf-8"))
final = pd.read_csv(OUT / "triage_results.csv", keep_default_na=False)
demo = pd.read_csv(OUT / "triage_budget_demo.csv", keep_default_na=False)

assert report["input_tickets"] == 164
assert report["empty_tickets"] == 4 and report["long_tickets"] == 4
assert report["eligible_tickets"] == 156
assert report["redaction_counts"] == {"email": 55, "phone": 91, "order_id": 132, "total": 278}
assert report["residual_pii_matches_after_redaction"] == 0
for name in ("zero_shot", "few_shot"):
    r = report[name]
    assert r["retry_count"] == 19
    assert r["mock_call_attempts"] == 175
    assert r["successful_mock_responses"] == 156
    assert r["classified_tickets"] == 147
    assert r["status_counts"].get("classified_json") == 135
    assert r["status_counts"].get("classified_keyword") == 12
    assert r["status_counts"].get("flagged_invalid_output") == 9
assert abs(report["zero_shot"]["accuracy_classified_only"] - 126 / 147) < 1e-9
assert abs(report["few_shot"]["accuracy_classified_only"] - 1.0) < 1e-9
assert report["budget_demo"]["budget_exhausted"] > 0
assert len(final) == 164 and len(demo) == 164
assert list(final.columns) == ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]

raw = pd.read_csv(ROOT / "data/raw/tickets.csv", keep_default_na=False)
segments = pd.read_csv(OUT / "customer_segments.csv", keep_default_na=False)
segment_lookup = segments.set_index("customer_id").to_dict("index")
for row in final.itertuples(index=False):
    segment = segment_lookup.get(row.customer_id, {})
    premium = str(segment.get("is_premium", "False")).lower() == "true"
    if row.status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "error"}:
        assert (row.route, row.priority) == ("human_review", "NORMAL")
        assert row.category == ""
    elif row.category == "fraud":
        assert (row.route, row.priority) == ("security_team", "HIGH")
    elif premium:
        assert (row.route, row.priority) == ("premium_desk", "HIGH")
    else:
        assert (row.route, row.priority) == ("support_queue", "NORMAL")

# The run script asserts that every user message is exactly the redacted ticket text,
# and checks all three PII expressions both before every model call and after redaction.
assert "error" not in report["zero_shot"]["status_counts"]
assert "error" not in report["few_shot"]["status_counts"]


for log_name in ["part_c_zero_shot_retry.log", "part_c_few_shot_retry.log", "part_c_final_few_shot_retry.log"]:
    lines = (OUT / log_name).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 19
    for line in lines:
        match = re.fullmatch(r"ticket_id=T\d{4} retry=(\d) wait_seconds=([0-9.]+)", line)
        assert match is not None and match.group(1) == "0"
        wait = float(match.group(2))
        assert 0.045 <= wait <= 0.0575

assert final.status.value_counts().to_dict() == {
    "classified_json": 135,
    "classified_keyword": 12,
    "flagged_invalid_output": 9,
    "flagged_long": 4,
    "skipped_empty": 4,
}
assert report["final_few_shot"]["routing_counts"] == {
    "support_queue": 102, "security_team": 34, "human_review": 17, "premium_desk": 11
}

print("ALL PART C CHECKPOINTS PASSED")
