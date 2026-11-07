"""Part C: privacy-safe, retryable, validated support-ticket triage using mock_llm."""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402

TICKETS_PATH = ROOT / "data/raw/tickets.csv"
SEGMENTS_PATH = ROOT / "outputs/customer_segments.csv"
OUT = ROOT / "outputs"
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET_DEFAULT = 100_000
ALLOWED = {"billing", "delivery", "technical", "fraud"}

# Required patterns and replacement order.
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """You are a support-ticket triage classifier. CLASSIFY the user's ticket into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply with exactly this form: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown, prose, or any text outside the JSON object."""

FEW_SHOT_PROMPT = """You are a support-ticket triage classifier. CLASSIFY the user's ticket into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply with exactly this form: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown, prose, or any text outside the JSON object.
EXAMPLES:
Example 1 (billing): {\"ticket\": \"I was charged twice and need an invoice correction.\", \"category\": \"billing\", \"reason\": \"duplicate charge\"}
Example 2 (fraud): {\"ticket\": \"There is an unauthorized transaction; my account was hacked.\", \"category\": \"fraud\", \"reason\": \"unauthorized account activity\"}
Example 3 (technical): {\"ticket\": \"The app crashes when I try to log in.\", \"category\": \"technical\", \"reason\": \"application error\"}
Example 4 (delivery): {\"ticket\": \"My order has not arrived and the rider is late.\", \"category\": \"delivery\", \"reason\": \"late delivery\"}"""


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    redacted = text or ""
    counts: dict[str, int] = {}
    for name, pattern, replacement in PII_PATTERNS:
        redacted, n = pattern.subn(replacement, redacted)
        counts[name] = n
    return redacted, counts


def assert_no_pii(text: str) -> None:
    for name, pattern, _ in PII_PATTERNS:
        assert pattern.search(text) is None, f"Unredacted {name} found in model input"


def validate_answer(raw: str | None) -> tuple[str, str]:
    """Return (status, category); invalid answers never receive a guessed category."""
    if raw is None:
        return "flagged_invalid_output", ""
    try:
        parsed = json.loads(raw)
        category = parsed.get("category") if isinstance(parsed, dict) else None
        if isinstance(category, str) and category in ALLOWED:
            return "classified_json", category
        # A valid JSON response with an unsupported category is always invalid.
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError, AttributeError):
        words = re.findall(r"[a-z]+", raw.lower())
        matches = [word for word in words if word in ALLOWED]
        if len(matches) == 1:
            return "classified_keyword", matches[0]
        return "flagged_invalid_output", ""


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, int]]:
    tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_PATH, keep_default_na=False)
    premium = {
        str(row.customer_id): {
            "segment_name": str(row.segment_name),
            "is_premium": str(row.is_premium).strip().lower() == "true",
        }
        for row in segments.itertuples(index=False)
    }
    redacted_texts: list[str] = []
    totals = {name: 0 for name, _, _ in PII_PATTERNS}
    for raw in tickets["text"].astype(str):
        redacted, counts = redact_text(raw)
        redacted_texts.append(redacted)
        for name, n in counts.items():
            totals[name] += n
        assert_no_pii(redacted)
    tickets["redacted_text"] = redacted_texts
    totals["total"] = sum(totals.values())
    # Reapplying every regex to every redacted text must find zero matches.
    for text in redacted_texts:
        for name, pattern, _ in PII_PATTERNS:
            assert pattern.search(text) is None, f"Residual {name} after redaction"
    return tickets, premium, totals


def retry_chat(messages: list[dict[str, str]], ticket_id: str,
               rng: random.Random, retry_log: list[str]):
    """Retry transient errors at most three times; retry numbers are 0, 1, 2."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            # The user's message content is exactly the redacted ticket text.
            assert len(messages) == 2 and messages[1]["role"] == "user"
            assert_no_pii(messages[1]["content"])
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * rng.uniform(0.9, 1.15)
            line = f"ticket_id={ticket_id} retry={retry_number} wait_seconds={wait:.6f}"
            retry_log.append(line)
            print(line)
            time.sleep(wait)
    raise RuntimeError("Unreachable retry state")


def route_ticket(status: str, category: str, is_premium: bool) -> tuple[str, str]:
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "error"}:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(tickets: pd.DataFrame, premium: dict[str, dict[str, object]],
                 prompt_name: str, token_budget: int | None,
                 run_name: str) -> dict[str, object]:
    # Required: reset immediately before every independent full run.
    mock_llm.reset()
    prompt = ZERO_SHOT_PROMPT if prompt_name == "zero_shot" else FEW_SHOT_PROMPT
    assert "CLASSIFY" in prompt
    assert all(category in prompt for category in ALLOWED)
    assert '{"category": "...", "reason": "..."}' in prompt
    if prompt_name == "zero_shot":
        assert "EXAMPLES:" not in prompt
    else:
        assert "\nEXAMPLES:" in prompt and "fraud" in prompt
    rng = random.Random(42)
    retry_log: list[str] = []
    rows: list[dict[str, object]] = []
    total_tokens = 0

    for row in tickets.itertuples(index=False):
        ticket_id = str(row.ticket_id)
        customer_id = str(row.customer_id)
        text = str(row.redacted_text)
        seg = premium.get(customer_id, {})
        segment_name = str(seg.get("segment_name", ""))
        is_premium = bool(seg.get("is_premium", False))
        status, category, tokens_used = "", "", 0

        if not text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
            try:
                response = retry_chat(messages, ticket_id, rng, retry_log)
                tokens_used = int(response.usage.total_tokens)
                total_tokens += tokens_used
                raw = response.choices[0].message.content
                status, category = validate_answer(raw)
            except mock_llm.TransientLLMError:
                # With the supplied mock this path should never occur.
                status = "error"

        route, priority = route_ticket(status, category, is_premium)
        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "true_category": str(row.true_category),
        })

    result = pd.DataFrame(rows)
    classified = result[result.status.isin(["classified_json", "classified_keyword"])]
    correct = int((classified.category == classified.true_category).sum())
    n_classified = len(classified)
    report: dict[str, object] = {
        "run_name": run_name,
        "prompt": prompt_name,
        "token_budget": token_budget,
        "status_counts": {str(k): int(v) for k, v in result.status.value_counts().items()},
        "classified_tickets": int(n_classified),
        "correct_classifications": correct,
        "accuracy_classified_only": correct / n_classified if n_classified else 0.0,
        "total_tokens": int(total_tokens),
        "tokens_per_classified_ticket": total_tokens / n_classified if n_classified else 0.0,
        "retry_count": len(retry_log),
        "mock_call_attempts": int(mock_llm._state["calls"]),
        "successful_mock_responses": int(mock_llm._state["ok"]),
        "retry_log": retry_log,
        "redaction_counts": {},
        "budget_exhausted": int((result.status == "budget_exhausted").sum()),
        "routing_counts": {str(k): int(v) for k, v in result.route.value_counts().items()},
    }
    return {"rows": result, "report": report}


def confusion_table(run: dict[str, object]) -> dict[str, dict[str, int]]:
    df = run["rows"]
    classified = df[df.status.isin(["classified_json", "classified_keyword"])]
    table = pd.crosstab(classified.true_category, classified.category, dropna=False)
    table = table.reindex(index=["billing", "delivery", "technical", "fraud"],
                          columns=["billing", "delivery", "technical", "fraud"], fill_value=0)
    return {str(i): {str(c): int(table.loc[i, c]) for c in table.columns} for i in table.index}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tickets, premium, redaction_counts = load_inputs()
    eligible = tickets[~tickets.redacted_text.str.strip().eq("") &
                       tickets.redacted_text.map(mock_llm.est_tokens).le(MAX_INPUT_TOKENS)]
    assert len(tickets) == 164
    assert int(tickets.redacted_text.str.strip().eq("").sum()) == 4
    assert int((tickets.redacted_text.map(mock_llm.est_tokens) > MAX_INPUT_TOKENS).sum()) == 4
    assert len(eligible) == 156
    assert redaction_counts == {"email": 55, "phone": 91, "order_id": 132, "total": 278}

    zero = run_pipeline(tickets, premium, "zero_shot", None, "zero_shot_full")
    few = run_pipeline(tickets, premium, "few_shot", None, "few_shot_full")
    zero["report"]["redaction_counts"] = redaction_counts
    few["report"]["redaction_counts"] = redaction_counts
    assert zero["report"]["retry_count"] == 19
    assert few["report"]["retry_count"] == 19
    assert zero["report"]["status_counts"].get("classified_json", 0) == 135
    assert zero["report"]["status_counts"].get("classified_keyword", 0) == 12
    assert zero["report"]["status_counts"].get("flagged_invalid_output", 0) == 9
    assert few["report"]["status_counts"] == zero["report"]["status_counts"]
    assert zero["report"]["mock_call_attempts"] == 175
    assert few["report"]["mock_call_attempts"] == 175
    assert zero["report"]["successful_mock_responses"] == 156
    assert few["report"]["successful_mock_responses"] == 156
    assert "error" not in zero["report"]["status_counts"]
    assert "error" not in few["report"]["status_counts"]

    # The 6,000-token demonstration is preferred; lower to 5,000 only if needed.
    budget = 6000
    demo = run_pipeline(tickets, premium, "zero_shot", budget, "zero_shot_budget_demo")
    if demo["report"]["budget_exhausted"] == 0:
        budget = 5000
        demo = run_pipeline(tickets, premium, "zero_shot", budget, "zero_shot_budget_demo")
    assert demo["report"]["budget_exhausted"] > 0

    final = run_pipeline(tickets, premium, "few_shot", TOKEN_BUDGET_DEFAULT, "few_shot_final")
    final_df = final["rows"].drop(columns=["true_category"])
    final_df.to_csv(OUT / "triage_results.csv", index=False)
    demo["rows"].drop(columns=["true_category"]).to_csv(OUT / "triage_budget_demo.csv", index=False)
    zero["rows"].to_csv(OUT / "triage_zero_shot_full.csv", index=False)
    few["rows"].to_csv(OUT / "triage_few_shot_full.csv", index=False)

    report = {
        "input_tickets": len(tickets),
        "empty_tickets": 4,
        "long_tickets": 4,
        "eligible_tickets": len(eligible),
        "long_ticket_estimated_tokens": sorted({int(mock_llm.est_tokens(t)) for t in tickets.loc[tickets.redacted_text.map(mock_llm.est_tokens) > MAX_INPUT_TOKENS, "redacted_text"]}),
        "redaction_counts": redaction_counts,
        "residual_pii_matches_after_redaction": 0,
        "zero_shot": zero["report"],
        "few_shot": few["report"],
        "zero_shot_confusion_matrix": confusion_table(zero),
        "budget_demo": {**demo["report"], "budget_used": budget},
        "final_few_shot": final["report"],
        "final_routing_counts": final["report"]["routing_counts"],
    }
    # The budget demo run is also reset independently and records only retries.
    (OUT / "part_c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, run in [("zero_shot", zero), ("few_shot", few), ("budget_demo", demo), ("final_few_shot", final)]:
        (OUT / f"part_c_{name}_retry.log").write_text("\n".join(run["report"]["retry_log"]) + "\n", encoding="utf-8")

    print("\nPART C RUN SUMMARY")
    for name in ("zero_shot", "few_shot", "budget_demo", "final_few_shot"):
        r = report[name] if name in report else report["budget_demo"]
        print(f"{name}: classified={r['classified_tickets']} accuracy={r['accuracy_classified_only']:.4f} "
              f"tokens={r['total_tokens']} tokens_per_classified={r['tokens_per_classified_ticket']:.2f} "
              f"retries={r['retry_count']} budget_exhausted={r['budget_exhausted']}")
    print("zero-shot confusion matrix:")
    print(pd.DataFrame(report["zero_shot_confusion_matrix"]).T.to_string())
    print(f"budget demo used={budget}; exhausted={demo['report']['budget_exhausted']}")
    print("final output rows:", len(final_df))


if __name__ == "__main__":
    main()
