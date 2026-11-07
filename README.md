# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains a reproducible, four-part capstone built entirely from deterministic synthetic and fictional Instamart-style data. The project uses Python 3.11+, pandas, NumPy, scikit-learn, joblib, SQLite and the supplied offline `mock_llm.py`. No paid API, account, network connection or real customer data is required. All monetary amounts are Indian Rupees (INR).

The pipeline is designed so that Part A produces the trusted clean CSVs and SQLite database consumed by Parts B and D; Part B customer segments are consumed by Part C ticket routing; Part D uses the Part A database and local policy documents.

## Reproduce

Run these commands from the repository root. The activation command shown is for Windows PowerShell; on macOS/Linux use `source .venv/bin/activate` instead.

1. Create an isolated environment: `python -m venv .venv`
2. Activate it: `.venv\Scripts\Activate.ps1`
3. Install the frozen dependencies: `python -m pip install -r requirements.txt`
4. Generate the deterministic raw data: `python generate_data.py`
5. Run the Part A audit, cleaning, SQLite load, SQL and pandas analysis: `python src/part_a/run_part_a.py`
6. Validate Part A reference checkpoints: `python src/part_a/validate_part_a.py`
7. Run Part B RFM, clustering, PCA, lapse modelling and retention scoring: `python src/part_b/run_part_b.py`
8. Validate Part B artifacts and checkpoints: `python src/part_b/validate_part_b.py`
9. Run Part C privacy-safe ticket triage, prompt comparison, retries and budget demo: `python src/part_c/run_part_c.py`
10. Validate Part C artifacts and checkpoints: `python src/part_c/validate_part_c.py`
11. Run Part D policy retrieval, evaluation, refund scenarios, guardrail demos and crash/resume: `python src/part_d/run_part_d.py`
12. Validate Part D outputs and checkpoints: `python src/part_d/validate_part_d.py`

The generator must finish with exactly: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.

## Part A

### Raw-data audit

The raw CSV dimensions, pandas data types and missing-value audit are below. Missing percentages are calculated per column. `tickets.csv` is read with `keep_default_na=False`, preserving empty ticket text as an empty string.

**customers.csv — shape: 1,200 × 5**

| Column | pandas dtype | Missing | Missing % |
|---|---:|---:|---:|
| customer_id | object | 0 | 0.00% |
| city | object | 0 | 0.00% |
| signup_date | object | 0 | 0.00% |
| age | float64 | 61 | 5.08% |
| gender | object | 30 | 2.50% |

**products.csv — shape: 40 × 4**

| Column | pandas dtype | Missing | Missing % |
|---|---:|---:|---:|
| product_id | object | 0 | 0.00% |
| product_name | object | 0 | 0.00% |
| category | object | 0 | 0.00% |
| price_inr | int64 | 0 | 0.00% |

**orders.csv — shape: 13,084 × 9**

| Column | pandas dtype | Missing | Missing % |
|---|---:|---:|---:|
| order_id | object | 0 | 0.00% |
| customer_id | object | 0 | 0.00% |
| product_id | object | 0 | 0.00% |
| quantity | int64 | 0 | 0.00% |
| order_ts | object | 0 | 0.00% |
| status | object | 0 | 0.00% |
| discount_pct | int64 | 0 | 0.00% |
| order_amount | object | 53 | 0.41% |
| delivery_minutes | float64 | 2,541 | 19.42% |

**tickets.csv — shape: 164 × 5**

| Column | pandas dtype | Missing | Missing % |
|---|---:|---:|---:|
| ticket_id | object | 0 | 0.00% |
| customer_id | object | 0 | 0.00% |
| order_id | object | 0 | 0.00% |
| text | object | 0 | 0.00% |
| true_category | object | 0 | 0.00% |

Specific raw audit figures:

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with `pending` status | 656 |
| Distinct `order_id` values | 12,309 |
| `order_amount` values not convertible to numeric | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 (excluding blanks) | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

The machine-readable audit is saved at `outputs/part_a_audit.json`.

### Customer cleaning

1. City values were stripped and converted to title case. **476 of 1,200 rows** changed representation; exactly six cities remain.
2. Gender spellings were mapped case-insensitively: `male`/`m` to `male`, `female`/`f` to `female`, and blank or unrecognised values to `unknown`. **950 values** changed to their standard category, including the 30 blanks.
3. Ages outside 10–100 were made missing. The **61 blank ages plus 6 invalid ages (67 total)** were imputed using each customer's city median via `groupby(city).transform('median')`.
4. `signup_date` was parsed as a pandas datetime.

Clean customer city counts:

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Clean gender counts: **female 610, male 560, unknown 30**. There are no missing ages; all cleaned ages are between 18 and 60.

### Order cleaning

Rules were applied in the required order:

1. Parse `order_ts` as datetime.
2. Remove **119 exact duplicate rows**, leaving 12,965 rows.
3. Sort by `order_id` and timestamp, then retain the latest timestamp per order. This removes the **656 earlier pending records**, leaving 12,309 unique orders and no pending status.
4. Convert `order_amount` to numeric. After exact and order-level deduplication, **112 final-order amounts** remained unparseable/missing; each was filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)` using its product catalogue price.
5. Convert negative delivery durations to missing. Of the final-order records, **105 negative durations** remained after deduplication. For completed orders, **669 missing delivery durations** (including negative durations) were imputed with the completed-order median for the relevant product category using `groupby(category).transform('median')`. All **1,199 cancelled orders** retain missing delivery duration because they were not delivered.

Clean order checkpoints:

| Check | Result |
|---|---:|
| Total orders | 12,309 |
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| Missing `order_amount` | 0 |
| Completed orders missing delivery time | 0 |
| Cancelled orders missing delivery time | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |

### Storage and validation

The clean files are `data/clean/customers.csv`, `data/clean/products.csv` and `data/clean/orders.csv`. `data/instamart.db` contains SQLite tables `customers`, `products` and `orders`. `orders.order_ts` is stored as text in `YYYY-MM-DD HH:MM:SS` format. The validation script checks more than eight required checkpoints and prints exactly `ALL CHECKPOINTS PASSED` when successful.

### SQL analysis

All seven statements are stored in `sql/queries.sql` and executed from Python using `pandas.read_sql_query` against `data/instamart.db`.

| Query | SQL concept | Result |
|---|---|---|
| Q1 | `WHERE` then `HAVING` | 30 customers have completed orders above INR 500 whose qualifying-order total exceeds INR 8,000. |
| Q2 | Three-table join | 30 city/category rows; total revenue INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45. |
| Q3 | Scalar subquery | 11 products are priced above the average catalogue price. |
| Q4 | `LEFT JOIN` | 25 customers have no completed order. |
| Q5 | Two CTEs | 550 customers exceed average completed spend; average customer spend is INR 2,552.06. |
| Q6 | `RANK`, `DENSE_RANK`, `ROW_NUMBER` | 1,175 customers with completed spend; 8 rows have `RANK()` different from `ROW_NUMBER()`. |
| Q7 | Running-total window | 183 daily rows; final cumulative revenue is INR 2,998,668.80. |

Top customer by city in Q6:

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

Tied customers share the same `RANK()` value, while `ROW_NUMBER()` assigns a distinct sequential number to every row; therefore, ties make the two functions differ for some rows.

### pandas cross-check and exploratory analysis

A `pivot_table` by city and product category with `margins=True` reproduces the SQL Q2 grand total: **INR 2,998,668.80**.

| City | Completed revenue sum | Mean order amount | Completed order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed `order_amount` distribution:

| Statistic | Result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q2 (median) | INR 135.00 |
| Q3 | INR 297.00 |
| IQR | INR 232.20 |
| Lower 1.5×IQR fence | INR -283.50 |
| Upper 1.5×IQR fence | INR 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

The correlation matrix is calculated over completed orders for `order_amount`, `discount_pct`, `quantity` and `delivery_minutes`. The largest absolute correlation is between **order amount and discount percentage (approximately −0.095)**; order amount and quantity is approximately +0.087.

**Exploratory interpretation (under 200 words):** The data describes synthetic customer purchases across six cities and five product categories over 183 days. Completed order amounts are strongly right-skewed: the mean (INR 269.91) is well above the median (INR 135.00), with skewness 4.61. The upper tail contains 897 values above the INR 645.30 upper IQR fence and none below the lower fence. The raw exports also contain duplicate order rows, earlier pending records, non-numeric amounts, impossible negative delivery durations and inconsistent customer labels; these are addressed by the cleaning rules above. Among the measured numeric variables, order amount and discount percentage have the largest absolute correlation (about −0.095), while amount and quantity correlate about +0.087. Both relationships are weak; correlation of this size does not establish causation.

## Part B

### RFM table and segmentation

RFM is calculated for customers with at least one completed order, using the snapshot `2025-10-01 00:00:00`. Recency is the whole-day difference from the snapshot to the latest completed order; frequency is the distinct completed-order count; monetary is the sum of completed `order_amount`.

| RFM checkpoint | Result |
|---|---:|
| Customers with completed orders | 1,175 |
| Recency median / minimum / maximum | 19 / 0 / 180 days |
| Frequency mean / maximum | 9.46 / 33 |
| Monetary mean / maximum | INR 2,552.06 / INR 14,766.55 |

The three RFM variables were standardised with `StandardScaler`. K-means was fitted for K=2 through K=8 with `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

**K=4 was selected.** It is at the visible elbow and has the strongest silhouette score of the business-actionable choices K=3, 4 and 5; four groups also support distinct retention and growth campaigns. Inertia always decreases as K increases because additional centroids can only keep or reduce each observation's distance to its assigned centroid; K=8 therefore has lower inertia than K=2, but that mechanical improvement does not account for over-fragmentation, separation or whether the business can act on the resulting groups.

Segment profiles and campaign actions:

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Growing Customers | 436 | 28.86 | 3.77 | INR 1,057.62 | Encourage second-category discovery and basket expansion. |
| 1 | Premium Champions (**Premium**) | 138 | 24.12 | 6.77 | INR 7,129.70 | Protect with priority service and tailored loyalty rewards. |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | INR 3,410.02 | Reinforce habits with replenishment reminders and bundles. |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | INR 685.99 | Trigger win-back outreach and a time-limited return incentive. |

Cluster IDs are arbitrary labels, not an ordered ranking. The Premium flag is assigned only to the segment with the highest mean monetary value. PCA on the scaled RFM matrix explains **63.43%** and **21.80%** of variance in its first two components, **85.23%** combined.

The required segment output is `outputs/customer_segments.csv` with columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`. Supporting diagnostics are `outputs/kmeans_metrics.csv`, `outputs/segment_profile.csv` and `outputs/pca_variance.csv`.

### Lapse cohort and leakage-safe features

The cohort includes each customer with at least one completed order before `2025-09-01`. A customer is labelled lapsed if they have no completed order from `2025-09-01` through `2025-09-30` inclusive. The cohort contains **1,156 customers**, of whom **443 (38.32%)** are lapsed. The stratified split (`test_size=0.2`, `random_state=42`) contains **924 training** and **232 test** rows.

Features use only orders dated before the cutoff plus the clean customer table: recency, distinct completed-order frequency, completed monetary spend, mean completed-order discount, cancellation rate among final-status pre-cutoff orders, mean completed delivery time, favourite completed-order category, city, age, gender and signup tenure. No order dated on or after 2025-09-01 enters feature construction, and the Part B segment labels are not model features. The label alone uses September completed-order activity. All engineered feature columns are checked to contain zero missing values.

The fitted numeric scaler's training-row frequency mean is **8.115**, compared with the full-cohort frequency mean of **8.029**. They differ because the scaler is fitted only on the 80% training partition, while the latter mean includes both training and held-out test customers. The one-hot encoder and scaler are inside the model pipeline and are fitted only on training folds during cross-validation and on the full training split for final fitting; neither saw test rows. The oversampling experiment duplicates minority-class training rows only; no resampler saw test rows.

### Model tuning and held-out evaluation

The always-predict-not-lapsed baseline has **61.64% accuracy** and lapsed-class **F1=0.000**. Both model searches used five-fold cross-validation and `scoring='f1'` on the training partition only.

| Model | Search grid | Best parameters | Best CV F1 | Test confusion matrix (actual rows: 0,1; predicted columns: 0,1) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | C = 0.01, 0.1, 1, 10, 100 | C=100 | 0.644 | [[108, 35], [33, 56]] | 0.615 | 0.629 | 0.622 |
| Random forest | max_depth = 3, 5, 8; min_samples_leaf = 5, 10, 20 | max_depth=8, min_samples_leaf=10 | 0.683 | [[111, 32], [30, 59]] | 0.648 | 0.663 | 0.656 |

The **random forest** is selected, with held-out lapsed-class F1 **0.656**, above the 0.55 acceptance threshold. Missing a lapsing customer can mean losing future orders and delaying a useful intervention, while wrongly targeting a loyal customer wastes incentive budget and may train them to wait for discounts. The forest has the stronger held-out F1 and better precision/recall balance here, so it is the preferred prioritisation model; campaign decisions should still be reviewed rather than treated as certainty.

### Class-balance experiment

The original training set had 570 non-lapsed and 354 lapsed customers. Random oversampling with replacement (`random_state=42`) retained all original rows and added lapsed training examples until both classes had **570** rows. The balanced logistic model was evaluated against the original, unchanged test set.

| Logistic model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.615 | 0.629 | 0.622 |
| Oversampled training distribution | 0.592 | 0.865 | 0.703 |

Recall moved the most, increasing by about **23.6 percentage points**, while precision decreased by about **2.3 points**. The retention team would identify more customers who are actually lapsing, at the cost of sending more unnecessary interventions to customers who would have ordered anyway.

### Interpretation, persistence and retention list

The random forest's five largest feature importances are:

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.2843 | Prior ordering cadence is the strongest signal in this model. |
| Monetary | 0.2748 | Historical spend materially helps distinguish future inactivity. |
| Recency | 0.1229 | How recently a customer ordered is an important warning signal. |
| Average delivery minutes | 0.0624 | Delivery experience carries additional predictive information. |
| Cancellation rate | 0.0480 | A history of cancellations contributes to lapse risk. |

The fitted logistic regression coefficient signs are **recency positive (+0.0425)**, **frequency negative (−1.6266)** and **monetary positive (+0.2224)**. Holding other encoded features constant, greater recency raises the model's lapse log-odds, while higher frequency lowers them; the positive monetary coefficient is a conditional model association, not a causal claim.

The selected random-forest pipeline is saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and produces predictions identical to the original fitted model on the test rows. `outputs/retention_list.csv` contains the top 10 test-set Premium customers ranked by predicted lapse probability; **0 of these 10 actually lapsed** in the held-out label window. The retention list is illustrative and not a leakage-free evaluation: its Premium flag is a post-hoc whole-window segment label based on RFM through 2025-10-01, which includes the September period used to define the lapse label. The model features themselves remain strictly pre-cutoff.

The complete metrics and diagnostics are saved in `outputs/part_b_report.json` and supporting CSV/JSON files under `outputs/`. Run `python src/part_b/validate_part_b.py` to verify the Part B acceptance checkpoints; it prints `ALL PART B CHECKPOINTS PASSED`.

## Part C

### Input guardrails and privacy redaction

The triage pipeline reads all **164** tickets in their original file order. It checks for empty/whitespace-only text and then estimates input tokens with the supplied `mock_llm.est_tokens`. **4 empty tickets** receive `skipped_empty`; **4 long tickets** receive `flagged_long` and are never sent to the model. Each long ticket estimates to **734 tokens**, above `MAX_INPUT_TOKENS=400`. The remaining **156** tickets are eligible for model calls.

Before any model call, the pipeline applies these exact substitutions in the required order: email `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, and order ID `ORD-\d{6}` to `[ORDER_ID]`. Across all 164 source tickets the counts are **55 emails, 91 phone numbers and 132 order IDs**, or **278 substitutions**. The code re-runs all three expressions over every redacted text and asserts zero residual matches. It also checks immediately before every model call that no expression matches the user message. The user message content is exactly the redacted ticket text; no ticket metadata or instructions are appended to it.

### Prompts, retries and output validation

Two system prompts are defined in `src/part_c/run_part_c.py`. Both include `CLASSIFY`, enumerate `billing`, `delivery`, `technical` and `fraud`, and require a JSON-only `{"category": "...", "reason": "..."}` response. The zero-shot prompt has no `EXAMPLES:` section. The few-shot prompt has labelled examples for billing, fraud, technical and delivery.

Every independent full run calls `mock_llm.reset()` immediately before processing. Calls are wrapped in a retry helper with at most three retries and wait `0.05 × 2**n × random.uniform(0.9, 1.15)` seconds before retry `n`. Each retry writes a line containing the ticket ID, retry number and measured wait to stdout and a run-specific file under `outputs/`. For each full 156-ticket run the mock produced **19 transient errors**; all recovered on retry number **0**, with observed waits from **0.045313 to 0.056152 seconds**. The mock saw 175 call attempts (156 successful responses plus 19 recovered transient failures), and no ticket ended in a permanent error.

Responses are parsed as JSON first. A valid allowed category is `classified_json`. If JSON parsing fails, lowercase alphabetic words are extracted with `re.findall(r"[a-z]+", text.lower())`; exactly one allowed category word gives `classified_keyword`. A valid JSON response with an unsupported category, or any other ambiguous/invalid response, becomes `flagged_invalid_output` with no guessed category. Both full prompt runs produced **135 `classified_json`, 12 `classified_keyword` and 9 `flagged_invalid_output`** results. The 135 JSON categories and 12 single-keyword categories passed the stated output rules and were eligible for automated routing; the 9 invalid outputs were not assigned a category and went to `human_review`. The 12 chatty keyword responses are accepted because hyphenated category words are split correctly by the regular expression.

### Zero-shot versus few-shot results

Accuracy below is calculated only over the **147 classified tickets**; the 17 empty, long or invalid-output tickets are excluded from the accuracy denominator and routed to human review.

| Measure | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct | 126 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 13,144 | 28,432 |
| Tokens per classified ticket | 89.41 | 193.41 |
| Transient errors recovered | 19 | 19 |

Zero-shot confusion matrix (actual categories in rows; predicted categories in columns):

| Actual \\ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt costs **15,288 additional tokens overall**, or **104.00 more tokens per classified ticket**, but corrects all 21 zero-shot errors and raises accuracy by **14.29 percentage points**. In this mock run the zero-shot errors were 11 fraud tickets and 10 technical tickets misclassified as billing, so the additional examples are worth paying for when misrouting security-related or technically urgent tickets has a meaningful operational cost. If token cost dominates and a human can reliably review uncertain tickets, the shorter zero-shot prompt may be a reasonable first-pass alternative; these synthetic mock results do not establish production model performance.

### Token budget, segment-aware routing and outputs

The budget check happens immediately before every model call: if accumulated `usage.total_tokens` is already at least the configured budget, the ticket is `budget_exhausted` and is routed to human review. The zero-shot demonstration used `TOKEN_BUDGET=6000`; **84 tickets** were budget-exhausted. The last permitted response can take the running total beyond the threshold because the check is made before a call; the demo consumed **6,076 tokens** before subsequent calls were stopped.

The final deliverable is a few-shot run with `TOKEN_BUDGET=100000`, reading `outputs/customer_segments.csv`. The output has 164 rows and columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Routing precedence is: guardrail/invalid/budget statuses to `human_review` with NORMAL priority; otherwise fraud to `security_team` with HIGH priority; otherwise Premium customers to `premium_desk` with HIGH priority; all others to `support_queue` with NORMAL priority. Missing segment rows are treated as not Premium. In this final run the routes are **102 support_queue, 34 security_team, 11 premium_desk and 17 human_review**. All 17 human-review rows comprise the 4 empty, 4 long and 9 invalid-output tickets; no ticket exhausted the 100,000-token final budget.

Key artifacts:

- `outputs/triage_results.csv` — final 164-row few-shot routing result.
- `outputs/triage_budget_demo.csv` — zero-shot budget demonstration (84 budget-exhausted tickets).
- `outputs/triage_zero_shot_full.csv` and `outputs/triage_few_shot_full.csv` — full-run comparison details, including true labels for evaluation.
- `outputs/part_c_report.json` — redaction counts, status counts, accuracy, token usage, confusion matrix, retry counts and routing counts.
- `outputs/part_c_*_retry.log` — one safe retry log line per recovered transient failure; ticket text and PII are not logged.

Run `python src/part_c/validate_part_c.py` to verify the Part C acceptance checkpoints; it prints `ALL PART C CHECKPOINTS PASSED`.

## Part D

### Policy chunking and retrieval

The five policy documents are read in a fixed order and split on whitespace into **40-word chunks with 10-word overlap** (next chunk starts 30 words after the prior start). The resulting 15 chunks are: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. Each chunk retains its source filename. `TfidfVectorizer()` is fitted on the chunk texts; queries are transformed with that same vectorizer and ranked by cosine similarity. TF (term frequency) measures how often a term occurs in a chunk, while IDF downweights terms that occur in many chunks and gives more weight to distinctive terms. TF-IDF is lexical rather than semantic, so two differently worded questions with the same meaning may have low similarity if they share few words.

For `refund policy late delivery`, all top three chunks come from `refund_policy.txt`. `answer(question)` sends a `POLICY ASSISTANT` system instruction and a user message containing exactly `CONTEXT:`, the three source-labelled chunks, `QUESTION:` and the question; it returns the model answer and the distinct source filenames used. The system instruction requires context-only answers and a fixed fallback when the documents do not contain the answer.

| Retrieval evaluation | Expected document in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Unanswerable fallback (2) |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the question **“What delivery time makes a completed order eligible for a full refund?”** loses its expected fact. The phrase “more than 45 minutes” falls at the end of refund chunk 1 in the overlapped version, while in the zero-overlap version the chunk boundary separates “more than 45 minutes” from “receives a full refund”; the mock answerer then selects only the second fragment. The payment timing question also fails the expected-fact substring check in the zero-overlap run. The 10-word overlap preserves the cross-boundary rule as a complete sentence in a retrieved chunk.

### Refund tools and ReAct agent

`TOOLS_SPEC` defines the five tool names, descriptions and JSON-style argument schemas; `TOOL_MAP` dispatches tool names to Python functions. `get_order` reads the Part A SQLite database, `search_policy` calls the same RAG retriever, and `calculate_refund` returns the full order amount only when delivery took more than 45 minutes. The agent checks that the order is completed before calculating a refund. `issue_refund` records amounts up to INR 1000; `escalate_to_human` records amounts above INR 1000 without issuing them. Both ledger tools append timestamped JSON lines to `outputs/refund_ledger.jsonl`. All seven simulated requests are treated as raised within the policy's 7-day window.

The ReAct loop sends an `AGENT` system message and the request, then alternates model calls and tool observations. It stops on a final answer or safely returns a human-handoff summary on iteration, token or wall-clock limits. After each tool observation, the complete messages, token count, step number, executed tools and model-call count are checkpointed under `state/`; `resume_agent(run_id)` reloads this state and continues without replaying completed tools.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,282 |
| S2 | REFUND | 135.00 | 5 | 1,282 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,269 |
| S4 | DENIED | 0.00 | 4 | 913 |
| S5 | DENIED | 0.00 | 3 | 568 |
| S6 | NOT_FOUND | 0.00 | 2 | 226 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,462 |

The S1 token-limit demonstration (`token_limit=200`) ends with **FALLBACK_TOKEN_LIMIT** after its first model response (256 accumulated tokens); the requested tool is not executed after the limit is crossed. The zero-second time-limit demonstration ends with **FALLBACK_TIMEOUT** before the first model call. Under the default `token_limit=6000`, S7 completes all six model calls and stops at `FALLBACK_MAX_ITERATIONS` rather than being cut off by the token limit.

For crash/resume, the S1 run was deliberately interrupted immediately after the second tool observation had been checkpointed. The crash portion executed `get_order`, `search_policy`; the resumed portion executed `calculate_refund`, `issue_refund`. Across both portions there were exactly four tool executions, no repeats, and the resumed final result was **REFUND: INR 128.25 issued.** (five model calls total).

### ReAct versus plan-and-execute (under 250 words)

In this mock, ReAct takes 5 model calls for S1, 3 for S5 and 2 for S6 because each observation is returned to the model before the next action. That adds prompt and observation tokens compared with a plan-and-execute design that asks for a complete tool sequence once and replans only after an error. However, ReAct can inspect `NOT_FOUND`, cancelled status, delivery time and calculated refund before choosing the next tool, whereas a fixed plan risks attempting an invalid action or needs a replan. For refunds I would choose ReAct: the extra calls are justified by conditional decisions and the ability to stop safely before an unauthorized or incorrect payment.

### Part D artifacts

- `outputs/part_d_report.json` — chunk counts, retrieval metrics, scenario results and safety demonstrations.
- `outputs/agent_runs.jsonl` — one JSON record per scenario, including all tool names, arguments and observations.
- `outputs/refund_ledger.jsonl` — simulated issued and escalated refund actions.
- `state/*.json` — resumable agent checkpoints.
- `src/part_d/run_part_d.py` and `src/part_d/validate_part_d.py` — implementation and acceptance checks.

Run `python src/part_d/validate_part_d.py` to verify Part D; it prints `ALL PART D CHECKPOINTS PASSED`. No real network call or API key is used.

## Limitations and Next Steps

The dataset and policy files are fictional and deterministic; results should not be interpreted as real Instamart performance or policy. The Premium flag used to select the retention-list customers is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation. The lapse model itself uses only pre-2025-09-01 features and does not use Part B segment labels as predictors. Part C uses the supplied offline mock model; its synthetic accuracy is not a production performance estimate. Part D uses lexical TF-IDF retrieval and a deterministic mock LLM, so semantic paraphrases may be missed and the agent's decisions do not establish production refund suitability. The refund ledger is a local simulation, not a payment-system integration. Production next steps would include human approval controls, audited policy versioning, semantic retrieval evaluation, authentication, observability, and testing against real-world edge cases. All four capstone parts and their local acceptance checks are implemented; a clean-clone reproduction should be run before publishing the repository.
