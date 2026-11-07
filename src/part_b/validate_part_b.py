"""Validate required Part B artifacts and deterministic reference checkpoints."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
segments = pd.read_csv(OUT / "customer_segments.csv")
metrics = pd.read_csv(OUT / "kmeans_metrics.csv")
profile = pd.read_csv(OUT / "segment_profile.csv")
pca = pd.read_csv(OUT / "pca_variance.csv")
report = json.loads((OUT / "part_b_report.json").read_text(encoding="utf-8"))
retention = pd.read_csv(OUT / "retention_list.csv")

assert len(segments) == 1175
assert segments.columns.tolist() == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
assert segments.groupby("segment_id").size().sort_values().tolist() == [138, 182, 419, 436]
assert segments["is_premium"].sum() == 138
assert segments.loc[segments.is_premium, "segment_name"].nunique() == 1
assert metrics.k.tolist() == list(range(2, 9))
assert np.isclose(metrics.loc[metrics.k.eq(4), "inertia"].iloc[0], 743.1, rtol=0.05)
assert np.isclose(pca.explained_variance_ratio.sum(), 0.852, atol=0.02)
assert report["cohort_rows"] == 1156
assert report["lapsed_count"] == 443
assert report["train_rows"] == 924 and report["test_rows"] == 232
assert report["feature_missing_values"] == 0
assert report["reloaded_predictions_identical"] is True
assert max(report["logistic"]["test"]["f1"], report["random_forest"]["test"]["f1"]) >= 0.55
assert len(retention) <= 10
assert retention.columns.tolist() == ["customer_id", "lapse_probability", "actually_lapsed"]
assert joblib.load(OUT / "lapse_model.joblib") is not None
print("ALL PART B CHECKPOINTS PASSED")
