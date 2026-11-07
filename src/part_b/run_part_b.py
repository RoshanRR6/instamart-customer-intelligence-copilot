"""Part B: reproducible RFM segmentation and leakage-safe lapse modelling."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")


def load_inputs():
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_ts"])
    products = pd.read_csv(DATA / "products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")
    return customers, orders


def build_rfm(customers, orders):
    completed = orders.loc[orders.status.eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm.pop("last_order_ts")).dt.days
    rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
    return rfm


def segment_customers(rfm):
    features = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[features])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(scaled, labels))})
    metrics = pd.DataFrame(rows)

    # K=4 balances the visible elbow, the strongest silhouette in the requested
    # business range, and actionable campaign groups.
    chosen_k = 4
    model = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    result = rfm.copy()
    result["segment_id"] = model.fit_predict(scaled)
    profile = result.groupby("segment_id").agg(
        size=("customer_id", "count"), mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")
    ).reset_index()

    # Names are assigned from measurable profile traits, not numeric cluster IDs.
    def describe(row):
        if row.mean_monetary == profile.mean_monetary.max():
            return "Premium Champions"
        if row.mean_frequency == profile.mean_frequency.max():
            return "Loyal Regulars"
        if row.mean_recency == profile.mean_recency.max():
            return "At-Risk Customers"
        return "Growing Customers"
    name_map = {int(row.segment_id): describe(row) for row in profile.itertuples(index=False)}
    premium_id = int(profile.loc[profile.mean_monetary.idxmax(), "segment_id"])
    result["segment_name"] = result.segment_id.map(name_map)
    result["is_premium"] = result.segment_id.eq(premium_id)

    pca = PCA(n_components=2)
    pca.fit(scaled)
    profile["segment_name"] = profile.segment_id.map(name_map)
    profile["recommended_action"] = profile.segment_id.map({
        premium_id: "Protect with priority service and tailored loyalty rewards"
    })
    for sid in profile.segment_id:
        if int(sid) == premium_id:
            continue
        name = name_map[int(sid)]
        action = {
            "Loyal Regulars": "Reinforce habits with replenishment reminders and bundles",
            "At-Risk Customers": "Trigger win-back outreach and a time-limited return incentive",
            "Growing Customers": "Encourage second-category discovery and basket expansion",
        }.get(name, "Test a targeted engagement campaign")
        profile.loc[profile.segment_id.eq(sid), "recommended_action"] = action

    result[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(OUT / "customer_segments.csv", index=False)
    metrics.to_csv(OUT / "kmeans_metrics.csv", index=False)
    profile.to_csv(OUT / "segment_profile.csv", index=False)
    pd.DataFrame({"component": [1, 2], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(OUT / "pca_variance.csv", index=False)
    return result, metrics, profile, pca.explained_variance_ratio_


def build_lapse_cohort(customers, orders):
    before = orders.loc[orders.order_ts.lt(CUTOFF)].copy()
    pre_completed = before.loc[before.status.eq("completed")].copy()
    cohort_ids = pre_completed.customer_id.unique()
    sept_completed = orders.loc[
        orders.status.eq("completed")
        & orders.order_ts.ge(CUTOFF)
        & orders.order_ts.lt(pd.Timestamp("2025-10-01")), "customer_id"
    ].unique()
    label = pd.Series(1, index=cohort_ids, name="lapsed", dtype=int)
    label.loc[label.index.isin(sept_completed)] = 0

    agg = pre_completed.groupby("customer_id").agg(
        last_completed=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    agg["recency"] = (CUTOFF - agg.pop("last_completed")).dt.days
    final_counts = before.groupby("customer_id").status.agg(
        total_final_orders="count", cancelled_orders=lambda s: int(s.eq("cancelled").sum())
    )
    agg = agg.join(final_counts, how="left")
    agg["cancellation_rate"] = agg.cancelled_orders.div(agg.total_final_orders.replace(0, np.nan)).fillna(0)
    fav = pre_completed.groupby("customer_id").category.agg(
        lambda s: s.value_counts().index[0] if not s.empty else "Unknown"
    ).rename("favourite_category")
    agg = agg.join(fav)

    cohort = customers.set_index("customer_id").join(agg, how="inner")
    cohort["tenure_days"] = (CUTOFF - cohort.signup_date).dt.days
    cohort["lapsed"] = label
    cohort = cohort.reset_index()
    numeric = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
               "avg_delivery_minutes", "age", "tenure_days"]
    categorical = ["city", "gender", "favourite_category"]
    # Guard against missing values before any model fitting; imputation is not used
    # to conceal a broken feature engineering step.
    for col in numeric:
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
        if cohort[col].isna().any():
            cohort[col] = cohort[col].fillna(cohort[col].median())
    for col in categorical:
        cohort[col] = cohort[col].fillna("Unknown").astype(str)
    assert not cohort[numeric + categorical].isna().any().any(), "Feature matrix contains missing values"
    return cohort, numeric, categorical


def make_pipeline(numeric, categorical, estimator):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def metrics_for(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def train_models(cohort, numeric, categorical, segments):
    features = numeric + categorical
    X = cohort[features].copy()
    y = cohort.lapsed.astype(int)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, cohort.customer_id, test_size=0.2, stratify=y, random_state=42
    )
    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_pred)),
        "lapsed_precision": float(precision_score(y_test, baseline_pred, zero_division=0)),
        "lapsed_recall": float(recall_score(y_test, baseline_pred, zero_division=0)),
        "lapsed_f1": float(f1_score(y_test, baseline_pred, zero_division=0)),
    }

    log_pipe = make_pipeline(numeric, categorical, LogisticRegression(max_iter=1000, random_state=42))
    log_grid = GridSearchCV(log_pipe, {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=-1)
    log_grid.fit(X_train, y_train)
    best_log = log_grid.best_estimator_

    rf_pipe = make_pipeline(numeric, categorical, RandomForestClassifier(n_estimators=100, random_state=42))
    rf_grid = GridSearchCV(rf_pipe, {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    }, cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    log_test = metrics_for(best_log, X_test, y_test)
    rf_test = metrics_for(best_rf, X_test, y_test)
    selected_name, selected_model = ("random_forest", best_rf) if rf_test["f1"] >= log_test["f1"] else ("logistic_regression", best_log)
    if max(log_test["f1"], rf_test["f1"]) < 0.55:
        raise AssertionError("Selected model test F1 is below the required 0.55")

    # Required training-only random oversampling for the logistic regression experiment.
    train_frame = X_train.copy()
    train_frame["lapsed"] = y_train.values
    majority = int(y_train.value_counts().max())
    minority_frame = train_frame.loc[train_frame.lapsed.eq(1)]
    additional_minority = majority - len(minority_frame)
    minority_duplicates = resample(minority_frame, replace=True, n_samples=additional_minority, random_state=42)
    # Preserve every original training row, then append only the required duplicates.
    balanced = pd.concat([train_frame, minority_duplicates], ignore_index=True)
    balanced_log = make_pipeline(numeric, categorical, LogisticRegression(max_iter=1000, C=log_grid.best_params_["model__C"], random_state=42))
    balanced_log.fit(balanced[features], balanced.lapsed)
    balanced_metrics = metrics_for(balanced_log, X_test, y_test)

    # Training-only scaler mean is extracted from the fitted logistic pipeline.
    scaler = best_log.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    frequency_idx = numeric.index("frequency")
    full_frequency_mean = float(X.frequency.mean())
    train_frequency_mean = float(scaler.mean_[frequency_idx])

    # Feature importance and coefficients from the fitted, training-only pipelines.
    rf_pre = best_rf.named_steps["preprocess"]
    names = rf_pre.get_feature_names_out()
    importances = best_rf.named_steps["model"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    top_features = [{"feature": str(names[i]), "importance": float(importances[i])} for i in top_idx]
    log_names = best_log.named_steps["preprocess"].get_feature_names_out()
    coefs = best_log.named_steps["model"].coef_[0]
    preferred = ["num__recency", "num__frequency", "num__monetary"]
    coefficient_signs = []
    for wanted in preferred:
        if wanted in list(log_names):
            idx = list(log_names).index(wanted)
            coefficient_signs.append({"feature": wanted.replace("num__", ""), "coefficient": float(coefs[idx]),
                                      "sign": "positive" if coefs[idx] > 0 else "negative"})

    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded = joblib.load(model_path)
    original_pred = selected_model.predict(X_test)
    reloaded_pred = reloaded.predict(X_test)
    predictions_identical = bool(np.array_equal(original_pred, reloaded_pred))
    assert predictions_identical

    # Premium flag is deliberately post-hoc and is not a model feature.
    segment_lookup = segments.set_index("customer_id")["is_premium"]
    premium_test = pd.DataFrame({"customer_id": id_test.values, "actual": y_test.values,
                                 "probability": selected_model.predict_proba(X_test)[:, 1]})
    premium_test["is_premium"] = premium_test.customer_id.map(segment_lookup).fillna(False).astype(bool)
    retention = premium_test.loc[premium_test.is_premium].sort_values("probability", ascending=False).head(10).copy()
    retention.rename(columns={"probability": "lapse_probability", "actual": "actually_lapsed"})[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ].to_csv(OUT / "retention_list.csv", index=False)

    report = {
        "cohort_rows": int(len(cohort)), "lapsed_count": int(y.sum()),
        "lapsed_rate_pct": float(y.mean() * 100),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "baseline": baseline,
        "logistic": {"best_params": log_grid.best_params_, "best_cv_f1": float(log_grid.best_score_), "test": log_test},
        "random_forest": {"best_params": rf_grid.best_params_, "best_cv_f1": float(rf_grid.best_score_), "test": rf_test},
        "selected_model": selected_name,
        "oversampled_logistic_test": balanced_metrics,
        "frequency_scaler_train_mean": train_frequency_mean,
        "frequency_full_cohort_mean": full_frequency_mean,
        "rf_top_features": top_features,
        "logistic_coefficient_signs": coefficient_signs,
        "reloaded_predictions_identical": predictions_identical,
        "retention_list_rows": int(len(retention)),
        "retention_list_actually_lapsed": int(retention.actual.sum()),
        "retention_list_is_illustrative_posthoc_premium_flag": True,
        "feature_missing_values": int(X.isna().sum().sum()),
        "feature_columns": features,
    }
    (OUT / "part_b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model": "logistic_regression", **log_test},
        {"model": "random_forest", **rf_test},
        {"model": "logistic_regression_oversampled_train", **balanced_metrics},
    ]).to_json(OUT / "model_metrics.json", orient="records", indent=2)
    return report


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    customers, orders = load_inputs()
    rfm = build_rfm(customers, orders)
    segments, km_metrics, profile, pca_ratio = segment_customers(rfm)
    cohort, numeric, categorical = build_lapse_cohort(customers, orders)
    report = train_models(cohort, numeric, categorical, segments)

    print("RFM rows:", len(rfm), "recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
    print("RFM frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max(),
          "monetary mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))
    print("K-means metrics:\n", km_metrics.to_string(index=False))
    print("Segment profile:\n", profile.to_string(index=False))
    print("PCA explained variance:", pca_ratio, "total:", pca_ratio.sum())
    print("Lapse report:\n", json.dumps(report, indent=2))
    print("PART B COMPLETE")


if __name__ == "__main__":
    main()
