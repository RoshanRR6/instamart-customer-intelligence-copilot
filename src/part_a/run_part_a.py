"""Part A: audit, clean, store and analyse the synthetic Instamart exports."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
DB = ROOT / "data" / "instamart.db"


def audit_frame(df: pd.DataFrame) -> dict:
    missing = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean() * 100), 4) for c in df.columns],
    })
    return {"shape": list(df.shape), "columns": missing.to_dict(orient="records")}


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    df["city"] = df["city"].astype("string").str.strip().str.title()
    gender_map = {
        "male": "male", "m": "male",
        "female": "female", "f": "female",
    }
    raw_gender = df["gender"].fillna("").astype(str).str.strip().str.lower()
    df["gender"] = raw_gender.map(gender_map).fillna("unknown")
    age = pd.to_numeric(df["age"], errors="coerce")
    age = age.where(age.between(10, 100))
    city_medians = age.groupby(df["city"]).transform("median")
    df["age"] = age.fillna(city_medians).astype("int64")
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="raise")
    return df


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="raise")
    # Remove exact row duplicates first, then retain the latest event per order.
    df = df.drop_duplicates().copy()
    df = df.sort_values(["order_id", "order_ts"], kind="mergesort")
    df = df.drop_duplicates(subset=["order_id"], keep="last").copy()

    for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    product_lookup = products[["product_id", "price_inr", "category"]].copy()
    df = df.merge(product_lookup, on="product_id", how="left", validate="many_to_one")
    missing_amount = df["order_amount"].isna()
    df.loc[missing_amount, "order_amount"] = (
        df.loc[missing_amount, "price_inr"]
        * df.loc[missing_amount, "quantity"]
        * (1 - df.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    completed = df["status"].eq("completed")
    category_medians = df.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    completed_index = df.index[completed]
    df.loc[completed_index, "delivery_minutes"] = (
        df.loc[completed_index, "delivery_minutes"].fillna(category_medians)
    )
    # Cancelled orders remain missing: they were not delivered.
    df = df.drop(columns=["price_inr", "category"])
    df["order_amount"] = df["order_amount"].round(2)
    df["quantity"] = df["quantity"].astype("int64")
    df["discount_pct"] = df["discount_pct"].astype("int64")
    df = df.sort_values("order_id").reset_index(drop=True)
    return df


def write_sqlite(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        db_orders = orders.copy()
        db_orders["order_ts"] = pd.to_datetime(db_orders["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        db_orders.to_sql("orders", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)")
        conn.commit()


def run_queries() -> dict[str, pd.DataFrame]:
    sql_text = (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    current = None
    for line in sql_text.splitlines():
        if line.startswith("-- Q"):
            current = line[3:].strip().split(" — ")[0].split(" - ")[0]
            blocks[current] = ""
        elif current:
            blocks[current] += line + "\n"
    results = {}
    with sqlite3.connect(DB) as conn:
        for name, query in blocks.items():
            if query.strip():
                results[name] = pd.read_sql_query(query, conn)
    return results


def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    customers_raw = pd.read_csv(RAW / "customers.csv", keep_default_na=True)
    products_raw = pd.read_csv(RAW / "products.csv", keep_default_na=True)
    orders_raw = pd.read_csv(RAW / "orders.csv", keep_default_na=True)
    tickets_raw = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

    audit = {
        "customers": audit_frame(customers_raw),
        "products": audit_frame(products_raw),
        "orders": audit_frame(orders_raw),
        "tickets": audit_frame(tickets_raw),
        "specific": {
            "orders_raw_rows": int(len(orders_raw)),
            "orders_exact_duplicate_rows": int(orders_raw.duplicated().sum()),
            "orders_pending_rows": int(orders_raw["status"].eq("pending").sum()),
            "orders_distinct_order_id": int(orders_raw["order_id"].nunique()),
            "orders_non_numeric_amount": int(pd.to_numeric(orders_raw["order_amount"], errors="coerce").isna().sum()),
            "orders_negative_delivery": int((pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce") < 0).sum()),
            "orders_blank_delivery": int(orders_raw["delivery_minutes"].isna().sum() + orders_raw["delivery_minutes"].astype(str).str.strip().eq("").sum()),
            "customers_blank_age": int(customers_raw["age"].isna().sum() + customers_raw["age"].astype(str).str.strip().eq("").sum()),
            "customers_age_outside_10_100": int((pd.to_numeric(customers_raw["age"], errors="coerce").notna() & ~pd.to_numeric(customers_raw["age"], errors="coerce").between(10, 100)).sum()),
            "customers_raw_city_spellings": int(customers_raw["city"].nunique()),
            "customers_raw_nonblank_gender_spellings": int(customers_raw.loc[customers_raw["gender"].notna() & customers_raw["gender"].astype(str).str.strip().ne(""), "gender"].nunique()),
            "customers_blank_gender": int(customers_raw["gender"].isna().sum() + customers_raw["gender"].fillna("").astype(str).str.strip().eq("").sum() - customers_raw["gender"].isna().sum()),
        },
    }
    (OUTPUTS / "part_a_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    customers = clean_customers(customers_raw)
    products = products_raw.copy()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    orders = clean_orders(orders_raw, products)

    customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_sqlite(customers, products, orders)

    query_results = run_queries()
    summaries = {k: {"rows": int(len(v)), "columns": list(v.columns)} for k, v in query_results.items()}
    # Include exact SQL results for review and README reporting.
    for name, frame in query_results.items():
        frame.to_csv(OUTPUTS / f"{name.lower()}_results.csv", index=False)
    (OUTPUTS / "part_a_sql_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # pandas cross-checks and distribution/correlation analysis.
    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed = completed.merge(customers[["customer_id", "city"]], on="customer_id", how="left")
    completed = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
    pivot = pd.pivot_table(completed, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, fill_value=0)
    pivot.to_csv(OUTPUTS / "part_a_pivot_crosscheck.csv")
    city_stats = completed.groupby("city").agg(
        sum_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"), order_count=("order_amount", "count")
    ).reset_index()
    city_stats.to_csv(OUTPUTS / "part_a_city_stats.csv", index=False)

    x = completed["order_amount"].astype(float)
    q1, median, q3 = x.quantile([0.25, 0.50, 0.75]).tolist()
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    stats = {
        "mean_inr": float(x.mean()), "median_inr": float(median), "skewness": float(x.skew()),
        "q1_inr": float(q1), "q2_inr": float(median), "q3_inr": float(q3),
        "iqr_inr": float(iqr), "lower_fence_inr": float(lower), "upper_fence_inr": float(upper),
        "outliers_below_lower_fence": int((x < lower).sum()), "outliers_above_upper_fence": int((x > upper).sum()),
        "correlation_matrix": completed[corr_cols].corr().round(6).to_dict(),
    }
    corr_pairs = completed[corr_cols].corr().abs().where(np.triu(np.ones((len(corr_cols), len(corr_cols))), k=1).astype(bool)).stack()
    stats["largest_abs_correlation_pair"] = " / ".join(corr_pairs.idxmax())
    stats["largest_abs_correlation"] = float(corr_pairs.max())
    (OUTPUTS / "part_a_distribution_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Part A pipeline completed: audit, clean CSVs, SQLite, seven SQL queries and pandas analysis.")


if __name__ == "__main__":
    main()
