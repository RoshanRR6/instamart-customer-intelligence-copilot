"""Assert the required Part A reference checkpoints."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
customers = pd.read_csv(ROOT / "data/clean/customers.csv")
products = pd.read_csv(ROOT / "data/clean/products.csv")
orders = pd.read_csv(ROOT / "data/clean/orders.csv")

def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

check(len(customers) == 1200, "customer row count")
check(customers["city"].nunique() == 6, "six standardised cities")
check(customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts")
check(customers["age"].notna().all() and customers["age"].between(18, 60).all(), "customer ages")
check(len(products) == 40, "product row count")
check(len(orders) == 12309 and orders["order_id"].nunique() == 12309, "one row per order")
check(orders["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199}, "order statuses")
check(orders["order_amount"].notna().all(), "no missing order amounts")
check(orders.loc[orders.status == "completed", "delivery_minutes"].notna().all(), "completed delivery times imputed")
check(orders.loc[orders.status == "cancelled", "delivery_minutes"].isna().all(), "cancelled delivery times remain missing")
check(round(orders.loc[orders.status == "completed", "order_amount"].sum(), 2) == 2998668.80, "completed revenue")

with sqlite3.connect(ROOT / "data/instamart.db") as conn:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check({"customers", "products", "orders"}.issubset(names), "SQLite tables")
    db_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    check(db_count == 12309, "SQLite order count")

print("ALL CHECKPOINTS PASSED")
