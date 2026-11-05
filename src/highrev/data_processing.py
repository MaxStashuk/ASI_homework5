"""Exercise 2: extract transformations from your baseline.py.

All public functions return a new DataFrame and leave their inputs unchanged.
Keep file access and model training outside this module.
"""

import pandas as pd
import user_agents

CATEGORY_MAP = {
    "Bl Tshirt F": "Black T-Shirt F",
    "Bl Tshirt M": "Black T-Shirt M",
    "Wh Tshirt M": "White T-Shirt M",
    "Wh Tshirt F": "White T-Shirt F",
}

LEAP_YEAR_FIXES = {
    "1993/2/29": "1993/2/28",
    "1947/2/29": "1947/2/28",
    "1965/2/29": "1965/2/28",
}

UNUSED_CUSTOMER_COLS = ["birthdate", "order_date", "order_id", "ip_address", "user_agent"]


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Return customer aggregates indexed by customer_id.

    Accept baseline year/month/day date strings, including unpadded month/day.
    Preserve the baseline category normalization, parse order_date, and derive
    total from tshirt_quantity and tshirt_price. Aggregate order_date (min),
    total (sum), pages_visited (mean), and order_id (count).
    Exercise 5 adds your documented negative-quantity policy.
    """
    cleaned = orders.loc[orders["tshirt_quantity"] >= 0].copy()
    cleaned["tshirt_category"] = cleaned["tshirt_category"].replace(CATEGORY_MAP)
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], format="%Y/%m/%d")
    cleaned["total"] = cleaned["tshirt_quantity"] * cleaned["tshirt_price"]
    summary = cleaned.groupby("customer_id", as_index=True).agg(
        order_date=("order_date", "min"),
        total=("total", "sum"),
        pages_visited=("pages_visited", "mean"),
        order_id=("order_id", "count"),
    )
    return summary


def prepare_customers(
    customers: pd.DataFrame, orders_by_customer: pd.DataFrame, max_age: int = 100
) -> pd.DataFrame:
    """Join customerID to the aggregate index using an inner join.

    Birthdate strings follow the baseline year/month/day slash convention.
    Preserve the baseline's three birthdate repairs and fractional age in days
    divided by 365. Keep rows with age_first_order strictly below max_age.
    Parse browser and os from user_agent. Drop birthdate, order_date, order_id,
    ip_address, and user_agent. Keep total for the separate labeling step.
    """
    joined = customers.merge(
        orders_by_customer, left_on="customerID", right_index=True, how="inner"
    )
    normalized_birth = joined["birthdate"].replace(LEAP_YEAR_FIXES)
    joined["birthdate"] = pd.to_datetime(normalized_birth, format="%Y/%m/%d")
    joined["age_first_order"] = (joined["order_date"] - joined["birthdate"]).dt.days / 365.0

    filtered = joined.loc[joined["age_first_order"] < max_age].copy()
    ua_objects = [user_agents.parse(ua) for ua in filtered["user_agent"]]
    filtered["browser"] = [ua.browser.family for ua in ua_objects]
    filtered["os"] = [ua.os.family for ua in ua_objects]
    return filtered.drop(columns=UNUSED_CUSTOMER_COLS)


def label_customers(customers_prepared: pd.DataFrame, threshold: int = 300) -> pd.DataFrame:
    """Add boolean high_revenue using strictly greater than threshold; drop total."""
    labeled = customers_prepared.assign(
        high_revenue=customers_prepared["total"] > threshold
    )
    return labeled.drop(columns=["total"])