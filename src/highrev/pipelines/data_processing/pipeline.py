"""Exercise 1: raw tables to labeled customers."""

from kedro.pipeline import Pipeline, node

from highrev import data_processing as dp


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=dp.prepare_orders,
                inputs="orders",
                outputs="orders_by_customer",
                name="aggregate_customer_orders",
            ),
            node(
                func=dp.prepare_customers,
                inputs=["customers", "orders_by_customer", "params:max_age"],
                outputs="customers_prepared",
                name="join_and_clean_customers",
            ),
            node(
                func=dp.label_customers,
                inputs=["customers_prepared", "params:high_revenue_threshold"],
                outputs="customers_labeled",
                name="create_revenue_labels",
            ),
        ]
    )