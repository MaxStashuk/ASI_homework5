"""Exercise 1: labeled customers to model and metrics."""

from kedro.pipeline import Pipeline, node

from highrev import modeling as mdl


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=mdl.split_data,
                inputs=["customers_labeled", "params:test_size", "params:random_state"],
                outputs=["train_set", "test_set"],
                name="partition_train_test",
            ),
            node(
                func=mdl.train_model,
                inputs=["train_set", "params:model_options"],
                outputs="classifier",
                name="fit_revenue_classifier",
            ),
            node(
                func=mdl.evaluate_model,
                inputs=["classifier", "test_set"],
                outputs="metrics",
                name="compute_test_metrics",
            ),
        ]
    )