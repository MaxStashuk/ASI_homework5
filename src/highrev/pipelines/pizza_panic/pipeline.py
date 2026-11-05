"""Wire the supplied functions into the Pizza Panic Kedro graph."""

from kedro.pipeline import Pipeline, node

from .nodes import prepare_features, search_candidates, split_data


def create_pipeline(**kwargs) -> Pipeline:
    """Build three nodes using the contracts in INTEGRATION_TEMPLATE.md.

    Raw catalog inputs -> pizza_features -> pizza_split -> results + models.
    Pass configuration with params: inputs; do not call the functions here.
    Give every node a descriptive unique name.
    """
    return Pipeline(
        nodes=[
            node(
                func=prepare_features,
                inputs=["pizza_orders", "pizza_restaurants"],
                outputs="pizza_features",
                name="join_pizza_features_node",
            ),
            node(
                func=split_data,
                inputs=["pizza_features", "params:pizza_settings"],
                outputs="pizza_split",
                name="split_pizza_data_node",
            ),
            node(
                func=search_candidates,
                inputs=["pizza_split", "params:pizza_settings", "params:pizza_selection"],
                outputs=["pizza_results", "pizza_models"],
                name="search_pizza_candidates_node",
            ),
        ]
    )


