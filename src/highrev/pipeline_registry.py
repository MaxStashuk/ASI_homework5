"""Exercise 1: give each pipeline a name that the command line can select."""

from kedro.pipeline import Pipeline

from highrev.pipelines.data_processing.pipeline import create_pipeline as build_data_processing_pipeline
from highrev.pipelines.modeling.pipeline import create_pipeline as build_modeling_pipeline
from highrev.pipelines.pizza_panic.pipeline import create_pipeline as build_pizza_panic_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    dp_pipe = build_data_processing_pipeline()
    mdl_pipe = build_modeling_pipeline()
    pizza_pipe = build_pizza_panic_pipeline()
    return {
        "__default__": dp_pipe + mdl_pipe,
        "data_processing": dp_pipe,
        "modeling": mdl_pipe,
        "pizza_panic": pizza_pipe,
    }