"""
This is a boilerplate pipeline 'supervised_model'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import data_preprocessing, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=data_preprocessing,
                inputs=[
                    "pokemon_raw",
                ],
                outputs=[
                    "boosting.X_dataset",
                    "boosting.y_dataset"
                ],
                name="data_preprocessing",
            ),
        node(
                func=split_data,
                inputs=[
                    "boosting.X_dataset",
                    "boosting.y_dataset",
                    "params:model_options"
                ],
                outputs=[
                    "boosting.X_train",
                    "boosting.X_test",
                    "boosting.y_train",
                    "boosting.y_test"
                ],
                name="split_data",
            ),
        node(
                func=train_model,
                inputs=[
                    "boosting.X_train",
                    "boosting.y_train",
                    "params:model_options"
                ],
                outputs= "boosting.model",
                name="train_model",
            ),
        node(
                func=evaluate_model,
                inputs=[
                    "boosting.model",
                    "boosting.X_test",
                    "boosting.y_test"
                ],
                outputs= "boosting.classification_report",
                name="evaluate_model",
            ),
    ])
