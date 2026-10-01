"""H&M multi-input hybrid recommendation model."""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Unit tests can exercise non-TensorFlow modules.
    tf = None  # type: ignore[assignment]


class TensorFlowUnavailableError(RuntimeError):
    pass


def tensorflow_available() -> bool:
    return tf is not None


def require_tensorflow() -> Any:
    if tf is None:
        raise TensorFlowUnavailableError(
            "TensorFlow is required for model construction and training"
        )
    return tf


def build_hybrid_model(
    num_customers: int,
    num_articles: int,
    feature_cardinalities: dict[str, int],
    *,
    embedding_dim: int = 32,
    dense_1: int = 64,
    dense_2: int = 32,
    learning_rate: float = 0.001,
) -> Any:
    """Build two towers from identity, customer profile, and article metadata."""
    tensorflow = require_tensorflow()
    if min(num_customers, num_articles, embedding_dim, dense_1, dense_2) <= 0:
        raise ValueError("Model dimensions must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    from .features import ARTICLE_FEATURES, CUSTOMER_FEATURES, FEATURE_NAMES

    if set(feature_cardinalities) != set(FEATURE_NAMES):
        raise ValueError("Feature cardinalities do not match H&M hybrid schema")
    if any(value <= 0 for value in feature_cardinalities.values()):
        raise ValueError("Feature cardinalities must be positive")

    customer_input = tensorflow.keras.Input(shape=(), dtype="int32", name="customer_index")
    article_input = tensorflow.keras.Input(shape=(), dtype="int32", name="article_index")
    inputs = {"customer_index": customer_input, "article_index": article_input}
    customer_parts = [
        tensorflow.keras.layers.Embedding(
            num_customers, embedding_dim, name="customer_embedding"
        )(customer_input)
    ]
    article_parts = [
        tensorflow.keras.layers.Embedding(
            num_articles, embedding_dim, name="article_embedding"
        )(article_input)
    ]
    for name in FEATURE_NAMES:
        input_value = tensorflow.keras.Input(shape=(), dtype="int32", name=name)
        inputs[name] = input_value
        cardinality = feature_cardinalities[name]
        width = min(8, max(2, int(cardinality**0.25) + 1))
        encoded = tensorflow.keras.layers.Embedding(
            cardinality, width, name=f"{name}_embedding"
        )(input_value)
        (customer_parts if name in CUSTOMER_FEATURES else article_parts).append(encoded)
    customer_joined = tensorflow.keras.layers.Concatenate(name="customer_features")(
        customer_parts
    )
    article_joined = tensorflow.keras.layers.Concatenate(name="article_features")(
        article_parts
    )
    customer_tower = tensorflow.keras.layers.Dense(
        embedding_dim, activation="relu", name="customer_tower"
    )(customer_joined)
    article_tower = tensorflow.keras.layers.Dense(
        embedding_dim, activation="relu", name="article_tower"
    )(article_joined)
    affinity = tensorflow.keras.layers.Dot(axes=1, name="tower_affinity")(
        [customer_tower, article_tower]
    )
    joined = tensorflow.keras.layers.Concatenate(name="hybrid_concatenation")(
        [customer_tower, article_tower, affinity]
    )
    first_hidden = tensorflow.keras.layers.Dense(
        dense_1, activation="relu", name="dense_64"
    )(joined)
    second_hidden = tensorflow.keras.layers.Dense(
        dense_2, activation="relu", name="dense_32"
    )(first_hidden)
    score = tensorflow.keras.layers.Dense(1, activation="sigmoid", name="score")(
        second_hidden
    )
    model = tensorflow.keras.Model(
        inputs=inputs,
        outputs=score,
        name="hm_hybrid_recommender",
    )
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tensorflow.keras.losses.BinaryCrossentropy(),
        metrics=[tensorflow.keras.metrics.AUC(name="auc")],
    )
    return model
