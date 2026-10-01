import importlib.util

import pytest

from app.models.features import FEATURE_NAMES
from app.models.ncf import TensorFlowUnavailableError, build_hybrid_model


CARDINALITIES = {name: 3 for name in FEATURE_NAMES}


def test_model_module_imports_without_tensorflow():
    if importlib.util.find_spec("tensorflow") is None:
        with pytest.raises(TensorFlowUnavailableError):
            build_hybrid_model(2, 3, CARDINALITIES)


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed")
def test_hybrid_model_uses_customer_and_article_metadata():
    import numpy as np

    model = build_hybrid_model(
        3, 4, CARDINALITIES, embedding_dim=2, dense_1=5, dense_2=3
    )
    assert model.get_layer("customer_embedding").output_dim == 2
    assert model.get_layer("article_embedding").output_dim == 2
    assert model.get_layer("age_band_embedding").input_dim == 3
    assert model.get_layer("product_group_name_embedding").input_dim == 3
    assert model.get_layer("dense_64").units == 5
    assert model.get_layer("dense_32").units == 3
    inputs = {name: np.asarray([1]) for name in FEATURE_NAMES}
    inputs.update(customer_index=np.asarray([0]), article_index=np.asarray([1]))
    score = model.predict(inputs, verbose=0)
    assert score.shape == (1, 1)
    assert 0 <= float(score[0, 0]) <= 1
