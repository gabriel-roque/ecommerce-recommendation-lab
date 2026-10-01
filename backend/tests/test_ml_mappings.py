import numpy as np

from app.models.mappings import DenseMappings


def test_dense_mappings_are_sorted_reversible_and_persisted(tmp_path):
    mappings = DenseMappings.build([99, 7, 99, 42], [500, 2, 100, 2])

    assert mappings.customer_keys.tolist() == [7, 42, 99]
    assert mappings.article_keys.tolist() == [2, 100, 500]
    assert mappings.customer_indices([99, 7, 123]).tolist() == [2, 0, -1]
    assert mappings.article_indices([500, 2, 8]).tolist() == [2, 0, -1]

    mappings.save(tmp_path)
    restored = DenseMappings.load(tmp_path)
    assert restored.customer_keys.tolist() == mappings.customer_keys.tolist()
    assert restored.article_keys.tolist() == mappings.article_keys.tolist()


def test_dense_mappings_reject_non_integral_ids():
    try:
        DenseMappings.build(np.asarray([1.5]), [1])
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("Non-integral ID was accepted")
