import numpy as np
import pytest

from app.models.features import ARTICLE_FEATURES, CUSTOMER_FEATURES, HybridFeatures
from app.models.mappings import DenseMappings
from app.training.metrics import evaluate_sampled_ranking, hit_rate_at_k, ndcg_at_k, recall_at_k
from app.training.sampling import InteractionIndex


class ItemScoreModel:
    def predict(self, features, **kwargs):
        del kwargs
        return np.asarray(features["article_index"], dtype=float).reshape(-1, 1) / 10


def hybrid_features():
    mappings = DenseMappings.build([1], range(10))
    customers = [{"customer_key": 1, **{name: "known" for name in CUSTOMER_FEATURES}}]
    articles = [
        {"article_key": key, **{name: "known" for name in ARTICLE_FEATURES}}
        for key in range(10)
    ]
    return HybridFeatures.build(mappings, customers, articles)


def test_ranking_metric_formulas():
    relevant = {2, 5}
    ranked = [2, 9, 5]
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert hit_rate_at_k(relevant, ranked, 2) == 1
    assert 0 < ndcg_at_k(relevant, ranked, 3) <= 1


def test_sampled_ranking_is_reproducible_and_identifies_top_positive():
    interactions = InteractionIndex.from_pairs([0], [0], num_items=10)
    first = evaluate_sampled_ranking(
        ItemScoreModel(), [0], [9], interactions, hybrid_features(), k=1, negative_candidates=8, seed=4
    )
    second = evaluate_sampled_ranking(
        ItemScoreModel(), [0], [9], interactions, hybrid_features(), k=1, negative_candidates=8, seed=4
    )

    assert first == second
    assert first.evaluated_positives == 1
    assert first.recall_at_k == first.hit_rate_at_k == first.ndcg_at_k == 1
    assert "sampled" in first.candidate_strategy


def test_sampled_ranking_requires_at_least_k_negatives():
    interactions = InteractionIndex.from_pairs([0], [0], num_items=10)

    with pytest.raises(ValueError, match="at least k"):
        evaluate_sampled_ranking(
            ItemScoreModel(), [0], [9], interactions, hybrid_features(), k=10, negative_candidates=9
        )
